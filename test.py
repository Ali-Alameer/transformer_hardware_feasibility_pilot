import os
import glob
import math
from typing import List

import torch
from collections import OrderedDict
import av
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from video_swin_transformer import SwinTransformer3D


def decode_video_pyav(path: str) -> torch.Tensor:
    """Decode a video file into a uint8 tensor [T, H, W, C] (RGB)."""
    container = av.open(path)
    frames: List[torch.Tensor] = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(arr))
    if len(frames) == 0:
        # fallback to at least one black frame
        frames = [torch.zeros((224, 224, 3), dtype=torch.uint8)]
    video = torch.stack(frames, dim=0)
    return video  # [T, H, W, C]


def resize_shorter_side(video: torch.Tensor, side_size: int) -> torch.Tensor:
    # video: [T, H, W, C]
    t, h, w, c = video.shape
    if h < w:
        new_h = side_size
        new_w = int(w * side_size / h)
    else:
        new_w = side_size
        new_h = int(h * side_size / w)
    frames = []
    for i in range(t):
        frames.append(TF.resize(video[i], [new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR))
    return torch.stack(frames, dim=0)


def center_crop(video: torch.Tensor, size: int) -> torch.Tensor:
    t, h, w, c = video.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    frames = []
    for i in range(t):
        frames.append(video[i, top:top+size, left:left+size, :])
    return torch.stack(frames, dim=0)


def temporal_sample_uniform(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    t = video.shape[0]
    if t == num_frames:
        return video
    if t < num_frames:
        reps = math.ceil(num_frames / t)
        return video.repeat(reps, 1, 1, 1)[:num_frames]
    # t > num_frames: uniform indices
    idx = torch.linspace(0, t - 1, steps=num_frames).round().long()
    return video.index_select(0, idx)


def preprocess_for_swin(video: torch.Tensor, num_frames: int = 32, side: int = 224) -> torch.Tensor:
    # input: uint8 [T, H, W, C]; output: float [1, 3, T, H, W]
    video = resize_shorter_side(video, 256)
    video = temporal_sample_uniform(video, num_frames)
    video = center_crop(video, side)
    video = video.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    video = (video - mean) / std
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    return video


def load_pretrained_backbone() -> SwinTransformer3D:
    model = SwinTransformer3D(embed_dim=128,
                              depths=[2, 2, 18, 2],
                              num_heads=[4, 8, 16, 32],
                              patch_size=(2, 4, 4),
                              window_size=(16, 7, 7),
                              drop_path_rate=0.4,
                              patch_norm=True)
    ckpt = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
    for k, v in state_dict.items():
        if 'backbone' in k:
            name = k[9:]
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_pretrained_backbone().to(device)

    data_dir = './data'
    exts = ('*.mp4', '*.avi', '*.mov', '*.mkv')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    files = sorted(files)
    if not files:
        print('No videos found in ./data')
        return

    with torch.no_grad():
        for path in files:
            try:
                raw = decode_video_pyav(path)
                inp = preprocess_for_swin(raw, num_frames=32, side=224).to(device)
                feats = model(inp)  # [1, C, T', H', W']
                pooled = feats.mean(dim=(2, 3, 4))  # [1, C]
                print(f'{os.path.basename(path)} -> feats {tuple(feats.shape)}, pooled {tuple(pooled.shape)}')
            except Exception as e:
                print(f'Failed on {path}: {e}')


if __name__ == '__main__':
    main()