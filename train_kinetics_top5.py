import os
import math
import random
import argparse
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from datasets import load_dataset
from datasets import Video as HFVideo
import av
from tqdm import tqdm
import fsspec
import io
import tempfile
import platform

from video_swin_transformer import SwinTransformer3D
import torchvision.io as tvio

# set at runtime from CLI
DATASET_VARIANT = "full"  # or "tiny"
DECODER = "pyav"  # or "torchvision"


class VideoTransforms:
    def __init__(self, side_size: int = 224, num_frames: int = 16, is_train: bool = True):
        self.side_size = side_size
        self.num_frames = num_frames
        self.is_train = is_train

    def _resize_shorter_side(self, video: torch.Tensor, side_size: int) -> torch.Tensor:
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

    def _random_crop(self, video: torch.Tensor, size: int) -> torch.Tensor:
        t, h, w, c = video.shape
        if h == size and w == size:
            return video
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        frames = []
        for i in range(t):
            frames.append(video[i, top:top+size, left:left+size, :])
        return torch.stack(frames, dim=0)

    def _center_crop(self, video: torch.Tensor, size: int) -> torch.Tensor:
        t, h, w, c = video.shape
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        frames = []
        for i in range(t):
            frames.append(video[i, top:top+size, left:left+size, :])
        return torch.stack(frames, dim=0)

    def _temporal_sample(self, video: torch.Tensor, num_frames: int, is_train: bool) -> torch.Tensor:
        # video: [T, H, W, C]
        t = video.shape[0]
        if t == num_frames:
            return video
        if t < num_frames:
            # loop pad
            reps = math.ceil(num_frames / t)
            video = video.repeat(reps, 1, 1, 1)[:num_frames]
            return video
        # uniform sample with random offset during training
        if is_train:
            start_max = max(0, t - num_frames)
            start = random.randint(0, start_max)
            return video[start:start+num_frames]
        else:
            # uniform spread
            idx = torch.linspace(0, t - 1, steps=num_frames).round().long()
            return video.index_select(0, idx)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        # expect uint8 [T, H, W, C]
        video = self._resize_shorter_side(video, 256)
        if self.is_train:
            video = self._temporal_sample(video, self.num_frames, True)
            video = self._random_crop(video, self.side_size)
            if random.random() < 0.5:
                video = torch.flip(video, dims=[2])  # horizontal flip W axis
        else:
            video = self._temporal_sample(video, self.num_frames, False)
            video = self._center_crop(video, self.side_size)

        # to float and normalize
        video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        video = video.float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        video = (video - mean) / std
        # to [C, T, H, W]
        video = video.permute(1, 0, 2, 3)
        return video


def decode_video(sample) -> Tuple[torch.Tensor, int]:
    # The dataset provides 'video' field as path-like or bytes. Use PyAV to decode.
    # Return frames as uint8 tensor [T, H, W, C] and label int.
    video_obj = sample["video"]
    label = int(sample["label"]) if "label" in sample else int(sample.get("class", 0))

    # HuggingFace datasets often expose local file paths or zip-backed URIs
    path = None
    if isinstance(video_obj, str):
        path = video_obj
    elif isinstance(video_obj, dict) and "path" in video_obj:
        path = video_obj["path"]

    # If tiny variant requested, rewrite the zip filename
    if path and path.startswith("zip://") and DATASET_VARIANT == "tiny" and "kinetics_top5.zip" in path:
        path = path.replace("kinetics_top5.zip", "kinetics_top5_tiny.zip")

    # Decoder selection
    if DECODER == "torchvision":
        try:
            # Ensure we have a filesystem path for torchvision
            tmp_path = None
            if path and path.startswith("zip://"):
                with fsspec.open(path, "rb") as fsrc:
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                        tf.write(fsrc.read())
                        tf.flush()
                        tmp_path = tf.name
                        video, _, _ = tvio.read_video(tmp_path, pts_unit="sec")  # [T,H,W,C] uint8
            else:
                if isinstance(video_obj, (bytes, bytearray)):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tf:
                        tf.write(video_obj if isinstance(video_obj, (bytes, bytearray)) else bytes(video_obj))
                        tf.flush()
                        tmp_path = tf.name
                        video, _, _ = tvio.read_video(tmp_path, pts_unit="sec")
                else:
                    # video_obj or path should be a string path
                    target_path = path if path is not None else str(video_obj)
                    video, _, _ = tvio.read_video(target_path, pts_unit="sec")
            frames = [frame for frame in video]
        except Exception:
            frames = []
    else:
        container = None
        try:
            if path and path.startswith("zip://"):
                # Use a spooled temp file to keep memory low; spills to disk if large
                with fsspec.open(path, "rb") as fsrc:
                    spooled = tempfile.SpooledTemporaryFile(max_size=8_000_000)
                    while True:
                        chunk = fsrc.read(1_048_576)
                        if not chunk:
                            break
                        spooled.write(chunk)
                    spooled.seek(0)
                    container = av.open(spooled)
            else:
                target = path if path is not None else video_obj
                if isinstance(target, (bytes, bytearray)):
                    container = av.open(io.BytesIO(target))
                else:
                    container = av.open(target)
        except Exception:
            # If tiny zip path failed, try the original full path
            if DATASET_VARIANT == "tiny":
                orig_path = None
                if isinstance(video_obj, str):
                    orig_path = video_obj.replace("kinetics_top5_tiny.zip", "kinetics_top5.zip")
                elif isinstance(video_obj, dict) and "path" in video_obj and isinstance(video_obj["path"], str):
                    orig_path = video_obj["path"].replace("kinetics_top5_tiny.zip", "kinetics_top5.zip")
                if orig_path and orig_path.startswith("zip://"):
                    try:
                        with fsspec.open(orig_path, "rb") as fsrc:
                            data = fsrc.read()
                        container = av.open(io.BytesIO(data))
                    except Exception:
                        container = None
            if container is None:
                if isinstance(video_obj, dict) and "bytes" in video_obj and video_obj["bytes"] is not None:
                    container = av.open(io.BytesIO(video_obj["bytes"]))
                else:
                    # return a dummy black video if completely unreadable
                    dummy = torch.zeros((16, 224, 224, 3), dtype=torch.uint8)
                    return dummy, label

        frames = []
        try:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")  # H, W, C uint8
                frames.append(torch.from_numpy(img))
        except Exception:
            # decoding error, fallback to empty frames
            frames = []
    if len(frames) == 0:
        # fallback to at least one frame of zeros
        frames = [torch.zeros((224, 224, 3), dtype=torch.uint8)]
    video = torch.stack(frames, dim=0)  # [T, H, W, C]
    return video, label


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, num_frames: int = 16, side_size: int = 224, limit_samples: Optional[int] = None):
        self.dataset = load_dataset("innat/KineticsTop5", split="train")
        # Disable HF auto-decoding that requires 'torchcodec'; we'll decode via PyAV
        self.dataset = self.dataset.cast_column("video", HFVideo(decode=False))
        # Dataset only has a single split; we will make a deterministic train/val split
        indices = list(range(len(self.dataset)))
        random.Random(42).shuffle(indices)
        val_ratio = 0.1
        val_count = int(len(indices) * val_ratio)
        if split == "train":
            self.indices = indices[val_count:]
            self.transforms = VideoTransforms(side_size=side_size, num_frames=num_frames, is_train=True)
        else:
            self.indices = indices[:val_count]
            self.transforms = VideoTransforms(side_size=side_size, num_frames=num_frames, is_train=False)
        if limit_samples is not None:
            self.indices = self.indices[:int(limit_samples)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.dataset[int(real_idx)]
        video, label = decode_video(sample)
        video = self.transforms(video)  # [C, T, H, W]
        return video, label


class SwinForVideoClassification(nn.Module):
    def __init__(self, num_classes: int = 5, 
                 backbone_kwargs: Optional[dict] = None, 
                 pretrained_backbone_path: Optional[str] = None):
        super().__init__()
        # Auto-select backbone config: if a base checkpoint is provided, use base config
        if backbone_kwargs is None:
            if pretrained_backbone_path is not None and os.path.isfile(pretrained_backbone_path):
                backbone_kwargs = {
                    "embed_dim": 128,
                    "depths": [2, 2, 18, 2],
                    "num_heads": [4, 8, 16, 32],
                    "patch_size": (2, 4, 4),
                    "window_size": (16, 7, 7),
                    "drop_path_rate": 0.4,
                    "patch_norm": True,
                }
            else:
                backbone_kwargs = {
                    "embed_dim": 96,
                    "depths": [2, 2, 6, 2],
                    "num_heads": [3, 6, 12, 24],
                    "patch_size": (2, 4, 4),
                    "window_size": (8, 7, 7),
                    "drop_path_rate": 0.2,
                    "patch_norm": True,
                }
        self.backbone = SwinTransformer3D(**backbone_kwargs)
        # init from 2D official if provided
        if pretrained_backbone_path is not None and os.path.isfile(pretrained_backbone_path):
            try:
                state = torch.load(pretrained_backbone_path, map_location="cpu")
                state_dict = state.get("state_dict", state.get("model", state))
                new_state = {}
                for k, v in state_dict.items():
                    if k.startswith("backbone."):
                        new_state[k[len("backbone."):]] = v
                missing, unexpected = self.backbone.load_state_dict(new_state, strict=False)
                print(f"Loaded pretrained backbone. missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                print(f"Warning: failed to load pretrained backbone: {e}")

        # classification head: global average over [D, H, W] then linear
        self.norm = nn.LayerNorm(self.backbone.num_features)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        feats = self.backbone(x)  # [B, C, T', H', W']
        feats = feats.mean(dim=(2, 3, 4))  # [B, C]
        feats = self.norm(feats)
        logits = self.head(feats)
        return logits


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pred_labels = preds.argmax(dim=1)
    return (pred_labels == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc="train", leave=False)
    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(videos)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits.detach(), labels) * batch_size
        total_count += batch_size
        pbar.set_postfix({"loss": f"{total_loss/total_count:.4f}", "acc": f"{total_acc/total_count:.3f}"})

    return total_loss / max(1, total_count), total_acc / max(1, total_count)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    criterion = nn.CrossEntropyLoss()

    for videos, labels in tqdm(loader, desc="val", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(videos)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy(logits, labels) * batch_size
        total_count += batch_size

    return total_loss / max(1, total_count), total_acc / max(1, total_count)


def collate_fn(batch):
    videos, labels = zip(*batch)
    videos = torch.stack(videos, dim=0)  # [B, C, T, H, W]
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels


def is_wsl() -> bool:
    try:
        return "microsoft" in platform.release().lower() or "wsl" in platform.version().lower()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=-1, help="-1 auto; 0 for debug on Windows")
    parser.add_argument("--max_train_samples", type=int, default=0, help="limit train samples; 0 means full")
    parser.add_argument("--max_val_samples", type=int, default=0, help="limit val samples; 0 means full")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--dataset_variant", type=str, choices=["full","tiny"], default="full")
    parser.add_argument("--decoder", type=str, choices=["pyav","torchvision"], default="pyav")
    parser.add_argument("--pretrained_backbone", type=str, default="checkpoints/swin_base_patch244_window1677_sthv2.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global DATASET_VARIANT
    DATASET_VARIANT = args.dataset_variant
    global DECODER
    DECODER = args.decoder

    # auto workers: default to 0 on Windows to avoid multiprocessing decode issues
    if args.workers < 0:
        args.workers = 0 if (os.name == "nt" or is_wsl()) else 2

    train_limit = args.max_train_samples if args.max_train_samples > 0 else None
    val_limit = args.max_val_samples if args.max_val_samples > 0 else None

    train_ds = HFDataset(split="train", num_frames=args.num_frames, side_size=args.img_size, limit_samples=train_limit)
    val_ds = HFDataset(split="val", num_frames=args.num_frames, side_size=args.img_size, limit_samples=val_limit)

    pin = (device.type == "cuda")
    dl_kwargs = {"num_workers": args.workers, "pin_memory": pin, "collate_fn": collate_fn, "persistent_workers": False, "timeout": 0}
    if args.workers > 0:
        dl_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **dl_kwargs)

    print(f"config: variant={DATASET_VARIANT} decoder={DECODER} workers={args.workers} batch={args.batch_size} frames={args.num_frames}")

    model = SwinForVideoClassification(num_classes=5, pretrained_backbone_path=args.pretrained_backbone)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        ckpt_path = os.path.join(args.checkpoint_dir, f"swin_kinetics_top5_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_acc": val_acc,
        }, ckpt_path)
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.checkpoint_dir, "swin_kinetics_top5_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path} (acc={best_acc:.3f})")


if __name__ == "__main__":
    main()


