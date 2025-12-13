import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryQueue:
    def __init__(self, max_length):
        self.max_length = max_length
        self.queue = []

    def push(self, tokens):
        self.queue.append(tokens.detach())
        if len(self.queue) > self.max_length:
            self.queue.pop(0)

    def get_all(self):
        if not self.queue:
            return None
        return torch.cat(self.queue, dim=0)


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x, mem):
        if mem is None:
            return x
        mem = mem.unsqueeze(0)
        out, _ = self.attn(x, mem, mem)
        return x + out


class MemoryTokenGenerator(nn.Module):
    def __init__(self, dim, num_tokens=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        pooled = x.mean(dim=1)
        t = self.proj(pooled).unsqueeze(1).repeat(1, self.num_tokens, 1)
        return t.reshape(-1, x.size(-1))


class FAMVideoSwin(nn.Module):
    def __init__(self, backbone, mem_length=20, dim=768):
        super().__init__()
        self.backbone = backbone
        self.memory = MemoryQueue(mem_length)
        self.cross = CrossAttention(dim)
        self.mem_token_gen = MemoryTokenGenerator(dim)
        self.cls = nn.Linear(dim, 2)

    def forward(self, clip):
        tokens = self.backbone(clip)
        mem = self.memory.get_all()
        tokens = self.cross(tokens, mem)
        summary = self.mem_token_gen(tokens)
        self.memory.push(summary)
        pooled = tokens.mean(dim=1)
        return self.cls(pooled)


def stream_inference(model, frames, stride=8, clip_size=32):
    out = []
    T = len(frames)
    for i in range(0, T - clip_size, stride):
        clip = frames[i:i+clip_size]
        clip = torch.tensor(clip).permute(0, 3, 1, 2).unsqueeze(0).float()
        out.append(model(clip).softmax(-1))
    return torch.stack(out)


class DummySwin(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.conv = nn.Conv3d(3, dim, 3, padding=1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        f = self.conv(x)
        f = f.mean(dim=[3, 4])
        return f.permute(0, 2, 1)


if __name__ == "__main__":
    model = FAMVideoSwin(DummySwin(), mem_length=25, dim=768)
    fake = torch.randn(500, 128, 128, 3)
    preds = stream_inference(model, fake)
    print(preds.shape)
