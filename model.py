"""Embedding network: torchvision backbone -> linear projection -> L2-normalized embedding, plus ArcFace head.

    python model.py                       # prints parameter count and state_dict size for the configured model
    python model.py --set model.backbone=convnext_tiny
"""
from __future__ import annotations

import argparse
import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from fpverif.config import add_config_args, load_config

BACKBONES = {
    "resnet18": ("resnet18", "ResNet18_Weights", "fc"),
    "resnet50": ("resnet50", "ResNet50_Weights", "fc"),
    "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights", "classifier"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", "classifier"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "classifier"),
}


def _first_conv(module: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            parent_name, _, attr = name.rpartition(".")
            parent = module.get_submodule(parent_name) if parent_name else module
            return parent, attr, m
    raise RuntimeError("no Conv2d found")


def adapt_in_channels(backbone: nn.Module, in_channels: int) -> None:
    """Sum the pretrained RGB filters of the first conv into `in_channels` channels, keeping the features."""
    parent, attr, conv = _first_conv(backbone)
    if conv.in_channels == in_channels:
        return
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                    conv.dilation, conv.groups, conv.bias is not None)
    with torch.no_grad():
        w = conv.weight.sum(dim=1, keepdim=True)
        new.weight.copy_(w.repeat(1, in_channels, 1, 1) / in_channels)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    setattr(parent, attr, new)


def build_backbone(name: str, weights: str | None, in_channels: int) -> tuple[nn.Module, int]:
    fn_name, enum_name, head_attr = BACKBONES[name]
    enum = getattr(torchvision.models, enum_name)
    w = getattr(enum, weights) if weights else None
    net = getattr(torchvision.models, fn_name)(weights=w)
    head = getattr(net, head_attr)
    feat_dim = head.in_features if isinstance(head, nn.Linear) else [m for m in head if isinstance(m, nn.Linear)][0].in_features
    if isinstance(head, nn.Sequential) and any("LayerNorm" in type(m).__name__ for m in head):
        # ConvNeXt's pretrained LayerNorm after pooling belongs to the feature path; drop only the Linear.
        head[-1] = nn.Identity()
    else:
        setattr(net, head_attr, nn.Identity())
    adapt_in_channels(net, in_channels)
    return net, feat_dim


class EmbeddingNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, embedding_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat_dim, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.backbone(x).flatten(1)), dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin softmax (Deng et al. 2019). Input embeddings must be L2-normalized."""

    def __init__(self, embedding_dim: int, n_classes: int, margin: float, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.m, self.s = margin, scale

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = F.linear(emb, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        # Keep the logit monotonic when theta + m passes pi.
        target = torch.where(theta + self.m < math.pi, target, cos - self.m * math.sin(self.m))
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        return self.s * torch.where(onehot, target, cos)


def build_model(cfg: dict) -> EmbeddingNet:
    m = cfg["model"]
    backbone, feat_dim = build_backbone(m["backbone"], m["weights"], m["in_channels"])
    return EmbeddingNet(backbone, feat_dim, m["embedding_dim"])


def freeze_until(model: EmbeddingNet, stage: str | None) -> list[nn.Module]:
    """Freeze backbone parameters up to and including the named child module (ResNet: layer1..layer4).
    Returns the frozen modules so the caller can keep them in eval mode with `train_mode`."""
    if not stage:
        return []
    stages = _stages(model.backbone, stage)
    names = [name for name, _ in stages]
    if stage not in names:
        raise ValueError(f"freeze_until={stage!r} is not a backbone stage; options: {names}")
    frozen, done = [], False
    for name, child in stages:
        for p in child.parameters():
            p.requires_grad = done
        if not done:
            frozen.append(child)
        if name == stage:
            done = True
    return frozen


def _stages(backbone: nn.Module, stage: str) -> list[tuple[str, nn.Module]]:
    """Ordered top-level children; a Sequential named in a dotted stage ("features.3") is expanded into its blocks."""
    out = []
    for name, child in backbone.named_children():
        if stage.startswith(name + ".") and isinstance(child, nn.Sequential):
            out += [(f"{name}.{i}", block) for i, block in child.named_children()]
        else:
            out.append((name, child))
    return out


def train_mode(model: nn.Module, frozen: list[nn.Module]) -> None:
    """model.train() with the frozen stages in eval mode, so their BatchNorm running statistics stay fixed."""
    model.train()
    for m in frozen:
        m.eval()


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def state_dict_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    model = build_model(cfg)
    freeze_until(model, cfg["train"]["freeze_until"])
    print(f"backbone={cfg['model']['backbone']} weights={cfg['model']['weights']} dim={cfg['model']['embedding_dim']}")
    print(f"params total={count_params(model):,} trainable={count_params(model, True):,} "
          f"state_dict={state_dict_bytes(model) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
