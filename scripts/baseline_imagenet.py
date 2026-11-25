"""Baseline without fine-tuning: ImageNet features of the backbone, L2-normalized, compared by cosine.

    python scripts/baseline_imagenet.py        # writes runs/imagenet_<backbone>/seed0/emb_<dataset>.npz
    python scripts/baseline_imagenet.py --set model.backbone=resnet50 --set model.weights=IMAGENET1K_V2

The pooled backbone output (768-d for ConvNeXt-Tiny, 2048-d for ResNet-50) replaces the trained 256-d projection. Same preprocessed images
and same pair files as the fine-tuned runs; evaluate.py then scores the .npz files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.config import add_config_args, load_config  # noqa: E402
from fpverif.data import index_dataset  # noqa: E402
from fpverif.seed import seed_everything  # noqa: E402
from model import build_backbone  # noqa: E402
from train import embed_frame, pick_device  # noqa: E402


class Pooled(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x).flatten(1), dim=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(ROOT / args.config, args.set)
    m = cfg["model"]
    seed_everything(cfg["seed"], cfg["train"]["deterministic"])
    device = pick_device(cfg["train"]["device"])
    backbone, _ = build_backbone(m["backbone"], m["weights"], m["in_channels"])
    model = Pooled(backbone).to(device).eval()
    out = ROOT / "runs" / f"imagenet_{m['backbone']}" / "seed0"
    out.mkdir(parents=True, exist_ok=True)
    pp = cfg["preprocess"]
    for dataset in [cfg["main_dataset"], cfg["second_dataset"]]:
        root = Path(pp["output_dir"]) / dataset / pp["normalization"]
        frame = index_dataset(root)
        frame = frame[frame["path"].str.endswith(".png")]
        emb = embed_frame(model, frame, root, device, cfg["eval"]["batch_size"])
        paths = np.array(list(emb))
        np.savez(out / f"emb_{dataset}.npz", paths=paths, embeddings=np.stack([emb[p] for p in paths]).astype(np.float32))
        print(dataset, len(paths), emb[paths[0]].shape)


if __name__ == "__main__":
    main()
