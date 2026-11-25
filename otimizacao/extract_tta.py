"""Embeddings averaged over rotated copies of each image (test-time augmentation).

    python otimizacao/extract_tta.py --dataset fvc2002_db1 --checkpoint runs/.../best.pt --out emb.npz --angles -10 0 10

Each image is rotated by every angle (background fill as in preprocessing), embedded, and the unit embeddings
are averaged and renormalised. Output format matches extract_embeddings.py, so evaluate.py reads it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from extract_embeddings import load_model  # noqa: E402
from fpverif.config import add_config_args, load_config  # noqa: E402
from fpverif.data import index_dataset  # noqa: E402
from train import eval_transform, pick_device  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--angles", type=float, nargs="+", default=[-10.0, 0.0, 10.0])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pp = cfg["preprocess"]
    device = pick_device(cfg["train"]["device"])
    root = Path(pp["output_dir"]) / args.dataset / pp["normalization"]
    frame = index_dataset(root)
    paths = [p for p in frame["path"] if p.endswith(".png")]
    model = load_model(cfg, args.checkpoint, device)
    to_tensor: v2.Compose = eval_transform()
    out = []
    for p in paths:
        img = Image.open(root / p).convert("L")
        batch = torch.stack([to_tensor(TF.rotate(img, a, fill=int(pp["background_fill"]))) for a in args.angles])
        out.append(F.normalize(model(batch.to(device)).float().mean(0), dim=0).cpu().numpy())
    np.savez(args.out, paths=np.array(paths), embeddings=np.stack(out).astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
