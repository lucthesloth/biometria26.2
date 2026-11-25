"""Embed every processed image of a dataset with a frozen checkpoint and time the network.

    python extract_embeddings.py --dataset fvc2004_db1 --checkpoint runs/<run>/seed42/best.pt --out runs/<run>/seed42/emb_fvc2004_db1.npz

Writes an .npz with `paths` (str array, e.g. 12_3.png) and `embeddings` (N x D float32), plus <out stem>_timing.json
beside it with median/std latency per image at batch size 1 and the hardware string.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fpverif.config import add_config_args, load_config
from fpverif.data import index_dataset
from fpverif.seed import seed_everything
from model import build_model, count_params, state_dict_bytes
from train import _hardware, embed_frame, pick_device


def load_model(cfg: dict, checkpoint: str | Path, device: torch.device):
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def time_inference(model, size: int, device: torch.device, repeats: int, warmup: int) -> dict:
    x = torch.randn(1, 1, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times = np.array(times) * 1000
    return {"median_ms": float(np.median(times)), "mean_ms": float(times.mean()), "std_ms": float(times.std()),
            "repeats": repeats, "batch_size": 1, "input_size": size, "device": str(device),
            "hardware": _hardware(device), "torch": torch.__version__}


def extract(cfg: dict, dataset: str, checkpoint: Path, out: Path, image_root: Path | None = None) -> Path:
    device = pick_device(cfg["train"]["device"])
    pp = cfg["preprocess"]
    image_root = Path(image_root or Path(pp["output_dir"]) / dataset / pp["normalization"])
    frame = index_dataset(image_root)
    frame = frame[frame["path"].str.endswith(".png")]
    model = load_model(cfg, checkpoint, device)
    emb = embed_frame(model, frame, image_root, device, cfg["eval"]["batch_size"])
    paths = np.array(list(emb))
    np.savez(out, paths=paths, embeddings=np.stack([emb[p] for p in paths]).astype(np.float32))
    timing = time_inference(model, pp["output_size"], device, cfg["eval"]["timing_repeats"], cfg["eval"]["timing_warmup"])
    timing.update({"params_total": count_params(model), "model_size_bytes": state_dict_bytes(model),
                   "checkpoint_bytes": Path(checkpoint).stat().st_size, "n_images": len(paths)})
    out.with_name(out.stem + "_timing.json").write_text(json.dumps(timing, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root", default=None, help="override processed image dir (used by robustness.py)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"], cfg["train"]["deterministic"])
    out = extract(cfg, args.dataset, Path(args.checkpoint), Path(args.out), args.image_root)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
