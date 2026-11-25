"""ROI segmentation and quality control -> grayscale -> normalization -> resize of the square around the ROI.

    python preprocess.py --dataset fvc2002_db1
    python preprocess.py --dataset fvc2002_db1 --set preprocess.normalization=none   # ablation arm

Writes data/processed/<dataset>/<normalization>/<finger>_<impression>.png and qc.csv next to them.
A sample whose usable area is below preprocess.min_foreground_mm2 is marked acquired=false (FTA)
and still written, so downstream code can decide how to count it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from fpverif.config import add_config_args, dataset_cfg, load_config
from fpverif.data import dataset_index
from fpverif.imageops import hong_normalize, roi_square, segment_foreground


def normalize(img: np.ndarray, mask: np.ndarray, pp: dict) -> np.ndarray:
    norm = pp["normalization"]
    if norm == "none":
        return img.astype(np.float32)
    if norm == "hong":
        return hong_normalize(img, mask, pp["hong_mean"], pp["hong_var"])
    raise ValueError(f"unknown normalization {norm!r}")


def preprocess_image(img: np.ndarray, pp: dict, dpi: float) -> tuple[np.ndarray, dict]:
    """Apply the standard pipeline to one grayscale uint8 image. Returns (uint8 image, qc dict).
    `dpi` only converts the ROI area to mm2 for quality control."""
    mask = segment_foreground(img, pp["block_size"], pp["fg_std_threshold"])
    fg_ratio = float(mask.mean())
    fg_mm2 = float(mask.sum()) / (dpi / 25.4) ** 2
    acquired = fg_mm2 >= pp["min_foreground_mm2"]
    if not mask.any():
        mask = np.ones_like(img, dtype=bool)

    fill = float(pp["background_fill"])
    out = np.where(mask, normalize(img, mask, pp), fill).astype(np.float32)
    out = roi_square(out, mask, fill)
    size = pp["output_size"]
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_CUBIC)
    qc = {"foreground_ratio": fg_ratio, "foreground_mm2": fg_mm2, "acquired": acquired,
          "sharpness": float(cv2.Laplacian(img, cv2.CV_32F).var()),
          "mean": float(img.mean()), "std": float(img.std())}
    return np.clip(out, 0, 255).astype(np.uint8), qc


def run(cfg: dict, dataset: str, src_root: Path | None = None, out_root: Path | None = None,
        index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Preprocess every image of `index` (default: the whole configured dataset). Rows need path, finger,
    impression and optionally dpi; without a dpi column the dataset's configured dpi is used."""
    dataset_cfg(cfg, dataset)
    pp = cfg["preprocess"]
    full_index, default_root = dataset_index(cfg, dataset)
    src_root = Path(src_root or default_root)
    out_root = Path(out_root or Path(pp["output_dir"]) / dataset / pp["normalization"])
    out_root.mkdir(parents=True, exist_ok=True)
    index = full_index if index is None else index
    if "dpi" not in index.columns:
        index = index.merge(full_index[["path", "dpi"]], on="path", how="left")
    rows = []
    for row in tqdm(index.itertuples(index=False), total=len(index), desc=f"preprocess {dataset}"):
        img = cv2.imread(str(src_root / row.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src_root / row.path)
        out, qc = preprocess_image(img, pp, row.dpi)
        name = f"{row.finger}_{row.impression}.png"
        cv2.imwrite(str(out_root / name), out)
        rows.append({"path": name, "source": row.path, "finger": row.finger, "impression": row.impression, **qc})
    qc_frame = pd.DataFrame(rows)
    qc_frame.to_csv(out_root / "qc.csv", index=False)
    return qc_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    qc = run(cfg, args.dataset)
    print(f"{len(qc)} images, FTA = {1 - qc['acquired'].mean():.4f}, "
          f"mean foreground ratio = {qc['foreground_ratio'].mean():.3f}")


if __name__ == "__main__":
    main()
