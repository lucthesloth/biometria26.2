"""Figure with the preprocessing steps for one image per dataset: raw, block-std map, ROI mask, normalized, network input.

    python scripts/pipeline_figure.py --dataset fvc2002_db1 --sample Db1_b/101_1.tif --out artigo/figures/pipeline.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpverif.config import add_config_args, dataset_cfg, load_config  # noqa: E402
from fpverif.imageops import block_std_map, segment_foreground  # noqa: E402
from preprocess import normalize, preprocess_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--sample", action="append", required=True, help="path relative to the dataset root, one per --dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pp = cfg["preprocess"]

    titles = ["bruto", "desvio-padrão por bloco", "máscara (ROI)", "normalização", f"entrada da rede, {pp['output_size']}×{pp['output_size']}"]
    fig, axes = plt.subplots(len(args.dataset), 5, figsize=(11, 2.4 * len(args.dataset)), squeeze=False)
    for row, (name, sample) in zip(axes, zip(args.dataset, args.sample)):
        ds = dataset_cfg(cfg, name)
        img = cv2.imread(str(Path(ds["root"]) / sample), cv2.IMREAD_GRAYSCALE)
        std = block_std_map(img, pp["block_size"])
        mask = segment_foreground(img, pp["block_size"], pp["fg_std_threshold"])
        norm = np.where(mask, normalize(img, mask, {**pp, "normalization": "hong"}), pp["background_fill"])
        final, _ = preprocess_image(img, pp, ds["dpi"])
        for ax, im, title in zip(row, (img, std, mask, norm, final), titles):
            ax.imshow(im, cmap="gray", vmin=0 if im is not std else None, vmax=255 if im is not std and im.dtype != bool else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        row[0].set_title(f"{name.upper().replace('_', ' ')}, {Path(sample).stem}", fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
