"""Render the worst false matches and false non-matches of one evaluation as the preprocessed network inputs, with scores.

    python scripts/error_figure.py --result-dir results/fvc2002_db1/convnext_tiny_gabor/seed2026 \
        --dataset fvc2002_db1 --out artigo/figures/errors_fvc2002_db1.png

Reads <result-dir>/errors.csv (written by evaluate.py). Also writes <out>.csv with the cases shown and
their quality numbers from qc.csv, which is where the error hypotheses in the paper start from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fpverif.config import add_config_args, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    pp = cfg["preprocess"]
    proc = Path(pp["output_dir"]) / args.dataset / pp["normalization"]
    qc = pd.read_csv(proc / "qc.csv").set_index("path")

    errors = pd.read_csv(Path(args.result_dir) / "errors.csv")
    cases = pd.concat([errors[errors["error"] == "false_match"].head(args.n),
                       errors[errors["error"] == "false_non_match"].head(args.n)])
    # Tiles are the 320x320 network inputs, which also crop away the empty frame of large sensors.
    fig, axes = plt.subplots(2, 2 * args.n, figsize=(7.16, 2.75), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    rows = []
    per_kind = {"false_match": 0, "false_non_match": 0}
    for case in cases.itertuples(index=False):
        r, c = (0 if case.error == "false_match" else 1), per_kind[case.error]
        per_kind[case.error] += 1
        for k, sample in enumerate((case.amostra_1, case.amostra_2)):
            ax = axes[r, 2 * c + k]
            ax.imshow(cv2.imread(str(proc / (Path(sample).stem + ".png")), cv2.IMREAD_GRAYSCALE),
                      cmap="gray", vmin=0, vmax=255)
            ax.set_title(Path(sample).stem, fontsize=7, pad=2)
        axes[r, 2 * c].text(1.02, -0.06, f"score {case.score:.3f}".replace(".", ","), transform=axes[r, 2 * c].transAxes,
                            fontsize=7, ha="center", va="top")
        q1, q2 = qc.loc[Path(case.amostra_1).stem + ".png"], qc.loc[Path(case.amostra_2).stem + ".png"]
        rows.append({"error": case.error, "amostra_1": case.amostra_1, "amostra_2": case.amostra_2,
                     "score": case.score, "threshold": case.threshold,
                     "area_mm2_1": q1["foreground_mm2"], "area_mm2_2": q2["foreground_mm2"],
                     "sharpness_1": q1["sharpness"], "sharpness_2": q2["sharpness"]})
    thr = cases["threshold"].iloc[0]
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.07, wspace=0.06, hspace=0.32)
    fig.text(0.024, 0.69, "falsos\nmatches", fontsize=7, ha="center", va="center", rotation=90)
    fig.text(0.024, 0.27, "falsos\nnão-matches", fontsize=7, ha="center", va="center", rotation=90)
    fig.suptitle(f"{args.dataset.upper().replace('_', ' ')}: limiar congelado = " + f"{thr:.3f}".replace(".", ","), fontsize=8, y=0.985)
    for c in range(1, args.n):
        x = (axes[0, 2 * c - 1].get_position().x1 + axes[0, 2 * c].get_position().x0) / 2
        fig.add_artist(plt.Line2D([x, x], [0.04, 0.92], color="#b0afab", linewidth=0.6))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    pd.DataFrame(rows).to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
