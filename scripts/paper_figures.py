"""Figures for the paper from results/ and runs/: ROC, score distributions, robustness, training curves.

    python scripts/paper_figures.py [--seed 2026] [--out artigo/figures]

Colours are a colour-blind-safe categorical palette; series also differ by line style or marker.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.metrics import error_curves  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight",
})
MAIN = "convnext_tiny_hong"
RUNS = {MAIN: "com normalização", "convnext_tiny_none": "sem normalização"}
BASES = {"fvc2002_db1": "FVC2002 DB1", "fvc2004_db1": "FVC2004 DB1"}


def scores(evaluation: str, run: str, seed: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    d = ROOT / "results" / evaluation / run / f"seed{seed}"
    if not (d / "scores.csv").exists():
        return None
    s = pd.read_csv(d / "scores.csv").dropna(subset=["score"])
    thr = json.loads((d / "threshold.json").read_text())["threshold"]
    return s.loc[s["tipo"] == "genuine", "score"].to_numpy(), s.loc[s["tipo"] == "impostor", "score"].to_numpy(), thr


def save(fig, out: Path, name: str) -> None:
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(comma)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {out / name}.pdf")


def fig_roc(out: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (run, label), color in zip(RUNS.items(), (BLUE, ORANGE)):
        for (ev, base), style in zip(BASES.items(), ("-", "--")):
            s = scores(ev, run, seed)
            if s is None:
                continue
            _, fmr, fnmr = error_curves(s[0], s[1])
            ax.plot(np.clip(fmr, 1e-4, 1), 1 - fnmr, style, color=color, linewidth=1.3, label=f"{base}, {label}")
    ax.axvline(0.01, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("FMR (escala log)")
    ax.set_ylabel("TAR = 1 − FNMR")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=1, fontsize=6.5, handlelength=1.8)
    save(fig, out, "roc")


def fig_distributions(out: Path, seed: int, run: str = MAIN) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, (ev, base) in zip(axes, BASES.items()):
        s = scores(ev, run, seed)
        if s is None:
            continue
        gen, imp, thr = s
        edges = np.linspace(min(gen.min(), imp.min()), max(gen.max(), imp.max()), 50)
        ax.hist(imp, bins=edges, density=True, color=ORANGE, alpha=0.75, label="impostor",
                edgecolor="white", linewidth=0.4)
        ax.hist(gen, bins=edges, density=True, color=BLUE, alpha=0.75, label="genuíno",
                edgecolor="white", linewidth=0.4)
        ax.axvline(thr, color=INK, linestyle="--", linewidth=1, label=f"limiar congelado = {thr:.3f}".replace(".", ","))
        ax.set_title(f"{base} ({len(gen)} genuínos, {len(imp)} impostores)")
        ax.set_xlabel("similaridade de cosseno")
    axes[0].set_ylabel("densidade")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2.0)
    save(fig, out, "distributions")


def fig_robustness(out: Path, run: str = MAIN) -> None:
    rob = ROOT / "results" / "robustness_summary.csv"
    if not rob.exists():
        return
    r = pd.read_csv(rob)
    r = r[r["run"] == run]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9), sharey=True)
    titles = {"blur": ("desfoque gaussiano", "σ (px)"), "rotation": ("rotação", "ângulo (graus)")}
    for ax, (setting, (title, xlabel)) in zip(axes, titles.items()):
        g = r[r["perturbation"] == setting].sort_values("level")
        for key, label, color, marker in (("eer", "EER", BLUE, "o"), ("fnmr", "FNMR no limiar", ORANGE, "s"),
                                          ("fmr", "FMR no limiar", AQUA, "^"), ("fta", "FTA", YELLOW, "D")):
            ax.errorbar(g["level"], g[f"{key}_mean"] * 100, yerr=g[f"{key}_std"].fillna(0) * 100, color=color,
                        marker=marker, markersize=4, capsize=2, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("taxa (%)")
        ax.set_xticks(g["level"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.5,
               columnspacing=0.9, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.0)
    save(fig, out, "robustness")


def fig_training(out: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for (run, label), color, style in zip(RUNS.items(), (BLUE, ORANGE), ("-", "--")):
        f = ROOT / "runs" / run / f"seed{seed}" / "train_log.csv"
        if f.exists():
            log = pd.read_csv(f)
            ax.plot(log["epoch"], log["val_eer"] * 100, style, color=color, label=label)
    ax.set_xlabel("época")
    ax.set_ylabel("EER na validação (%)")
    ax.legend(loc="upper right")
    save(fig, out, "training")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "artigo" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_roc(out, args.seed)
    fig_distributions(out, args.seed)
    fig_robustness(out)
    fig_training(out, args.seed)


if __name__ == "__main__":
    main()
