from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import error_curves


def plot_roc(genuine, impostor, out: Path, label: str = "") -> None:
    _, fmr, fnmr = error_curves(genuine, impostor)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fmr, 1 - fnmr, label=label or None)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("FMR (FAR)")
    ax.set_ylabel("TAR = 1 - FNMR")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_det(genuine, impostor, out: Path, label: str = "") -> None:
    _, fmr, fnmr = error_curves(genuine, impostor)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fmr, fnmr, label=label or None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.grid(True, which="both", alpha=0.3)
    if label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distributions(genuine, impostor, threshold: float, out: Path, bins: int = 60,
                       score_name: str = "cosine similarity") -> None:
    lo = min(np.min(genuine), np.min(impostor))
    hi = max(np.max(genuine), np.max(impostor))
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(impostor, bins=edges, density=True, alpha=0.6, label=f"impostor (n={len(impostor)})")
    ax.hist(genuine, bins=edges, density=True, alpha=0.6, label=f"genuine (n={len(genuine)})")
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold = {threshold:.3f}")
    ax.set_xlabel(score_name)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
