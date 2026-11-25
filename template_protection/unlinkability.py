"""Unlinkability measures of Gomez-Barrero et al. (IEEE TIFS 13(6), 2018), used in Section 5.4 of the paper.

Mated scores compare templates of the same subject protected with different keys; non-mated scores compare
different subjects with different keys. With LR(s) = p(s | mated) / p(s | non-mated) and equal priors,
D(s) = 2 LR(s) / (1 + LR(s)) - 1 where LR(s) > 1, else 0. D_sys is the integral of D(s) p(s | mated):
0 means fully unlinkable, 1 fully linkable.
"""
from __future__ import annotations

import numpy as np


def d_sys(mated: np.ndarray, non_mated: np.ndarray, bins: int = 64, omega: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns D_sys, the bin centres and the local measure D(s)."""
    lo, hi = min(mated.min(), non_mated.min()), max(mated.max(), non_mated.max())
    edges = np.linspace(lo, hi + 1e-9, bins + 1)
    pm = np.histogram(mated, edges)[0] / len(mated)
    pn = np.histogram(non_mated, edges)[0] / len(non_mated)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where(pn > 0, pm / pn, np.where(pm > 0, np.inf, 0.0)) * omega
        local = np.where(lr > 1, np.where(np.isinf(lr), 1.0, 2 * lr / (1 + lr) - 1), 0.0)
    return float((local * pm).sum()), (edges[:-1] + edges[1:]) / 2, local
