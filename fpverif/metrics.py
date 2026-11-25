"""Verification metrics on raw scores. Convention: higher score = more similar; decision is score >= threshold."""
from __future__ import annotations

import numpy as np


def error_curves(genuine: np.ndarray, impostor: np.ndarray):
    """Return thresholds (ascending), FMR(t) and FNMR(t) evaluated at every distinct score plus a top sentinel."""
    genuine = np.sort(np.asarray(genuine, dtype=float))
    impostor = np.sort(np.asarray(impostor, dtype=float))
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.append(thresholds, np.nextafter(thresholds[-1], np.inf))
    fmr = 1.0 - np.searchsorted(impostor, thresholds, side="left") / len(impostor)
    fnmr = np.searchsorted(genuine, thresholds, side="left") / len(genuine)
    return thresholds, fmr, fnmr


def fmr_at(impostor: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(impostor) >= threshold))


def fnmr_at(genuine: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(genuine) < threshold))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """EER and its threshold, linearly interpolated between the two thresholds where FNMR crosses FMR."""
    t, fmr, fnmr = error_curves(genuine, impostor)
    diff = fnmr - fmr
    idx = int(np.argmax(diff >= 0))
    if idx == 0:
        return float((fmr[0] + fnmr[0]) / 2), float(t[0])
    d0, d1 = diff[idx - 1], diff[idx]
    w = 0.0 if d1 == d0 else -d0 / (d1 - d0)
    thr = t[idx - 1] + w * (t[idx] - t[idx - 1])
    rate = fmr[idx - 1] + w * (fmr[idx] - fmr[idx - 1])
    return float(rate), float(thr)


def threshold_for_fmr(impostor: np.ndarray, target: float) -> float:
    """Smallest threshold whose FMR is <= target."""
    impostor = np.sort(np.asarray(impostor, dtype=float))
    k = int(np.ceil((1.0 - target) * len(impostor)))
    if k >= len(impostor):
        return float(np.nextafter(impostor[-1], np.inf))
    return float(impostor[k])


def fnmr_at_fmr(genuine: np.ndarray, impostor: np.ndarray, target: float) -> float:
    """Lowest FNMR with FMR <= target (FMR100 for 0.01, FMR1000 for 0.001). TAR@FAR = 1 - this."""
    return fnmr_at(genuine, threshold_for_fmr(impostor, target))


def roc_auc(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Exact area under the ROC: P(genuine score > impostor score) with ties counted as one half."""
    impostor = np.sort(np.asarray(impostor, dtype=float))
    genuine = np.asarray(genuine, dtype=float)
    left = np.searchsorted(impostor, genuine, side="left")
    right = np.searchsorted(impostor, genuine, side="right")
    return float(np.mean((left + right) / 2) / len(impostor))


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    g, i = np.asarray(genuine), np.asarray(impostor)
    return float(abs(g.mean() - i.mean()) / np.sqrt((g.var() + i.var()) / 2))


def system_rates(fmr: float, fnmr: float, fta: float) -> dict[str, float]:
    """ISO/IEC 19795-1 combination of acquisition failures with matching errors."""
    return {"far": fmr * (1 - fta), "frr": fta + fnmr * (1 - fta)}


def summarize(genuine: np.ndarray, impostor: np.ndarray, threshold: float, fta: float = 0.0) -> dict:
    eer_val, eer_thr = eer(genuine, impostor)
    fmr, fnmr = fmr_at(impostor, threshold), fnmr_at(genuine, threshold)
    out = {
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "threshold": float(threshold),
        "fmr": fmr,
        "fnmr": fnmr,
        "eer": eer_val,
        "eer_threshold": eer_thr,
        "auc": roc_auc(genuine, impostor),
        "d_prime": d_prime(genuine, impostor),
        "fmr100_fnmr": fnmr_at_fmr(genuine, impostor, 0.01),
        "fmr1000_fnmr": fnmr_at_fmr(genuine, impostor, 0.001),
        "tar_at_far_1pct": 1.0 - fnmr_at_fmr(genuine, impostor, 0.01),
        "tar_at_far_0_1pct": 1.0 - fnmr_at_fmr(genuine, impostor, 0.001),
        "fta": float(fta),
    }
    out.update(system_rates(fmr, fnmr, fta))
    return out
