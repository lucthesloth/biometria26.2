"""Subject-level bootstrap confidence intervals on the stored score files.

    python scripts/bootstrap_ci.py            # writes results/bootstrap_ci.csv

Fingers are resampled with replacement (10 000 replicates, seed 42). A finger drawn m times contributes its
genuine pairs m times and its impostor pairs with finger g m_f * m_g times, so pairs of the same finger stay
together. Each replicate uses the same fingers for every training seed and every run, and the metric is the
mean over the training seeds, which makes the differences between runs paired. FMR and FNMR use each seed's
frozen validation threshold. Intervals are the 2.5 and 97.5 percentiles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.metrics import eer, fmr_at, fnmr_at  # noqa: E402

EVALS = ["fvc2002_db1", "fvc2004_db1", "fvc2004_db1_official"]
BASE = "convnext_tiny_gabor"
RUNS = [BASE, "resnet50_gabor", "efficientnet_b0_gabor", "convnext_tiny_hong", "imagenet_convnext_tiny", "sourceafis",
        f"{BASE}_fusion"]
# Paired EER differences: every run against the main one, and the fusion against the minutiae matcher alone.
DIFFS = [(run, BASE) for run in RUNS[1:]] + [(f"{BASE}_fusion", "sourceafis"),
                                            ("efficientnet_b0_gabor", "resnet50_gabor")]
N_BOOT = 10_000


def finger(name: str) -> str:
    return name.split("_")[0]


def load(evaluation: str, run: str, fingers: list[str]) -> list[tuple]:
    index = {f: k for k, f in enumerate(fingers)}
    out = []
    for d in sorted((ROOT / "results" / evaluation / run).glob("seed*")):
        s = pd.read_csv(d / "scores.csv").dropna(subset=["score"])
        out.append((s["amostra_1"].map(finger).map(index).to_numpy(), s["amostra_2"].map(finger).map(index).to_numpy(),
                    (s["tipo"] == "genuine").to_numpy(), s["score"].to_numpy(),
                    json.loads((d / "threshold.json").read_text())["threshold"]))
    return out


def replicate(seeds: list[tuple], mult: np.ndarray) -> np.ndarray:
    vals = []
    for f1, f2, gen, scores, thr in seeds:
        w = mult[f1] * np.where(gen, 1, mult[f2])
        s, is_gen = np.repeat(scores, w), np.repeat(gen, w)
        g, i = s[is_gen], s[~is_gen]
        vals.append([eer(g, i)[0], fmr_at(i, thr), fnmr_at(g, thr)])
    return np.mean(vals, axis=0)


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for evaluation in EVALS:
        first = pd.read_csv(next((ROOT / "results" / evaluation / BASE).glob("seed*")) / "scores.csv")
        fingers = sorted(set(first["amostra_1"].map(finger)) | set(first["amostra_2"].map(finger)))
        data = {run: load(evaluation, run, fingers) for run in RUNS}
        boot = {run: [] for run in RUNS}
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, len(fingers), len(fingers)), minlength=len(fingers))
            if (mult > 0).sum() < 2:
                continue
            for run in RUNS:
                boot[run].append(replicate(data[run], mult))
        boot = {run: np.array(v) * 100 for run, v in boot.items()}
        ones = np.ones(len(fingers), dtype=int)
        point = {run: replicate(data[run], ones) * 100 for run in RUNS}
        for run in RUNS:
            lo, hi = np.percentile(boot[run], [2.5, 97.5], axis=0)
            for k, m in enumerate(["EER", "FMR", "FNMR"]):
                rows.append({"avaliacao": evaluation, "execucao": run, "metrica": m, "valor": round(point[run][k], 2),
                             "ic95_inf": round(lo[k], 2), "ic95_sup": round(hi[k], 2)})
        for run, ref in DIFFS:
            lo, hi = np.percentile(boot[run][:, 0] - boot[ref][:, 0], [2.5, 97.5])
            rows.append({"avaliacao": evaluation, "execucao": f"{run} - {ref}", "metrica": "dEER",
                         "valor": round(point[run][0] - point[ref][0], 2), "ic95_inf": round(lo, 2),
                         "ic95_sup": round(hi, 2)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "bootstrap_ci.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
