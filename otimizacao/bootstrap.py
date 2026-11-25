"""Paired bootstrap over fingers of the optimised configurations against the base ConvNeXt-Tiny.

    python otimizacao/bootstrap.py        # writes results/otimizacao/bootstrap_ci.csv

Same procedure as scripts/bootstrap_ci.py (fingers resampled with replacement, the same fingers for every run
and training seed, metrics averaged over the seeds), with 2000 replicates. FMR and FNMR use each run's own
threshold, frozen on the validation fingers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from bootstrap_ci import finger, replicate  # noqa: E402

EVALS = ["fvc2002_db1", "fvc2004_db1", "fvc2004_db1_official"]
BASE = ("results", "convnext_tiny_gabor")
RUNS = {"base_gabor_hong": ("results/otimizacao", "base_gabor_hong"), "pool": ("results/otimizacao", "pool"),
        "pool_gabor_hong": ("results/otimizacao", "pool_gabor_hong"), "wide_gh": ("results/otimizacao", "wide_gh"),
        "wide_4_as": ("results/otimizacao", "wide_4_as"), "wide_12_as": ("results/otimizacao", "wide_12_as")}
N_BOOT = 2000


def load(root: str, evaluation: str, run: str, fingers: list[str]) -> list[tuple]:
    index = {f: k for k, f in enumerate(fingers)}
    out = []
    for d in sorted((ROOT / root / evaluation / run).glob("seed*")):
        s = pd.read_csv(d / "scores.csv").dropna(subset=["score"])
        out.append((s["amostra_1"].map(finger).map(index).to_numpy(), s["amostra_2"].map(finger).map(index).to_numpy(),
                    (s["tipo"] == "genuine").to_numpy(), s["score"].to_numpy(),
                    json.loads((d / "threshold.json").read_text())["threshold"]))
    return out


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for evaluation in EVALS:
        first = pd.read_csv(next((ROOT / BASE[0] / evaluation / BASE[1]).glob("seed*")) / "scores.csv")
        fingers = sorted(set(first["amostra_1"].map(finger)) | set(first["amostra_2"].map(finger)))
        data = {"base": load(*BASE[:1], evaluation, BASE[1], fingers)}
        data.update({name: load(root, evaluation, run, fingers) for name, (root, run) in RUNS.items()})
        boot = {name: [] for name in data}
        for _ in range(N_BOOT):
            mult = np.bincount(rng.integers(0, len(fingers), len(fingers)), minlength=len(fingers))
            if (mult > 0).sum() < 2:
                continue
            for name in data:
                boot[name].append(replicate(data[name], mult))
        boot = {name: np.array(v) * 100 for name, v in boot.items()}
        ones = np.ones(len(fingers), dtype=int)
        point = {name: replicate(data[name], ones) * 100 for name in data}
        for name in RUNS:
            diff = boot[name] - boot["base"]
            lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
            for k, m in enumerate(["EER", "FMR", "FNMR"]):
                rows.append({"avaliacao": evaluation, "execucao": name, "metrica": m, "base": round(point["base"][k], 2),
                             "valor": round(point[name][k], 2), "diferenca": round(point[name][k] - point["base"][k], 2),
                             "ic95_inf": round(lo[k], 2), "ic95_sup": round(hi[k], 2)})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "otimizacao" / "bootstrap_ci.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
