"""Table of the optimisation attempts: validation and test metrics per variant, mean over the available seeds.

    python otimizacao/table.py            # writes results/otimizacao/summary.csv and prints it

The base configuration is read from results/ (the main pipeline); every other row from results/otimizacao/.
FMR and FNMR are measured at each run's own threshold, frozen on the validation fingers.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# wide_val is the multi-sensor validation of the fuse.py systems; the other rows validate on fvc2002_db1_val.
EVALS = ["fvc2002_db1_val", "wide_val", "fvc2002_db1", "fvc2004_db1", "fvc2004_db1_official"]
BASE = [("base (gabor)", "convnext_tiny_gabor"), ("base (hong)", "convnext_tiny_hong")]


def rows_for(root: Path, label: str, run: str) -> list[dict]:
    out = []
    for ev in EVALS:
        for f in sorted((root / ev / run).glob("seed*/metrics.json")):
            m = json.loads(f.read_text())
            out.append({"variant": label, "evaluation": ev, "seed": f.parent.name,
                        **{k: 100 * m[k] for k in ("eer", "fmr", "fnmr", "tar_at_far_1pct")}})
    return out


def main() -> None:
    rows = []
    for label, run in BASE:
        rows += rows_for(ROOT / "results", label, run)
    opt = ROOT / "results" / "otimizacao"
    for run in sorted({p.name for p in opt.glob("*/*") if p.is_dir()}):
        rows += rows_for(opt, run, run)
    raw = pd.DataFrame(rows)
    g = raw.groupby(["variant", "evaluation"], sort=False)
    summary = g[["eer", "fmr", "fnmr", "tar_at_far_1pct"]].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary.insert(0, "seeds", g["seed"].agg(lambda s: " ".join(x.removeprefix("seed") for x in s)))
    summary = summary.reset_index()
    summary.to_csv(opt / "summary.csv", index=False)
    wide = summary.pivot(index=["variant", "seeds"], columns="evaluation", values=["eer_mean", "fmr_mean", "fnmr_mean"])
    wide = wide.reindex(columns=EVALS, level=1)
    print(wide.round(2).to_string())


if __name__ == "__main__":
    main()
