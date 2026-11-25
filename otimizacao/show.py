"""Prints validation and test metrics of single runs, for comparing attempts.

    python otimizacao/show.py pool pool_aug --seed 2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ["fvc2002_db1_val", "fvc2002_db1", "fvc2004_db1", "fvc2004_db1_official"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    print(f"{'run':22s} val EER | FVC2002 EER FMR FNMR | FVC2004 EER FMR FNMR | official EER FMR FNMR")
    for run in args.runs:
        cells = []
        for ev in EVALS:
            f = ROOT / "results" / "otimizacao" / ev / run / f"seed{args.seed}" / "metrics.json"
            if not f.exists():
                cells.append("missing")
                continue
            m = json.loads(f.read_text())
            keys = ["eer"] if ev.endswith("_val") else ["eer", "fmr", "fnmr"]
            cells.append(" ".join(f"{100 * m[k]:5.2f}" for k in keys))
        print(f"{run:22s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
