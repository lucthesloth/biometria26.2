"""Score-level ensemble of ConvNeXt-Tiny models, tuned on the validation fingers only.

    python otimizacao/ensemble.py --name gabor_hong --a results:convnext_tiny_gabor --b results:convnext_tiny_hong
    python otimizacao/ensemble.py --name x --a results/otimizacao:pool_gabor --b results/otimizacao:pool_hong

Each member is <results root>:<run>. Scores are z-normalised with the member's validation impostor scores, as
in scripts/fusion.py. The fused score is w * z_a + (1 - w) * z_b; w (grid of 0.1) and the threshold come from
the validation EER and are reused unchanged on the three tests. Writes results/otimizacao/<evaluation>/<name>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import choose_threshold, evaluate  # noqa: E402
from fpverif.metrics import eer  # noqa: E402

VAL = "fvc2002_db1_val"
TESTS = ["fvc2002_db1", "fvc2004_db1", "fvc2004_db1_official"]
OUT = ROOT / "results" / "otimizacao"


def load(member: str, evaluation: str, seed: str) -> pd.DataFrame:
    root, run = member.split(":")
    s = pd.read_csv(ROOT / root / evaluation / run / seed / "scores.csv")
    key = s["amostra_1"].map(lambda p: Path(p).stem) + "|" + s["amostra_2"].map(lambda p: Path(p).stem)
    return s.assign(key=key).set_index("key")


def z(scores: pd.Series, ref: pd.DataFrame) -> pd.Series:
    imp = ref.loc[ref["tipo"] == "impostor", "score"].dropna()
    return (scores - imp.mean()) / imp.std()


def fuse(a, b, a_ref, b_ref, w: float) -> pd.DataFrame:
    b = b.reindex(a.index)
    return a.assign(score=w * z(a["score"], a_ref) + (1 - w) * z(b["score"], b_ref)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", nargs="+", default=["seed2026", "seed1999", "seed1234"])
    args = ap.parse_args()
    for seed in args.seeds:
        a_val, b_val = load(args.a, VAL, seed), load(args.b, VAL, seed)
        grid = {}
        for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
            f = fuse(a_val, b_val, a_val, b_val, w).dropna(subset=["score"])
            grid[float(w)] = eer(f.loc[f["tipo"] == "genuine", "score"].to_numpy(),
                                 f.loc[f["tipo"] == "impostor", "score"].to_numpy())[0]
        w = min(grid, key=grid.get)
        fused = fuse(a_val, b_val, a_val, b_val, w)
        thr = choose_threshold(fused, "eer", 0.0)
        thr.update({"weight_a": w, "a": args.a, "b": args.b})
        out = OUT / VAL / args.name / seed
        m = evaluate(fused, thr["threshold"], out, score_name="score")
        (out / "threshold.json").write_text(json.dumps(thr, indent=2))
        line = [f"{seed} w={w} val EER {100 * m['eer']:.2f}"]
        for test in TESTS:
            fused = fuse(load(args.a, test, seed), load(args.b, test, seed), a_val, b_val, w)
            out = OUT / test / args.name / seed
            m = evaluate(fused, thr["threshold"], out, score_name="score")
            (out / "threshold.json").write_text(json.dumps(thr, indent=2))
            line.append(f"{test} EER {100 * m['eer']:.2f} FMR {100 * m['fmr']:.2f} FNMR {100 * m['fnmr']:.2f}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
