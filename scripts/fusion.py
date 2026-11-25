"""Score-level fusion of a CNN run with the SourceAFIS baseline, tuned on validation only.

    python scripts/fusion.py --run convnext_tiny_gabor

Each matcher's scores are z-normalized with the mean and standard deviation of its validation impostor scores.
The fused score is w * z_cnn + (1 - w) * z_minutiae. The weight w (grid of 0.1) and the threshold both come from
the validation EER and are frozen for the FVC test protocols. Writes results/<evaluation>/<run>_fusion/seed<N>/.
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
from fpverif.config import load_config  # noqa: E402
from fpverif.metrics import eer  # noqa: E402

EVALS = load_config(ROOT / "config.yaml")["evaluations"]
TESTS = list(EVALS["tests"])


def key(frame: pd.DataFrame) -> pd.Series:
    return frame["amostra_1"].map(lambda p: Path(p).stem) + "|" + frame["amostra_2"].map(lambda p: Path(p).stem)


def load(evaluation: str, run: str, seed: str) -> pd.DataFrame:
    s = pd.read_csv(ROOT / "results" / evaluation / run / seed / "scores.csv")
    return s.assign(key=key(s)).set_index("key")


def znorm(scores: pd.Series, ref: pd.DataFrame) -> pd.Series:
    imp = ref.loc[ref["tipo"] == "impostor", "score"].dropna()
    return (scores - imp.mean()) / imp.std()


def fuse(cnn: pd.DataFrame, afis: pd.DataFrame, cnn_ref: pd.DataFrame, afis_ref: pd.DataFrame, w: float) -> pd.DataFrame:
    afis = afis.reindex(cnn.index)
    assert afis["score"].notna().all() or cnn["score"].isna().any(), "pair files differ"
    return cnn.assign(score=w * znorm(cnn["score"], cnn_ref) + (1 - w) * znorm(afis["score"], afis_ref)).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--validation", default=EVALS["validation"])
    ap.add_argument("--afis", default="sourceafis")
    args = ap.parse_args()
    afis_val = load(args.validation, args.afis, "seed0")
    for d in sorted((ROOT / "results" / args.validation / args.run).glob("seed*")):
        seed = d.name
        cnn_val = load(args.validation, args.run, seed)
        best = None
        for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
            f = fuse(cnn_val, afis_val, cnn_val, afis_val, w).dropna(subset=["score"])
            e = eer(f.loc[f["tipo"] == "genuine", "score"].to_numpy(), f.loc[f["tipo"] == "impostor", "score"].to_numpy())[0]
            if best is None or e < best[1]:
                best = (float(w), e)
        w = best[0]
        fused_val = fuse(cnn_val, afis_val, cnn_val, afis_val, w)
        thr = choose_threshold(fused_val, "eer", 0.0)
        thr.update({"weight_cnn": w, "source": f"results/{args.validation}/{args.run}/{seed}"})
        out = ROOT / "results" / args.validation / f"{args.run}_fusion" / seed
        m = evaluate(fused_val, thr["threshold"], out, score_name="score")
        (out / "threshold.json").write_text(json.dumps(thr, indent=2))
        line = [f"{seed} w={w} val EER {m['eer'] * 100:.2f}"]
        for test in TESTS:
            fused = fuse(load(test, args.run, seed), load(test, args.afis, "seed0"), cnn_val, afis_val, w)
            out = ROOT / "results" / test / f"{args.run}_fusion" / seed
            m = evaluate(fused, thr["threshold"], out, score_name="score")
            (out / "threshold.json").write_text(json.dumps(thr, indent=2))
            line.append(f"{test} EER {m['eer'] * 100:.2f} FMR {m['fmr'] * 100:.2f} FNMR {m['fnmr'] * 100:.2f}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
