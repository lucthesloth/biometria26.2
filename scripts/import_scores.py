"""Import another team's scores.csv (Experiment B) after checking it scores exactly our pairs.

    python scripts/import_scores.py --scores outra_equipe/scores_fvc2002.csv --protocol-dir protocols/fvc2002_db1 \
        --threshold <limiar> --out results/fvc2002_db1/<run>/seed42

Their file must have columns amostra_1, amostra_2, tipo, score (assignment section 11). Pairs are compared as
unordered pairs of file names, so path prefixes and the order inside a pair do not matter. Any missing, extra or
relabelled pair stops the import. The threshold must come from their own validation data, never from our test set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate  # noqa: E402


def pair_key(a: str, b: str) -> tuple[str, str]:
    a, b = Path(str(a)).stem, Path(str(b)).stem
    return (a, b) if a <= b else (b, a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="their frozen validation threshold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    theirs = pd.read_csv(args.scores)
    ours = pd.read_csv(Path(args.protocol_dir) / "test_pairs.csv")
    t = {pair_key(r.amostra_1, r.amostra_2): r.tipo for r in theirs.itertuples(index=False)}
    o = {pair_key(r.sample_1, r.sample_2): r.type for r in ours.itertuples(index=False)}
    missing, extra = set(o) - set(t), set(t) - set(o)
    relabelled = [k for k in set(o) & set(t) if o[k] != t[k]]
    if missing or extra or relabelled or len(theirs) != len(t):
        raise SystemExit(f"pairs differ: {len(missing)} missing, {len(extra)} extra, {len(relabelled)} relabelled, "
                         f"{len(theirs) - len(t)} duplicated. Both teams must score the same test_pairs.csv.")
    metrics = evaluate(theirs[["amostra_1", "amostra_2", "tipo", "score"]], args.threshold, Path(args.out), score_name="score")
    print({k: round(metrics[k], 4) for k in ("eer", "fmr", "fnmr", "auc", "tar_at_far_1pct")})


if __name__ == "__main__":
    main()
