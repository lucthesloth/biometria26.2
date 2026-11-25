"""Training protocols of the optimisation attempt: FVC2002 DB1 train fingers plus every finger of the other sources.

    python otimizacao/make_pool_protocol.py                       # writes protocols/fvc2002_pool/
    python otimizacao/make_pool_protocol.py --dataset wide_pool   # adds FVC2004 DB2 to DB4 and U.are.U
    python otimizacao/make_pool_protocol.py --dataset wide_pool --holdout 20   # writes protocols/wide_val/

Validation is the same 20 FVC2002 DB1 fingers as the base protocol, so validation EERs are comparable. The 30
DB1 test fingers and every FVC2004 DB1 image stay out.

--holdout N moves N fingers of FVC2004 DB2 and N of FVC2004 DB3 (seed 42) from training to validation, so the
validation set also has distorted, dry and wet captures from sensors other than the DB1 one. Its impostor pairs
stay inside each source.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.config import load_config  # noqa: E402
from fpverif.data import dataset_index  # noqa: E402
from fpverif.protocol import build_pairs  # noqa: E402

OFFSET = 1000  # fvc2002_db1 is the first source of both pools
HOLDOUT_SOURCES = ["fvc2004_db2", "fvc2004_db3"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="fvc2002_pool", choices=["fvc2002_pool", "wide_pool"])
    ap.add_argument("--holdout", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    index, _ = dataset_index(cfg, args.dataset)
    base = ROOT / "protocols" / "fvc2002_db1"
    db1_train = set(pd.read_csv(base / "train.csv")["finger"] + OFFSET)
    db1_val = set(pd.read_csv(base / "validation.csv")["finger"] + OFFSET)
    is_db1 = index["source"] == "fvc2002_db1"
    train = index[~is_db1 | index["finger"].isin(db1_train)].assign(split="train")
    val = index[is_db1 & index["finger"].isin(db1_val)].assign(split="validation")
    out = ROOT / "protocols" / args.dataset
    if args.holdout:
        rng = np.random.default_rng(cfg["seed"])
        held = set()
        for src in HOLDOUT_SOURCES:
            fingers = np.sort(index.loc[index["source"] == src, "finger"].unique())
            held |= set(rng.choice(fingers, args.holdout, replace=False).tolist())
        val = pd.concat([val, train[train["finger"].isin(held)].assign(split="validation")], ignore_index=True)
        train = train[~train["finger"].isin(held)]
        out = ROOT / "protocols" / "wide_val"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "validation.csv", index=False)
    pairs = pd.concat([build_pairs(g, cfg["protocol"]["impostor_mode"]) for _, g in val.groupby("source", sort=False)],
                      ignore_index=True)
    pairs.to_csv(out / "validation_pairs.csv", index=False)
    summary = {"train_fingers": int(train["finger"].nunique()), "train_images": len(train),
               "train_fingers_by_source": train.groupby("source")["finger"].nunique().to_dict(),
               "validation_fingers": int(val["finger"].nunique()),
               "validation_fingers_by_source": val.groupby("source")["finger"].nunique().to_dict(),
               "validation_genuine": int((pairs["type"] == "genuine").sum()),
               "validation_impostor": int((pairs["type"] == "impostor").sum())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
