"""Write identity-disjoint splits and evaluation pairs for one dataset.

    python make_protocols.py --dataset fvc2002_db1 --mode split      # 60/20/30 identity split, also writes <dataset>_val/
    python make_protocols.py --dataset fvc2002_db1 --mode test       # every finger is a test finger, all pairs
    python make_protocols.py --dataset fvc2002_db1 --mode official   # FVC rule: impostor = impression 1 vs 1

Split sizes come from datasets.<name>.split, else protocol.split. A split without `test` writes no test pairs.
With set A present, `--mode split` on an FVC database gives the original 60/20/30 protocol and `--mode official`
gives the official 2800 genuine / 4950 impostor comparisons on set A. With only set B, `official` applies the
same rule to set B (280 genuine / 45 impostor).

Outputs under protocols/<dataset>[_<mode>]/ (split mode has no suffix):
    train.csv, validation.csv, test.csv           one row per image: path, finger, impression, subset, dpi, split
    validation_pairs.csv                          pairs used to pick the threshold
    genuine_pairs.csv, impostor_pairs.csv         test pairs by type
    test_pairs.csv                                both, columns sample_1, sample_2, type
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fpverif.config import add_config_args, dataset_cfg, load_config
from fpverif.data import dataset_index
from fpverif.protocol import build_pairs, official_protocol, split_identities


def write_test_pairs(pairs, out: Path) -> dict:
    pairs[pairs["type"] == "genuine"].to_csv(out / "genuine_pairs.csv", index=False)
    pairs[pairs["type"] == "impostor"].to_csv(out / "impostor_pairs.csv", index=False)
    pairs.to_csv(out / "test_pairs.csv", index=False)
    return {"n_genuine": int((pairs["type"] == "genuine").sum()), "n_impostor": int((pairs["type"] == "impostor").sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["split", "test", "official"], default="split")
    ap.add_argument("--out", default=None, help="defaults to protocols/<dataset>[_<mode>]")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ds = dataset_cfg(cfg, args.dataset)
    index, _ = dataset_index(cfg, args.dataset)
    suffix = "" if args.mode == "split" else f"_{args.mode}"
    out = Path(args.out or f"protocols/{args.dataset}{suffix}")
    out.mkdir(parents=True, exist_ok=True)
    impostor_mode = cfg["protocol"]["impostor_mode"]
    summary = {"mode": args.mode, "dataset": args.dataset, "n_images": int(len(index))}

    if args.mode == "official":
        has_a = bool((index["subset"] == "A").any())
        test = index[index["subset"] == "A"] if has_a else index
        test.assign(split="test").to_csv(out / "test.csv", index=False)
        summary.update({"rule": "fvc", "subset": "A" if has_a else "all available (set B only)",
                        "test_fingers": int(test["finger"].nunique()), **write_test_pairs(official_protocol(index), out)})
    elif args.mode == "test":
        index.assign(split="test").to_csv(out / "test.csv", index=False)
        summary.update({"impostor_mode": impostor_mode, "test_fingers": int(index["finger"].nunique()),
                        **write_test_pairs(build_pairs(index, impostor_mode), out)})
    else:
        sizes = ds.get("split") or cfg["protocol"]["split"]
        split = split_identities(index["finger"].tolist(), sizes, cfg["seed"])
        frames = {name: index[index["finger"].isin(f)].assign(split=name) for name, f in split.items()}
        for name, frame in frames.items():
            frame.to_csv(out / f"{name}.csv", index=False)
        val_pairs = build_pairs(frames["validation"], impostor_mode)
        val_pairs.to_csv(out / "validation_pairs.csv", index=False)
        summary.update({"seed": cfg["seed"], "impostor_mode": impostor_mode,
                        **{f"{k}_fingers": len(v) for k, v in split.items()},
                        "validation_genuine": int((val_pairs["type"] == "genuine").sum()),
                        "validation_impostor": int((val_pairs["type"] == "impostor").sum())})
        if "test" in frames:
            summary.update(write_test_pairs(build_pairs(frames["test"], impostor_mode), out))
            # Same validation fingers without test pairs, so evaluate.py reports the validation split itself.
            val_dir = out.with_name(out.name + "_val")
            val_dir.mkdir(parents=True, exist_ok=True)
            frames["validation"].to_csv(val_dir / "validation.csv", index=False)
            val_pairs.to_csv(val_dir / "validation_pairs.csv", index=False)
        if "source" in index.columns:
            summary["fingers_per_source_and_split"] = {
                name: frame.groupby("source")["finger"].nunique().to_dict() for name, frame in frames.items()}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
