"""Experiment D: degrade raw test images at four levels, re-run the frozen pipeline, keep the frozen threshold.

    python robustness.py --dataset fvc2002_db1 --checkpoint runs/convnext_tiny_hong/seed2026/best.pt \
        --protocol-dir protocols/fvc2002_db1 --threshold-json results/fvc2002_db1_val/convnext_tiny_hong/seed2026/threshold.json \
        --out results/robustness/rotation/convnext_tiny_hong/seed2026 --set robustness.perturbation=rotation

For each level: perturb raw images -> preprocess.py pipeline -> embeddings -> metrics on test_pairs.csv.
robustness.apply_to=both degrades both samples of a pair; probe scores clean sample_1 against degraded sample_2.
With robustness.drop_fta=false no pair is dropped and the per-sample FTA is reported beside the matching metrics.
Writes <out>/<level>/{scores.csv,metrics.json,...} and <out>/robustness.json plus robustness.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import evaluate, load_embeddings, sample_fta, score_pairs
from extract_embeddings import extract
from fpverif.config import add_config_args, dataset_cfg, load_config
from fpverif.imageops import gaussian_blur, rotate
from fpverif.seed import seed_everything
from preprocess import run as preprocess_run


def perturb_dataset(index: pd.DataFrame, src_root: Path, dst_root: Path, kind: str, level: float, seed: int) -> None:
    """Write a perturbed copy of every image in `index`."""
    rng = np.random.default_rng(seed)
    dst_root.mkdir(parents=True, exist_ok=True)
    for row in index.itertuples(index=False):
        img = cv2.imread(str(src_root / row.path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(src_root / row.path)
        if level:
            if kind == "blur":
                img = gaussian_blur(img, level)
            elif kind == "rotation":
                img = rotate(img, float(rng.choice([-1, 1]) * level), fill=float(np.median(img)))
            else:
                raise ValueError(kind)
        out = dst_root / row.path
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--threshold-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"], cfg["train"]["deterministic"])
    rb, pp = cfg["robustness"], cfg["preprocess"]
    kind = rb["perturbation"]
    ds = dataset_cfg(cfg, args.dataset)
    protocol = Path(args.protocol_dir)
    out = Path(args.out)
    threshold = json.loads(Path(args.threshold_json).read_text())["threshold"]

    test_index = pd.read_csv(protocol / "test.csv")
    pairs = pd.read_csv(protocol / "test_pairs.csv")
    if rb["apply_to"] not in ("both", "probe"):
        raise ValueError(f"robustness.apply_to must be both or probe, got {rb['apply_to']!r}")
    probe_only, drop = rb["apply_to"] == "probe", bool(rb["drop_fta"])

    def run_level(name: str, level: float) -> tuple[dict, pd.DataFrame]:
        work = Path(pp["output_dir"]) / "robustness" / args.dataset / kind / name
        raw_dir, proc_dir = work / "raw", work / pp["normalization"]
        perturb_dataset(test_index, Path(ds["root"]), raw_dir, kind, level, cfg["seed"])
        qc = preprocess_run(cfg, args.dataset, src_root=raw_dir, out_root=proc_dir, index=test_index)
        emb_path = extract(cfg, args.dataset, Path(args.checkpoint), work / "emb.npz", image_root=proc_dir)
        return load_embeddings(emb_path), qc

    clean = run_level("clean", 0.0) if probe_only else None
    results = {}
    for level_name, level in rb["levels"][kind].items():
        emb, qc = clean if probe_only and not level else run_level(level_name, level)
        emb_1, qc_1 = clean if probe_only else (emb, qc)
        scores = score_pairs(emb_1, pairs, qc_1 if drop else None, emb, qc if drop else None)
        metrics = evaluate(scores, threshold, out / level_name, fta=sample_fta(pairs, qc_1, qc))
        metrics["fta_dropped_pairs"] = drop
        results[level_name] = {"level": level, **metrics}
        print(level_name, {k: round(metrics[k], 4) for k in ("eer", "fmr", "fnmr", "auc", "fta", "pairs_dropped")})

    (out / "robustness.json").write_text(json.dumps({"perturbation": kind, "apply_to": rb["apply_to"],
                                                     "drop_fta": drop, "threshold": threshold,
                                                     "levels": results}, indent=2))
    names = list(results)
    fig, ax = plt.subplots(figsize=(5, 4))
    for key in ("eer", "fmr", "fnmr"):
        ax.plot(names, [results[n][key] for n in names], marker="o", label=key.upper())
    ax.set_xlabel(f"{kind} level")
    ax.set_ylabel("rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)


if __name__ == "__main__":
    main()
