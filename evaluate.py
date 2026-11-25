"""Score pairs with cosine similarity, fix the threshold on validation, report metrics on test.

Experiment A (main dataset, threshold from its own validation pairs):
    python evaluate.py --embeddings runs/<run>/seed42/emb_fvc2002_db1.npz --protocol-dir protocols/fvc2002_db1 \
        --out results/fvc2002_db1/<run>/seed42

Experiment C (cross-dataset, frozen model AND frozen threshold from the source dataset):
    python evaluate.py --embeddings runs/<run>/seed42/emb_fvc2004_db1.npz --protocol-dir protocols/fvc2004_db1_official \
        --threshold-json results/fvc2002_db1/<run>/seed42/threshold.json --out results/fvc2004_db1_official/<run>/seed42

Writes scores.csv (amostra_1, amostra_2, tipo, score), metrics.json, threshold.json, roc.png, det.png,
distributions.png and errors.csv (worst false matches and false non-matches for the error analysis).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpverif.config import add_config_args, load_config
from fpverif.metrics import eer, summarize, threshold_for_fmr
from fpverif.plots import plot_det, plot_distributions, plot_roc


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return dict(zip(z["paths"].tolist(), z["embeddings"]))


def to_png(name: str) -> str:
    return Path(name).stem + ".png"


def score_pairs(emb: dict[str, np.ndarray], pairs: pd.DataFrame, qc: pd.DataFrame | None = None,
                emb_2: dict[str, np.ndarray] | None = None, qc_2: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cosine similarity per pair. Pairs touching a non-acquired sample (FTA) get score NaN.

    sample_1 is looked up in `emb`/`qc` and sample_2 in `emb_2`/`qc_2` (default: the same), so robustness.py
    can compare a clean reference against a degraded probe."""
    emb_2 = emb if emb_2 is None else emb_2
    qc_2 = qc if qc_2 is None else qc_2
    a = pairs["sample_1"].map(to_png)
    b = pairs["sample_2"].map(to_png)
    missing = sorted((set(a) - set(emb)) | (set(b) - set(emb_2)))
    if missing:
        raise KeyError(f"{len(missing)} pair samples have no embedding, e.g. {missing[:3]}")
    ea, eb = np.stack([emb[p] for p in a]), np.stack([emb_2[p] for p in b])
    scores = np.einsum("ij,ij->i", ea, eb)
    out = pd.DataFrame({"amostra_1": pairs["sample_1"], "amostra_2": pairs["sample_2"],
                        "tipo": pairs["type"], "score": scores})
    if qc is not None:
        out.loc[a.isin(_failed(qc)), "score"] = np.nan
    if qc_2 is not None:
        out.loc[b.isin(_failed(qc_2)), "score"] = np.nan
    return out


def _failed(qc: pd.DataFrame) -> set[str]:
    return set(qc.loc[~qc["acquired"].astype(bool), "path"].map(to_png))


def sample_fta(pairs: pd.DataFrame, qc: pd.DataFrame | None, qc_2: pd.DataFrame | None = None) -> float:
    """Per-sample failure-to-acquire rate over the samples these pairs use (sample_1 from qc, sample_2 from qc_2)."""
    if qc is None:
        return 0.0
    qc_2 = qc if qc_2 is None else qc_2
    used_1, used_2 = set(pairs["sample_1"].map(to_png)), set(pairs["sample_2"].map(to_png))
    if qc_2 is qc:
        rows = [qc[qc["path"].map(to_png).isin(used_1 | used_2)]]
    else:
        rows = [qc[qc["path"].map(to_png).isin(used_1)], qc_2[qc_2["path"].map(to_png).isin(used_2)]]
    acquired = pd.concat(rows)["acquired"].astype(bool)
    return float(1.0 - acquired.mean()) if len(acquired) else 0.0


def choose_threshold(scores: pd.DataFrame, rule: str, fmr_target: float) -> dict:
    ok = scores.dropna(subset=["score"])
    gen = ok.loc[ok["tipo"] == "genuine", "score"].to_numpy()
    imp = ok.loc[ok["tipo"] == "impostor", "score"].to_numpy()
    if rule == "eer":
        val_eer, thr = eer(gen, imp)
        return {"rule": "eer", "threshold": thr, "validation_eer": val_eer}
    if rule == "fmr_target":
        return {"rule": "fmr_target", "fmr_target": fmr_target, "threshold": threshold_for_fmr(imp, fmr_target)}
    raise ValueError(rule)


def evaluate(scores: pd.DataFrame, threshold: float, out: Path, fta: float = 0.0, n_errors: int = 10,
             score_name: str = "cosine similarity") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / "scores.csv", index=False)
    acquired = scores.dropna(subset=["score"])
    gen = acquired.loc[acquired["tipo"] == "genuine", "score"].to_numpy()
    imp = acquired.loc[acquired["tipo"] == "impostor", "score"].to_numpy()
    metrics = summarize(gen, imp, threshold, fta=fta)
    metrics["pairs_dropped"] = 1.0 - len(acquired) / len(scores)
    metrics["fta_note"] = ("fta is the per-sample failure-to-acquire rate from qc.csv over the samples in these pairs; "
                           "pairs_dropped is the fraction of pairs removed because one of their samples failed")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_roc(gen, imp, out / "roc.png")
    plot_det(gen, imp, out / "det.png")
    plot_distributions(gen, imp, threshold, out / "distributions.png", score_name=score_name)

    fm = acquired[(acquired["tipo"] == "impostor") & (acquired["score"] >= threshold)].nlargest(n_errors, "score")
    fnm = acquired[(acquired["tipo"] == "genuine") & (acquired["score"] < threshold)].nsmallest(n_errors, "score")
    pd.concat([fm.assign(error="false_match"), fnm.assign(error="false_non_match")]).assign(threshold=threshold) \
        .to_csv(out / "errors.csv", index=False)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--protocol-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold-json", default=None, help="reuse a frozen threshold instead of validation pairs")
    ap.add_argument("--qc", default=None, help="qc.csv from preprocess.py, to drop FTA samples")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emb = load_embeddings(Path(args.embeddings))
    qc = pd.read_csv(args.qc) if args.qc else None
    protocol = Path(args.protocol_dir)

    if args.threshold_json:
        thr_info = json.loads(Path(args.threshold_json).read_text())
        thr_info["source"] = args.threshold_json
    else:
        val_scores = score_pairs(emb, pd.read_csv(protocol / "validation_pairs.csv"), qc)
        thr_info = choose_threshold(val_scores, cfg["eval"]["threshold_rule"], cfg["eval"]["fmr_target"])
        thr_info["source"] = str(protocol / "validation_pairs.csv")
    (out / "threshold.json").write_text(json.dumps(thr_info, indent=2))

    # A train/validation-only protocol has no test pairs: report the validation pairs.
    pairs_file = protocol / "test_pairs.csv"
    if not pairs_file.exists():
        pairs_file = protocol / "validation_pairs.csv"
    test_pairs = pd.read_csv(pairs_file)
    test_scores = score_pairs(emb, test_pairs, qc)
    metrics = evaluate(test_scores, thr_info["threshold"], out, fta=sample_fta(test_pairs, qc))
    metrics.update({"evaluated_on": str(pairs_file), "threshold_source": thr_info["source"]})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    keys = ("threshold", "eer", "fmr", "fnmr", "auc", "tar_at_far_1pct", "fta", "pairs_dropped")
    print(json.dumps({k: metrics[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
