"""Scores, fuses and evaluates ConvNeXt-Tiny runs trained on wide_pool, with a chosen validation protocol.

    python otimizacao/fuse.py --name w2_gh --members w2 w2_hong
    python otimizacao/fuse.py --name w2_gh_all --members w2 w2_hong --pool-seeds --asnorm 100

Each member is a run under runs/otimizacao/ with emb_wide_pool.npz, emb_fvc2002_db1.npz and emb_fvc2004_db1.npz
per seed. A member's score is the cosine, or with --asnorm K the adaptive symmetric normalisation of the cosine
against the K closest training-finger centroids of each image. Members are z-normalised with their validation
impostor scores and averaged with equal weights. The threshold is the EER point of the validation protocol
(default protocols/wide_val: 20 fingers each of FVC2002 DB1, FVC2004 DB2 and FVC2004 DB3) and is reused
unchanged on the three tests. --pool-seeds fuses every seed of every member into one system (folder seed_all).
Writes results/otimizacao/<evaluation>/<name>/<seed>/ in the format of evaluate.py.
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

RUNS = ROOT / "runs" / "otimizacao"
OUT = ROOT / "results" / "otimizacao"
TESTS = {"fvc2002_db1": "fvc2002_db1", "fvc2004_db1": "fvc2004_db1", "fvc2004_db1_official": "fvc2004_db1"}


def png(name: str) -> str:
    return Path(name).stem + ".png"


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return dict(zip([Path(p).name for p in z["paths"].tolist()], z["embeddings"]))


def cohort(emb: dict[str, np.ndarray], train: pd.DataFrame) -> np.ndarray:
    names = train["finger"].astype(str) + "_" + train["impression"].astype(str) + ".png"
    e = np.stack([emb[n] for n in names])
    fingers = train["finger"].to_numpy()
    c = np.stack([e[fingers == f].mean(0) for f in np.unique(fingers)])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def member_scores(emb: dict[str, np.ndarray], pairs: pd.DataFrame, centroids: np.ndarray | None, k: int) -> np.ndarray:
    a, b = pairs["sample_1"].map(png).to_numpy(), pairs["sample_2"].map(png).to_numpy()
    s = np.einsum("ij,ij->i", np.stack([emb[x] for x in a]), np.stack([emb[x] for x in b]))
    if centroids is None:
        return s
    names = sorted(set(a) | set(b))
    top = -np.sort(-(np.stack([emb[n] for n in names]) @ centroids.T), axis=1)[:, :k]
    mu, sd = dict(zip(names, top.mean(1))), dict(zip(names, top.std(1)))
    za = (s - np.array([mu[x] for x in a])) / np.array([sd[x] for x in a])
    zb = (s - np.array([mu[x] for x in b])) / np.array([sd[x] for x in b])
    return 0.5 * (za + zb)


def failed(dataset: str) -> set[str]:
    qc = pd.read_csv(ROOT / "data" / "processed" / dataset / "gabor" / "qc.csv")
    return set(qc.loc[~qc["acquired"].astype(bool), "path"].map(png))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 1999, 1234])
    ap.add_argument("--val-protocol", default="protocols/wide_val")
    ap.add_argument("--asnorm", type=int, default=0, help="top-K training centroids; 0 keeps the plain cosine")
    ap.add_argument("--pool-seeds", action="store_true")
    args = ap.parse_args()

    val_dir = ROOT / args.val_protocol
    train = pd.read_csv(val_dir / "train.csv")
    pairs = {"val": pd.read_csv(val_dir / "validation_pairs.csv")}
    pairs.update({t: pd.read_csv(ROOT / "protocols" / t / "test_pairs.csv") for t in TESTS})
    source = {"val": "wide_pool", **TESTS}
    bad = {d: failed(d) for d in set(TESTS.values())}

    def scores_of(run: str, seed: int) -> dict[str, np.ndarray]:
        r = RUNS / run / f"seed{seed}"
        emb = {d: load_npz(r / f"emb_{d}.npz") for d in set(source.values())}
        centroids = cohort(emb["wide_pool"], train) if args.asnorm else None
        out = {ev: member_scores(emb[source[ev]], pairs[ev], centroids, args.asnorm) for ev in pairs}
        imp = out["val"][(pairs["val"]["type"] == "impostor").to_numpy()]
        return {ev: (s - imp.mean()) / imp.std() for ev, s in out.items()}

    groups = {"seed_all": [(m, s) for m in args.members for s in args.seeds]} if args.pool_seeds else \
        {f"seed{s}": [(m, s) for m in args.members] for s in args.seeds}
    val_name = Path(args.val_protocol).name
    for label, members in groups.items():
        parts = [scores_of(m, s) for m, s in members]
        fused = {ev: np.mean([p[ev] for p in parts], axis=0) for ev in pairs}
        frames = {}
        for ev, p in pairs.items():
            f = pd.DataFrame({"amostra_1": p["sample_1"], "amostra_2": p["sample_2"], "tipo": p["type"], "score": fused[ev]})
            if ev in TESTS:
                drop = p["sample_1"].map(png).isin(bad[TESTS[ev]]) | p["sample_2"].map(png).isin(bad[TESTS[ev]])
                f.loc[drop, "score"] = np.nan
            frames[ev] = f
        thr = choose_threshold(frames["val"], "eer", 0.0)
        thr.update({"members": [f"{m}/seed{s}" for m, s in members], "asnorm": args.asnorm,
                    "validation": args.val_protocol})
        line = [label]
        for ev, f in frames.items():
            out = OUT / (val_name if ev == "val" else ev) / args.name / label
            m = evaluate(f, thr["threshold"], out, score_name="score")
            (out / "threshold.json").write_text(json.dumps(thr, indent=2))
            line.append(f"{ev} EER {100 * m['eer']:.2f} FMR {100 * m['fmr']:.2f} FNMR {100 * m['fnmr']:.2f}")
        print(" | ".join(line))


if __name__ == "__main__":
    main()
