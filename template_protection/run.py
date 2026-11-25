"""Protected fingerprint templates from the ConvNeXt-Tiny embeddings of the main project.

    python template_protection/run.py         # writes template_protection/results/

Features are the 256-d embeddings in runs/convnext_tiny_gabor/seed*/emb_<base>.npz. For each base, the 30 test
fingers enrol with impressions 1 to 4 and are probed with impressions 5 to 8: every enrolment template against
every probe template, 480 genuine and 13 920 impostor comparisons. Hyperparameters (ridge weight, Bloom word
and block sizes) are chosen on the 20 validation fingers of FVC2002 DB1 and reused unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import bloom  # noqa: E402
import ladh  # noqa: E402
from fpverif.metrics import eer, fnmr_at_fmr  # noqa: E402
from unlinkability import d_sys  # noqa: E402

RUN = "convnext_tiny_gabor"
BASES = ["fvc2002_db1", "fvc2004_db1"]
SEEDS = [2026, 1999, 1234]
BITS = [32, 64, 128, 256]
KEYS = [((11, 12), (21, 22)), ((31, 32), (41, 42)), ((51, 52), (61, 62))]   # (application A, application B)
ENROL, PROBE = [1, 2, 3, 4], [5, 6, 7, 8]
N_PSEUDO = 50
OUT = HERE / "results"


def load(base: str, seed: int, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings, finger and impression of one split of one base."""
    z = np.load(ROOT / "runs" / RUN / f"seed{seed}" / f"emb_{base}.npz")
    emb = dict(zip(z["paths"].tolist(), z["embeddings"]))
    frame = pd.read_csv(ROOT / "protocols" / base / f"{split}.csv").sort_values(["finger", "impression"])
    X = np.stack([emb[f"{f}_{i}.png"] for f, i in zip(frame["finger"], frame["impression"])])
    return X.astype(np.float32), frame["finger"].to_numpy(), frame["impression"].to_numpy()


def rates(similarity: np.ndarray, genuine: np.ndarray) -> dict:
    g, i = similarity[genuine], similarity[~genuine]
    return {"eer": eer(g, i)[0], "gmr_fmr1": 1 - fnmr_at_fmr(g, i, 0.01), "gmr_fmr01": 1 - fnmr_at_fmr(g, i, 0.001),
            "gmr_fmr001": 1 - fnmr_at_fmr(g, i, 0.0001)}


def closed_set(X, finger, imp, q, keys, lam):
    """Section 2.2: one projection learned from the labelled enrolment set, used for enrolment and probes."""
    e, p = np.isin(imp, ENROL), np.isin(imp, PROBE)
    W = ladh.fit(X[e], finger[e], q, keys, lam)
    sim = -ladh.hamming(ladh.protect(X[e], W), ladh.protect(X[p], W))
    return sim, finger[e][:, None] == finger[p][None, :], W


def unseen_impostors(X, finger, imp, q, keys, lam, n_enrolled=20):
    """Not in the paper: impostors are fingers that were never in the enrolment set."""
    ids = np.unique(finger)
    known = np.isin(finger, ids[:n_enrolled])
    e, p = known & np.isin(imp, ENROL), known & np.isin(imp, PROBE)
    W = ladh.fit(X[e], finger[e], q, keys, lam)
    te = ladh.protect(X[e], W)
    genuine = -ladh.hamming(te, ladh.protect(X[p], W))[finger[e][:, None] == finger[p][None, :]]
    impostor = -ladh.hamming(te, ladh.protect(X[~known], W)).ravel()
    sim = np.concatenate([genuine, impostor])
    return sim, np.arange(len(sim)) < len(genuine)


def pseudo_user(X, finger, imp, Xa, fa, q, keys, lam):
    """Section 2.4: each user gets a projection learned from that user's enrolment samples plus pseudo-users."""
    e, p = np.isin(imp, ENROL), np.isin(imp, PROBE)
    sims, gens, templates = [], [], {}
    for u in np.unique(finger):
        mine = e & (finger == u)
        W = ladh.fit(np.concatenate([Xa, X[mine]]), np.concatenate([fa, np.full(mine.sum(), -1)]), q, keys, lam)
        sims.append(-ladh.hamming(ladh.protect(X[mine], W), ladh.protect(X[p], W)))
        gens.append(np.broadcast_to(finger[p][None, :] == u, sims[-1].shape))
        templates[u] = ladh.protect(X[finger == u], W)
    return np.concatenate(sims), np.concatenate(gens), templates


def linkage(A: np.ndarray, B: np.ndarray, finger: np.ndarray, distance) -> tuple[float, np.ndarray, np.ndarray]:
    """Templates of the same samples under two applications: mated = same finger, non-mated = other fingers."""
    d = distance(A, B)
    same = finger[:, None] == finger[None, :]
    return d_sys(d[same], d[~same], bins=32)[0], d[same], d[~same]


def tune(base: str = "fvc2002_db1", seed: int = 2026) -> dict:
    X, finger, imp = load(base, seed, "validation")
    lam = {lam: rates(*closed_set(X, finger, imp, 64, KEYS[0][0], lam)[:2])["eer"] for lam in (0.01, 0.1, 1.0, 10.0)}
    e, p = np.isin(imp, ENROL), np.isin(imp, PROBE)
    gen = finger[e][:, None] == finger[p][None, :]
    bits = bloom.binarize(X)
    blm = {}
    for w in (3, 4, 5):
        for l in (4, 8, 16):
            t = bloom.protect(bits, w, l, 7)
            blm[f"{w},{l}"] = rates(-bloom.dissimilarity(t[e], t[p]), gen)["eer"]
    w, l = map(int, min(blm, key=blm.get).split(","))
    return {"lam": min(lam, key=lam.get), "bloom_word_bits": w, "bloom_block_words": l,
            "validation_eer_by_lam": lam, "validation_eer_by_bloom": blm}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg = tune()
    (OUT / "tuning.json").write_text(json.dumps(cfg, indent=2))
    lam, bw, bl = cfg["lam"], cfg["bloom_word_bits"], cfg["bloom_block_words"]
    rows, link_plot = [], {}
    for base in BASES:
        for seed in SEEDS:
            X, finger, imp = load(base, seed, "test")
            e, p = np.isin(imp, ENROL), np.isin(imp, PROBE)
            gen = finger[e][:, None] == finger[p][None, :]
            # Pseudo-users: fingers of the same base outside the test split, drawn once with seed 42.
            Xn, fn, im = (np.concatenate(v) for v in zip(load(base, seed, "train"), load(base, seed, "validation")))
            chosen = np.random.default_rng(42).choice(np.unique(fn), N_PSEUDO, replace=False)
            aux = np.isin(fn, chosen) & np.isin(im, ENROL)
            Xa, fa = Xn[aux], fn[aux]

            def add(method: str, q, key_id, r: dict, **extra) -> None:
                rows.append({"base": base, "seed": seed, "method": method, "bits": q, "keys": key_id, **r, **extra})

            add("unprotected, cosine", 256 * 32, -1, rates(X[e] @ X[p].T, gen))
            sign = bloom.binarize(X)
            add("unprotected, sign bits", 256, -1, rates(-ladh.hamming(sign[e], sign[p]), gen),
                active_bits=float(sign.sum(1).mean()))
            for k, (ka, kb) in enumerate(KEYS):
                ta, tb = bloom.protect(sign, bw, bl, ka[0]), bloom.protect(sign, bw, bl, kb[0])
                d, _, _ = linkage(ta, tb, finger, bloom.dissimilarity)
                add("Bloom filter", ta[0].size, k, rates(-bloom.dissimilarity(ta[e], ta[p]), gen), d_sys=d,
                    active_bits=float(ta.reshape(len(ta), -1).sum(1).mean()))
                for q in BITS:
                    sim, g, Wa = closed_set(X, finger, imp, q, ka, lam)
                    Wb = closed_set(X, finger, imp, q, kb, lam)[2]
                    d, mated, non_mated = linkage(ladh.protect(X, Wa), ladh.protect(X, Wb), finger, ladh.hamming)
                    add("LADH, closed set", q, k, rates(sim, g), d_sys=d, active_bits=float(ladh.protect(X, Wa).sum(1).mean()))
                    if (base, seed, k, q) == (BASES[1], SEEDS[0], 0, 256):
                        link_plot["LADH, conjunto fechado"] = (mated, non_mated)
                    add("LADH, closed set, unseen impostors", q, k, rates(*unseen_impostors(X, finger, imp, q, ka, lam)))
                    if k == 0:      # one projection per user: 30 fits per setting, so one key pair
                        sim, g, ta_u = pseudo_user(X, finger, imp, Xa, fa, q, ka, lam)
                        tb_u = pseudo_user(X, finger, imp, Xa, fa, q, kb, lam)[2]
                        # Linkage here compares each user's own two templates (mated) with two users' templates.
                        A = np.concatenate([ta_u[u] for u in np.unique(finger)])
                        B = np.concatenate([tb_u[u] for u in np.unique(finger)])
                        d, mated, non_mated = linkage(A, B, np.sort(finger), ladh.hamming)
                        add("LADH, pseudo-users", q, k, rates(sim, g), d_sys=d, active_bits=float(A.sum(1).mean()))
                        if (base, seed, q) == (BASES[1], SEEDS[0], 256):
                            link_plot["LADH, pseudousuários"] = (mated, non_mated)
            print(base, seed, "done", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "raw.csv", index=False)
    cols = ["eer", "gmr_fmr1", "gmr_fmr01", "gmr_fmr001", "d_sys", "active_bits"]
    # Mean over key pairs first, then mean and standard deviation over the three CNN training seeds.
    per_seed = raw.groupby(["base", "method", "bits", "seed"], sort=False)[cols].mean()
    summary = per_seed.groupby(["base", "method", "bits"], sort=False).agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)

    def pm(r, c, k=100.0, d=2):
        return "n/d" if pd.isna(r[f"{c}_mean"]) else f"{k * r[f'{c}_mean']:.{d}f} ± {k * r[f'{c}_std']:.{d}f}"

    lines = ["# Templates protegidos: resultados", "",
             f"Embeddings de `{RUN}`, média ± desvio sobre as sementes de treino {SEEDS}.", ""]
    for base in BASES:
        lines += [f"## {base}", "", "| Método | Bits | EER (%) | GMR@FMR=1% | GMR@FMR=0,1% | GMR@FMR=0,01% | D_sys | Bits ativos |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["base"] == base].iterrows():
            lines.append(f"| {r['method']} | {r['bits']} | {pm(r, 'eer')} | {pm(r, 'gmr_fmr1')} | {pm(r, 'gmr_fmr01')} | "
                         f"{pm(r, 'gmr_fmr001')} | {pm(r, 'd_sys', 1.0, 3)} | {pm(r, 'active_bits', 1.0, 1)} |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, len(link_plot), figsize=(4.2 * len(link_plot), 3.0), squeeze=False)
    for ax, (name, (mated, non_mated)) in zip(axes[0], link_plot.items()):
        edges = np.linspace(0.25, 0.75, 41)
        ax.hist(non_mated, edges, density=True, alpha=0.6, label="dedos diferentes")
        ax.hist(mated, edges, density=True, alpha=0.6, label="mesmo dedo")
        ax.set(title=f"{name}, q = 256", xlabel="distância de Hamming entre aplicações", ylabel="densidade")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "unlinkability.png", dpi=160)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for base, style in zip(BASES, ("-", "--")):
        for method in ("LADH, closed set", "LADH, closed set, unseen impostors", "LADH, pseudo-users"):
            s = summary[(summary["base"] == base) & (summary["method"] == method)]
            ax.plot(s["bits"], 100 * s["eer_mean"], style, marker="o", label=f"{method} ({base})")
    ax.set(xscale="log", xticks=BITS, xticklabels=BITS, xlabel="bits do template (q)", ylabel="EER (%)")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "eer_by_bits.png", dpi=160)
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
