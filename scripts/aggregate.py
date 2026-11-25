"""Mean and standard deviation over the training seeds of every result, written to results/summary.{md,csv}.

    python scripts/aggregate.py

Reads results/<evaluation>/<run>/seed*/metrics.json, results/robustness/<perturbation>/<run>/seed*/robustness.json
and the timing and run files under runs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVALS = {"fvc2002_db1_val": "Validação, FVC2002 DB1 (20 dedos)", "fvc2002_db1": "Teste, FVC2002 DB1 (30 dedos), exp. A",
         "fvc2004_db1": "Teste, FVC2004 DB1 (30 dedos), exp. C", "fvc2004_db1_official": "FVC2004 DB1-A, protocolo oficial, exp. C"}
RUNS = {"convnext_tiny_hong": "ConvNeXt-Tiny", "convnext_tiny_none": "ConvNeXt-Tiny sem normalização (ablação)"}
KEYS = ["threshold", "eer", "auc", "fmr", "fnmr", "tar_at_far_1pct", "fta"]
PCT = {"eer", "fmr", "fnmr", "tar_at_far_1pct", "fta"}


def cell(mean: float, std: float, key: str) -> str:
    if key in PCT:
        return f"{100 * mean:.2f} ± {100 * std:.2f}"
    return f"{mean:.4f} ± {std:.4f}"


def main() -> None:
    rows = []
    for ev in EVALS:
        for run in RUNS:
            for f in sorted((RESULTS / ev / run).glob("seed*/metrics.json")):
                m = json.loads(f.read_text())
                rows.append({"evaluation": ev, "run": run, "seed": f.parent.name, **{k: m[k] for k in KEYS}})
    raw = pd.DataFrame(rows)
    stats = raw.groupby(["evaluation", "run"], sort=False)[KEYS].agg(["mean", "std"])
    stats.columns = [f"{k}_{s}" for k, s in stats.columns]
    stats = stats.reset_index()
    stats.insert(2, "seeds", stats.apply(lambda r: int(((raw["evaluation"] == r["evaluation"]) & (raw["run"] == r["run"])).sum()), axis=1))

    rob_rows = []
    for f in sorted((RESULTS / "robustness").glob("*/*/seed*/robustness.json")):
        r = json.loads(f.read_text())
        for name, m in r["levels"].items():
            rob_rows.append({"perturbation": r["perturbation"], "run": f.parts[-3], "seed": f.parts[-2], "level_name": name,
                             "level": m["level"], **{k: m[k] for k in ("eer", "auc", "fmr", "fnmr", "fta")}})
    rob = pd.DataFrame(rob_rows)

    cost = []
    for run in RUNS:
        t = [json.loads(f.read_text()) for f in sorted((ROOT / "runs" / run).glob("seed*/emb_fvc2002_db1_timing.json"))]
        j = [json.loads(f.read_text()) for f in sorted((ROOT / "runs" / run).glob("seed*/run.json"))]
        if t and j:
            cost.append({"run": run, "params_total": t[0]["params_total"], "params_trainable": j[0]["params_trainable"],
                         "model_size_mb": t[0]["model_size_bytes"] / 1e6,
                         "inference_mean_ms": sum(x["mean_ms"] for x in t) / len(t),
                         "train_minutes": sum(x["train_time_s"] for x in j) / len(j) / 60, "hardware": t[0]["hardware"],
                         "input_size": t[0]["input_size"], "torch": t[0]["torch"]})
    cost = pd.DataFrame(cost)

    raw.to_csv(RESULTS / "per_seed.csv", index=False)
    stats.to_csv(RESULTS / "summary.csv", index=False)
    if len(rob):
        rs = rob.groupby(["perturbation", "run", "level_name", "level"], sort=False)[["eer", "auc", "fmr", "fnmr", "fta"]].agg(["mean", "std"])
        rs.columns = [f"{k}_{s}" for k, s in rs.columns]
        rs = rs.reset_index().sort_values(["perturbation", "level"])
        rs.to_csv(RESULTS / "robustness_summary.csv", index=False)
    if len(cost):
        cost.to_csv(RESULTS / "cost.csv", index=False)

    lines = ["# Resultados", "", "Média ± desvio-padrão amostral de três sementes de treino. Taxas em %. FMR e FNMR no limiar congelado na validação.", ""]
    for ev, title in EVALS.items():
        lines += [f"## {title}", "", "| Modelo | Limiar | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |", "|---|---|---|---|---|---|---|---|"]
        for _, r in stats[stats["evaluation"] == ev].iterrows():
            lines.append(f"| {RUNS[r['run']]} | " + " | ".join(cell(r[f"{k}_mean"], r[f"{k}_std"], k) for k in KEYS) + " |")
        lines.append("")
    if len(rob):
        lines += ["## Experimento D: robustez, 30 dedos de teste da FVC2002 DB1", "",
                  "| Perturbação | Nível | EER | AUC | FMR | FNMR | FTA |", "|---|---|---|---|---|---|---|"]
        for _, r in rs.iterrows():
            lines.append(f"| {r['perturbation']} | {r['level_name']} ({r['level']:g}) | " +
                         " | ".join(cell(r[f"{k}_mean"], r[f"{k}_std"], k) for k in ("eer", "auc", "fmr", "fnmr", "fta")) + " |")
        lines.append("")
    if len(cost):
        lines += ["## Custo", "", "| Modelo | Parâmetros | Treináveis | Pesos (MB) | Inferência média (ms) | Treino (min) | Hardware |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in cost.iterrows():
            lines.append(f"| {RUNS[r['run']]} | {r['params_total']:,} | {r['params_trainable']:,} | {r['model_size_mb']:.1f} | "
                         f"{r['inference_mean_ms']:.2f} | {r['train_minutes']:.1f} | {r['hardware']} |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
