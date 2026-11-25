"""Mean ± std across seeds for every evaluation, plus the assignment's standard results table (section 26).

    python scripts/aggregate.py            # reads results/ and runs/, writes results/summary.{csv,md}

Layout it expects (written by scripts/run_pipeline.sh):
    results/<evaluation>/<run>/seed<S>/metrics.json                 evaluation = protocol folder name
    results/robustness/<perturbation>_<apply_to>/<run>/seed<S>/robustness.json
    runs/<run>/seed<S>/emb_<dataset>_timing.json                     latency, parameters, model size
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.config import load_config  # noqa: E402

METRICS = ["eer", "auc", "fmr", "fnmr", "tar_at_far_1pct", "fmr100_fnmr", "d_prime", "fta", "threshold", "eer_threshold"]
EVAL_LABELS = {
    "fvc2002_db1_val": "validação (FVC2002 DB1, 20 dedos)",
    "fvc2002_db1": "FVC2002 DB1, 30 dedos de teste (experimento A)",
    "fvc2004_db1": "FVC2004 DB1, 30 dedos de teste (experimento C)",
    "fvc2004_db1_official": "FVC2004 DB1, protocolo oficial no conjunto A (experimento C)",
}


def pm(mean: float, std: float, digits: int = 4, pct: bool = False) -> str:
    if pd.isna(mean):
        return "n/d"
    k = 100 if pct else 1
    s = f"{mean * k:.{digits}f}"
    return s + (f" ± {std * k:.{digits}f}" if not pd.isna(std) else "") + ("%" if pct else "")


def load_metrics(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(results.glob("*/*/seed*/metrics.json")):
        evaluation, run, seed = f.parts[-4], f.parts[-3], int(f.parts[-2].removeprefix("seed"))
        if evaluation == "robustness":
            continue
        m = json.loads(f.read_text())
        rows.append({"evaluation": evaluation, "run": run, "seed": seed, "n_genuine": m["n_genuine"],
                     "n_impostor": m["n_impostor"], **{k: m.get(k) for k in METRICS}})
    return pd.DataFrame(rows)


def load_timing(runs: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(runs.glob("*/seed*/emb_*_timing.json")):
        t = json.loads(f.read_text())
        rows.append({"run": f.parts[-3], "seed": int(f.parts[-2].removeprefix("seed")),
                     "median_ms": t["median_ms"], "mean_ms": t["mean_ms"], "params": t["params_total"], "size_mb": t["model_size_bytes"] / 1e6,
                     "hardware": t["hardware"], "device": t["device"], "input": t["input_size"]})
    return pd.DataFrame(rows)


def load_robustness(results: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((results / "robustness").glob("*/*/seed*/robustness.json")):
        r = json.loads(f.read_text())
        for level, m in r["levels"].items():
            rows.append({"setting": f.parts[-4], "run": f.parts[-3], "seed": int(f.parts[-2].removeprefix("seed")),
                         "level_name": level, "level": m["level"],
                         **{k: m.get(k) for k in ("eer", "fmr", "fnmr", "auc", "fta")}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index()


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    results, runs = ROOT / cfg["eval"]["results_dir"], ROOT / cfg["train"]["runs_dir"]
    metrics = load_metrics(results)
    if metrics.empty:
        raise SystemExit("no metrics.json under results/")
    summary = summarize(metrics, ["evaluation", "run"], METRICS)
    pairs = metrics.groupby(["evaluation", "run"])[["n_genuine", "n_impostor"]].first().reset_index()
    summary = summary.merge(pairs, on=["evaluation", "run"])
    timing = load_timing(runs)
    summary.to_csv(results / "summary.csv", index=False)

    lines = ["# Resultados agregados", "", "Sementes de treino 2026, 1999 e 1234 (média ± desvio quando há mais de uma); divisão e pares com semente 42.", ""]
    lines += ["## Verificação 1:1", "",
              "| Avaliação | Execução | Seeds | Gen./Imp. | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary.itertuples(index=False):
        # FAR = 1% needs at least 100 impostor pairs; below that the operating point does not exist.
        tar = pm(r.tar_at_far_1pct_mean, r.tar_at_far_1pct_std, 2, True) if r.n_impostor >= 100 else "n/d"
        lines.append(f"| {EVAL_LABELS.get(r.evaluation, r.evaluation)} | {r.run} | {r.n_seeds} | {r.n_genuine}/{r.n_impostor} | "
                     f"{pm(r.eer_mean, r.eer_std, 2, True)} | {pm(r.auc_mean, r.auc_std)} | {pm(r.fmr_mean, r.fmr_std, 2, True)} | "
                     f"{pm(r.fnmr_mean, r.fnmr_std, 2, True)} | {tar} | {pm(r.fta_mean, r.fta_std, 2, True)} |")

    if not timing.empty:
        t = timing.groupby("run").agg(median_ms=("median_ms", "median"), mean_ms=("mean_ms", "mean"), params=("params", "first"),
                                      size_mb=("size_mb", "first"), hardware=("hardware", "first"),
                                      device=("device", "first"), input=("input", "first")).reset_index()
        t.to_csv(results / "efficiency.csv", index=False)
        lines += ["", "## Custo computacional", "", "| Execução | Parâmetros | Tamanho (state_dict) | Inferência (batch 1, média) | Inferência (batch 1, mediana) | Hardware |",
                  "|---|---|---|---|---|---|"]
        for r in t.itertuples(index=False):
            lines.append(f"| {r.run} | {r.params / 1e6:.2f} M | {r.size_mb:.1f} MB | {r.mean_ms:.2f} ms | {r.median_ms:.2f} ms ({r.device}, {r.input}x{r.input}) | {r.hardware} |")

    rob = load_robustness(results)
    if not rob.empty:
        rs = summarize(rob, ["setting", "run", "level_name", "level"], ["eer", "fmr", "fnmr", "auc", "fta"])
        rs.to_csv(results / "robustness_summary.csv", index=False)
        lines += ["", "## Robustez (limiar congelado)", "", "| Perturbação | Execução | Nível | EER | FMR | FNMR | AUC | FTA |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rs.itertuples(index=False):
            lines.append(f"| {r.setting} | {r.run} | {r.level_name} ({r.level}) | {pm(r.eer_mean, r.eer_std, 2, True)} | "
                         f"{pm(r.fmr_mean, r.fmr_std, 2, True)} | {pm(r.fnmr_mean, r.fnmr_std, 2, True)} | "
                         f"{pm(r.auc_mean, r.auc_std)} | {pm(r.fta_mean, r.fta_std, 2, True)} |")

    main_run = f"{cfg['model']['backbone']}_{cfg['preprocess']['normalization']}"
    lines += ["", f"## Tabela padronizada (seção 26), execução `{main_run}`", ""] + standard_table(cfg, summary, timing, main_run)
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def standard_table(cfg: dict, summary: pd.DataFrame, timing: pd.DataFrame, run: str) -> list[str]:
    cols = [("fvc2002_db1", "FVC2002 DB1 (base principal)"), ("fvc2004_db1", "FVC2004 DB1 (cross-dataset)")]
    get = {e: summary[(summary["evaluation"] == e) & (summary["run"] == run)] for e, _ in cols}
    tm = timing[timing["run"] == run] if not timing.empty else timing

    def val(e: str, key: str, digits: int = 2, pct: bool = True) -> str:
        r = get[e]
        return pm(r[f"{key}_mean"].iloc[0], r[f"{key}_std"].iloc[0], digits, pct) if len(r) else "n/d"

    m = cfg["model"]
    lines = ["| Informação | " + " | ".join(label for _, label in cols) + " |", "|---|---|---|"]
    fixed = {
        "Modalidade": "Impressão digital, verificação 1:1",
        "CNN": m["backbone"],
        "Pré-treinamento": f"ImageNet-1k (torchvision {m['weights']}), fine-tuning em 60 dedos da FVC2002 DB1",
        "Nº de parâmetros": f"{tm['params'].iloc[0] / 1e6:.2f} M" if len(tm) else "n/d",
        "Dimensão do embedding": f"{m['embedding_dim']}, normalizado L2",
        "Matching": "similaridade de cosseno",
    }
    lines.append("| Dataset | FVC2002 DB1, 30 dedos de teste | FVC2004 DB1, 30 dedos de teste |")
    for k, v in fixed.items():
        lines.append(f"| {k} | {v} | {v} |")
    lines.append(f"| Threshold | {val(cols[0][0], 'threshold', 4, False)} (EER na validação, congelado) | mesmo limiar |")
    for label, key, digits, pct in (("EER", "eer", 2, True), ("AUC", "auc", 4, False), ("FMR", "fmr", 2, True),
                                    ("FNMR", "fnmr", 2, True), ("TAR@FAR=1%", "tar_at_far_1pct", 2, True), ("FTA", "fta", 2, True)):
        lines.append(f"| {label} | {val(cols[0][0], key, digits, pct)} | {val(cols[1][0], key, digits, pct)} |")
    if len(tm):
        t = (f"média {tm['mean_ms'].mean():.2f} ms, mediana {tm['median_ms'].median():.2f} ms por imagem "
             f"(batch 1, {tm['device'].iloc[0]})")
        lines += [f"| Tempo de inferência | {t} | {t} |",
                  f"| Tamanho do modelo | {tm['size_mb'].iloc[0]:.1f} MB | {tm['size_mb'].iloc[0]:.1f} MB |",
                  f"| Hardware | {tm['hardware'].iloc[0]} | {tm['hardware'].iloc[0]} |"]
    return lines


if __name__ == "__main__":
    main()
