"""Classical baseline: SourceAFIS 3.18.1 (minutiae matcher, Apache 2.0) on the raw images and the same pair files.

    mvn -q -B -f baselines/sourceafis/pom.xml package dependency:copy-dependencies     # once, needs JDK 17+
    python scripts/baseline_sourceafis.py

The threshold is the EER threshold on the FVC2002 DB1 validation fingers, frozen for every test protocol, as for the CNNs.
Writes results/<evaluation>/sourceafis/seed0/{scores.csv,metrics.json,threshold.json,...} through evaluate.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import choose_threshold, evaluate  # noqa: E402
from fpverif.config import load_config  # noqa: E402

JAVA_DIR = ROOT / "baselines" / "sourceafis" / "target"
RUN = "sourceafis"


def match(samples: pd.DataFrame, raw_root: Path, pairs: Path, work: Path) -> tuple[pd.DataFrame, float]:
    work.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame({"sample": samples["sample"], "path": [str(raw_root / p) for p in samples["path"]],
                             "dpi": samples["dpi"]})
    manifest.to_csv(work / "manifest.csv", index=False)
    cp = os.pathsep.join([str(JAVA_DIR / "classes"), str(JAVA_DIR / "dependency" / "*")])
    start = time.perf_counter()
    subprocess.run(["java", "-cp", cp, "Match", str(work / "manifest.csv"), str(pairs), str(work / "scores.csv")],
                   check=True)
    return pd.read_csv(work / "scores.csv"), time.perf_counter() - start


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    ds = cfg["datasets"]
    VAL = cfg["evaluations"]["validation"]
    work = ROOT / "runs" / RUN
    val = pd.read_csv(ROOT / "protocols" / VAL / "validation.csv")
    scores, secs = match(val, ROOT / ds[cfg["train_dataset"]]["root"], ROOT / "protocols" / VAL / "validation_pairs.csv",
                         work / VAL)
    thr = choose_threshold(scores, "eer", 0.0)
    thr["source"] = f"protocols/{VAL}/validation_pairs.csv"
    out = ROOT / "results" / VAL / RUN / "seed0"
    metrics = {VAL: evaluate(scores, thr["threshold"], out, score_name="score")}
    (out / "threshold.json").write_text(json.dumps(thr, indent=2))
    timing = {VAL: {"seconds": secs, "images": len(val), "pairs": len(scores)}}
    for proto, base in cfg["evaluations"]["tests"].items():
        pdir = ROOT / "protocols" / proto
        samples = pd.read_csv(pdir / "test.csv")
        scores, secs = match(samples, ROOT / ds[base]["root"], pdir / "test_pairs.csv", work / proto)
        out = ROOT / "results" / proto / RUN / "seed0"
        metrics[proto] = evaluate(scores, thr["threshold"], out, score_name="score")
        (out / "threshold.json").write_text(json.dumps(thr, indent=2))
        timing[proto] = {"seconds": secs, "images": len(samples), "pairs": len(scores)}
    (work / "timing.json").write_text(json.dumps(timing, indent=2))
    for k, m in metrics.items():
        print(k, {x: round(m[x] * 100, 2) for x in ["eer", "fmr", "fnmr", "tar_at_far_1pct"]}, round(m["auc"], 4),
              round(m["threshold"], 2))


if __name__ == "__main__":
    main()
