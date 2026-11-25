"""EER on the two 30-finger test splits with and without the set-B fingers that fall in them.

    python scripts/sensitivity_setb.py        # writes results/sensitivity_setb.csv

The training configuration was chosen in an earlier phase whose test set was set B (fingers 101 to 110). The
seed-42 split puts fingers 107 and 109 in the test split of both databases. This recomputes the EER from the
stored score files without every pair that touches one of them, mean over the training seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fpverif.metrics import eer  # noqa: E402

EVALS = ["fvc2002_db1", "fvc2004_db1"]
RUNS = ["convnext_tiny_gabor", "resnet50_gabor", "efficientnet_b0_gabor", "convnext_tiny_hong", "sourceafis",
        "convnext_tiny_gabor_fusion"]


def main() -> None:
    rows = []
    for evaluation in EVALS:
        test = pd.read_csv(ROOT / "protocols" / evaluation / "test.csv")
        set_b = sorted(str(f) for f in test.loc[test["subset"] == "B", "finger"].unique())
        for run in RUNS:
            full, kept, n = [], [], (0, 0)
            for f in sorted((ROOT / "results" / evaluation / run).glob("seed*/scores.csv")):
                s = pd.read_csv(f).dropna(subset=["score"])
                touches = s["amostra_1"].str.split("_").str[0].isin(set_b) | s["amostra_2"].str.split("_").str[0].isin(set_b)
                for frame, out in ((s, full), (s[~touches], kept)):
                    genuine = frame["tipo"] == "genuine"
                    out.append(eer(frame.loc[genuine, "score"].to_numpy(), frame.loc[~genuine, "score"].to_numpy())[0])
                n = (int(((s["tipo"] == "genuine") & ~touches).sum()), int(((s["tipo"] != "genuine") & ~touches).sum()))
            if full:
                rows.append({"evaluation": evaluation, "run": run, "set_b_fingers": " ".join(set_b), "n_seeds": len(full),
                             "eer_all": sum(full) / len(full), "eer_without_set_b": sum(kept) / len(kept),
                             "n_genuine_without": n[0], "n_impostor_without": n[1]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "sensitivity_setb.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{100 * v:.2f}"))


if __name__ == "__main__":
    main()
