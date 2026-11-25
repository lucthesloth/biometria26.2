"""Figure of the optimisation steps: EER, FMR and FNMR of each configuration on two tests.

    python otimizacao/figure.py        # writes artigo/figures/otimizacao.{pdf,png}

Reads results/otimizacao/summary.csv (otimizacao/table.py). FMR and FNMR are at each configuration's frozen
threshold. Style and colours come from scripts/paper_figures.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_figures import AQUA, BLUE, ORANGE, save  # noqa: E402

STEPS = [("base (gabor)", "Base"), ("base_gabor_hong", "Base\n2 redes"), ("pool", "390\n1 rede"),
         ("pool_gabor_hong", "390\n2 redes"), ("wide_gh", "745\n2 redes"), ("wide_4_as", "745\n4 redes"),
         ("wide_12_as", "745\n12 redes")]
PANELS = {"fvc2002_db1": "FVC2002 DB1, teste", "fvc2004_db1_official": "FVC2004 DB1-A, oficial"}
SERIES = (("eer_mean", "EER", BLUE, "o"), ("fmr_mean", "FMR no limiar", AQUA, "^"), ("fnmr_mean", "FNMR no limiar", ORANGE, "s"))


def main() -> None:
    s = pd.read_csv(ROOT / "results" / "otimizacao" / "summary.csv").set_index(["variant", "evaluation"])
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True, sharey=True)
    for ax, (ev, title) in zip(axes, PANELS.items()):
        for key, label, color, marker in SERIES:
            ax.plot(range(len(STEPS)), [s.loc[(v, ev), key] for v, _ in STEPS], color=color, marker=marker,
                    markersize=4, label=label)
        ax.set_title(title)
        ax.set_ylabel("taxa (%)")
        ax.set_ylim(0, 15)
        ax.set_xticks(range(len(STEPS)))
        ax.set_xticklabels([label for _, label in STEPS], fontsize=6.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.8)
    save(fig, ROOT / "artigo" / "figures", "otimizacao")


if __name__ == "__main__":
    main()
