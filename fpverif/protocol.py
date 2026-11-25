"""Identity-disjoint splits and genuine/impostor pair generation (FVC rule or all cross pairs)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def split_identities(fingers: list[int], sizes: dict[str, int], seed: int) -> dict[str, list[int]]:
    fingers = sorted(set(fingers))
    total = sum(sizes.values())
    if total != len(fingers):
        raise ValueError(f"split sizes sum to {total} but there are {len(fingers)} fingers")
    perm = np.random.default_rng(seed).permutation(fingers)
    out, start = {}, 0
    for name, n in sizes.items():
        out[name] = sorted(int(f) for f in perm[start:start + n])
        start += n
    return out


def _ids(frame: pd.DataFrame) -> str:
    """Column that names samples in pair files: `sample` when the index has it, else the raw path."""
    return "sample" if "sample" in frame.columns else "path"


def genuine_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in frame.sort_values(["finger", "impression"]).groupby("finger"):
        for a, b in combinations(g[_ids(frame)].tolist(), 2):
            rows.append((a, b))
    return pd.DataFrame(rows, columns=["sample_1", "sample_2"]).assign(type="genuine")


def impostor_pairs(frame: pd.DataFrame, mode: str = "fvc") -> pd.DataFrame:
    frame = frame.sort_values(["finger", "impression"])
    if mode == "fvc":
        first = frame.groupby("finger").head(1)
        rows = list(combinations(first[_ids(frame)].tolist(), 2))
    elif mode == "all":
        paths = frame[_ids(frame)].to_numpy()
        fingers = frame["finger"].to_numpy()
        i, j = np.triu_indices(len(frame), k=1)
        keep = fingers[i] != fingers[j]
        rows = list(zip(paths[i][keep], paths[j][keep]))
    else:
        raise ValueError(f"unknown impostor mode {mode!r}")
    return pd.DataFrame(rows, columns=["sample_1", "sample_2"]).assign(type="impostor")


def build_pairs(frame: pd.DataFrame, impostor_mode: str) -> pd.DataFrame:
    return pd.concat([genuine_pairs(frame), impostor_pairs(frame, impostor_mode)], ignore_index=True)


def official_protocol(index: pd.DataFrame) -> pd.DataFrame:
    """FVC rule on set A: 100 fingers -> 2800 genuine and 4950 impostor comparisons.
    Without set A the same rule runs on what is available (set B: 280 genuine, 45 impostor)."""
    set_a = index[index["subset"] == "A"]
    return build_pairs(set_a if len(set_a) else index, "fvc")
