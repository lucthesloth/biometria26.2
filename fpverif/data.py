"""Indexing of FVC-style datasets and PyTorch dataset/sampler helpers."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# FVC: <finger>_<impression>.tif.
NAME_RE = re.compile(r"^(\d+)_(\d+)\.(tif|tiff|png|bmp|jpg|jpeg)$", re.IGNORECASE)


def parse_name(path: str | Path) -> tuple[int, int]:
    m = NAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not an FVC file name: {path}")
    return int(m.group(1)), int(m.group(2))


def index_dataset(root: str | Path, subdirs: list[str] | None = None) -> pd.DataFrame:
    """Return one row per image: path (relative to root), finger, impression, subset (A, B or unknown).

    `subdirs` limits the search to those folders of `root` (the FVC DVDs keep four databases side by side)."""
    root = Path(root)
    rows = []
    files = root.rglob("*") if subdirs is None else (p for d in subdirs for p in (root / d).rglob("*"))
    for p in sorted(files):
        if not p.is_file() or not NAME_RE.match(p.name):
            continue
        finger, impression = parse_name(p)
        parent = p.parent.name.upper()
        subset = parent[-1] if parent.endswith(("_A", "_B")) else "unknown"
        rows.append({"path": p.relative_to(root).as_posix(), "finger": finger, "impression": impression,
                     "subset": subset})
    if not rows:
        raise FileNotFoundError(f"no FVC images under {root}")
    return pd.DataFrame(rows)


def dataset_index(cfg: dict, name: str) -> tuple[pd.DataFrame, Path]:
    """Index a configured dataset. Returns (frame with dpi and sample columns, root that frame paths are relative to).
    `sample` is what pair files and scores.csv name an image by: the original file name (101_1.tif)."""
    ds = cfg["datasets"][name]
    frame = index_dataset(ds["root"], ds.get("subdirs"))
    return frame.assign(sample=frame["path"].map(lambda p: Path(p).name), dpi=float(ds["dpi"]), source=name), \
        Path(ds["root"])



class PKBatchSampler:
    """Yields batches of P identities x K samples. Identities with fewer than K samples are oversampled."""

    def __init__(self, labels: np.ndarray, p: int, k: int, seed: int):
        self.labels = np.asarray(labels)
        self.p, self.k = p, k
        self.rng = np.random.default_rng(seed)
        self.by_label = {int(l): np.flatnonzero(self.labels == l) for l in np.unique(self.labels)}
        self.n_batches = max(1, math.ceil(len(self.labels) / (p * k)))

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        ids = np.array(list(self.by_label))
        for _ in range(self.n_batches):
            chosen = self.rng.choice(ids, size=min(self.p, len(ids)), replace=False)
            batch = []
            for c in chosen:
                pool = self.by_label[int(c)]
                batch.extend(self.rng.choice(pool, size=self.k, replace=len(pool) < self.k).tolist())
            yield batch


class ImageFrameDataset:
    """Map-style dataset over a frame of processed images. Module-level so DataLoader workers can pickle it
    on Windows, where workers are spawned rather than forked."""

    def __init__(self, frame: pd.DataFrame, image_root: str | Path, transform, label_map: dict[int, int],
                 label_col: str = "finger"):
        self.paths = frame["path"].tolist()
        self.labels = [label_map[int(f)] for f in frame[label_col]]
        self.image_root, self.transform = Path(image_root), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.open(self.image_root / self.paths[i]).convert("L")
        return self.transform(img), self.labels[i]


def make_torch_dataset(frame: pd.DataFrame, image_root: str | Path, transform, label_col: str = "finger"):
    label_map = {int(f): i for i, f in enumerate(sorted(frame[label_col].unique()))}
    return ImageFrameDataset(frame, image_root, transform, label_map, label_col), label_map
