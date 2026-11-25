"""Pack what the Colab notebook needs into one zip, so the GPU work runs on Colab instead of this machine.

    python scripts/make_colab_bundle.py        # writes colab/biometria_colab.zip

The zip holds the code, protocols and the images the GPU stage reads (preprocessed here, so every arm sees the
same inputs). SourceAFIS, fusion, tables and figures run locally afterwards (STAGE=cpu). The images are
licensed for research use: keep the zip in your own Drive and never commit it.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "biometria_colab.zip"
FILES = ["config.yaml", "pyproject.toml", "uv.lock", "requirements.txt", "make_protocols.py", "preprocess.py",
         "model.py", "train.py", "extract_embeddings.py", "evaluate.py", "robustness.py"]
# The GPU stage (STAGE=gpu) reads only these: preprocessed images of both bases, and the raw FVC2002 DB1 images
# that robustness.py degrades and preprocesses again.
DIRS = ["fpverif", "scripts", "tests", "protocols", "data/raw/FVC2002/Dbs/Db1_a",
        "data/raw/FVC2002/Dbs/Db1_b"] + [
    f"data/processed/{d}/{n}" for d in ("fvc2002_db1", "fvc2004_db1") for n in ("gabor", "hong")]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    paths = [ROOT / f for f in FILES]
    for d in DIRS:
        paths += [p for p in sorted((ROOT / d).rglob("*")) if p.is_file() and not SKIP_PARTS & set(p.parts)]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{OUT} ({len(paths)} files, {OUT.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
