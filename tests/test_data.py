import numpy as np
from PIL import Image

from fpverif.data import dataset_index, parse_name
from fpverif.protocol import build_pairs


def test_parse_name_handles_fvc_and_neurotechnology_names():
    assert parse_name("DB1_B/101_3.tif") == (101, 3)
    assert parse_name("012_1_8.tif") == (121, 8)
    assert parse_name("012_2_8.tif") == (122, 8)


def test_pool_renumbers_fingers_per_source(tmp_path):
    for sub in ("a/DB1_B", "b/DB2_B"):
        (tmp_path / sub).mkdir(parents=True)
        for i in (1, 2):
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / sub / f"101_{i}.tif")
    cfg = {"datasets": {"a": {"root": str(tmp_path / "a"), "dpi": 500},
                        "b": {"root": str(tmp_path / "b"), "dpi": 569},
                        "pool": {"root": str(tmp_path), "sources": ["a", "b"]}}}
    frame, root = dataset_index(cfg, "pool")
    assert sorted(frame["finger"].unique()) == [1101, 2101]
    assert set(frame["path"]) == {"a/DB1_B/101_1.tif", "a/DB1_B/101_2.tif", "b/DB2_B/101_1.tif", "b/DB2_B/101_2.tif"}
    assert frame.groupby("source")["dpi"].first().to_dict() == {"a": 500.0, "b": 569.0}
    assert all((root / p).exists() for p in frame["path"])
    # Pair files name samples by `sample`, whose stem must be the processed file name <finger>_<impression>.
    pairs = build_pairs(frame, "all")
    names = set(pairs["sample_1"]) | set(pairs["sample_2"])
    assert names == {f"{f}_{i}.png" for f, i in zip(frame["finger"], frame["impression"])}

    single, _ = dataset_index(cfg, "a")
    assert set(single["sample"]) == {"101_1.tif", "101_2.tif"}


def test_subdirs_limit_the_index(tmp_path):
    for sub in ("Db1_a", "Db1_b", "Db2_a"):
        (tmp_path / sub).mkdir()
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / sub / "1_1.tif")
    cfg = {"datasets": {"db1": {"root": str(tmp_path), "subdirs": ["Db1_a", "Db1_b"], "dpi": 500}}}
    frame, _ = dataset_index(cfg, "db1")
    assert sorted(frame["path"]) == ["Db1_a/1_1.tif", "Db1_b/1_1.tif"]
    assert sorted(frame["subset"]) == ["A", "B"]
