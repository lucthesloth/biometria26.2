import numpy as np
from PIL import Image

from fpverif.data import dataset_index, parse_name
from fpverif.protocol import build_pairs


def test_parse_name_reads_finger_and_impression():
    assert parse_name("Db1_a/12_3.tif") == (12, 3)
    assert parse_name("101_8.TIF") == (101, 8)


def test_subdirs_limit_the_index(tmp_path):
    for sub in ("Db1_a", "Db1_b", "Db2_a"):
        (tmp_path / sub).mkdir()
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path / sub / "1_1.tif")
    cfg = {"datasets": {"db1": {"root": str(tmp_path), "subdirs": ["Db1_a", "Db1_b"], "dpi": 500}}}
    frame, _ = dataset_index(cfg, "db1")
    assert sorted(frame["path"]) == ["Db1_a/1_1.tif", "Db1_b/1_1.tif"]
    assert sorted(frame["subset"]) == ["A", "B"]
