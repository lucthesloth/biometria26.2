import pandas as pd
import pytest

from fpverif.protocol import build_pairs, genuine_pairs, impostor_pairs, official_protocol, split_identities


def fake_index(n_fingers=110, n_impressions=8):
    rows = []
    for f in range(1, n_fingers + 1):
        for i in range(1, n_impressions + 1):
            subset = "A" if f <= 100 else "B"
            rows.append({"path": f"DB1_{subset}/{f}_{i}.tif", "finger": f, "impression": i, "subset": subset})
    return pd.DataFrame(rows)


def test_official_protocol_counts_match_fvc():
    pairs = official_protocol(fake_index())
    assert (pairs["type"] == "genuine").sum() == 2800
    assert (pairs["type"] == "impostor").sum() == 4950


def test_split_is_identity_disjoint_and_reproducible():
    fingers = list(range(1, 111))
    sizes = {"train": 60, "validation": 20, "test": 30}
    a = split_identities(fingers, sizes, 42)
    b = split_identities(fingers, sizes, 42)
    assert a == b
    assert not set(a["train"]) & set(a["test"])
    assert not set(a["train"]) & set(a["validation"])
    assert sorted(a["train"] + a["validation"] + a["test"]) == fingers
    assert split_identities(fingers, sizes, 123) != a


def test_split_rejects_wrong_total():
    with pytest.raises(ValueError):
        split_identities(list(range(10)), {"train": 5, "test": 3}, 42)


def test_pairs_never_mix_types():
    idx = fake_index(n_fingers=5, n_impressions=3)
    gen = genuine_pairs(idx)
    assert len(gen) == 5 * 3
    imp_all = impostor_pairs(idx, "all")
    assert len(imp_all) == (15 * 14 // 2) - 15
    imp_fvc = impostor_pairs(idx, "fvc")
    assert len(imp_fvc) == 10
    both = build_pairs(idx, "all")
    assert not both.duplicated(subset=["sample_1", "sample_2"]).any()
