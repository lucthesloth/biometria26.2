import numpy as np
import pandas as pd

from evaluate import sample_fta, score_pairs


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


PAIRS = pd.DataFrame({"sample_1": ["DB1_A/1_1.tif", "DB1_A/1_1.tif", "DB1_A/1_2.tif"],
                      "sample_2": ["DB1_A/1_2.tif", "DB1_A/2_1.tif", "DB1_A/2_1.tif"],
                      "type": ["genuine", "impostor", "impostor"]})


def test_fta_is_per_sample_and_drop_rate_is_per_pair():
    emb = {"1_1.png": unit([1, 0]), "1_2.png": unit([1, 1]), "2_1.png": unit([0, 1])}
    qc = pd.DataFrame({"path": ["1_1.png", "1_2.png", "2_1.png"], "acquired": [True, True, False]})
    scores = score_pairs(emb, PAIRS, qc)
    assert scores["score"].isna().tolist() == [False, True, True]
    assert abs(sample_fta(PAIRS, qc) - 1 / 3) < 1e-12


def test_probe_scoring_reads_sample_2_from_the_second_set():
    clean = {"1_1.png": unit([1, 0]), "1_2.png": unit([1, 0]), "2_1.png": unit([0, 1])}
    degraded = {"1_1.png": unit([0, 1]), "1_2.png": unit([0, 1]), "2_1.png": unit([1, 0])}
    scores = score_pairs(clean, PAIRS, emb_2=degraded)["score"].to_numpy()
    np.testing.assert_allclose(scores, [0.0, 1.0, 1.0], atol=1e-6)

    qc_clean = pd.DataFrame({"path": ["1_1.png", "1_2.png", "2_1.png"], "acquired": [True, True, True]})
    qc_deg = pd.DataFrame({"path": ["1_1.png", "1_2.png", "2_1.png"], "acquired": [False, False, True]})
    # sample_1 uses 1_1 and 1_2 (clean, both fine); sample_2 uses 1_2 and 2_1 (degraded, 1_2 failed).
    assert abs(sample_fta(PAIRS, qc_clean, qc_deg) - 1 / 4) < 1e-12
    dropped = score_pairs(clean, PAIRS, qc_clean, degraded, qc_deg)["score"].isna().tolist()
    assert dropped == [True, False, False]
