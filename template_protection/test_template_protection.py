import numpy as np

import bloom
import ladh
from unlinkability import d_sys

LABELS = np.repeat(np.arange(12), 4)


def test_key_matrix_is_the_xor_of_two_keyed_matrices():
    a, b = ladh.key_matrix(5, 16, (1,)), ladh.key_matrix(5, 16, (2,))
    both = ladh.key_matrix(5, 16, (1, 2))
    assert set(np.unique(both)) <= {-1.0, 1.0}
    assert np.array_equal(both == 1, a != b)


def test_codes_lower_the_loss_and_agree_within_a_subject():
    S = ladh.similarity(LABELS)
    B = ladh.learn_codes(LABELS, 32, (1, 2))
    assert ladh.loss(B, S) < 0.2 * ladh.loss(ladh.key_matrix(len(LABELS), 32, (1, 2)), S)
    assert all((B[LABELS == c] == B[LABELS == c][0]).all() for c in range(12))


def test_other_keys_give_unrelated_codes():
    a, b = ladh.learn_codes(LABELS, 64, (1, 2)), ladh.learn_codes(LABELS, 64, (3, 4))
    assert 0.4 < (a == b).mean() < 0.6


def test_protected_templates_separate_subjects():
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((12, 40))
    X = (centres[LABELS] + 0.3 * rng.standard_normal((48, 40))).astype(np.float32)
    P = (centres[LABELS] + 0.3 * rng.standard_normal((48, 40))).astype(np.float32)
    W = ladh.fit(X, LABELS, 32, (1, 2), lam=1.0)
    d = ladh.hamming(ladh.protect(X, W), ladh.protect(P, W))
    same = LABELS[:, None] == LABELS[None, :]
    assert d[same].mean() < 0.1 < 0.3 < d[~same].mean()


def test_hamming_counts_disagreeing_bits():
    a = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    b = np.array([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
    assert np.allclose(ladh.hamming(a, b), [[0.5, 0.0]])


def test_bloom_filter_sets_one_bit_per_distinct_word():
    bits = np.array([[0, 0, 1, 0, 1, 1, 0, 0]], dtype=np.uint8)     # words 0, 1, 3, 0 with word_bits = 2
    t = bloom.protect(bits, word_bits=2, block_words=4, key=5)
    assert t.shape == (1, 1, 4) and t.sum() == 3
    assert bloom.dissimilarity(t, t)[0, 0] == 0


def test_d_sys_is_zero_for_equal_distributions_and_one_for_disjoint():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.05, 200_000)
    assert d_sys(x[:100_000], x[100_000:])[0] < 0.02
    assert d_sys(np.full(100, 0.1), np.full(100, 0.9))[0] == 1.0
