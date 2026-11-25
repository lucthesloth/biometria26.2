"""Bloom-filter protected templates (Rathgeb, Breitinger and Busch, ICB 2013), the baseline of the paper.

A binary code is cut into words of `word_bits` bits. Each block of `block_words` words becomes one Bloom
filter of 2^word_bits bits: the filter bit indexed by (word XOR key) is set. The dissimilarity of two templates
is HD(b1, b2) / (|b1| + |b2|) per block, averaged over blocks.
"""
from __future__ import annotations

import numpy as np


def binarize(X: np.ndarray) -> np.ndarray:
    return (X >= 0).astype(np.uint8)


def protect(bits: np.ndarray, word_bits: int, block_words: int, key: int) -> np.ndarray:
    """bits: n x L in {0, 1}. Returns n x n_blocks x 2^word_bits Bloom filters."""
    n, length = bits.shape
    n_words = length // word_bits
    n_blocks = n_words // block_words
    words = bits[:, :n_words * word_bits].reshape(n, n_words, word_bits) @ (1 << np.arange(word_bits))
    # One secret per block (Gomez-Barrero et al. 2016), so the same word maps to a different bit in each block.
    secret = np.random.default_rng(int(key)).integers(0, 1 << word_bits, size=n_blocks)
    words = words[:, :n_blocks * block_words].reshape(n, n_blocks, block_words) ^ secret[None, :, None]
    out = np.zeros((n, n_blocks, 1 << word_bits), dtype=np.uint8)
    rows, blocks = np.meshgrid(np.arange(n), np.arange(n_blocks), indexing="ij")
    for j in range(block_words):
        out[rows, blocks, words[:, :, j]] = 1
    return out


def dissimilarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Every template of A against every template of B."""
    a, b = A.astype(np.float32), B.astype(np.float32)
    inter = np.einsum("ikw,jkw->ijk", a, b)
    size = a.sum(-1)[:, None, :] + b.sum(-1)[None, :, :]
    return ((size - 2 * inter) / size).mean(-1)
