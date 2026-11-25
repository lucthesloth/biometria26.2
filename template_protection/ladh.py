"""Label-assisted discrete hashing for protected biometric templates (Raja, Raghavendra and Busch, IET Biometrics
11(1), 2022, doi 10.1049/bme2.12043).

The enrolment features X (n x d) and their subject labels give a similarity matrix S: 1 for two samples of
the same subject and 0 otherwise (endnote 2 of the paper; its Section 2.2 writes {-1, 1}). The binary
code matrix B in {-1, +1}^(n x q) minimises ||q S - B B^T||_F^2 (Eq. 1), starting from the XOR of two random
binary matrices drawn from the application keys k1 and k2 (Eq. 3). A linear projection W maps features to the
codes (Eq. 7), and the protected template is the sign of X W as q bits (Eq. 8). Templates are compared with the
Hamming distance.

The paper takes the optimiser from Kang, Li and Zhou (COSDISH, AAAI 2016) without restating it. This file
follows that scheme: sample a column subset, update its rows bit by bit as a binary quadratic problem, then
set the remaining rows in closed form. The bit-wise problem is solved here by coordinate ascent, not by the
2-approximation of the COSDISH paper.
"""
from __future__ import annotations

import numpy as np


def key_matrix(n: int, q: int, keys: tuple[int, ...]) -> np.ndarray:
    """Eq. 3 and 9: XOR of one thresholded normal matrix per key, in {-1, +1}."""
    out = np.ones((n, q), dtype=np.float32)
    for i, k in enumerate(keys):
        bits = np.where(np.random.default_rng(int(k)).standard_normal((n, q)) >= 0, 1.0, -1.0)
        out = bits if i == 0 else -out * bits            # XOR of two bits is +1 where they differ
    return out.astype(np.float32)


def similarity(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(np.float32)


def loss(B: np.ndarray, S: np.ndarray) -> float:
    q = B.shape[1]
    return float(((q * S - B @ B.T) ** 2).sum())


def learn_codes(labels: np.ndarray, q: int, keys: tuple[int, ...], n_iter: int = 5, n_alt: int = 2,
                sweeps: int = 2) -> np.ndarray:
    """Minimise Eq. 6 by column sampling. Returns B in {-1, +1}^(n x q)."""
    S = q * similarity(labels)
    n = len(S)
    B = key_matrix(n, q, keys)
    rng = np.random.default_rng([int(k) for k in keys] + [q])
    m = min(q, max(2, n // 2))
    for _ in range(n_iter):
        omega = np.sort(rng.choice(n, size=m, replace=False))
        gamma = np.setdiff1d(np.arange(n), omega)
        S_oo, S_go = S[np.ix_(omega, omega)], S[np.ix_(gamma, omega)]
        for _ in range(n_alt):
            B_o, B_g = B[omega].copy(), B[gamma]
            P_oo, P_go = B_o @ B_o.T, B_g @ B_o.T
            for k in range(q):
                b, bg = B_o[:, k].copy(), B_g[:, k]
                # Residuals without bit k; maximise b^T R b + g^T b over b in {-1, +1}^m.
                R = S_oo - P_oo + np.outer(b, b)
                np.fill_diagonal(R, 0.0)
                g = (S_go - P_go + np.outer(bg, b)).T @ bg
                old = b.copy()
                for _ in range(sweeps):
                    changed = False
                    for i in range(m):
                        v = 2.0 * R[i] @ b + g[i]
                        new = b[i] if v == 0 else (1.0 if v > 0 else -1.0)
                        if new != b[i]:
                            b[i], changed = new, True
                    if not changed:
                        break
                P_oo += np.outer(b, b) - np.outer(old, old)
                P_go += np.outer(bg, b) - np.outer(bg, old)
                B_o[:, k] = b
            B[omega] = B_o
            if len(gamma):
                vote = S_go @ B_o                       # q times the sum of the same-subject codes in omega
                B[gamma] = np.where(vote > 0, 1.0, np.where(vote < 0, -1.0, B[gamma]))
    return B


def learn_projection(X: np.ndarray, B: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression from features to codes: the out-of-sample hash of Kang et al., W in Eq. 7."""
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d, dtype=X.dtype), X.T @ B).astype(np.float32)


def fit(X: np.ndarray, labels: np.ndarray, q: int, keys: tuple[int, ...], lam: float = 1.0, **kw) -> np.ndarray:
    return learn_projection(X.astype(np.float32), learn_codes(labels, q, keys, **kw), lam)


def protect(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Eq. 7 and 8: q-bit protected template, one row per sample."""
    return (X @ W >= 0).astype(np.uint8)


def hamming(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Normalised Hamming distance between every row of A and every row of B."""
    A, B = A.astype(np.float32), B.astype(np.float32)
    return (A @ (1 - B).T + (1 - A) @ B.T) / A.shape[1]
