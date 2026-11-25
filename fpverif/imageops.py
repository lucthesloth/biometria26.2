"""Image operations shared by preprocess.py and robustness.py."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label


def block_std_map(img: np.ndarray, block: int) -> np.ndarray:
    h, w = img.shape
    ph, pw = (-h) % block, (-w) % block
    padded = np.pad(img.astype(np.float32), ((0, ph), (0, pw)), mode="reflect")
    blocks = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block)
    std = blocks.std(axis=(1, 3))
    return np.kron(std, np.ones((block, block), dtype=np.float32))[:h, :w]


def segment_foreground(img: np.ndarray, block: int = 16, std_threshold: float | None = None) -> np.ndarray:
    """Boolean mask of the fingerprint area: block std thresholding (Otsu), morphology, largest component, holes
    filled. Uses std, not variance, so Otsu keeps the whole print on low-contrast images."""
    std = block_std_map(img, block)
    thr = threshold_otsu(std) if std_threshold is None else std_threshold
    mask = (std > thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block * 2 + 1, block * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    labels = cc_label(mask, connectivity=2)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(labels == largest)


def hong_normalize(img: np.ndarray, mask: np.ndarray | None, m0: float, v0: float) -> np.ndarray:
    """Hong, Wan and Jain (1998) normalization to desired mean m0 and variance v0, statistics taken inside mask."""
    x = img.astype(np.float32)
    region = x[mask] if mask is not None and mask.any() else x
    m, v = float(region.mean()), float(region.var())
    if v < 1e-6:
        return np.full_like(x, m0)
    dev = np.sqrt(v0 * (x - m) ** 2 / v)
    out = np.where(x > m, m0 + dev, m0 - dev)
    return np.clip(out, 0, 255)


def roi_square(img: np.ndarray, mask: np.ndarray, fill: float) -> np.ndarray:
    """Smallest square around the bounding box of `mask`, padded with `fill` past the image border."""
    h, w = img.shape
    if mask.any():
        ys, xs = np.nonzero(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    else:
        y0, y1, x0, x1 = 0, h, 0, w
    side = int(max(y1 - y0, x1 - x0))
    top, left = (y0 + y1 - side) // 2, (x0 + x1 - side) // 2
    out = np.full((side, side), fill, dtype=np.float32)
    sy0, sx0, sy1, sx1 = max(top, 0), max(left, 0), min(top + side, h), min(left + side, w)
    out[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = img[sy0:sy1, sx0:sx1]
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def rotate(img: np.ndarray, degrees: float, fill: float) -> np.ndarray:
    if degrees == 0:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(fill))
