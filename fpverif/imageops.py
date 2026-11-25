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


def zscore(img: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    x = img.astype(np.float32)
    region = x[mask] if mask is not None and mask.any() else x
    out = (x - region.mean()) / (region.std() + 1e-6)
    return np.clip(out * 64 + 128, 0, 255)


def center_window(img: np.ndarray, mask: np.ndarray, size: int, fill: float) -> tuple[np.ndarray, np.ndarray]:
    """size x size window centred on the mask centroid, padded with `fill` past the image border. Never rescales,
    so ridge spacing stays at the sensor's native resolution."""
    h, w = img.shape
    if mask.any():
        ys, xs = np.nonzero(mask)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    else:
        cy, cx = h // 2, w // 2
    y0, x0 = cy - size // 2, cx - size // 2
    sy0, sx0, sy1, sx1 = max(y0, 0), max(x0, 0), min(y0 + size, h), min(x0 + size, w)
    out = np.full((size, size), fill, dtype=np.float32)
    out_mask = np.zeros((size, size), dtype=bool)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    out_mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = mask[sy0:sy1, sx0:sx1]
    return out, out_mask


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


def ridge_period(img: np.ndarray, mask: np.ndarray, lo: float = 5.0, hi: float = 15.0) -> float:
    """Dominant ridge period in px: peak of the radially averaged power spectrum between `lo` and `hi`."""
    x = np.where(mask, img.astype(np.float32) - float(img[mask].mean()), 0.0)
    n = 512
    power = np.abs(np.fft.fftshift(np.fft.fft2(x, s=(n, n)))) ** 2
    yy, xx = np.indices((n, n)) - n // 2
    radius = np.hypot(yy, xx).round().astype(int)
    radial = np.bincount(radius.ravel(), power.ravel()) / np.maximum(np.bincount(radius.ravel()), 1)
    r_lo, r_hi = int(np.ceil(n / hi)), int(np.floor(n / lo))
    return n / float(r_lo + np.argmax(radial[r_lo:r_hi + 1]))


def gabor_enhance(img: np.ndarray, mask: np.ndarray, n_orient: int = 16, fill: float = 128.0) -> np.ndarray:
    """Hong, Wan and Jain (1998) contextual filtering, simplified: one ridge period per image and a bank of
    `n_orient` Gabor filters picked per pixel by the local ridge orientation. The output keeps ridge geometry and
    drops sensor-specific contrast and texture. Background and pixels outside `mask` get `fill`."""
    if not mask.any():
        return np.full(img.shape, fill, dtype=np.float32)
    x = img.astype(np.float32)
    period = ridge_period(x, mask)
    gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    sigma = 1.5 * period
    gxx, gyy, gxy = (cv2.GaussianBlur(g, (0, 0), sigma) for g in (gx * gx, gy * gy, gx * gy))
    normal = 0.5 * np.arctan2(2 * gxy, gxx - gyy)  # direction across the ridges, in (-pi/2, pi/2]
    bins = np.round(np.mod(normal, np.pi) / (np.pi / n_orient)).astype(int) % n_orient
    ksize = int(2 * round(1.5 * period) + 1)
    centred = x - cv2.GaussianBlur(x, (0, 0), 2 * period)
    out = np.zeros_like(x)
    for k in range(n_orient):
        kernel = cv2.getGaborKernel((ksize, ksize), 0.5 * period, k * np.pi / n_orient, period, 1.0, 0, cv2.CV_32F)
        kernel -= kernel.mean()
        sel = bins == k
        if sel.any():
            out[sel] = cv2.filter2D(centred, cv2.CV_32F, kernel)[sel]
    scale = float(out[mask].std()) + 1e-6
    return np.where(mask, fill + 127.0 * np.tanh(out / scale), fill).astype(np.float32)
