import numpy as np

from fpverif.config import load_config
from fpverif.imageops import hong_normalize, roi_square, segment_foreground
from preprocess import preprocess_image


def ridges(h=300, w=260, period=9):
    """Synthetic print: sinusoidal ridges inside an ellipse on a flat background."""
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), 200.0)
    inside = ((yy - h / 2) / (h * 0.35)) ** 2 + ((xx - w / 2) / (w * 0.3)) ** 2 < 1
    img[inside] = 128 + 80 * np.sin(2 * np.pi * xx[inside] / period)
    return img.astype(np.uint8), inside


def test_hong_reaches_target_mean_and_variance():
    img, mask = ridges()
    out = hong_normalize(img, mask, 128.0, 2500.0)
    assert abs(out[mask].mean() - 128) < 6
    assert abs(out[mask].std() - 50) < 6


def test_segmentation_finds_the_ridge_area():
    img, inside = ridges()
    mask = segment_foreground(img, 16)
    overlap = (mask & inside).sum() / (mask | inside).sum()
    assert overlap > 0.7


def test_roi_square_is_square_and_pads_with_fill():
    img = np.zeros((100, 60), dtype=np.float32)
    mask = np.zeros((100, 60), dtype=bool)
    mask[10:90, 20:40] = True
    out = roi_square(img, mask, 128.0)
    assert out.shape == (80, 80)
    assert out[0, 0] == 128.0 and out[40, 40] == 0.0


def test_pipeline_output_size_and_qc():
    cfg = load_config("config.yaml")
    img, _ = ridges()
    out, qc = preprocess_image(img, cfg["preprocess"], 500.0)
    size = cfg["preprocess"]["output_size"]
    assert out.shape == (size, size) and out.dtype == np.uint8
    assert qc["acquired"] and qc["foreground_mm2"] > 40


def test_small_roi_is_a_failure_to_acquire():
    cfg = load_config("config.yaml")
    img = np.full((300, 260), 200, dtype=np.uint8)
    img[140:160, 120:140] = (128 + 80 * np.sin(np.arange(20) * 0.7)).astype(np.uint8)
    _, qc = preprocess_image(img, cfg["preprocess"], 500.0)
    assert not qc["acquired"]
