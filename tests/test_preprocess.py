from pathlib import Path

import numpy as np
import yaml

from fpverif.imageops import center_window, hong_normalize
from preprocess import preprocess_image

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def synthetic_print(h=374, w=388, center=(200, 180), radius=(150, 120), contrast=40.0, seed=0):
    """Sinusoidal ridges (period 9 px, about 500 dpi) inside an ellipse on a bright background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    inside = ((yy - center[0]) / radius[0]) ** 2 + ((xx - center[1]) / radius[1]) ** 2 <= 1
    ridges = 170 + contrast * np.sin(2 * np.pi * (xx + 0.3 * yy) / 9.0)
    img = np.where(inside, ridges, 235) + rng.normal(0, 2, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8), inside


def config_pp():
    with open(CONFIG) as f:
        return yaml.safe_load(f)["preprocess"]


def test_hong_output_keeps_every_gray_level_and_widens_contrast():
    img, mask = synthetic_print()
    pp = config_pp()
    out = hong_normalize(img, mask, pp["hong_mean"], pp["hong_var"])
    fg = np.round(out[mask]).astype(np.uint8)
    assert abs(fg.mean() - pp["hong_mean"]) < 5
    assert abs(fg.std() - np.sqrt(pp["hong_var"])) < 5
    # Hong is linear in the input, so it cannot add levels; a target variance that is too small merges them.
    assert len(np.unique(fg)) >= 0.95 * len(np.unique(img[mask]))
    p1, p99 = np.percentile(fg, [1, 99])
    assert p99 - p1 >= 120


def test_center_window_keeps_native_scale_and_pads_with_fill():
    img = np.arange(100 * 120, dtype=np.float32).reshape(100, 120)
    mask = np.zeros_like(img, dtype=bool)
    mask[10:30, 15:45] = True
    out, out_mask = center_window(img, mask, 64, fill=-1.0)
    assert out.shape == (64, 64) and out_mask.shape == (64, 64)
    cy, cx = 20, 30  # centroid of the mask, rounded
    # The window is a pixel-exact copy where it overlaps the image, so nothing was resampled.
    np.testing.assert_array_equal(out[32:, 32:], img[cy:cy + 32, cx:cx + 32])
    assert (out[:32 - cy, :] == -1.0).all()
    ys, xs = np.nonzero(out_mask)
    assert abs(ys.mean() - 32) <= 1 and abs(xs.mean() - 32) <= 1


def test_preprocess_image_shape_spread_and_background():
    img, _ = synthetic_print()
    pp = config_pp()
    out, qc = preprocess_image(img, pp)
    assert out.dtype == np.uint8
    assert out.shape == (pp["output_size"], pp["output_size"])
    assert qc["acquired"]
    assert len(np.unique(out)) >= 64
    corners = np.concatenate([out[:4, :4].ravel(), out[-4:, -4:].ravel()])
    assert (corners == pp["background_fill"]).all()


def test_ridge_period_is_the_same_for_different_sensor_sizes():
    """Two sensors at the same dpi must give the same ridge spacing after preprocessing."""
    pp = config_pp()
    small, _ = synthetic_print(h=374, w=388, center=(190, 190))
    large, _ = synthetic_print(h=480, w=640, center=(240, 320), radius=(200, 160))

    def period(x):
        row = x[pp["output_size"] // 2].astype(float)
        row -= row.mean()
        spec = np.abs(np.fft.rfft(row))
        spec[0] = 0
        return len(row) / np.argmax(spec)

    p_small = period(preprocess_image(small, pp)[0])
    p_large = period(preprocess_image(large, pp)[0])
    assert abs(p_small - p_large) < 0.5


def test_ridge_period_and_gabor_enhance_on_a_synthetic_grating():
    from fpverif.imageops import gabor_enhance, ridge_period

    yy, xx = np.mgrid[0:256, 0:256]
    img = (128 + 60 * np.sin(2 * np.pi * (xx * 0.8 + yy * 0.6) / 9.0)).astype(np.float32)
    mask = np.zeros((256, 256), dtype=bool)
    mask[32:224, 32:224] = True
    assert abs(ridge_period(img, mask) - 9.0) < 1.0
    out = gabor_enhance(img, mask)
    assert out.shape == img.shape and out.min() >= 0 and out.max() <= 255
    assert np.all(out[~mask] == 128)
    # The enhanced ridges keep the phase of the input grating inside the mask.
    inner = np.zeros_like(mask)
    inner[64:192, 64:192] = True
    assert np.corrcoef(out[inner], img[inner])[0, 1] > 0.9
