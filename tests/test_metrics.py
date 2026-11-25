import numpy as np

from fpverif.metrics import eer, error_curves, fnmr_at_fmr, roc_auc, summarize, system_rates, threshold_for_fmr


def test_perfectly_separable_scores_have_zero_eer_and_unit_auc():
    gen = np.array([0.8, 0.9, 0.95])
    imp = np.array([0.1, 0.2, 0.3])
    rate, thr = eer(gen, imp)
    assert rate == 0.0
    assert 0.3 < thr <= 0.8
    assert roc_auc(gen, imp) == 1.0


def test_identical_distributions_give_half_eer():
    rng = np.random.default_rng(0)
    gen = rng.normal(size=5000)
    imp = rng.normal(size=5000)
    rate, _ = eer(gen, imp)
    assert abs(rate - 0.5) < 0.03
    assert abs(roc_auc(gen, imp) - 0.5) < 0.03


def test_eer_interpolates_between_thresholds():
    gen = np.array([0.4, 0.6, 0.7, 0.8])
    imp = np.array([0.1, 0.3, 0.5, 0.65])
    rate, thr = eer(gen, imp)
    assert 0.0 < rate < 0.5
    assert 0.3 <= thr <= 0.7


def test_curves_are_monotone():
    rng = np.random.default_rng(1)
    gen = rng.normal(1, 1, 300)
    imp = rng.normal(0, 1, 500)
    _, fmr, fnmr = error_curves(gen, imp)
    assert np.all(np.diff(fmr) <= 0)
    assert np.all(np.diff(fnmr) >= 0)
    assert fmr[-1] == 0.0 and fnmr[-1] == 1.0


def test_threshold_for_fmr_meets_target():
    imp = np.linspace(0, 1, 1000)
    thr = threshold_for_fmr(imp, 0.01)
    assert np.mean(imp >= thr) <= 0.01
    gen = np.linspace(0.5, 1.5, 1000)
    assert 0.0 <= fnmr_at_fmr(gen, imp, 0.01) <= 1.0


def test_system_rates_iso_formula():
    r = system_rates(fmr=0.01, fnmr=0.05, fta=0.1)
    assert abs(r["far"] - 0.009) < 1e-12
    assert abs(r["frr"] - (0.1 + 0.05 * 0.9)) < 1e-12


def test_auc_and_eer_agree_with_scikit_learn():
    from sklearn.metrics import roc_auc_score, roc_curve

    rng = np.random.default_rng(3)
    gen, imp = rng.normal(1.5, 1, 800), rng.normal(0, 1, 3000)
    y, s = np.r_[np.ones(len(gen)), np.zeros(len(imp))], np.r_[gen, imp]

    tied_gen, tied_imp = np.round(gen, 1), np.round(imp, 1)
    assert abs(roc_auc(tied_gen, tied_imp) - roc_auc_score(y, np.r_[tied_gen, tied_imp])) < 1e-12
    assert abs(roc_auc(gen, imp) - roc_auc_score(y, s)) < 1e-12

    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    rate, _ = eer(gen, imp)
    assert abs(rate - (fpr[i] + fnr[i]) / 2) <= 1 / len(gen)


def test_summarize_keys():
    rng = np.random.default_rng(2)
    s = summarize(rng.normal(1, 1, 200), rng.normal(0, 1, 400), threshold=0.5)
    for key in ("eer", "auc", "fmr", "fnmr", "tar_at_far_1pct", "fmr100_fnmr", "far", "frr"):
        assert key in s
