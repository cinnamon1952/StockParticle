"""Run:  pytest -q tests/        (from the repo root)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from flr_core import (mfdfa_fast, delta_alpha, spectrum, spectrum_direct, iaaft, phase_randomise,
                      binomial_cascade, binomial_hq)

Q = [float(q) for q in range(-30, 31, 5)]
SET = dict(smin=20, smax=800, n_scales=30, poly=1, overlap=0.75)


def width(x, q=Q, method="sg", **kw):
    s = {**SET, **kw}
    _, _, hq = mfdfa_fast(x, q, s["smin"], s["smax"], s["n_scales"], s["poly"], s["overlap"])
    return delta_alpha(q, hq, method), hq


def fgn_h05(n, rng):
    """Fractional Gaussian noise with H = 0.5 is white Gaussian noise."""
    return rng.standard_normal(n)


# (a) monofractal series -> small width -------------------------------------------------------------------
def test_monofractal_small_width_and_h_near_half():
    rng = np.random.default_rng(0)
    w, hq = width(fgn_h05(4091, rng))
    assert w < 0.40, w                                  # finite-size floor (about 0.24 at N=4091), far below real assets (~0.55)
    h2 = mfdfa_fast(fgn_h05(4091, np.random.default_rng(1)), [2.0], 20, 800, 30)[2][0]
    assert abs(h2 - 0.5) < 0.05, h2


def test_monofractal_narrower_than_fat_tailed():
    rng = np.random.default_rng(2)
    wg = np.mean([width(rng.standard_normal(4091))[0] for _ in range(8)])
    wt = np.mean([width(rng.standard_t(4, 4091))[0] for _ in range(8)])
    assert wg < wt


# (b) binomial cascade: analytical h(q) and width ---------------------------------------------------------
@pytest.mark.parametrize("a", [0.3, 0.2])
def test_binomial_cascade_hq_matches_analytic(a):
    x = binomial_cascade(13, a)
    q = [-10., -5., -2., -1., 1., 2., 5., 10.]
    _, _, h = mfdfa_fast(x, q, 10, 800, 30, 1, 0.75)
    assert np.max(np.abs(h - binomial_hq(q, a))) < 0.10


def test_binomial_cascade_width_matches_analytic_under_same_pipeline():
    a = 0.3; x = binomial_cascade(13, a)
    q = [float(v) for v in range(-10, 11, 1)]
    _, _, h = mfdfa_fast(x, q, 10, 800, 30, 1, 0.75)
    qa = np.array(q); ha = np.where(qa == 0, np.nan, binomial_hq(np.where(qa == 0, 1e-6, qa), a))
    ha[q.index(0.0)] = binomial_hq([1e-6], a)[0]
    w_est = delta_alpha(q, h, "direct"); w_ana = delta_alpha(q, ha, "direct")
    assert abs(w_est - w_ana) < 0.15 * w_ana, (w_est, w_ana)


def test_tau_zero_is_minus_one_and_f_peak_is_one_unsmoothed():
    a = 0.3; q = np.arange(-10, 11, 1.0)
    ha = binomial_hq(np.where(q == 0, 1e-6, q), a)
    tau, alpha, f = spectrum_direct(q, ha)
    assert abs(tau[list(q).index(0.0)] + 1) < 1e-5
    assert abs(f.max() - 1) < 0.02


# (c) shuffle preserves the marginal ----------------------------------------------------------------------
def test_shuffle_preserves_marginal():
    rng = np.random.default_rng(3); x = rng.standard_t(4, 2000)
    y = rng.permutation(x)
    assert np.array_equal(np.sort(x), np.sort(y)) and not np.array_equal(x, y)


# (d) phase randomization preserves the power spectrum -----------------------------------------------------
@pytest.mark.parametrize("n", [2000, 2001])
def test_phase_randomisation_preserves_spectrum(n):
    rng = np.random.default_rng(4)
    x = np.convolve(rng.standard_normal(n + 5), np.ones(6) / 6, "valid")[:n] + 0.1
    y = phase_randomise(x, rng)
    ax_, ay = np.abs(np.fft.rfft(x - x.mean())), np.abs(np.fft.rfft(y - y.mean()))
    assert np.allclose(ax_[1:], ay[1:], rtol=1e-8, atol=1e-8)
    assert abs(y.mean() - x.mean()) < 1e-10
    assert not np.allclose(x, y)


# (e) IAAFT preserves the marginal exactly and the spectrum approximately -----------------------------------
def test_iaaft_marginal_exact_spectrum_approximate():
    rng = np.random.default_rng(5); n = 2048
    z = np.convolve(rng.standard_normal(n + 9), np.ones(10) / 10, "valid")[:n]
    x = np.sign(z) * np.abs(z) ** 1.5                         # non-Gaussian marginal with linear dependence
    y = iaaft(x, rng, 50)
    assert np.array_equal(np.sort(x), np.sort(y))
    px, py = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(y)) ** 2
    rel = np.linalg.norm(px - py) / np.linalg.norm(px)
    assert rel < 0.25, rel
    # autocorrelation at lag 1 approximately preserved
    ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]
    assert abs(ac(x) - ac(y)) < 0.05
