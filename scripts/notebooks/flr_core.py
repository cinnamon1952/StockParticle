"""Core estimators for the FLR multifractal paper.

Everything here is copied verbatim from the original notebook (sections 2 and 3), so numbers are unchanged.
Additions (clearly marked NEW) never alter the default code path:
  spectrum_direct, delta_alpha(method=...), segment_rms, binomial_cascade, binomial_hq.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.special import logsumexp

_M_CACHE = {}


def _resid_maker(s, poly):
    """M = I - X(X'X)^-1 X' : residuals of a degree-`poly` least-squares fit on a length-s window."""
    key = (s, poly)
    if key not in _M_CACHE:
        t = np.linspace(-1, 1, s)
        X = np.vander(t, poly + 1)
        _M_CACHE[key] = np.eye(s) - X @ np.linalg.pinv(X)
    return _M_CACHE[key]


def make_scales(N, smin, smax, n_scales, poly):
    smax = min(smax, N // 4); smin = max(smin, poly + 2)
    sc = np.unique(np.logspace(np.log10(smin), np.log10(smax), n_scales).astype(int))
    return sc[sc >= smin]


def mfdfa_fast(x, q, smin, smax, n_scales, poly=1, overlap=0.75):
    """MF-DFA on log-returns x. Same algorithm as the original loop version; detrending in one matrix product."""
    y = np.cumsum(x - np.mean(x)); N = len(y)
    scales = make_scales(N, smin, smax, n_scales, poly)
    q = np.asarray(q, float)
    Fq = np.full((len(q), len(scales)), np.nan)
    for i, s in enumerate(scales):
        step = max(1, int(s * (1.0 - overlap)))
        starts = np.arange(0, N - s + 1, step)
        if len(starts) < 4: continue
        seg = y[starts[:, None] + np.arange(s)[None, :]]
        res = seg @ _resid_maker(s, poly)
        rms = np.sqrt((res ** 2).mean(axis=1)); rms = rms[rms > 0]
        if len(rms) == 0: continue
        lr = np.log(rms)
        for j, qq in enumerate(q):
            Fq[j, i] = np.exp(lr.mean()) if qq == 0 else np.exp((logsumexp(qq * lr) - np.log(len(lr))) / qq)
    hq = np.full(len(q), np.nan)
    for j in range(len(q)):
        m = np.isfinite(Fq[j]) & (Fq[j] > 0)
        if m.sum() >= 4: hq[j] = np.polyfit(np.log(scales[m]), np.log(Fq[j][m]), 1)[0]
    return scales, Fq, hq


def spectrum(q, hq):
    """tau(q), alpha(q), f(alpha): Legendre transform with Savitzky-Golay smoothing, as in the original."""
    q = np.asarray(q, float); tau = q * hq - 1.0
    wl = min(5, len(tau) if len(tau) % 2 else len(tau) - 1)
    ts = savgol_filter(tau, wl, 2) if len(tau) >= 5 else tau
    alpha = np.gradient(ts, q); return tau, alpha, q * alpha - ts


def spectrum_direct(q, hq):
    """NEW. Direct Legendre transform, no smoothing: alpha = d tau/dq (central differences), f = q*alpha - tau."""
    q = np.asarray(q, float); tau = q * hq - 1.0
    alpha = np.gradient(tau, q); return tau, alpha, q * alpha - tau


def delta_alpha(q, hq, method="sg"):
    if not np.all(np.isfinite(hq)): return np.nan
    a = (spectrum if method == "sg" else spectrum_direct)(q, hq)[1]
    return float(np.ptp(a[np.isfinite(a)]))


def iaaft(x, rng, n_iter=50):
    xs = np.sort(x); amp = np.abs(np.fft.rfft(x)); y = rng.permutation(x); prev = None
    for _ in range(n_iter):
        ph = np.angle(np.fft.rfft(y))
        ys = np.fft.irfft(amp * np.exp(1j * ph), n=len(x))
        ranks = np.argsort(np.argsort(ys)); y = xs[ranks]
        if prev is not None and np.array_equal(ranks, prev): break
        prev = ranks
    return y


def phase_randomise(x, rng):
    F = np.fft.rfft(x - x.mean()); ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0
    if len(x) % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x)) + x.mean()


def segment_rms(x, s, poly=1, overlap=0.75):
    """NEW. Detrended segment RMS values at scale s (the quantities whose q-th moments form F_q)."""
    y = np.cumsum(x - np.mean(x)); N = len(y)
    step = max(1, int(s * (1.0 - overlap))); starts = np.arange(0, N - s + 1, step)
    seg = y[starts[:, None] + np.arange(s)[None, :]]
    rms = np.sqrt(((seg @ _resid_maker(s, poly)) ** 2).mean(axis=1))
    return rms[rms > 0]


def binomial_cascade(n_levels, a):
    """NEW. Deterministic binomial multifractal series of length 2**n_levels (Kantelhardt et al. 2002)."""
    N = 2 ** n_levels; i = np.arange(N)
    n1 = np.array([bin(k).count("1") for k in i])
    return a ** n1 * (1 - a) ** (n_levels - n1)


def binomial_hq(q, a):
    """NEW. Analytical h(q) = 1/q - ln(a^q + (1-a)^q) / (q ln 2) of the binomial cascade (q != 0)."""
    q = np.asarray(q, float)
    return 1.0 / q - np.log(a ** q + (1 - a) ** q) / (q * np.log(2))
