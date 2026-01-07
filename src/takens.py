from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns r_t = log(p_t) - log(p_{t-1}).
    Returns array of length len(prices)-1.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be 1D")
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to compute returns")
    if np.any(prices <= 0):
        raise ValueError("Prices must be > 0 to compute log returns")
    return np.diff(np.log(prices))


def delay_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding:
      X_t = [x_t, x_{t+tau}, x_{t+2tau}, ..., x_{t+(m-1)tau}]
    Output shape: (N - (m-1)*tau, m)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if m < 1:
        raise ValueError("m must be >= 1")
    if tau < 1:
        raise ValueError("tau must be >= 1")
    n = len(x)
    k = n - (m - 1) * tau
    if k <= 1:
        raise ValueError(
            f"Not enough points for embedding: len(x)={n}, m={m}, tau={tau}"
        )
    return np.column_stack([x[i : i + k] for i in range(0, m * tau, tau)])


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized autocorrelation for lags 0..max_lag (inclusive).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom == 0:
        return np.concatenate([np.array([1.0]), np.zeros(max_lag)])
    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for lag in range(1, max_lag + 1):
        ac[lag] = np.dot(x[:-lag], x[lag:]) / denom
    return ac


def choose_tau_autocorr(x: np.ndarray, max_lag: int = 250) -> int:
    """
    Heuristic delay choice: first lag where autocorr crosses below 1/e,
    else first local minimum, else max_lag//2.
    """
    ac = autocorrelation(x, max_lag=max_lag)
    threshold = 1.0 / np.e
    below = np.where(ac[1:] < threshold)[0]
    if len(below) > 0:
        return int(below[0] + 1)
    # local minimum search (exclude lag 0)
    for lag in range(2, len(ac) - 1):
        if ac[lag] < ac[lag - 1] and ac[lag] < ac[lag + 1]:
            return int(lag)
    return int(max(1, max_lag // 2))


def mutual_information_hist(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
    """
    Estimate mutual information I(X;Y) via histogram binning.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")
    h2d, xedges, yedges = np.histogram2d(x, y, bins=bins)
    pxy = h2d / np.sum(h2d)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))


def choose_tau_ami(x: np.ndarray, max_lag: int = 250, bins: int = 32) -> int:
    """
    Heuristic delay choice: first local minimum of average mutual information (AMI).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < max_lag + 5:
        max_lag = max(5, min(max_lag, len(x) // 5))
    amis = np.empty(max_lag + 1, dtype=float)
    amis[0] = np.inf
    for lag in range(1, max_lag + 1):
        amis[lag] = mutual_information_hist(x[:-lag], x[lag:], bins=bins)
    for lag in range(2, len(amis) - 1):
        if amis[lag] < amis[lag - 1] and amis[lag] < amis[lag + 1]:
            return int(lag)
    # fallback: smallest AMI excluding lag 0
    return int(np.argmin(amis[1:]) + 1)


@dataclass(frozen=True)
class FNNResult:
    m_values: np.ndarray
    fnn_fraction: np.ndarray
    chosen_m: int


def false_nearest_neighbors(
    x: np.ndarray,
    tau: int,
    m_max: int = 12,
    rtol: float = 10.0,
    atol: float = 2.0,
    sample: int = 4000,
    seed: int = 0,
) -> FNNResult:
    """
    False Nearest Neighbors (Kennel et al.) heuristic to choose embedding dimension m.

    A neighbor is "false" going from m -> m+1 if:
      |x_{t+m*tau} - x_{t_nn+m*tau}| / ||X_m(t) - X_m(t_nn)|| > rtol
    OR
      ||X_{m+1}(t) - X_{m+1}(t_nn)|| / std(x) > atol

    Returns fractions for m=1..m_max (reporting falseness when augmenting to m+1).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    if tau < 1:
        raise ValueError("tau must be >= 1")
    if m_max < 2:
        raise ValueError("m_max must be >= 2")
    n = len(x)
    stdx = float(np.std(x)) or 1.0

    rng = np.random.default_rng(seed)
    m_values = np.arange(1, m_max + 1, dtype=int)
    fnn_frac = np.full_like(m_values, np.nan, dtype=float)

    # For each m, we need embeddings for m and (m+1)
    for m in range(1, m_max + 1):
        # Need at least (m)*tau to access x_{t+m*tau}
        k = n - m * tau
        if k <= 2:
            break
        Xm = delay_embed(x, m=m, tau=tau)  # (n - (m-1)tau, m)
        Xm = Xm[:k]  # align so that Xm[t] has corresponding x[t + m*tau]
        Xm1 = delay_embed(x, m=m + 1, tau=tau)[:k]

        idx_all = np.arange(k)
        if k > sample:
            idx = rng.choice(idx_all, size=sample, replace=False)
        else:
            idx = idx_all

        # brute-force nearest neighbor in Xm (exclude self); OK for sample<=4000
        false_count = 0
        total = 0
        for i in idx:
            d = Xm - Xm[i]
            dist2 = np.sum(d * d, axis=1)
            dist2[i] = np.inf
            j = int(np.argmin(dist2))
            dist = float(np.sqrt(dist2[j]))
            if not np.isfinite(dist) or dist == 0:
                continue

            # criterion 1
            delta_next = abs(x[i + m * tau] - x[j + m * tau])
            crit1 = (delta_next / dist) > rtol

            # criterion 2
            dist_m1 = float(np.linalg.norm(Xm1[i] - Xm1[j]))
            crit2 = (dist_m1 / stdx) > atol

            total += 1
            if crit1 or crit2:
                false_count += 1

        fnn_frac[m - 1] = (false_count / total) if total > 0 else np.nan

    # choose m: first m where FNN fraction < 1% (and defined), else argmin
    valid = np.isfinite(fnn_frac)
    chosen_m = 3
    if np.any(valid):
        for m, frac in zip(m_values[valid], fnn_frac[valid]):
            if m >= 2 and frac < 0.01:
                chosen_m = int(m + 1) if (m + 1) <= m_max else int(m)
                break
        else:
            chosen_m = int(m_values[valid][np.nanargmin(fnn_frac[valid])])
            chosen_m = min(m_max, max(2, chosen_m + 1))
    return FNNResult(m_values=m_values, fnn_fraction=fnn_frac, chosen_m=chosen_m)


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return x - mu
    return (x - mu) / sd


