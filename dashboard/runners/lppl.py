"""
LPPL-MFDFA v2 — self-contained runner.
Generates overview, rolling trajectory, and residual-surrogate 6-panel plots.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import yfinance as yf
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from plot_helpers import IMAGE_SUFFIXES, relative_to_project

try:
    from numba import njit
except ImportError:
    def njit(func=None, *args, **kwargs):
        if func is None:
            return lambda inner: inner
        return func

warnings.filterwarnings("ignore")

Q_VALS = np.arange(-10, 12, 2, dtype=float)
SCALE_MIN = 2
N_SCALES = 50
POLY_ORD = 1
OVERLAP = 0.75
MIN_N_DFA = 60
MIN_DECADES = 0.6
PAPER_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#000000", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


# ── LPPL core ─────────────────────────────────────────────────────────────────
@njit
def solve_lppl(tc, m, w, t, p):
    tau = np.maximum(tc - t, 1e-4)
    log_tau = np.log(tau)
    f = tau ** m
    g = tau ** m * np.cos(w * log_tau)
    h = tau ** m * np.sin(w * log_tau)
    X = np.ones((len(t), 4))
    X[:, 1] = f; X[:, 2] = g; X[:, 3] = h
    sol, _, _, _ = np.linalg.lstsq(X, p)
    return sol, f, g, h


@njit
def cost_fn(params, t, p):
    tc, m, w = params
    if tc <= t[-1] or tc > t[-1] + 30: return 1e22
    if m < 0.05 or m > 0.45: return 1e22
    if w < 5.0 or w > 15.0: return 1e22
    lin, f, g, h = solve_lppl(tc, m, w, t, p)
    if lin[1] >= 0: return 1e22
    osc_amp = np.sqrt(lin[2] ** 2 + lin[3] ** 2)
    if osc_amp > abs(lin[1]): return 1e22
    y_hat = lin[0] + lin[1] * f + lin[2] * g + lin[3] * h
    mse = np.sum((y_hat - p) ** 2) / len(t)
    return mse * (1.0 + (tc - t[-1]) / 5.0)


def _fit_sornette(t, p):
    tc_start = t[-1] + 1
    best_c = 1e30
    best_init = [tc_start + 5, 0.2, 8.0]
    for tc_g in np.linspace(tc_start + 2, tc_start + 30, 5):
        for m_g in [0.1, 0.3, 0.5]:
            for w_g in [6.0, 10.0, 14.0]:
                c = cost_fn(np.array([tc_g, m_g, w_g]), t, p)
                if c < best_c:
                    best_c = c
                    best_init = [tc_g, m_g, w_g]
    res = minimize(cost_fn, best_init, args=(t, p), method="Nelder-Mead", tol=1e-9)
    return res.x


# ── MF-DFA ────────────────────────────────────────────────────────────────────
def _overlap_starts(n, s, overlap):
    step = max(1, int(s * (1.0 - overlap)))
    return np.arange(0, n - s + 1, step)


def _mfdfa(x, q_vals=Q_VALS, scale_min=SCALE_MIN, n_scales=N_SCALES,
           poly_ord=POLY_ORD, overlap=OVERLAP):
    y = np.cumsum(x - x.mean())
    n = len(y)
    scale_max = max(scale_min + 1, n // 4)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales).astype(int))
    scales = scales[scales >= scale_min]
    fqs = np.full((len(q_vals), len(scales)), np.nan)
    for i, s in enumerate(scales):
        starts = _overlap_starts(n, s, overlap)
        if len(starts) < 4: continue
        t_seg = np.arange(s)
        rms = []
        for st in starts:
            seg = y[st: st + s]
            coef = np.polyfit(t_seg, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t_seg)) ** 2))
            if r > 0: rms.append(r)
        rms = np.asarray(rms)
        if len(rms) == 0: continue
        for j, q in enumerate(q_vals):
            if q == 0:
                fqs[j, i] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                fqs[j, i] = (np.mean(rms ** q)) ** (1.0 / q)
    hq = np.full(len(q_vals), np.nan)
    for j in range(len(q_vals)):
        row = fqs[j]; mask = np.isfinite(row) & (row > 0)
        if mask.sum() >= 4:
            hq[j] = np.polyfit(np.log(scales[mask]), np.log(row[mask]), 1)[0]
    return scales, fqs, hq


def _spectrum(hq):
    tau_q = Q_VALS * hq - 1.0
    wl = min(5, len(tau_q) if len(tau_q) % 2 == 1 else len(tau_q) - 1)
    tau_s = savgol_filter(tau_q, wl, 2) if len(tau_q) >= 5 else tau_q
    alpha = np.gradient(tau_s, Q_VALS)
    d_alpha = Q_VALS * alpha - tau_s
    return tau_q, alpha, d_alpha


def _delta_alpha(x):
    n = len(x)
    if n < MIN_N_DFA:
        return float("nan"), "too_short"
    scale_max = n // 4
    decades = np.log10(scale_max) - np.log10(SCALE_MIN)
    if decades < MIN_DECADES:
        return float("nan"), "too_few_decades"
    _, _, hq = _mfdfa(x)
    _, alpha, d_alp = _spectrum(hq)
    vs = np.isfinite(alpha) & np.isfinite(d_alp)
    if vs.sum() < 4:
        return float("nan"), "unstable_spectrum"
    return float(alpha[vs].max() - alpha[vs].min()), "ok"


def _surrogate_test(x, n_surrogates=100):
    obs_da, flag = _delta_alpha(x)
    if flag != "ok":
        return obs_da, float("nan"), float("nan"), flag
    rng = np.random.default_rng(42)
    fft_x = np.fft.rfft(x)
    amps = np.abs(fft_x)
    sur_das = []
    for _ in range(n_surrogates):
        phases = rng.uniform(0, 2 * np.pi, len(fft_x))
        phases[0] = 0
        if len(x) % 2 == 0:
            phases[-1] = 0
        sur = np.fft.irfft(amps * np.exp(1j * phases), n=len(x))
        da_s, _ = _delta_alpha(sur)
        if np.isfinite(da_s):
            sur_das.append(da_s)
    arr = np.array(sur_das)
    if len(arr) == 0:
        return obs_da, float("nan"), float("nan"), "surrogate_failed"
    return obs_da, float(np.mean(arr >= obs_da)), float(arr.mean()), "ok"


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _plot_mfdfa_6panel(ticker, window_label, series_label, raw, scales, fqs, hq,
                       tau_q, alpha, d_alpha, obs_da=None, p_val=None, sur_mean=None):
    valid = np.isfinite(hq)
    colors = (PAPER_COLORS * 4)[:len(Q_VALS)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    sig_str = ""
    if p_val is not None and np.isfinite(p_val):
        sig_str = f"  |  surrogate p={p_val:.3f}  (μ_sur={sur_mean:.3f})"
    fig.suptitle(f"MF-DFA · {ticker} · {window_label}d · {series_label}{sig_str}",
                 fontsize=12, fontweight="bold", y=1.01)

    axs[0, 0].plot(raw, color="#c0392b", lw=0.85)
    axs[0, 0].set_xlabel("$t$ (trading days)"); axs[0, 0].set_ylabel(series_label)
    axs[0, 0].set_title("(a) Input series")

    handles = []
    for idx_q, (q, c) in enumerate(zip(Q_VALS, colors)):
        row = fqs[idx_q]; mask = np.isfinite(row) & (row > 0)
        if mask.sum() < 2: continue
        h, = axs[0, 1].loglog(scales[mask], row[mask], "o-", color=c, markersize=3.5, lw=1.0)
        handles.append((h, f"$q={int(q)}$"))
    axs[0, 1].set_xlabel("$s$"); axs[0, 1].set_ylabel("$F_q(s)$")
    axs[0, 1].set_title("(b) Fluctuation functions")
    step = max(1, len(handles) // 8); shown = handles[::step]
    axs[0, 1].legend([h for h, _ in shown], [l for _, l in shown], fontsize=7, ncol=1)

    axs[0, 2].plot(Q_VALS[valid], hq[valid], "-", color="#1f4e8c", lw=2)
    axs[0, 2].axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    axs[0, 2].set_xlabel("$q$"); axs[0, 2].set_ylabel("$H(q)$")
    axs[0, 2].set_title("(c) Generalised Hurst $H(q)$")

    axs[1, 0].plot(Q_VALS[valid], tau_q[valid], "-", color="#1f4e8c", lw=2)
    axs[1, 0].axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)
    axs[1, 0].set_xlabel("$q$"); axs[1, 0].set_ylabel(r"$\tau(q)$")
    axs[1, 0].set_title(r"(d) Scaling exponent $\tau(q)$")

    vs = np.isfinite(alpha) & np.isfinite(d_alpha)
    axs[1, 1].plot(Q_VALS[vs], alpha[vs], "-", color="#1f4e8c", lw=2)
    axs[1, 1].set_xlabel("$q$"); axs[1, 1].set_ylabel(r"$\alpha(q)$")
    axs[1, 1].set_title(r"(e) Singularity strength $\alpha(q)$")

    axs[1, 2].plot(alpha[vs], d_alpha[vs], "-", color="#1f4e8c", lw=2)
    q0 = np.argmin(np.abs(Q_VALS))
    if np.isfinite(alpha[q0]) and np.isfinite(d_alpha[q0]):
        axs[1, 2].plot(alpha[q0], d_alpha[q0], "o", color="#c0392b", markersize=6, zorder=5)
    axs[1, 2].set_xlabel(r"$\alpha$"); axs[1, 2].set_ylabel(r"$f(\alpha)$")
    axs[1, 2].set_title(r"(f) Multifractal spectrum $f(\alpha)$")

    for a in axs.ravel():
        a.tick_params(direction="in", which="both", top=True, right=True)
        a.spines[["top", "right"]].set_visible(True)
    footer = fr"$\Delta\alpha = {obs_da:.4f}$" if (obs_da and np.isfinite(obs_da)) else ""
    if footer:
        fig.text(0.5, -0.01, footer, ha="center", fontsize=10, color="grey")
    plt.tight_layout()
    return fig


def _plot_rolling(ticker, dates, da_bw, tc_sig, js, target_date):
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    colors_w = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax1 = fig.add_subplot(gs[0])
    for (wlab, da_s), col in zip(da_bw.items(), colors_w):
        ax1.plot(dates, np.array(da_s, dtype=float), "o-", color=col, markersize=3, lw=1.2, label=f"{wlab}d window")
    ax1.axvline(target_date, color="black", ls="--", lw=1, label="Target date")
    ax1.set_ylabel(r"$\Delta\alpha$ (residuals)")
    ax1.set_title(f"{ticker}  —  Rolling Δα trajectory (LPPL residuals)")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, np.array(tc_sig, dtype=float), "s-", color="#9467bd", markersize=3, lw=1.2)
    ax2.axvline(target_date, color="black", ls="--", lw=1)
    ax2.set_ylabel(r"$\sigma(t_c)$ across windows")
    ax2.set_title("tc Stability index (lower = more consensus)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    js_arr = np.array(js, dtype=float)
    valid_js = js_arr[np.isfinite(js_arr)]
    js_norm = (js_arr - valid_js.min()) / (valid_js.ptp() + 1e-12) if len(valid_js) > 0 else js_arr
    ax3.fill_between(dates, js_norm, alpha=0.35, color="#d62728")
    ax3.plot(dates, js_norm, "-", color="#d62728", lw=1.5)
    ax3.axvline(target_date, color="black", ls="--", lw=1)
    ax3.set_ylabel("Joint score (normalised)")
    ax3.set_title(r"Joint Crash Score  =  $\Delta\alpha_{mean}$ × $(1 / \sigma_{t_c})$")
    ax3.set_ylim(0, 1.05); ax3.grid(alpha=0.3)
    for a in [ax1, ax2, ax3]: a.set_xlabel("Analysis date")
    plt.suptitle(f"{ticker}  ·  Rolling LPPL + MF-DFA joint signal", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ── Public entry point ────────────────────────────────────────────────────────
def generate(output_dir: str | Path, **overrides) -> dict:
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    before = _scan(output_dir)

    tickers = overrides.get("tickers", ["^GSPC", "JPM", "XLF", "C", "GE"])
    start_data = overrides.get("START_DATA", "2007-01-01")
    end_data = overrides.get("END_DATA", "2009-06-01")
    target_date = overrides.get("TARGET_DATE", "2008-09-28")
    window_offsets = overrides.get("window_start_offsets", [120, 100, 80, 60])
    plot_duration = overrides.get("plot_duration", 150)
    roll_days = overrides.get("ROLL_DAYS", 60)
    roll_step = overrides.get("ROLL_STEP", 5)
    surrogate_n = overrides.get("SURROGATE_N", 100)

    data = yf.download(tickers, start=start_data, end=end_data, progress=False, auto_adjust=True)

    for ticker in tickers:
        close_data = data["Close"]
        if hasattr(close_data, "columns") and ticker in list(close_data.columns):
            prices = close_data[ticker].dropna()
        elif hasattr(close_data, "ndim") and close_data.ndim == 2:
            prices = close_data.iloc[:, 0].dropna()
        else:
            prices = close_data.dropna()

        df = prices.to_frame(name="Close").reset_index()
        try:
            anchor_idx = df[df["Date"] <= target_date].index[-1]
        except (IndexError, KeyError):
            continue

        safe = ticker.replace("^", "").replace("=", "_")

        roll_dates = []
        da_by_window = {w: [] for w in window_offsets}
        tc_sigma_series = []
        joint_score_series = []

        for days_back in range(roll_days, -1, -roll_step):
            end_idx = anchor_idx - days_back
            if end_idx < max(window_offsets):
                continue
            as_of_date = df["Date"].iloc[end_idx]
            roll_dates.append(as_of_date)
            tc_deltas = []
            da_this_row = {}

            for start_offset in window_offsets:
                if end_idx - start_offset < 0:
                    da_by_window[start_offset].append(float("nan"))
                    continue
                df_fit = df.iloc[end_idx - start_offset: end_idx + 1].copy()
                t_f = np.arange(len(df_fit)).astype(np.float64)
                p_f = np.log(df_fit["Close"].values).astype(np.float64)
                tc_rel, m, w = _fit_sornette(t_f, p_f)
                tc_deltas.append(tc_rel - t_f[-1])
                lin_f, f_v, g_v, h_v = solve_lppl(tc_rel, m, w, t_f, p_f)
                y_fit = lin_f[0] + lin_f[1] * f_v + lin_f[2] * g_v + lin_f[3] * h_v
                residuals = p_f - y_fit
                da, flag = _delta_alpha(residuals)
                da_by_window[start_offset].append(da)
                da_this_row[start_offset] = da

                if days_back == 0:
                    obs_da, p_val, sur_mean, sflag = _surrogate_test(residuals, surrogate_n)
                    _, _, hq = _mfdfa(residuals)
                    tau_q, alpha, da2 = _spectrum(hq)
                    sc, fqs_r, _ = _mfdfa(residuals)
                    fig = _plot_mfdfa_6panel(
                        safe, start_offset, "LPPL Residuals", residuals,
                        sc, fqs_r, hq, tau_q, alpha, da2,
                        obs_da=obs_da, p_val=p_val, sur_mean=sur_mean,
                    )
                    fname = output_dir / f"{safe}_{start_offset}d_residuals_surrogate.png"
                    fig.savefig(str(fname), dpi=120, bbox_inches="tight")
                    plt.close(fig)

            sigma_tc = float(np.std(tc_deltas)) if len(tc_deltas) >= 2 else float("nan")
            tc_sigma_series.append(sigma_tc)
            valid_das = [v for v in da_this_row.values() if np.isfinite(v)]
            mean_da = float(np.mean(valid_das)) if valid_das else float("nan")
            if np.isfinite(mean_da) and np.isfinite(sigma_tc) and sigma_tc > 0:
                joint_score_series.append(mean_da / sigma_tc)
            else:
                joint_score_series.append(float("nan"))

        if len(roll_dates) >= 3:
            fig_roll = _plot_rolling(safe, roll_dates, da_by_window, tc_sigma_series,
                                     joint_score_series, df["Date"].iloc[anchor_idx])
            fig_roll.savefig(str(output_dir / f"{safe}_rolling_trajectory.png"), dpi=120, bbox_inches="tight")
            plt.close(fig_roll)

        # Overview
        end_idx = anchor_idx
        as_of_date = df["Date"].iloc[end_idx]
        tc_trading_deltas = []
        fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
        limit = min(len(df) - 1, end_idx + 40)
        view_df = df.iloc[max(0, end_idx - plot_duration): limit + 1]
        ax_ov.plot(view_df["Date"], np.log(view_df["Close"]), color="black", alpha=0.3, label="Price History")

        for start_offset in window_offsets:
            if end_idx - start_offset < 0: continue
            df_fit = df.iloc[end_idx - start_offset: end_idx + 1].copy()
            t_f = np.arange(len(df_fit)).astype(np.float64)
            p_f = np.log(df_fit["Close"].values).astype(np.float64)
            tc_rel, m, w = _fit_sornette(t_f, p_f)
            tc_trading_deltas.append(tc_rel - t_f[-1])
            lin_f, f_v, g_v, h_v = solve_lppl(tc_rel, m, w, t_f, p_f)
            y_fit = lin_f[0] + lin_f[1] * f_v + lin_f[2] * g_v + lin_f[3] * h_v
            ax_ov.plot(df_fit["Date"], y_fit, alpha=0.7, label=f"Fit {start_offset}d")

        sigma = float(np.std(tc_trading_deltas)) if tc_trading_deltas else 0.0
        ax_ov.axvline(as_of_date, color="blue", ls="--", label="Analysis Point")
        ax_ov.set_title(f"{safe}  |  σ(tc) = {sigma:.2f}")
        ax_ov.legend(loc="lower left", fontsize="small", ncol=2)
        ax_ov.grid(alpha=0.3)
        fig_ov.savefig(str(output_dir / f"{safe}_overview.png"), dpi=120, bbox_inches="tight")
        plt.close(fig_ov)

    after = _scan(output_dir)
    changed = [p for p, m in after.items() if p not in before or before[p] != m]
    return {
        "output_dir": relative_to_project(output_dir),
        "changed_count": len(changed),
        "changed_files": [relative_to_project(Path(p)) for p in sorted(changed)],
    }


def _scan(d: Path) -> dict:
    if not d.exists():
        return {}
    return {
        str(p.resolve()): (int(p.stat().st_mtime_ns), p.stat().st_size)
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
