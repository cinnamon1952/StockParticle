from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from dashboard.tab_api import TabSpec

try:
    from numba import njit
except ImportError:
    def njit(func=None, *args, **kwargs):
        if func is None:
            return lambda inner: inner
        return func

warnings.filterwarnings("ignore")

PAPER_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#000000", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]

Q_VALS = np.arange(-10, 12, 2, dtype=float)
SCALE_MIN = 2
N_SCALES = 50
POLY_ORD = 1
OVERLAP = 0.75
MIN_N_DFA = 60
MIN_DECADES = 0.6


@dataclass(frozen=True)
class LpplConfig:
    tickers: tuple[str, ...]
    start_data: str
    end_data: str
    target_date: str
    window_start_offsets: tuple[int, ...]
    plot_duration: int
    roll_days: int
    roll_step: int
    surrogate_n: int


# ---------------------------------------------------------------------------
# LPPL core
# ---------------------------------------------------------------------------
@njit
def solve_lppl(tc, m, w, t, p):
    tau = np.maximum(tc - t, 1e-4)
    log_tau = np.log(tau)
    f = tau**m
    g = tau**m * np.cos(w * log_tau)
    h = tau**m * np.sin(w * log_tau)
    X = np.ones((len(t), 4))
    X[:, 1] = f; X[:, 2] = g; X[:, 3] = h
    sol, _, _, _ = np.linalg.lstsq(X, p)
    return sol, f, g, h


@njit
def cost_fn(params, t, p):
    tc, m, w = params
    if tc <= t[-1] or tc > t[-1] + 30:
        return 1e22
    if m < 0.05 or m > 0.45:
        return 1e22
    if w < 5.0 or w > 15.0:
        return 1e22
    lin, f, g, h = solve_lppl(tc, m, w, t, p)
    if lin[1] >= 0:
        return 1e22
    osc_amp = np.sqrt(lin[2] ** 2 + lin[3] ** 2)
    if osc_amp > abs(lin[1]):
        return 1e22
    y_hat = lin[0] + lin[1] * f + lin[2] * g + lin[3] * h
    mse = np.sum((y_hat - p) ** 2) / len(t)
    return mse * (1.0 + (tc - t[-1]) / 5.0)


def fit_sornette_model(t, p):
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


# ---------------------------------------------------------------------------
# MF-DFA
# ---------------------------------------------------------------------------
def _overlap_starts(n, s, overlap):
    step = max(1, int(s * (1.0 - overlap)))
    return np.arange(0, n - s + 1, step)


def _mfdfa(x, q_vals=Q_VALS, scale_min=SCALE_MIN, n_scales=N_SCALES, poly_ord=POLY_ORD, overlap=OVERLAP):
    y = np.cumsum(x - x.mean())
    n = len(y)
    scale_max = max(scale_min + 1, n // 4)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales).astype(int))
    scales = scales[scales >= scale_min]
    fqs = np.full((len(q_vals), len(scales)), np.nan)
    for i, s in enumerate(scales):
        starts = _overlap_starts(n, s, overlap)
        if len(starts) < 4:
            continue
        t_seg = np.arange(s)
        rms: list[float] = []
        for st in starts:
            seg = y[st : st + s]
            coef = np.polyfit(t_seg, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t_seg)) ** 2))
            if r > 0:
                rms.append(r)
        rms_arr = np.asarray(rms)
        if len(rms_arr) == 0:
            continue
        for j, q in enumerate(q_vals):
            if q == 0:
                fqs[j, i] = np.exp(0.5 * np.mean(np.log(rms_arr**2)))
            else:
                fqs[j, i] = (np.mean(rms_arr**q)) ** (1.0 / q)
    hq = np.full(len(q_vals), np.nan)
    for j in range(len(q_vals)):
        row = fqs[j]
        mask = np.isfinite(row) & (row > 0)
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
    da = float(alpha[vs].max() - alpha[vs].min())
    return da, "ok"


def _surrogate_test(x, n_surrogates=100):
    obs_da, flag = _delta_alpha(x)
    if flag != "ok":
        return obs_da, float("nan"), float("nan"), flag
    rng = np.random.default_rng(42)
    fft_x = np.fft.rfft(x)
    amps = np.abs(fft_x)
    sur_das: list[float] = []
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


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_lppl_analysis(config: LpplConfig) -> dict[str, Any]:
    data = yf.download(list(config.tickers), start=config.start_data, end=config.end_data, progress=False, auto_adjust=True)
    all_results: dict[str, Any] = {}

    for ticker in config.tickers:
        if len(config.tickers) > 1:
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"][ticker].dropna()
            else:
                prices = data["Close"].dropna()
        else:
            close_data = data["Close"]
            if hasattr(close_data, "columns") and ticker in list(close_data.columns):
                prices = close_data[ticker].dropna()
            elif hasattr(close_data, "ndim") and close_data.ndim == 2:
                prices = close_data.iloc[:, 0].dropna()
            else:
                prices = close_data.dropna()

        df = prices.to_frame(name="Close").reset_index()
        try:
            anchor_idx = df[df["Date"] <= config.target_date].index[-1]
        except (IndexError, KeyError):
            continue

        ticker_safe = ticker.replace("^", "").replace("=", "_")

        roll_dates: list[Any] = []
        da_by_window: dict[int, list[float]] = {w: [] for w in config.window_start_offsets}
        tc_sigma_series: list[float] = []
        joint_score_series: list[float] = []
        surrogate_results: dict[int, dict[str, float]] = {}
        overview_data: dict[str, Any] = {}

        pivot_offsets = range(config.roll_days, -1, -config.roll_step)
        for days_back in pivot_offsets:
            end_idx = anchor_idx - days_back
            if end_idx < max(config.window_start_offsets):
                continue
            as_of_date = df["Date"].iloc[end_idx]
            roll_dates.append(as_of_date)
            tc_deltas: list[float] = []
            da_this_row: dict[int, float] = {}

            for start_offset in config.window_start_offsets:
                if end_idx - start_offset < 0:
                    da_by_window[start_offset].append(float("nan"))
                    continue
                df_fit = df.iloc[end_idx - start_offset : end_idx + 1].copy()
                t_f = np.arange(len(df_fit)).astype(np.float64)
                p_f = np.log(df_fit["Close"].values).astype(np.float64)
                tc_rel, m, w = fit_sornette_model(t_f, p_f)
                tc_deltas.append(tc_rel - t_f[-1])
                lin_f, f_v, g_v, h_v = solve_lppl(tc_rel, m, w, t_f, p_f)
                y_fit = lin_f[0] + lin_f[1] * f_v + lin_f[2] * g_v + lin_f[3] * h_v
                residuals = p_f - y_fit
                da, flag = _delta_alpha(residuals)
                da_by_window[start_offset].append(da)
                da_this_row[start_offset] = da

                if days_back == 0:
                    obs_da, p_val, sur_mean, sflag = _surrogate_test(residuals, config.surrogate_n)
                    surrogate_results[start_offset] = {"da": obs_da, "p": p_val, "sur_mean": sur_mean, "flag": sflag, "m": m, "w": w}

            sigma_tc = float(np.std(tc_deltas)) if len(tc_deltas) >= 2 else float("nan")
            tc_sigma_series.append(sigma_tc)
            valid_das = [v for v in da_this_row.values() if np.isfinite(v)]
            mean_da = float(np.mean(valid_das)) if valid_das else float("nan")
            if np.isfinite(mean_da) and np.isfinite(sigma_tc) and sigma_tc > 0:
                joint_score_series.append(mean_da / sigma_tc)
            else:
                joint_score_series.append(float("nan"))

        end_idx = anchor_idx
        as_of_date = df["Date"].iloc[end_idx]
        limit = min(len(df) - 1, end_idx + 40)
        view_df = df.iloc[max(0, end_idx - config.plot_duration) : limit + 1]
        tc_trading_deltas: list[float] = []
        fit_segments: list[dict[str, Any]] = []
        for start_offset in config.window_start_offsets:
            if end_idx - start_offset < 0:
                continue
            df_fit = df.iloc[end_idx - start_offset : end_idx + 1].copy()
            t_f = np.arange(len(df_fit)).astype(np.float64)
            p_f = np.log(df_fit["Close"].values).astype(np.float64)
            tc_rel, m, w = fit_sornette_model(t_f, p_f)
            tc_trading_deltas.append(tc_rel - t_f[-1])
            lin_f, f_v, g_v, h_v = solve_lppl(tc_rel, m, w, t_f, p_f)
            y_fit = lin_f[0] + lin_f[1] * f_v + lin_f[2] * g_v + lin_f[3] * h_v
            tc_cal = as_of_date + pd.tseries.offsets.BusinessDay(n=int(round(tc_rel - t_f[-1])))
            fit_segments.append({"offset": start_offset, "dates": df_fit["Date"].values, "y_fit": y_fit, "tc_date": str(tc_cal.date()), "m": m, "w": w})

        overview_data = {
            "view_dates": view_df["Date"].values,
            "view_log_close": np.log(view_df["Close"].values),
            "as_of_date": as_of_date,
            "sigma_tc": float(np.std(tc_trading_deltas)) if tc_trading_deltas else float("nan"),
            "fits": fit_segments,
        }

        all_results[ticker_safe] = {
            "roll_dates": roll_dates,
            "da_by_window": da_by_window,
            "tc_sigma_series": tc_sigma_series,
            "joint_score_series": joint_score_series,
            "surrogate_results": surrogate_results,
            "overview": overview_data,
            "target_date": df["Date"].iloc[anchor_idx],
        }
    return all_results


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_overview_figure(ticker: str, data: dict[str, Any]) -> plt.Figure:
    ov = data["overview"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ov["view_dates"], ov["view_log_close"], color="black", alpha=0.3, label="Price History")
    for seg in ov["fits"]:
        ax.plot(seg["dates"], seg["y_fit"], alpha=0.7, label=f"Fit {seg['offset']}d")
    ax.axvline(ov["as_of_date"], color="blue", ls="--", label="Analysis Point")
    ax.set_title(f"{ticker}  |  σ(tc) = {ov['sigma_tc']:.2f}")
    ax.legend(loc="lower left", fontsize="small", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def build_rolling_figure(ticker: str, data: dict[str, Any]) -> plt.Figure:
    dates = data["roll_dates"]
    da_bw = data["da_by_window"]
    tc_sig = data["tc_sigma_series"]
    js = data["joint_score_series"]
    target = data["target_date"]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, hspace=0.45)
    colors_w = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    ax1 = fig.add_subplot(gs[0])
    for (wlab, da_s), col in zip(da_bw.items(), colors_w):
        arr = np.array(da_s, dtype=float)
        ax1.plot(dates, arr, "o-", color=col, markersize=3, lw=1.2, label=f"{wlab}d window")
    ax1.axvline(target, color="black", ls="--", lw=1, label="Target date")
    ax1.set_ylabel(r"$\Delta\alpha$ (residuals)")
    ax1.set_title(f"{ticker}  —  Rolling Δα trajectory (LPPL residuals)")
    ax1.legend(fontsize=8, ncol=2); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, np.array(tc_sig, dtype=float), "s-", color="#9467bd", markersize=3, lw=1.2)
    ax2.axvline(target, color="black", ls="--", lw=1)
    ax2.set_ylabel(r"$\sigma(t_c)$ across windows")
    ax2.set_title("tc Stability index (lower = more consensus)")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    js_arr = np.array(js, dtype=float)
    valid_js = js_arr[np.isfinite(js_arr)]
    if len(valid_js) > 0:
        js_norm = (js_arr - valid_js.min()) / (valid_js.ptp() + 1e-12)
    else:
        js_norm = js_arr
    ax3.fill_between(dates, js_norm, alpha=0.35, color="#d62728")
    ax3.plot(dates, js_norm, "-", color="#d62728", lw=1.5)
    ax3.axvline(target, color="black", ls="--", lw=1)
    ax3.set_ylabel("Joint score (normalised)")
    ax3.set_title(r"Joint Crash Score  =  $\Delta\alpha_{mean}$ × $(1 / \sigma_{t_c})$")
    ax3.set_ylim(0, 1.05); ax3.grid(alpha=0.3)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Analysis date")
    plt.suptitle(f"{ticker}  ·  Rolling LPPL + MF-DFA joint signal", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        "Self-contained LPPL-MFDFA crash diagnostics with rolling trajectory, surrogate testing, "
        "and overview plots — all computed directly in this tab."
    )

    with st.form("lppl-controls"):
        tickers_str = st.text_input("Tickers (comma-separated)", value="^GSPC, JPM, XLF, C, GE")
        col1, col2 = st.columns(2)
        with col1:
            start_data = st.date_input("Start", value=pd.Timestamp("2007-01-01"))
            target_date = st.date_input("Target date", value=pd.Timestamp("2008-09-28"))
            roll_days = st.number_input("Roll days", value=60, min_value=10, step=5)
            surrogate_n = st.number_input("Surrogates", value=100, min_value=10, step=10)
        with col2:
            end_data = st.date_input("End", value=pd.Timestamp("2009-06-01"))
            offsets_str = st.text_input("Window offsets", value="120, 100, 80, 60")
            roll_step = st.number_input("Roll step", value=5, min_value=1, step=1)
            plot_duration = st.number_input("Plot duration", value=150, min_value=50, step=10)
        submitted = st.form_submit_button("Run LPPL-MFDFA", type="primary")

    tickers = tuple(t.strip() for t in tickers_str.split(",") if t.strip())
    offsets = tuple(int(o.strip()) for o in offsets_str.split(",") if o.strip())

    config = LpplConfig(
        tickers=tickers,
        start_data=str(start_data),
        end_data=str(end_data),
        target_date=str(target_date),
        window_start_offsets=offsets,
        plot_duration=int(plot_duration),
        roll_days=int(roll_days),
        roll_step=int(roll_step),
        surrogate_n=int(surrogate_n),
    )

    if "lppl_has_run" not in st.session_state:
        st.session_state["lppl_has_run"] = False
    if submitted:
        st.session_state["lppl_has_run"] = True
    if not st.session_state["lppl_has_run"]:
        st.info("Configure parameters and run.")
        return

    with st.spinner("Fitting LPPL models and running MF-DFA + surrogate tests..."):
        results = run_lppl_analysis(config)

    if not results:
        st.error("No results were returned. Check your tickers and date range.")
        return

    ticker_select = st.selectbox("Ticker", list(results.keys()))
    if not ticker_select:
        return
    ticker_data = results[ticker_select]

    st.subheader("LPPL Overview")
    st.pyplot(build_overview_figure(ticker_select, ticker_data), clear_figure=True, use_container_width=True)

    ov = ticker_data["overview"]
    rows = []
    for seg in ov["fits"]:
        rows.append({"Window": f"{seg['offset']}d", "tc date": seg["tc_date"], "m": f"{seg['m']:.3f}", "w": f"{seg['w']:.3f}"})
    if rows:
        st.table(pd.DataFrame(rows))

    if ticker_data["surrogate_results"]:
        st.subheader("Surrogate Test Results")
        sur_rows = []
        for offset, sr in ticker_data["surrogate_results"].items():
            sig = "GENUINE" if (np.isfinite(sr["p"]) and sr["p"] < 0.05) else "linear/artefact"
            sur_rows.append({
                "Window": f"{offset}d",
                "Δα": f"{sr['da']:.3f}" if np.isfinite(sr["da"]) else "—",
                "p-value": f"{sr['p']:.3f}" if np.isfinite(sr["p"]) else "—",
                "μ surrogate": f"{sr['sur_mean']:.3f}" if np.isfinite(sr["sur_mean"]) else "—",
                "Verdict": sig,
            })
        st.table(pd.DataFrame(sur_rows))

    if len(ticker_data["roll_dates"]) >= 3:
        st.subheader("Rolling Trajectory")
        st.pyplot(build_rolling_figure(ticker_select, ticker_data), clear_figure=True, use_container_width=True)


TAB_SPEC = TabSpec(
    slug="lppl",
    title="LPPL-MFDFA",
    description="LPPL crash fitting with rolling multifractal residual diagnostics and surrogate tests.",
    render=render,
)
