from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.signal import savgol_filter

from dashboard.tab_api import TabSpec

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "V", "MA",
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "PG", "KO", "PEP", "WMT", "COST",
    "XOM", "CVX", "COP", "BA", "CAT", "GE", "HON", "LMT", "RTX", "UPS",
    "GC=F", "SI=F", "CL=F", "BTC-USD", "ETH-USD",
    "EWJ", "EWZ", "FXI", "EWG", "EWU",
]

CORR_SECTORS: dict[str, dict[str, Any]] = {
    "Tech": {
        "tickers": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC"],
        "colors": ["#1f77b4", "#4e9fd4", "#2196f3", "#0d47a1", "#17becf",
                   "#00bcd4", "#006994", "#4fc3f7", "#0288d1", "#80d8ff"],
    },
    "Financials": {
        "tickers": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "V", "MA"],
        "colors": ["#2ca02c", "#27ae60", "#1abc9c", "#16a085", "#8db600",
                   "#4caf50", "#388e3c", "#00897b", "#6b8e23", "#a5d6a7"],
    },
    "Healthcare & Consumer": {
        "tickers": ["JNJ", "PFE", "UNH", "MRK", "ABBV", "PG", "KO", "PEP", "WMT", "COST"],
        "colors": ["#9467bd", "#8e44ad", "#6c3483", "#7c3aed", "#a855f7",
                   "#ba68c8", "#7b1fa2", "#ce93d8", "#76448a", "#e1bee7"],
    },
    "Energy & Industrials": {
        "tickers": ["XOM", "CVX", "COP", "BA", "CAT", "GE", "HON", "LMT", "RTX", "UPS"],
        "colors": ["#ff7f0e", "#e8631a", "#f39c12", "#d35400", "#ff6b6b",
                   "#e64a19", "#bf360c", "#ff8f00", "#f57c00", "#ffcc02"],
    },
    "Commodities & Crypto": {
        "tickers": ["GC=F", "SI=F", "CL=F", "BTC-USD", "ETH-USD"],
        "colors": ["#bcbd22", "#f1c40f", "#ffd600", "#ffab00", "#ffe57f"],
    },
    "International ETFs": {
        "tickers": ["EWJ", "EWZ", "FXI", "EWG", "EWU"],
        "colors": ["#d62728", "#c0392b", "#e74c3c", "#b71c1c", "#ef9a9a"],
    },
}

PAPER_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#000000", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]


@dataclass(frozen=True)
class MfConfig:
    tickers: tuple[str, ...]
    start: str
    end: str
    q_min: int
    q_max: int
    q_step: int
    scale_min: int
    scale_max: int
    n_scales: int
    poly_ord: int
    overlap: float


def _prepare_series(price: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    price = price[price > 0]
    log_price = np.log(price)
    returns = np.diff(log_price)
    return price, log_price, returns


def _integrate(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x - x.mean())


def _overlap_starts(n: int, s: int, overlap: float) -> np.ndarray:
    step = max(1, int(s * (1.0 - overlap)))
    return np.arange(0, n - s + 1, step)


def _mfdfa(
    x: np.ndarray,
    q_vals: np.ndarray,
    scale_min: int,
    scale_max: int,
    n_scales: int,
    poly_ord: int = 1,
    overlap: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _integrate(x)
    n = len(y)
    s_max = min(scale_max, n // 4)
    s_min = max(scale_min, poly_ord + 2)
    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int))
    scales = scales[scales >= s_min]
    fqs = np.full((len(q_vals), len(scales)), np.nan)

    for i, s in enumerate(scales):
        starts = _overlap_starts(n, s, overlap)
        if len(starts) < 4:
            continue
        t = np.arange(s)
        rms: list[float] = []
        for st in starts:
            seg = y[st : st + s]
            coef = np.polyfit(t, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2))
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


def _multifractal_spectrum(q_vals: np.ndarray, hq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_q = q_vals * hq - 1.0
    wl = min(5, len(tau_q) if len(tau_q) % 2 == 1 else len(tau_q) - 1)
    tau_smooth = savgol_filter(tau_q, window_length=wl, polyorder=2) if len(tau_q) >= 5 else tau_q
    alpha = np.gradient(tau_smooth, q_vals)
    d_alpha = q_vals * alpha - tau_smooth
    return tau_q, alpha, d_alpha


def _mfdfa_symmetric(
    x: np.ndarray,
    q_vals: np.ndarray,
    scale_min: int,
    scale_max: int,
    n_scales: int,
    poly_ord: int = 1,
    overlap: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _integrate(x)
    n = len(y)
    s_max = min(scale_max, n // 4)
    s_min = max(scale_min, poly_ord + 2)
    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int))
    scales = scales[scales >= s_min]
    fqs_pos = np.full((len(q_vals), len(scales)), np.nan)
    fqs_neg = np.full((len(q_vals), len(scales)), np.nan)

    for i, s in enumerate(scales):
        starts = _overlap_starts(n, s, overlap)
        if len(starts) < 4:
            continue
        t = np.arange(s)
        rms: list[float] = []
        for st in starts:
            seg = y[st : st + s]
            coef = np.polyfit(t, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2))
            if r > 0:
                rms.append(r)
        rms_arr = np.asarray(rms)
        if len(rms_arr) == 0:
            continue
        for j, q in enumerate(q_vals):
            fqs_pos[j, i] = (np.mean(rms_arr**q)) ** (1.0 / q)
            fqs_neg[j, i] = (np.mean(rms_arr ** (-q))) ** (1.0 / (-q))
    return scales, fqs_pos, fqs_neg


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 3:
        return float("nan")
    a_log, b_log = np.log(a[mask]), np.log(b[mask])
    cov = np.mean((a_log - a_log.mean()) * (b_log - b_log.mean()))
    denom = a_log.std() * b_log.std()
    if denom == 0:
        return float("nan")
    return float(cov / denom)


def _style_ax(ax: Any) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.spines[["top", "right"]].set_visible(True)


# ---------------------------------------------------------------------------
# Analysis runners
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_6panel_analysis(config: MfConfig) -> dict[str, Any]:
    q_vals = np.arange(config.q_min, config.q_max + 1, config.q_step, dtype=float)
    results: dict[str, Any] = {}

    for ticker in config.tickers:
        data = yf.download(ticker, start=config.start, end=config.end, progress=False)
        close = data["Close"].dropna().values.ravel().astype(float)
        if len(close) < 200:
            continue
        _, log_price, returns = _prepare_series(close)
        s_max_eff = min(config.scale_max, len(returns) // 4)
        s_min_eff = max(config.scale_min, config.poly_ord + 2)
        scales, fqs, hq = _mfdfa(returns, q_vals, s_min_eff, s_max_eff, config.n_scales, config.poly_ord, config.overlap)
        tau_q, alpha, d_alpha = _multifractal_spectrum(q_vals, hq)
        results[ticker] = {
            "returns": returns,
            "scales": scales,
            "fqs": fqs,
            "hq": hq,
            "q_vals": q_vals,
            "tau_q": tau_q,
            "alpha": alpha,
            "d_alpha": d_alpha,
        }
    return results


@st.cache_data(show_spinner=False)
def run_corr_analysis(config: MfConfig) -> dict[str, dict[str, np.ndarray]]:
    q_vals = np.arange(1, 31, dtype=float)
    results: dict[str, dict[str, np.ndarray]] = {}
    for sector_name, sector in CORR_SECTORS.items():
        sector_corrs: dict[str, np.ndarray] = {}
        for ticker in sector["tickers"]:
            data = yf.download(ticker, start=config.start, end=config.end, progress=False)
            close = data["Close"].dropna().values.ravel().astype(float)
            if len(close) < 200:
                continue
            returns = np.diff(np.log(close[close > 0]))
            s_min = max(config.scale_min, config.poly_ord + 2)
            s_max = min(config.scale_max, len(returns) // 4)
            scales, fqs_pos, fqs_neg = _mfdfa_symmetric(returns, q_vals, s_min, s_max, config.n_scales, config.poly_ord, config.overlap)
            corr_vals = np.array([_pearson_corr(fqs_pos[j], fqs_neg[j]) for j in range(len(q_vals))])
            sector_corrs[ticker] = corr_vals
        if sector_corrs:
            results[sector_name] = sector_corrs
    return results


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_6panel_figure(ticker: str, data: dict[str, Any]) -> plt.Figure:
    q_vals = data["q_vals"]
    hq = data["hq"]
    tau_q = data["tau_q"]
    alpha = data["alpha"]
    d_alpha = data["d_alpha"]
    scales = data["scales"]
    fqs = data["fqs"]
    raw = data["returns"]
    valid = np.isfinite(hq)
    colors = (PAPER_COLORS * 4)[: len(q_vals)]

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Multifractal analysis  ·  {ticker}  ·  Log-Returns  ·  MF-DFA", fontsize=13, fontweight="bold", y=1.01)

    ax = axs[0, 0]
    ax.plot(raw, color="#c0392b", linewidth=0.65)
    ax.set_xlabel("$t$", fontsize=11); ax.set_ylabel("$X(t)$", fontsize=11)
    ax.set_title("(a) Time series", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))

    ax = axs[0, 1]
    handles = []
    for idx_q, (q, c) in enumerate(zip(q_vals, colors)):
        row = fqs[idx_q]; mask = np.isfinite(row) & (row > 0)
        if mask.sum() < 2:
            continue
        h, = ax.loglog(scales[mask], row[mask], "o-", color=c, markersize=3.5, linewidth=1.0)
        handles.append((h, f"$q={int(q)}$"))
    ax.set_xlabel("$s$", fontsize=11); ax.set_ylabel("$F_q(s)$", fontsize=11)
    ax.set_title("(b) Fluctuation functions $F_q(s)$", fontsize=11)
    step = max(1, len(handles) // 8)
    shown = handles[::step]
    ax.legend([h for h, _ in shown], [l for _, l in shown], fontsize=7, ncol=1, loc="best")

    ax = axs[0, 2]
    ax.plot(q_vals[valid], hq[valid], "-", color="#1f4e8c", linewidth=2)
    ax.set_xlabel("$q$", fontsize=11); ax.set_ylabel("$H(q)$", fontsize=11)
    ax.set_title("(c) Generalised Hurst exponent $H(q)$", fontsize=11)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6, label="H=0.5")
    ax.axhline(1.0, color="blue", lw=0.8, ls=":", alpha=0.6, label="H=1.0")
    ax.legend(fontsize=7)

    ax = axs[1, 0]
    ax.plot(q_vals[valid], tau_q[valid], "-", color="#1f4e8c", linewidth=2)
    ax.set_xlabel("$q$", fontsize=11); ax.set_ylabel(r"$\tau(q)$", fontsize=11)
    ax.set_title(r"(d) Scaling exponent $\tau(q)$", fontsize=11)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.6)

    valid_s = np.isfinite(alpha) & np.isfinite(d_alpha)
    ax = axs[1, 1]
    ax.plot(q_vals[valid_s], alpha[valid_s], "-", color="#1f4e8c", linewidth=2)
    ax.set_xlabel("$q$", fontsize=11); ax.set_ylabel(r"$\alpha(q)$", fontsize=11)
    ax.set_title(r"(e) Singularity strength $\alpha(q)$", fontsize=11)

    ax = axs[1, 2]
    ax.plot(alpha[valid_s], d_alpha[valid_s], "-", color="#1f4e8c", linewidth=2)
    q0_idx = np.argmin(np.abs(q_vals))
    if np.isfinite(alpha[q0_idx]) and np.isfinite(d_alpha[q0_idx]):
        ax.plot(alpha[q0_idx], d_alpha[q0_idx], "o", color="#c0392b", markersize=6, zorder=5)
    ax.set_xlabel(r"$\alpha$", fontsize=11); ax.set_ylabel(r"$f(\alpha)$", fontsize=11)
    ax.set_title(r"(f) Multifractal spectrum $f(\alpha)$", fontsize=11)

    for a in axs.ravel():
        _style_ax(a)

    if valid_s.sum() >= 2:
        da = alpha[valid_s].max() - alpha[valid_s].min()
        fig.text(0.5, -0.01, fr"Multifractal width  $\Delta\alpha = {da:.4f}$", ha="center", fontsize=10, color="grey")
    fig.tight_layout()
    return fig


def build_corr_figure(corr_results: dict[str, dict[str, np.ndarray]], full_range: bool = False) -> plt.Figure:
    q_vals = np.arange(1, 31, dtype=float)
    n_sectors = len(corr_results)
    ncols = 2
    nrows = int(np.ceil(n_sectors / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for ax_idx, (sector_name, ticker_corrs) in enumerate(corr_results.items()):
        ax = axes[ax_idx]
        sector_def = CORR_SECTORS.get(sector_name)
        for t_idx, (ticker, corr_vals) in enumerate(ticker_corrs.items()):
            color = sector_def["colors"][t_idx % len(sector_def["colors"])] if sector_def else f"C{t_idx}"
            valid = np.isfinite(corr_vals)
            ax.plot(q_vals[valid], corr_vals[valid], "o-", color=color, linewidth=1.8, markersize=4.5, label=ticker)
        ax.set_title(sector_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("|q|", fontsize=11); ax.set_ylabel("Pearson $r$", fontsize=11)
        ax.set_xticks(q_vals[::2]); ax.set_xlim(1, 30)
        if full_range:
            ax.set_ylim(-0.1, 1.05)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
        else:
            ax.set_ylim(0.90, 1.005)
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.4)
        ax.axhline(0.0, color="grey", lw=0.8, ls=":", alpha=0.4)
        _style_ax(ax)
        ax.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.9)
    for ax in axes[n_sectors:]:
        ax.set_visible(False)
    range_label = " [Full Range]" if full_range else ""
    fig.suptitle(
        "Pearson correlation between $F_{+q}(s)$ and $F_{-q}(s)$\n"
        f"MF-DFA · Log-Returns · q = 1 to 30  ·  By Sector{range_label}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        "This tab runs MF-DFA on log-returns and generates a 6-panel diagnostic for each ticker, "
        "plus a sector-grouped Pearson correlation view between positive and negative fluctuation functions."
    )

    analysis_type = st.radio("Analysis", ["6-Panel MF-DFA", "Pearson Correlation (by sector)", "Both"], horizontal=True)

    with st.form("mf-controls"):
        tickers_str = st.text_area("Tickers (comma-separated)", value=", ".join(DEFAULT_TICKERS), height=80)
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start", value=pd.Timestamp("2010-01-01").date())
            q_min = st.number_input("q min", value=-30, step=5)
            scale_min = st.number_input("Scale min", value=20, min_value=4)
        with col2:
            end = st.date_input("End", value=pd.Timestamp("2026-01-01").date())
            q_max = st.number_input("q max", value=30, step=5)
            scale_max = st.number_input("Scale max", value=800, min_value=20)
        submitted = st.form_submit_button("Run Multifractal Analysis", type="primary")

    tickers = tuple(t.strip() for t in tickers_str.split(",") if t.strip())
    config = MfConfig(
        tickers=tickers,
        start=str(start),
        end=str(end),
        q_min=int(q_min), q_max=int(q_max), q_step=5,
        scale_min=int(scale_min), scale_max=int(scale_max),
        n_scales=30, poly_ord=1, overlap=0.75,
    )

    if "mf_has_run" not in st.session_state:
        st.session_state["mf_has_run"] = False
    if submitted:
        st.session_state["mf_has_run"] = True
    if not st.session_state["mf_has_run"]:
        st.info("Configure parameters and run.")
        return

    run_6 = analysis_type in ("6-Panel MF-DFA", "Both")
    run_corr = analysis_type in ("Pearson Correlation (by sector)", "Both")

    if run_6:
        with st.spinner("Running per-ticker MF-DFA..."):
            results_6 = run_6panel_analysis(config)
        st.metric("Tickers analysed", len(results_6))
        ticker_select = st.selectbox("Select ticker", list(results_6.keys()))
        if ticker_select:
            fig = build_6panel_figure(ticker_select, results_6[ticker_select])
            st.pyplot(fig, clear_figure=True, use_container_width=True)

    if run_corr:
        with st.spinner("Running symmetric MF-DFA correlation..."):
            corr_results = run_corr_analysis(config)
        full_range = st.checkbox("Full y-axis range", value=False)
        fig = build_corr_figure(corr_results, full_range=full_range)
        st.pyplot(fig, clear_figure=True, use_container_width=True)


TAB_SPEC = TabSpec(
    slug="multifractal",
    title="Multifractal MF-DFA",
    description="Per-ticker 6-panel MF-DFA diagnostics and sector-grouped Pearson correlation.",
    render=render,
)
