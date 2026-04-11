"""
MF-DFA Multifractal Analysis — self-contained runner.
Generates:
  - Per-ticker 6-panel MF-DFA diagnostic plots (mf_plots)
  - Sector-grouped symmetric Pearson correlation plots (mf_corr_plots)
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings
import yfinance as yf
from scipy.signal import savgol_filter

from plot_helpers import IMAGE_SUFFIXES, relative_to_project

warnings.filterwarnings("ignore")

TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "V", "MA",
    "JNJ", "PFE", "UNH", "MRK", "ABBV", "PG", "KO", "PEP", "WMT", "COST",
    "XOM", "CVX", "COP", "BA", "CAT", "GE", "HON", "LMT", "RTX", "UPS",
    "GC=F", "SI=F", "CL=F", "BTC-USD", "ETH-USD",
    "EWJ", "EWZ", "FXI", "EWG", "EWU",
]
START = "2010-01-01"
END = "2026-01-01"
Q_VALS = np.arange(-30, 31, 5, dtype=float)
SCALE_MIN = 20
N_SCALES = 30
POLY_ORD = 1
SCALE_MAX = 800
OVERLAP = 0.75

PAPER_COLORS = [
    "#d62728", "#2ca02c", "#1f77b4", "#000000", "#9467bd",
    "#ff7f0e", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
]

SECTORS = {
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


# ── Analysis core ─────────────────────────────────────────────────────────────
def _prepare_series(price):
    price = price[price > 0]
    log_price = np.log(price)
    returns = np.diff(log_price)
    return price, log_price, returns


def _integrate(x):
    return np.cumsum(x - x.mean())


def _overlap_starts(n, s, overlap):
    step = max(1, int(s * (1.0 - overlap)))
    return np.arange(0, n - s + 1, step)


def _mfdfa(x, q_vals, scale_min, scale_max, n_scales, poly_ord=1, overlap=0.75):
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
        rms = []
        for st in starts:
            seg = y[st: st + s]
            coef = np.polyfit(t, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2))
            if r > 0:
                rms.append(r)
        rms = np.asarray(rms)
        if len(rms) == 0:
            continue
        for j, q in enumerate(q_vals):
            if q == 0:
                fqs[j, i] = np.exp(0.5 * np.mean(np.log(rms ** 2)))
            else:
                fqs[j, i] = (np.mean(rms ** q)) ** (1.0 / q)
    hq = np.full(len(q_vals), np.nan)
    for j in range(len(q_vals)):
        row = fqs[j]
        mask = np.isfinite(row) & (row > 0)
        if mask.sum() >= 4:
            hq[j] = np.polyfit(np.log(scales[mask]), np.log(row[mask]), 1)[0]
    return scales, fqs, hq


def _multifractal_spectrum(q_vals, hq):
    tau_q = q_vals * hq - 1.0
    wl = min(5, len(tau_q) if len(tau_q) % 2 == 1 else len(tau_q) - 1)
    tau_smooth = savgol_filter(tau_q, window_length=wl, polyorder=2) if len(tau_q) >= 5 else tau_q
    alpha = np.gradient(tau_smooth, q_vals)
    d_alpha = q_vals * alpha - tau_smooth
    return tau_q, alpha, d_alpha


def _mfdfa_symmetric(x, q_vals, scale_min, scale_max, n_scales, poly_ord=1, overlap=0.75):
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
        rms = []
        for st in starts:
            seg = y[st: st + s]
            coef = np.polyfit(t, seg, poly_ord)
            r = np.sqrt(np.mean((seg - np.polyval(coef, t)) ** 2))
            if r > 0:
                rms.append(r)
        rms = np.asarray(rms)
        if len(rms) == 0:
            continue
        for j, q in enumerate(q_vals):
            fqs_pos[j, i] = (np.mean(rms ** q)) ** (1.0 / q)
            fqs_neg[j, i] = (np.mean(rms ** (-q))) ** (1.0 / (-q))
    return scales, fqs_pos, fqs_neg


def _pearson_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if mask.sum() < 3:
        return float("nan")
    a_, b_ = np.log(a[mask]), np.log(b[mask])
    cov = np.mean((a_ - a_.mean()) * (b_ - b_.mean()))
    denom = a_.std() * b_.std()
    if denom == 0:
        return float("nan")
    return float(cov / denom)


# ── 6-panel plot ──────────────────────────────────────────────────────────────
def _plot_6panel(ticker, method, series_key, raw_series, scales, fqs, hq,
                 q_vals, tau_q, alpha, d_alpha):
    labels = {"log_prices": "Log-Prices", "returns": "Log-Returns"}
    series_label = labels.get(series_key, series_key)
    valid = np.isfinite(hq)
    colors = (PAPER_COLORS * 4)[: len(q_vals)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Multifractal analysis  ·  {ticker}  ·  {series_label}  ·  {method}",
                 fontsize=13, fontweight="bold", y=1.01)

    ax = axs[0, 0]
    ax.plot(raw_series, color="#c0392b", linewidth=0.65)
    ax.set_xlabel("$t$", fontsize=11); ax.set_ylabel("$X(t)$", fontsize=11)
    ax.set_title("(a) Time series", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))

    ax = axs[0, 1]
    handles = []
    for idx_q, (q, c) in enumerate(zip(q_vals, colors)):
        row = fqs[idx_q]; mask = np.isfinite(row) & (row > 0)
        if mask.sum() < 2: continue
        h, = ax.loglog(scales[mask], row[mask], "o-", color=c, markersize=3.5, linewidth=1.0)
        handles.append((h, f"$q={int(q)}$"))
    ax.set_xlabel("$s$", fontsize=11); ax.set_ylabel("$F_q(s)$", fontsize=11)
    ax.set_title("(b) Fluctuation functions $F_q(s)$", fontsize=11)
    step = max(1, len(handles) // 8); shown = handles[::step]
    ax.legend([h for h, _ in shown], [l for _, l in shown], fontsize=7, ncol=1, loc="best",
              handlelength=1.2, handletextpad=0.4)

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
        a.tick_params(direction="in", which="both", top=True, right=True)
        a.spines[["top", "right"]].set_visible(True)

    if valid_s.sum() >= 2:
        da = alpha[valid_s].max() - alpha[valid_s].min()
        fig.text(0.5, -0.01, fr"Multifractal width  $\Delta\alpha = {da:.4f}$",
                 ha="center", fontsize=10, color="grey")
    plt.tight_layout()
    return fig


# ── Public entry points ───────────────────────────────────────────────────────
def generate_mf_plots(output_dir: str | Path, **overrides) -> dict:
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    before = _scan(output_dir)

    tickers = overrides.get("TICKERS", TICKERS)
    start = overrides.get("START", START)
    end = overrides.get("END", END)

    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end, progress=False)
        close = data["Close"].dropna().values.ravel().astype(float)
        if len(close) < 200:
            continue
        _, _, returns = _prepare_series(close)
        s_max_eff = min(SCALE_MAX, len(returns) // 4)
        s_min_eff = max(SCALE_MIN, POLY_ORD + 2)
        scales, fqs, hq = _mfdfa(returns, Q_VALS, s_min_eff, s_max_eff, N_SCALES, POLY_ORD, OVERLAP)
        tau_q, alpha, d_alpha = _multifractal_spectrum(Q_VALS, hq)
        fig = _plot_6panel(ticker, "MF-DFA", "returns", returns, scales, fqs, hq,
                           Q_VALS, tau_q, alpha, d_alpha)
        fname = output_dir / f"{ticker.replace('=', '_')}_MF_DFA_returns.png"
        fig.savefig(str(fname), dpi=130, bbox_inches="tight")
        plt.close(fig)

    after = _scan(output_dir)
    changed = [p for p, m in after.items() if p not in before or before[p] != m]
    return {
        "output_dir": relative_to_project(output_dir),
        "changed_count": len(changed),
        "changed_files": [relative_to_project(Path(p)) for p in sorted(changed)],
    }


def generate_mf_corr_plots(output_dir: str | Path, **overrides) -> dict:
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    before = _scan(output_dir)

    start = overrides.get("START", START)
    end = overrides.get("END", END)
    q_corr = np.arange(1, 31, dtype=float)

    for full_range in (False, True):
        n_sectors = len(SECTORS)
        ncols = 2
        nrows = int(np.ceil(n_sectors / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes).flatten()

        for ax_idx, (sector_name, sector) in enumerate(SECTORS.items()):
            ax = axes[ax_idx]
            tickers = sector["tickers"]
            colors = sector["colors"]
            for t_idx, ticker in enumerate(tickers):
                data = yf.download(ticker, start=start, end=end, progress=False)
                close = data["Close"].dropna().values.ravel().astype(float)
                if len(close) < 200:
                    continue
                returns = np.diff(np.log(close[close > 0]))
                s_min = max(SCALE_MIN, POLY_ORD + 2)
                s_max = min(SCALE_MAX, len(returns) // 4)
                scales, fqs_pos, fqs_neg = _mfdfa_symmetric(returns, q_corr, s_min, s_max, N_SCALES, POLY_ORD, OVERLAP)
                corr_vals = np.array([_pearson_corr(fqs_pos[j], fqs_neg[j]) for j in range(len(q_corr))])
                valid = np.isfinite(corr_vals)
                color = colors[t_idx % len(colors)]
                ax.plot(q_corr[valid], corr_vals[valid], "o-", color=color, linewidth=1.8, markersize=4.5, label=ticker)
            ax.set_title(sector_name, fontsize=13, fontweight="bold")
            ax.set_xlabel("|q|", fontsize=11)
            ax.set_ylabel("Pearson $r$", fontsize=11)
            ax.set_xticks(q_corr[::2])
            ax.set_xlim(1, 30)
            if full_range:
                ax.set_ylim(-0.1, 1.05)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
            else:
                ax.set_ylim(0.90, 1.005)
                ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
            ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.4)
            ax.axhline(0.0, color="grey", lw=0.8, ls=":", alpha=0.4)
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.legend(fontsize=8.5, ncol=2, loc="lower left", framealpha=0.9)

        for ax in axes[n_sectors:]:
            ax.set_visible(False)
        range_tag = " [Full Range]" if full_range else ""
        fig.suptitle(
            "Pearson correlation between $F_{+q}(s)$ and $F_{-q}(s)$\n"
            f"MF-DFA · Log-Returns · q = 1 to 30  ·  By Sector{range_tag}",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        suffix = "_fullrange" if full_range else ""
        fname = output_dir / f"mfdfa_pearson_by_sector{suffix}.png"
        fig.savefig(str(fname), dpi=130, bbox_inches="tight")
        plt.close(fig)

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
