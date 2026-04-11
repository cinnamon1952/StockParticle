"""
Correlation & Returns Analysis — Replicating Sornette's "Why Stock Markets Crash"
Self-contained runner: generates 7 figures, saves to output_dir.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import curve_fit

from plot_helpers import IMAGE_SUFFIXES, relative_to_project

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance")

DJIA_TICKER = "^DJI"
NASDAQ_TICKER = "^IXIC"
START_DATE = "1990-01-01"
END_DATE = "2024-01-01"
FIG_STYLE = "seaborn-v0_8-darkgrid"

DJIA_COMPONENTS_2008 = [
    "MMM", "AXP", "BA", "CAT", "CVX", "KO", "XOM",
    "GE", "IBM", "MCD", "MRK", "PG", "WMT", "JPM", "MSFT",
]
ROLL_WINDOW_2008 = 60


def _download(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df["Close"].squeeze().dropna()


def _compute_returns(price):
    log_ret = np.log(price / price.shift(1)).dropna()
    simp_ret = price.pct_change().dropna()
    return log_ret, simp_ret


def _resample_returns(log_ret, freq="ME"):
    return log_ret.resample(freq).sum().dropna()


def _autocorrelation(series, max_lag=60):
    series = (series - series.mean()) / series.std()
    lags, vals = [], []
    for lag in range(1, max_lag + 1):
        c = np.corrcoef(series[:-lag].values, series[lag:].values)[0, 1]
        lags.append(lag)
        vals.append(c)
    return np.array(lags), np.array(vals)


def _autocorrelation_abs(series, max_lag=60):
    return _autocorrelation(series.abs(), max_lag)


def _scaling_law_test(daily_log_ret):
    std_d = daily_log_ret.std()
    weekly = daily_log_ret.resample("W").sum().dropna()
    monthly = daily_log_ret.resample("ME").sum().dropna()
    quarterly = daily_log_ret.resample("QE").sum().dropna()
    return {
        "Daily (baseline)": (std_d, std_d),
        "Weekly  (√5  days)": (std_d * np.sqrt(5), weekly.std()),
        "Monthly (√21 days)": (std_d * np.sqrt(21), monthly.std()),
        "Quarterly(√63 days)": (std_d * np.sqrt(63), quarterly.std()),
    }


def _ccdf_counts(arr):
    arr_s = np.sort(arr)
    counts = np.arange(len(arr_s), 0, -1)
    return arr_s, counts


def _fit_exponential(x, a, b):
    return a * np.exp(-b * x)


def generate(output_dir: str | Path, **kwargs) -> dict:
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use(FIG_STYLE)
    before = _scan(output_dir)

    djia_price = _download(DJIA_TICKER, START_DATE, END_DATE)
    nasdaq_price = _download(NASDAQ_TICKER, START_DATE, END_DATE)
    djia_log, djia_simp = _compute_returns(djia_price)
    nasdaq_log, nasdaq_simp = _compute_returns(nasdaq_price)
    djia_monthly = _resample_returns(djia_log, "ME")
    nasdaq_monthly = _resample_returns(nasdaq_log, "ME")

    # ── FIG 1: distributions ──────────────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(9, 7))
    fig1.suptitle("FIG 2.7 Replica — Distribution of daily returns\n"
                  "DJIA and Nasdaq, January 2 1990 – September 29 2000",
                  fontsize=13, fontweight="bold")
    djia_book = djia_log["1990-01-02":"2000-09-29"]
    nasdaq_book = nasdaq_log["1990-01-02":"2000-09-29"]
    _plot_book_distribution(ax, djia_book, nasdaq_book)
    fig1.tight_layout()
    fig1.savefig(str(output_dir / "fig1_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # ── FIG 2: autocorrelation ────────────────────────────────────────────
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("Autocorrelation of Returns (Daily & Monthly)\n"
                  "Raw returns decorrelate instantly; |returns| show persistent volatility clustering",
                  fontsize=13, fontweight="bold")
    conf = 1.96 / np.sqrt(len(djia_log))

    for ax, (series, label, ml) in zip(axes.ravel(), [
        (djia_log, "DJIA Daily Returns", 30),
        (nasdaq_log, "Nasdaq Daily Returns", 30),
        (djia_monthly, "DJIA Monthly Returns", 24),
        (nasdaq_monthly, "Nasdaq Monthly Returns", 24),
    ]):
        lags_r, acf_r = _autocorrelation(series, ml)
        lags_a, acf_a = _autocorrelation_abs(series, ml)
        ax.bar(lags_r - 0.2, acf_r, width=0.4, alpha=0.7, color="steelblue", label="Returns ACF")
        ax.bar(lags_a + 0.2, acf_a, width=0.4, alpha=0.7, color="coral", label="|Returns| ACF")
        ci = 1.96 / np.sqrt(len(series))
        ax.axhline(ci, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axhline(-ci, color="gray", ls="--", lw=1, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("Autocorrelation", fontsize=10)
        ax.set_ylim(-0.15, 0.25)
        ax.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(str(output_dir / "fig2_autocorrelation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── FIG 3: scaling law ────────────────────────────────────────────────
    djia_scaling = _scaling_law_test(djia_log)
    nasdaq_scaling = _scaling_law_test(nasdaq_log)

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    fig3.suptitle("Square-Root-of-Time Scaling Law — DJIA & Nasdaq\n"
                  "If returns are i.i.d., volatility should scale as √T",
                  fontsize=13, fontweight="bold")

    for ax, (scaling, title) in zip(axes3, [(djia_scaling, "DJIA"), (nasdaq_scaling, "Nasdaq")]):
        labels = list(scaling.keys())
        expected = [v[0] * 100 for v in scaling.values()]
        actual = [v[1] * 100 for v in scaling.values()]
        x = np.arange(len(labels))
        ax.bar(x - 0.15, expected, 0.3, color="steelblue", alpha=0.8, label="Expected (√T)")
        ax.bar(x + 0.15, actual, 0.3, color="coral", alpha=0.8, label="Actual")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Volatility (σ, %)", fontsize=10)
        ax.set_title(f"{title} — Scaling Law Test", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        for i in range(len(labels)):
            ratio = actual[i] / expected[i]
            ax.text(i, max(expected[i], actual[i]) + 0.05, f"ratio={ratio:.2f}",
                    ha="center", fontsize=8, color="darkred")
    fig3.tight_layout()
    fig3.savefig(str(output_dir / "fig3_scaling_law.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── FIG 4: volatility clustering ──────────────────────────────────────
    fig4, axes4 = plt.subplots(3, 2, figsize=(16, 12))
    fig4.suptitle("Volatility Clustering — Self-Similar Bursts Across Scales\n"
                  "Extreme events (red dots) cluster in time at every frequency",
                  fontsize=13, fontweight="bold")
    for col, (log_ret, ticker_label) in enumerate([(djia_log, "DJIA"), (nasdaq_log, "Nasdaq")]):
        for row, (series, freq_label) in enumerate([
            (log_ret, "Daily"),
            (log_ret.resample("W").sum().dropna(), "Weekly"),
            (log_ret.resample("ME").sum().dropna(), "Monthly"),
        ]):
            ax = axes4[row, col]
            threshold = series.std() * 3
            extreme = series.abs() > threshold
            ax.plot(series.index, series.values, color="steelblue", lw=0.5, alpha=0.6)
            ax.scatter(series.index[extreme], series.values[extreme],
                       color="red", s=15, zorder=5, label=f"|ret| > 3σ (n={extreme.sum()})")
            ax.set_title(f"{ticker_label} — {freq_label} Returns", fontsize=10, fontweight="bold")
            ax.set_ylabel("Log Return", fontsize=9)
            ax.legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(str(output_dir / "fig4_volatility_clustering.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── FIG 5: 2008 pairwise correlation ──────────────────────────────────
    component_data = {}
    for t in DJIA_COMPONENTS_2008:
        try:
            s = _download(t, "2004-01-01", "2010-01-01")
            if len(s) > 100:
                component_data[t] = s
        except Exception:
            pass

    returns_2008 = pd.DataFrame({t: np.log(s / s.shift(1)).dropna() for t, s in component_data.items()}).dropna()

    def avg_pairwise_corr(df):
        corr = df.corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        vals = corr.where(mask).stack().values
        return vals.mean() if len(vals) > 0 else np.nan

    roll_pw = pd.Series(index=returns_2008.index, dtype=float)
    for i in range(ROLL_WINDOW_2008, len(returns_2008)):
        roll_pw.iloc[i] = avg_pairwise_corr(returns_2008.iloc[i - ROLL_WINDOW_2008:i])
    roll_pw = roll_pw.dropna()
    roll_pw_smooth = roll_pw.rolling(10, min_periods=1).mean()

    djia_2008 = _download(DJIA_TICKER, "2004-01-01", "2010-01-01")
    BUBBLE_2008_START = "2006-01-01"
    DJIA_2008_PEAK = "2007-10-09"
    CRASH_BOTTOM = "2009-03-09"

    fig5, axes5 = plt.subplots(2, 1, figsize=(14, 10))
    fig5.suptitle("2008 Crash — Rolling Average Pairwise Correlation (DJIA)\n"
                  "Rising stock synchronisation signals growing systemic risk before collapse",
                  fontsize=13, fontweight="bold")
    ax1 = axes5[0]
    ax1.plot(djia_2008.index, djia_2008.values, color="steelblue", lw=1.5)
    ax1.axvspan(pd.Timestamp(BUBBLE_2008_START), pd.Timestamp(DJIA_2008_PEAK), alpha=0.15, color="red", label="Bubble buildup")
    ax1.axvspan(pd.Timestamp(DJIA_2008_PEAK), pd.Timestamp(CRASH_BOTTOM), alpha=0.15, color="darkred", label="Crash & decline")
    ax1.axvline(pd.Timestamp(DJIA_2008_PEAK), color="red", lw=1.5, ls="--", label="DJIA Peak (Oct 9, 2007)")
    ax1.axvline(pd.Timestamp(CRASH_BOTTOM), color="darkred", lw=1.5, ls="--", label="Market Bottom (Mar 9, 2009)")
    ax1.set_title("DJIA Price — 2008 Bubble and Crash", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Index Level", fontsize=10)
    ax1.legend(fontsize=9)

    ax2 = axes5[1]
    ax2.plot(roll_pw_smooth.index, roll_pw_smooth.values, color="steelblue", lw=1.5,
             label=f"Avg pairwise correlation ({ROLL_WINDOW_2008}-day rolling)")
    ax2.axvspan(pd.Timestamp(BUBBLE_2008_START), pd.Timestamp(DJIA_2008_PEAK), alpha=0.15, color="red")
    ax2.axvspan(pd.Timestamp(DJIA_2008_PEAK), pd.Timestamp(CRASH_BOTTOM), alpha=0.15, color="darkred")
    ax2.axvline(pd.Timestamp(DJIA_2008_PEAK), color="red", lw=1.5, ls="--", label="DJIA Peak")
    ax2.axvline(pd.Timestamp(CRASH_BOTTOM), color="darkred", lw=1.5, ls="--", label="Market Bottom")

    pre_peak_2008 = roll_pw_smooth[BUBBLE_2008_START:"2007-09-01"].dropna()
    if len(pre_peak_2008) > 10:
        x_num = np.arange(len(pre_peak_2008))
        slope, intercept = np.polyfit(x_num, pre_peak_2008.values, 1)
        trend = slope * x_num + intercept
        ax2.plot(pre_peak_2008.index, trend, color="black", lw=2, ls="--", alpha=0.7,
                 label=f"Buildup trend (slope={slope * 252:.4f}/yr)")
        mid = len(pre_peak_2008) // 2
        ax2.annotate("Stocks increasingly\nmoving in lockstep\nbefore crash",
                     xy=(pre_peak_2008.index[mid], trend[mid]),
                     xytext=(pre_peak_2008.index[max(0, mid - 150)], trend[mid] + pre_peak_2008.std() * 1.5),
                     arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.5),
                     fontsize=9, color="darkblue", fontweight="bold")

    crash_peak_idx = roll_pw_smooth["2008-01-01":"2009-06-01"].idxmax()
    crash_peak_val = roll_pw_smooth[crash_peak_idx]
    ax2.annotate("Correlation spikes\nat crash peak\n(maximum cooperativity)",
                 xy=(crash_peak_idx, crash_peak_val),
                 xytext=(pd.Timestamp("2008-01-01"), crash_peak_val - pre_peak_2008.std() * 1.2),
                 arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                 fontsize=9, color="darkred", fontweight="bold")
    ax2.set_title("Average Pairwise Correlation of DJIA Components — Cooperativity Measure", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Pairwise Correlation", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.legend(fontsize=9)
    fig5.tight_layout()
    fig5.savefig(str(output_dir / "fig5_bubble_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5)

    # ── FIG 5b: heatmaps ─────────────────────────────────────────────────
    snapshots = [
        ("Pre-Bubble\n(Jan 2006)", "2005-07-01", "2006-01-01", "Blues"),
        ("Pre-Crash Peak\n(Oct 2007)", "2007-04-01", "2007-10-09", "Oranges"),
        ("During Crash\n(Oct 2008)", "2008-04-01", "2008-10-01", "Reds"),
    ]
    fig5b, axes5b = plt.subplots(1, 3, figsize=(18, 6))
    fig5b.suptitle("Pairwise Correlation Heatmaps — DJIA Components at Key Moments\n"
                   "Deeper colour = higher correlation = more lockstep movement",
                   fontsize=13, fontweight="bold")
    for ax, (label, start, end, cmap) in zip(axes5b, snapshots):
        window_df = returns_2008[start:end]
        corr_matrix = window_df.corr()
        avg_c = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().mean()
        im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.index)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr_matrix.index, fontsize=8)
        ax.set_title(f"{label}\nAvg pairwise corr = {avg_c:.3f}", fontsize=11, fontweight="bold")
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                val = corr_matrix.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 0.6 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig5b.tight_layout()
    fig5b.savefig(str(output_dir / "fig5b_pairwise_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig5b)

    # ── FIG 6: log vs simple additivity ───────────────────────────────────
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
    fig6.suptitle("Log Returns vs Simple Returns — Additivity Property\n"
                  "Log returns sum exactly to period total; simple returns do not",
                  fontsize=13, fontweight="bold")
    sample = djia_log["2020-01-01":"2020-12-31"]
    monthly_sum = sample.resample("ME").sum()
    monthly_direct = np.log(
        djia_price["2020-01-01":"2020-12-31"].resample("ME").last() /
        djia_price["2020-01-01":"2020-12-31"].resample("ME").last().shift(1)
    ).dropna()
    common = monthly_sum.index.intersection(monthly_direct.index)
    axes6[0].scatter(monthly_sum.loc[common] * 100, monthly_direct.loc[common] * 100,
                     color="steelblue", s=60, edgecolors="black", lw=0.5)
    lims = [min(monthly_sum.loc[common].min(), monthly_direct.loc[common].min()) * 100 - 1,
            max(monthly_sum.loc[common].max(), monthly_direct.loc[common].max()) * 100 + 1]
    axes6[0].plot(lims, lims, "r--", lw=1.5, label="Perfect additivity (y=x)")
    axes6[0].set_xlabel("Sum of daily log returns (%)", fontsize=10)
    axes6[0].set_ylabel("Direct monthly log return (%)", fontsize=10)
    axes6[0].set_title("Log Returns: Sum of Parts = Whole (DJIA 2020)", fontsize=11, fontweight="bold")
    axes6[0].legend(fontsize=9)

    sample_simp = djia_simp["2020-01-01":"2020-12-31"]
    monthly_simp_sum = sample_simp.resample("ME").sum()
    monthly_simp_direct = djia_price["2020-01-01":"2020-12-31"].resample("ME").last().pct_change().dropna()
    common2 = monthly_simp_sum.index.intersection(monthly_simp_direct.index)
    axes6[1].scatter(monthly_simp_sum.loc[common2] * 100, monthly_simp_direct.loc[common2] * 100,
                     color="darkorange", s=60, edgecolors="black", lw=0.5)
    lims2 = [min(monthly_simp_sum.loc[common2].min(), monthly_simp_direct.loc[common2].min()) * 100 - 1,
             max(monthly_simp_sum.loc[common2].max(), monthly_simp_direct.loc[common2].max()) * 100 + 1]
    axes6[1].plot(lims2, lims2, "r--", lw=1.5, label="Perfect additivity (y=x)")
    axes6[1].set_xlabel("Sum of daily simple returns (%)", fontsize=10)
    axes6[1].set_ylabel("Direct monthly simple return (%)", fontsize=10)
    axes6[1].set_title("Simple Returns: Sum ≠ Whole — Diverges at Extremes (DJIA 2020)", fontsize=11, fontweight="bold")
    axes6[1].legend(fontsize=9)
    fig6.tight_layout()
    fig6.savefig(str(output_dir / "fig6_log_vs_simple.png"), dpi=150, bbox_inches="tight")
    plt.close(fig6)

    after = _scan(output_dir)
    changed = [p for p, m in after.items() if p not in before or before[p] != m]
    return {
        "output_dir": relative_to_project(output_dir),
        "changed_count": len(changed),
        "changed_files": [relative_to_project(Path(p)) for p in sorted(changed)],
    }


def _plot_book_distribution(ax, djia_ret, nasdaq_ret):
    r_dj = djia_ret.dropna().values
    r_nas = nasdaq_ret.dropna().values
    series = {
        "return DJ>0": (r_dj[r_dj > 0], "black", "+", "-"),
        "return DJ<0": (-r_dj[r_dj < 0], "black", "o", "-"),
        "return NAS>0": (r_nas[r_nas > 0], "gray", "x", "--"),
        "return NAS<0": (-r_nas[r_nas < 0], "gray", "D", "--"),
    }
    for label, (arr, color, marker, linestyle) in series.items():
        x, counts = _ccdf_counts(arr)
        step = max(1, len(x) // 300)
        ax.plot(x[::step], counts[::step], marker=marker, linestyle="none",
                color=color, markersize=4, alpha=0.7, label=label)
        try:
            mask = x > np.median(x)
            popt, _ = curve_fit(_fit_exponential, x[mask], counts[mask], p0=[counts[mask][0], 50], maxfev=5000)
            x_fit = np.linspace(x[mask][0], x.max(), 200)
            ax.plot(x_fit, _fit_exponential(x_fit, *popt), color=color, linestyle=linestyle, lw=1.5, alpha=0.9)
        except Exception:
            pass
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=1)
    ax.set_xlabel("|Returns|", fontsize=12)
    ax.set_ylabel("Distribution Function", fontsize=12)
    ax.set_title("Distribution of Daily Returns — DJIA & Nasdaq\n(Replicating Fig 2.7)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(y=10, color="black", lw=0.8, ls="-", alpha=0.3)
    ax.axhline(y=100, color="black", lw=0.8, ls="-", alpha=0.3)


def _scan(d: Path) -> dict:
    if not d.exists():
        return {}
    return {
        str(p.resolve()): (int(p.stat().st_mtime_ns), p.stat().st_size)
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }
