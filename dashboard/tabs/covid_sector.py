from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

from dashboard.tab_api import TabSpec
warnings.filterwarnings("ignore")


SECTORS: dict[str, list[str]] = {
    "Tech": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD", "INTC"],
    "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "V", "MA"],
    "Healthcare": ["JNJ", "PFE", "UNH", "MRK", "ABBV", "PG", "KO", "PEP", "WMT", "COST"],
    "Energy": ["XOM", "CVX", "COP", "BA", "CAT", "GE", "HON", "LMT", "RTX", "UPS"],
    "Commodities": ["GC=F", "SI=F", "CL=F"],
    "Crypto": ["BTC-USD", "ETH-USD"],
    "Intl ETFs": ["EWJ", "EWZ", "FXI", "EWG", "EWU"],
}

DEFAULT_PERIODS = {
    "Pre-COVID": ("2018-01-01", "2020-02-19"),
    "Crisis": ("2020-02-01", "2020-06-30"),
    "Post-COVID": ("2020-05-01", "2022-01-01"),
}

COVID_EVENTS = {
    "WHO\nEmergency": "2020-01-30",
    "Market\nBottom": "2020-03-23",
    "Vaccine\nApproval": "2020-12-11",
}

PERIOD_COLORS = {
    "Pre-COVID": "#2196f3",
    "Crisis": "#f44336",
    "Post-COVID": "#4caf50",
}

SECTOR_COLORS = {
    "Tech": "#42a5f5",
    "Financials": "#ab47bc",
    "Healthcare": "#66bb6a",
    "Energy": "#ffa726",
    "Commodities": "#a1887f",
    "Crypto": "#ffee58",
    "Intl ETFs": "#26c6da",
}

TICKER_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
]

Q_SPECTRUM = np.concatenate([np.linspace(-10, -0.5, 40), np.linspace(0.5, 10, 40)])
Q_SCALING = np.array([-10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10], dtype=float)
SCALE_MIN = 5
SCALE_MAX = 100
N_SCALES = 20
POLY_ORD = 1
OVERLAP = 0.5


@dataclass(frozen=True)
class CovidTabConfig:
    sectors: tuple[str, ...]
    full_start: str
    full_end: str
    window_td: int
    step_td: int
    window_td_p6: int
    step_td_p6: int
    roll_w_p6: int


def get_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 2:
        return np.array([], dtype=float)
    return np.diff(np.log(prices))


def integrate(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x - np.mean(x))


def overlap_starts(length: int, scale: int, overlap: float) -> np.ndarray:
    step = max(1, int(scale * (1.0 - overlap)))
    return np.arange(0, length - scale + 1, step)


def mfdfa(
    x: np.ndarray,
    q_vals: np.ndarray,
    *,
    scale_min: int = SCALE_MIN,
    scale_max: int = SCALE_MAX,
    n_scales: int = N_SCALES,
    poly_ord: int = POLY_ORD,
    overlap: float = OVERLAP,
) -> tuple[np.ndarray | None, float | None, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    q_vals = np.asarray(q_vals, dtype=float)
    q_vals = q_vals[q_vals != 0]

    if len(x) < scale_min * 4:
        return None, None, None

    y = integrate(x)
    length = len(y)
    scale_max = min(scale_max, length // 4)
    scale_min = max(scale_min, poly_ord + 2)
    if scale_max <= scale_min:
        return None, None, None

    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales).astype(int))
    scales = scales[scales >= scale_min]
    if len(scales) == 0:
        return None, None, None

    fqs = np.full((len(q_vals), len(scales)), np.nan)
    for idx, scale in enumerate(scales):
        starts = overlap_starts(length, scale, overlap)
        if len(starts) < 3:
            continue
        t = np.arange(scale)
        rms_values: list[float] = []
        for start in starts:
            seg = y[start : start + scale]
            coeff = np.polyfit(t, seg, poly_ord)
            rms = np.sqrt(np.mean((seg - np.polyval(coeff, t)) ** 2))
            if rms > 0:
                rms_values.append(rms)
        rms_array = np.asarray(rms_values, dtype=float)
        if len(rms_array) == 0:
            continue
        for q_idx, q in enumerate(q_vals):
            val = np.mean(rms_array**q)
            if val > 0:
                fqs[q_idx, idx] = val ** (1.0 / q)

    hq = np.full(len(q_vals), np.nan)
    for q_idx, row in enumerate(fqs):
        valid = np.isfinite(row) & (row > 0)
        if valid.sum() >= 4:
            hq[q_idx] = np.polyfit(np.log(scales[valid]), np.log(row[valid]), 1)[0]

    valid_h = np.isfinite(hq)
    if valid_h.sum() < 4:
        return hq, None, None

    q_valid = q_vals[valid_h]
    hq_valid = hq[valid_h]
    tau_q = q_valid * hq_valid - 1.0

    window_len = min(7, len(tau_q) if len(tau_q) % 2 == 1 else len(tau_q) - 1)
    tau_smooth = savgol_filter(tau_q, window_len, 2) if len(tau_q) >= 5 else tau_q
    alpha = np.gradient(tau_smooth, q_valid)
    f_alpha = q_valid * alpha - tau_smooth

    valid_spec = np.isfinite(alpha) & np.isfinite(f_alpha)
    if valid_spec.sum() < 2:
        return hq, None, None

    da = float(np.max(alpha[valid_spec]) - np.min(alpha[valid_spec]))
    return hq, da, (q_valid[valid_spec], alpha[valid_spec], f_alpha[valid_spec])


def nearest_td_pos(trading_index: pd.DatetimeIndex, date_str: str) -> int:
    pos = trading_index.searchsorted(pd.Timestamp(date_str))
    return int(min(pos, len(trading_index) - 1))


def period_td_mask(trading_index: pd.DatetimeIndex, start_str: str, end_str: str) -> np.ndarray:
    start_idx = nearest_td_pos(trading_index, start_str)
    end_idx = nearest_td_pos(trading_index, end_str)
    mask = np.zeros(len(trading_index), dtype=bool)
    mask[start_idx : end_idx + 1] = True
    return mask


def style_ax(ax: Any) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.spines[["top", "right"]].set_visible(True)


def add_covid_lines(ax: Any) -> None:
    crisis_start = pd.Timestamp(DEFAULT_PERIODS["Crisis"][0])
    crisis_end = pd.Timestamp(DEFAULT_PERIODS["Crisis"][1])
    for _, date in COVID_EVENTS.items():
        ax.axvline(pd.Timestamp(date), color="red", lw=1.2, ls="--", alpha=0.6)
    ax.axvspan(crisis_start, crisis_end, alpha=0.10, color="red")


def smooth_savgol(values: np.ndarray, window_len: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 5:
        return arr
    idx = np.arange(len(arr))
    filled = np.interp(idx, idx[finite], arr[finite])
    max_window = len(arr) if len(arr) % 2 == 1 else len(arr) - 1
    actual = min(window_len, max_window)
    actual = max(actual, 5)
    if actual % 2 == 0:
        actual -= 1
    return savgol_filter(filled, actual, 2)


def cross_sectional_dispersion(sub_matrix: np.ndarray) -> np.ndarray:
    return np.nanstd(sub_matrix, axis=0)


def synchrony_score(sub_matrix: np.ndarray) -> np.ndarray:
    diff = np.diff(sub_matrix, axis=1)
    scores = np.full(diff.shape[1], np.nan)
    for t_idx in range(diff.shape[1]):
        col = diff[:, t_idx]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            continue
        median_dir = np.sign(np.median(finite))
        if median_dir == 0:
            scores[t_idx] = 0.5
            continue
        scores[t_idx] = np.sum(np.sign(finite) == median_dir) / len(finite)
    return scores


def avg_pairwise_corr_local(sub_matrix: np.ndarray, window_start: int, window_end: int) -> float:
    window_data = sub_matrix[:, window_start:window_end]
    values: list[float] = []
    for i in range(window_data.shape[0]):
        for j in range(i + 1, window_data.shape[0]):
            xi = window_data[i]
            xj = window_data[j]
            both = np.isfinite(xi) & np.isfinite(xj)
            if both.sum() < 3:
                continue
            xi_local = xi[both]
            xj_local = xj[both]
            xi_local = (xi_local - xi_local.mean()) / (xi_local.std() + 1e-9)
            xj_local = (xj_local - xj_local.mean()) / (xj_local.std() + 1e-9)
            corr, _ = pearsonr(xi_local, xj_local)
            if np.isfinite(corr):
                values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def _all_selected_tickers(sectors: tuple[str, ...]) -> list[str]:
    tickers: list[str] = []
    for sector in sectors:
        tickers.extend(SECTORS[sector])
    return tickers


@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple[str, ...], full_start: str, full_end: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=full_start, end=full_end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.dropna(how="all")


@st.cache_data(show_spinner=False)
def run_analysis(config: CovidTabConfig) -> dict[str, Any]:
    tickers = tuple(_all_selected_tickers(config.sectors))
    prices_df = download_prices(tickers, config.full_start, config.full_end)
    if prices_df.empty:
        raise ValueError("No price data was returned from yfinance for the selected configuration.")

    prices_df = prices_df[[col for col in prices_df.columns if col in tickers]]
    log_ret_df = np.log(prices_df / prices_df.shift(1)).dropna(how="all")
    period_names = tuple(DEFAULT_PERIODS.keys())

    period_bars: dict[str, dict[str, list[float]]] = {}
    rolling_da: dict[str, dict[str, dict[str, Any]]] = {}
    spectrum_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    heatmaps: dict[str, dict[str, Any]] = {}
    recovery_data: dict[str, dict[str, dict[str, Any]]] = {}

    for sector in config.sectors:
        sector_tickers = [ticker for ticker in SECTORS[sector] if ticker in prices_df.columns]
        if not sector_tickers:
            continue

        period_bars[sector] = {name: [] for name in period_names}
        rolling_da[sector] = {}
        spectrum_data[sector] = {}
        recovery_data[sector] = {}
        da_mat = np.full((len(sector_tickers), len(period_names)), np.nan)

        for ticker_idx, ticker in enumerate(sector_tickers):
            col = prices_df[ticker].dropna()
            col_vals = col.values.astype(float)
            col_idx = col.index

            for period_idx, (period_name, (start, end)) in enumerate(DEFAULT_PERIODS.items()):
                mask = period_td_mask(col_idx, start, end)
                ret = get_returns(col_vals[mask])
                if len(ret) < 35:
                    period_bars[sector][period_name].append(0.0)
                    continue
                _, da, spec = mfdfa(ret, Q_SCALING)
                da_value = float(da) if da is not None and np.isfinite(da) else 0.0
                period_bars[sector][period_name].append(da_value)
                da_mat[ticker_idx, period_idx] = da_value if da_value > 0 else np.nan

                if spec is not None:
                    _, alpha_values, f_values = spec
                    period_bucket = spectrum_data[sector].setdefault(period_name, {"alpha": [], "f": []})
                    period_bucket["alpha"].append(alpha_values)
                    period_bucket["f"].append(f_values)

            roll_dates: list[pd.Timestamp] = []
            roll_values: list[float] = []
            for start_idx in range(0, len(col_vals) - config.window_td, config.step_td):
                end_idx = start_idx + config.window_td
                ret = get_returns(col_vals[start_idx:end_idx])
                if len(ret) < 40:
                    continue
                _, da, _ = mfdfa(ret, Q_SCALING)
                if da is not None and np.isfinite(da):
                    roll_dates.append(col_idx[end_idx - 1])
                    roll_values.append(float(da))
            rolling_da[sector][ticker] = {"dates": roll_dates, "values": roll_values}

            pre_mask = period_td_mask(col_idx, DEFAULT_PERIODS["Pre-COVID"][0], DEFAULT_PERIODS["Pre-COVID"][1])
            pre_ret = get_returns(col_vals[pre_mask])
            if len(pre_ret) >= 40:
                _, pre_da, _ = mfdfa(pre_ret, Q_SCALING)
            else:
                pre_da = None
            if pre_da is not None and np.isfinite(pre_da):
                crisis_pos = nearest_td_pos(col_idx, "2020-02-20")
                diff_dates: list[pd.Timestamp] = []
                diff_values: list[float] = []
                for start_idx in range(crisis_pos, len(col_vals) - config.window_td, config.step_td):
                    end_idx = start_idx + config.window_td
                    ret = get_returns(col_vals[start_idx:end_idx])
                    if len(ret) < 40:
                        continue
                    _, da, _ = mfdfa(ret, Q_SCALING)
                    if da is not None and np.isfinite(da):
                        diff_dates.append(col_idx[end_idx - 1])
                        diff_values.append(float(da - pre_da))
                recovery_data[sector][ticker] = {"dates": diff_dates, "values": diff_values}

        heatmaps[sector] = {"tickers": sector_tickers, "matrix": da_mat}

        for period_name, period_bucket in list(spectrum_data[sector].items()):
            all_alpha = period_bucket["alpha"]
            all_f = period_bucket["f"]
            if not all_alpha:
                continue
            alpha_min = max(arr.min() for arr in all_alpha)
            alpha_max = min(arr.max() for arr in all_alpha)
            if alpha_max <= alpha_min:
                spectrum_data[sector][period_name] = {"alpha_grid": np.array([]), "avg_f": np.array([]), "min_f": np.array([]), "max_f": np.array([])}
                continue
            alpha_grid = np.linspace(alpha_min, alpha_max, 200)
            interpolated: list[np.ndarray] = []
            for alpha_values, f_values in zip(all_alpha, all_f):
                order = np.argsort(alpha_values)
                interpolated.append(np.interp(alpha_grid, alpha_values[order], f_values[order]))
            stack = np.vstack(interpolated)
            spectrum_data[sector][period_name] = {
                "alpha_grid": alpha_grid,
                "avg_f": np.mean(stack, axis=0),
                "min_f": np.min(stack, axis=0),
                "max_f": np.max(stack, axis=0),
            }

    available_tickers = [ticker for ticker in prices_df.columns if ticker in log_ret_df.columns]
    lr_idx = log_ret_df.index
    window_ends_p6 = list(range(config.window_td_p6, len(lr_idx), config.step_td_p6))
    roll_dates_p6 = np.array([lr_idx[i] for i in window_ends_p6], dtype="datetime64[ns]")

    da_matrix_dict: dict[str, np.ndarray] = {}
    for ticker in available_tickers:
        series = log_ret_df[ticker].values
        da_values: list[float] = []
        for window_end in window_ends_p6:
            seg = series[window_end - config.window_td_p6 : window_end]
            seg = seg[np.isfinite(seg)]
            if len(seg) < 32:
                da_values.append(np.nan)
                continue
            _, da, _ = mfdfa(seg, Q_SCALING, scale_max=min(SCALE_MAX, config.window_td_p6 // 4))
            da_values.append(float(da) if da is not None and np.isfinite(da) else np.nan)
        da_matrix_dict[ticker] = np.array(da_values, dtype=float)

    da_matrix_all = np.array([da_matrix_dict[t] for t in available_tickers], dtype=float) if available_tickers else np.empty((0, 0))
    sector_roll: dict[str, dict[str, Any]] = {}

    for sector in config.sectors:
        sector_tickers = [ticker for ticker in SECTORS[sector] if ticker in da_matrix_dict]
        if len(sector_tickers) < 2 or da_matrix_all.size == 0:
            continue
        index_list = [available_tickers.index(ticker) for ticker in sector_tickers]
        sub_mat = da_matrix_all[index_list]
        corr_dates: list[pd.Timestamp] = []
        corr_values: list[float] = []
        for window_end in range(config.roll_w_p6, len(roll_dates_p6)):
            window_start = window_end - config.roll_w_p6
            corr_dates.append(pd.Timestamp(roll_dates_p6[window_end]))
            corr_values.append(avg_pairwise_corr_local(sub_mat, window_start, window_end))
        sector_roll[sector] = {
            "tickers": sector_tickers,
            "n_pairs": len(sector_tickers) * (len(sector_tickers) - 1) // 2,
            "dates": pd.DatetimeIndex(corr_dates),
            "corr": np.array(corr_values, dtype=float),
            "all_dates": pd.DatetimeIndex(pd.to_datetime(roll_dates_p6)),
            "disp": cross_sectional_dispersion(sub_mat),
            "sync_dates": pd.DatetimeIndex(pd.to_datetime(roll_dates_p6[1:])),
            "sync": synchrony_score(sub_mat),
        }

    return {
        "prices_df": prices_df,
        "period_bars": period_bars,
        "rolling_da": rolling_da,
        "spectrum_data": spectrum_data,
        "heatmaps": heatmaps,
        "recovery_data": recovery_data,
        "sector_roll": sector_roll,
        "window_td": config.window_td,
        "step_td": config.step_td,
        "window_td_p6": config.window_td_p6,
        "step_td_p6": config.step_td_p6,
        "roll_w_p6": config.roll_w_p6,
    }


def _subplots_grid(num_rows: int, figsize: tuple[float, float]) -> tuple[Any, np.ndarray]:
    fig, axes = plt.subplots(num_rows, 2, figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes).reshape(num_rows, 2).flatten()
    return fig, axes


def build_period_bars_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    rows = int(np.ceil(len(sectors) / 2))
    fig, axes = _subplots_grid(rows, (14, 5 * rows))
    for ax_idx, sector in enumerate(sectors):
        ax = axes[ax_idx]
        tickers = analysis["heatmaps"][sector]["tickers"]
        x = np.arange(len(tickers))
        width = 0.25
        offsets = [-width, 0.0, width]
        for period_name, offset in zip(DEFAULT_PERIODS.keys(), offsets):
            values = analysis["period_bars"][sector][period_name]
            ax.bar(
                x + offset,
                values,
                width * 0.88,
                label=period_name,
                color=PERIOD_COLORS[period_name],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(sector, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, fontsize=8, rotation=20, ha="right")
        ax.set_xlabel("Ticker", fontsize=9)
        ax.set_ylabel(r"Multifractal Width $\Delta\alpha$", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.axhline(0, color="grey", lw=0.7, ls="--", alpha=0.5)
        style_ax(ax)
    for ax in axes[len(sectors) :]:
        ax.set_visible(False)
    fig.suptitle(
        r"Multifractal Width $\Delta\alpha$ by Ticker — Pre-COVID vs Crisis vs Post-COVID",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def build_rolling_da_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    fig, axes = plt.subplots(len(sectors), 1, figsize=(16, 4 * len(sectors)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax_idx, sector in enumerate(sectors):
        ax = axes[ax_idx]
        for ticker_idx, (ticker, series) in enumerate(analysis["rolling_da"][sector].items()):
            if len(series["dates"]) < 3:
                continue
            ax.plot(
                series["dates"],
                series["values"],
                "-o",
                markersize=2.5,
                linewidth=1.6,
                color=TICKER_COLORS[ticker_idx % len(TICKER_COLORS)],
                label=ticker,
            )
        add_covid_lines(ax)
        y_min, y_max = ax.get_ylim()
        for label, date in COVID_EVENTS.items():
            ax.text(pd.Timestamp(date), y_max, label, fontsize=6.5, color="red", ha="center", va="bottom")
        ax.set_ylabel(r"Multifractal Width $\Delta\alpha$", fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_title(sector, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, ncol=max(1, len(analysis["rolling_da"][sector])), loc="upper left", framealpha=0.85)
        style_ax(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30)
    fig.suptitle(
        "Time-Varying Multifractal Strength Around COVID-19\n"
        f"Rolling Window = {analysis['window_td']} trading days  |  "
        f"Step = {analysis['step_td']} trading days  |  Shaded = Acute Crisis",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    return fig


def build_spectrum_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    rows = int(np.ceil(len(sectors) / 2))
    fig, axes = _subplots_grid(rows, (14, 5 * rows))
    for ax_idx, sector in enumerate(sectors):
        ax = axes[ax_idx]
        for period_name, payload in analysis["spectrum_data"][sector].items():
            alpha_grid = payload["alpha_grid"]
            if len(alpha_grid) == 0:
                continue
            ax.plot(alpha_grid, payload["avg_f"], "-", linewidth=2.2, color=PERIOD_COLORS[period_name], label=period_name)
            ax.fill_between(alpha_grid, payload["min_f"], payload["max_f"], color=PERIOD_COLORS[period_name], alpha=0.12)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(sector, fontsize=12, fontweight="bold")
        ax.set_xlabel(r"Singularity Exponent $\alpha$  (width = multifractal complexity)", fontsize=9)
        ax.set_ylabel(r"Singularity Spectrum $f(\alpha)$  (fractal dimension)", fontsize=9)
        ax.legend(fontsize=8)
        style_ax(ax)
    for ax in axes[len(sectors) :]:
        ax.set_visible(False)
    fig.suptitle(
        r"Multifractal Spectrum $f(\alpha)$ by Sector — COVID Periods"
        "\nDense $q \in [-10, 10]$, shaded band = ticker range within sector",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def build_heatmap_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    rows = int(np.ceil(len(sectors) / 2))
    fig, axes = _subplots_grid(rows, (13, 4 * rows))
    for ax_idx, sector in enumerate(sectors):
        ax = axes[ax_idx]
        da_mat = analysis["heatmaps"][sector]["matrix"]
        tickers = analysis["heatmaps"][sector]["tickers"]
        if np.all(np.isnan(da_mat)):
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(da_mat))
            vmax = float(np.nanmax(da_mat))
        image = ax.imshow(da_mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(DEFAULT_PERIODS)))
        ax.set_xticklabels(list(DEFAULT_PERIODS.keys()), fontsize=9)
        ax.set_yticks(range(len(tickers)))
        ax.set_yticklabels(tickers, fontsize=9)
        ax.set_xlabel("Market Period", fontsize=9)
        ax.set_ylabel("Ticker", fontsize=9)
        ax.set_title(sector, fontsize=11, fontweight="bold")
        plt.colorbar(image, ax=ax, label=r"Multifractal Width $\Delta\alpha$", shrink=0.8)
        for ticker_idx in range(len(tickers)):
            for period_idx in range(len(DEFAULT_PERIODS)):
                value = da_mat[ticker_idx, period_idx]
                if np.isfinite(value):
                    ax.text(period_idx, ticker_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")
    for ax in axes[len(sectors) :]:
        ax.set_visible(False)
    fig.suptitle(r"$\Delta\alpha$ Heatmap by Sector — Pre-COVID / Crisis / Post-COVID", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def build_recovery_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    fig, axes = plt.subplots(len(sectors), 1, figsize=(16, 4 * len(sectors)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax_idx, sector in enumerate(sectors):
        ax = axes[ax_idx]
        for ticker_idx, (ticker, series) in enumerate(analysis["recovery_data"][sector].items()):
            if len(series["dates"]) < 3:
                continue
            ax.plot(
                series["dates"],
                series["values"],
                "-o",
                markersize=2.5,
                linewidth=1.6,
                color=TICKER_COLORS[ticker_idx % len(TICKER_COLORS)],
                label=ticker,
            )
        ax.axhline(0, color="black", lw=1.5, ls="-", alpha=0.7, label="Pre-COVID baseline")
        ax.axhline(0.05, color="grey", lw=0.8, ls=":", alpha=0.4)
        ax.axhline(-0.05, color="grey", lw=0.8, ls=":", alpha=0.4)
        add_covid_lines(ax)
        y_min, y_max = ax.get_ylim()
        for label, date in COVID_EVENTS.items():
            ax.text(pd.Timestamp(date), y_max, label, fontsize=6.5, color="red", ha="center", va="bottom")
        ax.set_ylabel(r"$\Delta\alpha_{roll} - \Delta\alpha_{pre}$" + "\n(deviation from pre-COVID baseline)", fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_title(sector, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, ncol=max(1, len(analysis["recovery_data"][sector])), loc="upper right", framealpha=0.85)
        style_ax(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=30)
    fig.suptitle(
        "Recovery of Market Efficiency After COVID-19 by Sector\n"
        "Deviation of Rolling Da from Pre-COVID Baseline  |  "
        f"Window = {analysis['window_td']} trading days  |  "
        f"Step = {analysis['step_td']} trading days  |  "
        "Shaded = Acute Crisis",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    return fig


def build_corr_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    sector_roll = {sector: analysis["sector_roll"][sector] for sector in sectors if sector in analysis["sector_roll"]}
    fig, axes = plt.subplots(len(sector_roll), 1, figsize=(16, 4.2 * max(1, len(sector_roll))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax_idx, (sector, payload) in enumerate(sector_roll.items()):
        ax = axes[ax_idx]
        color = SECTOR_COLORS.get(sector, "#888888")
        corr = payload["corr"]
        corr_smooth = smooth_savgol(corr, 9)
        ax.fill_between(payload["dates"], corr, alpha=0.12, color=color)
        ax.plot(payload["dates"], corr, color=color, lw=1.1, alpha=0.45, label="Raw")
        ax.plot(payload["dates"], corr_smooth, color=color, lw=2.2, alpha=0.95, label="Smoothed")
        if np.any(np.isfinite(corr_smooth)):
            peak_idx = int(np.nanargmax(corr_smooth))
            peak_val = corr_smooth[peak_idx]
            ax.scatter([payload["dates"][peak_idx]], [peak_val], color="black", s=60, zorder=6)
        ax.set_title(
            f"{sector}  —  {len(payload['tickers'])} tickers  ({payload['n_pairs']} pairs)  |  " + ", ".join(payload["tickers"]),
            fontsize=11,
            fontweight="bold",
            color=color,
            pad=5,
        )
        ax.set_ylabel("Mean Pairwise\nPearson r(Da)\n[-1 = anti-correlated, +1 = co-moving]", fontsize=8)
        ax.set_xlabel("Date", fontsize=9)
        ax.axhline(0, color="black", lw=0.5, ls="--", alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.8, loc="upper left", ncol=2)
        ax.tick_params(labelsize=8)
        style_ax(ax)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30)
    fig.suptitle(
        "Rolling Mean Pairwise Pearson Correlation of Multifractal Strength (Da) — By Sector\n"
        f"Da Window = {analysis['window_td_p6']} trading days  |  "
        f"Corr Window = {analysis['roll_w_p6']} steps x {analysis['step_td_p6']} trading days  |  "
        "LOCAL z-score (COVID spike preserved)",
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )
    fig.tight_layout(h_pad=1.8)
    return fig


def build_dashboard_figure(analysis: dict[str, Any], sectors: tuple[str, ...]) -> plt.Figure:
    sector_roll = [sector for sector in sectors if sector in analysis["sector_roll"]]
    fig, axes = plt.subplots(len(sector_roll), 3, figsize=(22, 3.8 * max(1, len(sector_roll))), sharex=True)
    axes = np.atleast_2d(axes)
    column_titles = [
        "(a) Mean Pairwise Corr of Da\n(high = co-moving fractal dynamics)",
        "(b) Cross-Sectional Dispersion of Da\n(low = assets clustered = co-moving)",
        "(c) Synchrony Score\n(>0.5 = majority moving same direction)",
    ]
    for ax_idx, sector in enumerate(sector_roll):
        payload = analysis["sector_roll"][sector]
        color = SECTOR_COLORS.get(sector, "#888888")

        ax_corr = axes[ax_idx, 0]
        corr_smooth = smooth_savgol(payload["corr"], 9)
        ax_corr.fill_between(payload["dates"], payload["corr"], alpha=0.15, color=color)
        ax_corr.plot(payload["dates"], payload["corr"], color=color, lw=1.0, alpha=0.4)
        ax_corr.plot(payload["dates"], corr_smooth, color=color, lw=2.0)
        ax_corr.axhline(0, color="black", lw=0.6, ls="--", alpha=0.3)
        add_covid_lines(ax_corr)
        ax_corr.set_ylabel(f"{sector}\n\nPearson r(Da)\n[-1 to +1]", fontsize=8, fontweight="bold", color=color)
        ax_corr.set_xlabel("Date", fontsize=8)
        ax_corr.tick_params(labelsize=7)
        style_ax(ax_corr)

        ax_disp = axes[ax_idx, 1]
        disp = payload["disp"]
        disp_smooth = smooth_savgol(disp, 9)
        ax_disp.fill_between(payload["all_dates"], disp, alpha=0.15, color=color)
        ax_disp.plot(payload["all_dates"], disp, color=color, lw=1.0, alpha=0.4)
        ax_disp.plot(payload["all_dates"], disp_smooth, color=color, lw=2.0)
        add_covid_lines(ax_disp)
        mean_disp = np.nanmean(disp)
        std_disp = np.nanstd(disp)
        ax_disp.axhline(mean_disp, color="grey", lw=0.8, ls="--", alpha=0.5)
        ax_disp.axhspan(mean_disp - std_disp, mean_disp + std_disp, color="grey", alpha=0.07)
        ax_disp.set_ylabel(r"Std of $\Delta\alpha$ across tickers" + "\n[low = clustered = co-moving]", fontsize=8)
        ax_disp.set_xlabel("Date", fontsize=8)
        ax_disp.tick_params(labelsize=7)
        style_ax(ax_disp)

        ax_sync = axes[ax_idx, 2]
        sync = payload["sync"]
        sync_smooth = smooth_savgol(sync, 9)
        ax_sync.fill_between(payload["sync_dates"], 0.5, sync, where=sync >= 0.5, alpha=0.20, color="green", label="Majority co-move up")
        ax_sync.fill_between(payload["sync_dates"], sync, 0.5, where=sync < 0.5, alpha=0.20, color="red", label="Majority diverge")
        ax_sync.plot(payload["sync_dates"], sync, color=color, lw=1.0, alpha=0.4)
        ax_sync.plot(payload["sync_dates"], sync_smooth, color=color, lw=2.0)
        ax_sync.axhline(0.5, color="black", lw=1.0, ls="--", alpha=0.5, label="Random (0.5)")
        add_covid_lines(ax_sync)
        ax_sync.set_ylim(0, 1)
        ax_sync.set_ylabel("Fraction of tickers\nmoving same direction\n[0.5 = random, 1 = perfect sync]", fontsize=8)
        ax_sync.set_xlabel("Date", fontsize=8)
        ax_sync.legend(fontsize=6, loc="lower right", framealpha=0.7)
        ax_sync.tick_params(labelsize=7)
        style_ax(ax_sync)

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=6)

    for row in range(len(sector_roll)):
        for col in range(3):
            axes[row, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            axes[row, col].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(axes[row, col].xaxis.get_majorticklabels(), rotation=30)

    fig.suptitle(
        "Multifractal Comovement Dashboard — Three Complementary Metrics\n"
        f"Da Window = {analysis['window_td_p6']} td  |  "
        f"Corr Window = {analysis['roll_w_p6']}x{analysis['step_td_p6']} td  |  "
        "Red band = COVID acute crisis",
        fontsize=13,
        fontweight="bold",
        y=1.003,
    )
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    return fig


def render_plot_section(title: str, caption: str, figure: plt.Figure) -> None:
    st.subheader(title)
    st.caption(caption)
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def render() -> None:
    st.markdown(
        """
        This tab packages the entire COVID sector analysis as one importable Python module.
        It downloads data, computes the multifractal metrics, and renders the figures directly
        in Streamlit instead of executing notebook cells or loading saved plot folders.
        """
    )

    with st.form("covid-sector-controls"):
        selected_sectors = st.multiselect(
            "Sectors",
            options=list(SECTORS.keys()),
            default=list(SECTORS.keys()),
        )
        col1, col2 = st.columns(2)
        with col1:
            full_start = st.date_input("Download start", value=pd.Timestamp("2017-01-01"))
            window_td = st.number_input("Plots 1-5 window (trading days)", min_value=63, max_value=504, value=252, step=21)
            window_td_p6 = st.number_input("Plots 6-7 window (trading days)", min_value=42, max_value=252, value=126, step=21)
        with col2:
            full_end = st.date_input("Download end", value=pd.Timestamp("2023-01-01"))
            step_td = st.number_input("Plots 1-5 step (trading days)", min_value=5, max_value=63, value=21, step=1)
            step_td_p6 = st.number_input("Plots 6-7 step (trading days)", min_value=5, max_value=30, value=10, step=1)
        roll_w_p6 = st.number_input("Plots 6-7 correlation window (steps)", min_value=4, max_value=24, value=12, step=1)
        submitted = st.form_submit_button("Run COVID Sector Analysis", type="primary")

    if not selected_sectors:
        st.info("Select at least one sector to render the tab.")
        return

    config = CovidTabConfig(
        sectors=tuple(selected_sectors),
        full_start=str(pd.Timestamp(full_start).date()),
        full_end=str(pd.Timestamp(full_end).date()),
        window_td=int(window_td),
        step_td=int(step_td),
        window_td_p6=int(window_td_p6),
        step_td_p6=int(step_td_p6),
        roll_w_p6=int(roll_w_p6),
    )

    if "covid_sector_has_run" not in st.session_state:
        st.session_state["covid_sector_has_run"] = False

    if submitted:
        st.session_state["covid_sector_has_run"] = True

    if not st.session_state["covid_sector_has_run"]:
        st.info("Run the analysis once to populate the tab.")
        return

    with st.spinner("Downloading market data and computing multifractal metrics..."):
        analysis = run_analysis(config)

    available_sectors = tuple(sector for sector in config.sectors if sector in analysis["heatmaps"])
    if not available_sectors:
        st.error("The selected sectors did not return enough data to build the tab.")
        return

    missing_sectors = [sector for sector in config.sectors if sector not in available_sectors]
    if missing_sectors:
        st.warning("Skipped sectors with insufficient downloaded data: " + ", ".join(missing_sectors))

    st.metric("Tickers loaded", value=len(analysis["prices_df"].columns))
    st.caption(
        "Plot 6 keeps the notebook's local z-score fix so the COVID co-movement spike is not normalized away."
    )

    render_plot_section(
        "1. Period Comparison Bar Chart",
        "Per-ticker multifractal width across pre-COVID, crisis, and post-COVID periods.",
        build_period_bars_figure(analysis, available_sectors),
    )
    render_plot_section(
        "2. Rolling Delta-Alpha",
        "Rolling multifractal strength for each ticker, grouped by sector.",
        build_rolling_da_figure(analysis, available_sectors),
    )
    render_plot_section(
        "3. f(alpha) Spectrum",
        "Sector-level average singularity spectrum for each COVID regime.",
        build_spectrum_figure(analysis, available_sectors),
    )
    render_plot_section(
        "4. Delta-Alpha Heatmap",
        "Sector heatmaps for the three analysis periods.",
        build_heatmap_figure(analysis, available_sectors),
    )
    render_plot_section(
        "5. Recovery Analysis",
        "Deviation from the pre-COVID multifractal baseline after the shock.",
        build_recovery_figure(analysis, available_sectors),
    )

    if analysis["sector_roll"]:
        render_plot_section(
            "6. Rolling Mean Pairwise Correlation",
            "Fixed local-window correlation view of rolling multifractal strength.",
            build_corr_figure(analysis, available_sectors),
        )
        render_plot_section(
            "7. Comovement Dashboard",
            "Correlation, dispersion, and synchrony together distinguish shared shocks from deeper coupling.",
            build_dashboard_figure(analysis, available_sectors),
        )
    else:
        st.warning("Not enough valid tickers were available to build the sector comovement plots.")


TAB_SPEC = TabSpec(
    slug="covid-sector",
    title="COVID Sector Analysis",
    description="Self-contained multifractal COVID analysis tab rendered directly in Streamlit.",
    render=render,
)
