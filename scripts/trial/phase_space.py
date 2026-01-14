from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

from src.takens import (
    choose_tau_ami,
    choose_tau_autocorr,
    delay_embed,
    false_nearest_neighbors,
    log_returns,
    standardize,
)


def load_prices_csv(path: Path, price_col: str, date_col: Optional[str]) -> pd.Series:
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
    if price_col not in df.columns:
        raise ValueError(f"price column '{price_col}' not found. Columns: {list(df.columns)}")
    s = pd.to_numeric(df[price_col], errors="coerce").dropna()
    if len(s) < 10:
        raise ValueError("Not enough price points after cleaning")
    return s


def load_prices_yfinance(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    interval: str,
    price_col: str,
) -> pd.Series:
    if yf is None:
        raise RuntimeError(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        )
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        raise ValueError("yfinance returned no data (check ticker/date range/interval).")

    # yfinance sometimes returns multi-index columns for multiple tickers; enforce single
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can order levels as (Ticker, Price) or (Price, Ticker) depending on version/options
        if (ticker, price_col) in df.columns:
            s = df[(ticker, price_col)]
        elif (price_col, ticker) in df.columns:
            s = df[(price_col, ticker)]
        else:
            # try to infer by level names
            lvl0 = df.columns.names[0] if len(df.columns.names) > 0 else None
            lvl1 = df.columns.names[1] if len(df.columns.names) > 1 else None
            raise ValueError(
                f"Could not find '{price_col}' for ticker '{ticker}' in yfinance columns "
                f"(levels: {lvl0}, {lvl1}). Columns: {df.columns}"
            )
    else:
        if price_col not in df.columns:
            raise ValueError(f"price column '{price_col}' not found. Columns: {list(df.columns)}")
        s = df[price_col]

    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 10:
        raise ValueError("Not enough price points after cleaning")
    return s


def choose_tau(x: np.ndarray, method: str, max_lag: int) -> int:
    if method == "ami":
        return choose_tau_ami(x, max_lag=max_lag)
    if method == "autocorr":
        return choose_tau_autocorr(x, max_lag=max_lag)
    raise ValueError("tau_method must be one of: ami, autocorr")


def plot_embedding(Y: np.ndarray, title: str, dims: int, out: Optional[Path]) -> None:
    if dims == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(Y[:, 0], Y[:, 1], s=6, alpha=0.35, edgecolors="none")
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+τ)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    elif dims == 3:
        fig = plt.figure(figsize=(7.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=4, alpha=0.25)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("x(t+τ)")
        ax.set_zlabel("x(t+2τ)")
        ax.set_title(title)
    else:
        raise ValueError("dims must be 2 or 3")

    fig.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
    else:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Takens delay embedding phase space for stock prices/log-returns.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to CSV file with prices.")
    src.add_argument("--ticker", type=str, help="Ticker symbol to fetch via yfinance, e.g. AAPL.")
    p.add_argument("--price-col", default="Close", help="Column name containing prices (default: Close).")
    p.add_argument("--date-col", default=None, help="Optional date/time column name for sorting.")
    p.add_argument("--start", default=None, help="yfinance start date, e.g. 2020-01-01 (only with --ticker).")
    p.add_argument("--end", default=None, help="yfinance end date, e.g. 2024-01-01 (only with --ticker).")
    p.add_argument("--interval", default="1d", help="yfinance interval, e.g. 1d, 1h (only with --ticker).")
    p.add_argument(
        "--series",
        choices=["price", "logret"],
        default="logret",
        help="Which time series to embed (default: logret).",
    )
    p.add_argument("--tau-method", choices=["ami", "autocorr"], default="ami")
    p.add_argument("--max-lag", type=int, default=250, help="Max lag to consider for tau choice.")
    p.add_argument("--tau", type=int, default=None, help="Override tau (delay) manually.")
    p.add_argument("--m-max", type=int, default=12, help="Max embedding dimension to consider for FNN.")
    p.add_argument("--m", type=int, default=None, help="Override embedding dimension m manually.")
    p.add_argument("--dims", type=int, choices=[2, 3], default=3, help="Plot 2D or 3D (default: 3).")
    p.add_argument("--no-standardize", action="store_true", help="Do not standardize series before embedding.")
    p.add_argument("--out", type=Path, default=None, help="If set, save plot to this path instead of showing.")
    args = p.parse_args()

    if args.csv is not None:
        prices = load_prices_csv(args.csv, price_col=args.price_col, date_col=args.date_col)
        src_label = args.csv.name
    else:
        prices = load_prices_yfinance(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            interval=args.interval,
            price_col=args.price_col,
        )
        src_label = f"{args.ticker} ({args.interval}, yfinance)"
    if args.series == "price":
        x = prices.to_numpy(dtype=float)
        label = "Price"
    else:
        x = log_returns(prices.to_numpy(dtype=float))
        label = "Log returns"

    if not args.no_standardize:
        x = standardize(x)

    tau = int(args.tau) if args.tau is not None else choose_tau(x, method=args.tau_method, max_lag=args.max_lag)

    if args.m is not None:
        m = int(args.m)
        fnn = None
    else:
        fnn = false_nearest_neighbors(x, tau=tau, m_max=args.m_max)
        m = int(fnn.chosen_m)

    # For plotting 2D/3D, ensure we have enough dimensions
    plot_m = max(m, args.dims)
    Y = delay_embed(x, m=plot_m, tau=tau)

    title = f"Takens embedding ({label}) [{src_label}] | τ={tau}, m={plot_m}"
    if fnn is not None:
        title += f" (FNN-chosen m≈{m})"

    plot_embedding(Y[:, : args.dims], title=title, dims=args.dims, out=args.out)

    if fnn is not None:
        print("FNN fractions (m -> m+1):")
        for m_i, frac in zip(fnn.m_values, fnn.fnn_fraction):
            if np.isfinite(frac):
                print(f"  m={m_i:2d}: {frac:.4f}")
        print(f"Chosen m (heuristic): {fnn.chosen_m}")
    print(f"Chosen τ: {tau}")


if __name__ == "__main__":
    main()


