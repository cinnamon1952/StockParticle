from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf  # type: ignore

from src.takens import (
    choose_tau_ami,
    choose_tau_autocorr,
    delay_embed,
    false_nearest_neighbors,
    log_returns,
    standardize,
)


def load_prices_yfinance(
    ticker: str,
    start: Optional[str],
    end: Optional[str],
    interval: str,
    price_col: str,
) -> np.ndarray:
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
        raise ValueError(f"yfinance returned no data for {ticker}.")

    if hasattr(df, "columns") and df.columns is not None and str(type(df.columns)).endswith("MultiIndex'>"):
        # handle both (Ticker, Price) and (Price, Ticker)
        if (ticker, price_col) in df.columns:
            s = df[(ticker, price_col)]
        elif (price_col, ticker) in df.columns:
            s = df[(price_col, ticker)]
        else:
            raise ValueError(f"Could not find {price_col} for {ticker} in columns: {df.columns}")
    else:
        if price_col not in df.columns:
            raise ValueError(f"price column '{price_col}' not found for {ticker}. Columns: {list(df.columns)}")
        s = df[price_col]

    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 20:
        raise ValueError(f"Not enough price data after cleaning for {ticker}.")
    return s


def choose_tau(x: np.ndarray, method: str, max_lag: int) -> int:
    if method == "ami":
        return choose_tau_ami(x, max_lag=max_lag)
    if method == "autocorr":
        return choose_tau_autocorr(x, max_lag=max_lag)
    raise ValueError("tau_method must be one of: ami, autocorr")


def plot_two(Y1: np.ndarray, Y2: np.ndarray, t1: str, t2: str, dims: int, out: Path) -> None:
    if dims == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
        axes[0].scatter(Y1[:, 0], Y1[:, 1], s=6, alpha=0.35, edgecolors="none")
        axes[0].set_title(t1)
        axes[0].set_xlabel("x(t)")
        axes[0].set_ylabel("x(t+τ)")
        axes[0].grid(True, alpha=0.25)

        axes[1].scatter(Y2[:, 0], Y2[:, 1], s=6, alpha=0.35, edgecolors="none")
        axes[1].set_title(t2)
        axes[1].set_xlabel("x(t)")
        axes[1].set_ylabel("x(t+τ)")
        axes[1].grid(True, alpha=0.25)
    elif dims == 3:
        fig = plt.figure(figsize=(13, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        ax1.scatter(Y1[:, 0], Y1[:, 1], Y1[:, 2], s=4, alpha=0.25)
        ax1.set_title(t1)
        ax1.set_xlabel("x(t)")
        ax1.set_ylabel("x(t+τ)")
        ax1.set_zlabel("x(t+2τ)")

        ax2.scatter(Y2[:, 0], Y2[:, 1], Y2[:, 2], s=4, alpha=0.25)
        ax2.set_title(t2)
        ax2.set_xlabel("x(t)")
        ax2.set_ylabel("x(t+τ)")
        ax2.set_zlabel("x(t+2τ)")
    else:
        raise ValueError("dims must be 2 or 3")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Takens phase-space plots for two tickers with shared τ and m.")
    p.add_argument("--ticker1", required=True, help="First ticker (e.g., AAPL).")
    p.add_argument("--ticker2", required=True, help="Second ticker (e.g., AMZN).")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d")
    p.add_argument("--price-col", default="Close")
    p.add_argument("--series", choices=["price", "logret"], default="logret")
    p.add_argument("--tau-method", choices=["ami", "autocorr"], default="ami")
    p.add_argument("--max-lag", type=int, default=250)
    p.add_argument("--tau", type=int, default=None, help="Force a shared τ for both.")
    p.add_argument("--m", type=int, default=None, help="Force a shared m for both.")
    p.add_argument("--m-max", type=int, default=12)
    p.add_argument("--dims", type=int, choices=[2, 3], default=3)
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs/compare.png"))
    args = p.parse_args()

    p1 = load_prices_yfinance(args.ticker1, args.start, args.end, args.interval, args.price_col)
    p2 = load_prices_yfinance(args.ticker2, args.start, args.end, args.interval, args.price_col)

    if args.series == "price":
        x1, x2 = p1, p2
        label = "Price"
    else:
        x1, x2 = log_returns(p1), log_returns(p2)
        label = "Log returns"

    if not args.no_standardize:
        x1, x2 = standardize(x1), standardize(x2)

    tau = int(args.tau) if args.tau is not None else choose_tau(x1, args.tau_method, args.max_lag)
    if args.m is not None:
        m = int(args.m)
    else:
        m = int(false_nearest_neighbors(x1, tau=tau, m_max=args.m_max).chosen_m)

    plot_m = max(m, args.dims)
    Y1 = delay_embed(x1, m=plot_m, tau=tau)[:, : args.dims]
    Y2 = delay_embed(x2, m=plot_m, tau=tau)[:, : args.dims]

    t1 = f"{args.ticker1} {label} | τ={tau}, m={plot_m}"
    t2 = f"{args.ticker2} {label} | τ={tau}, m={plot_m}"
    plot_two(Y1, Y2, t1, t2, dims=args.dims, out=args.out)
    print(f"Saved: {args.out}")
    print(f"Shared τ={tau}, shared m={plot_m}")


if __name__ == "__main__":
    main()


