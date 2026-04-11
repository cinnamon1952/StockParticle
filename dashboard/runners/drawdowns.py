"""
Drawdown Distribution Analysis — Sornette (2003)
Self-contained runner: generates one figure per market, saves to output_dir.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import IMAGE_SUFFIXES, relative_to_project

MARKETS = {
    "DJIA (USA)":      dict(sigma=0.0063, n=25000),
    "NASDAQ (USA)":    dict(sigma=0.0095, n=8000),
    "FTSE 100 (UK)":   dict(sigma=0.0075, n=10000),
    "CAC 40 (France)": dict(sigma=0.0080, n=8000),
    "DAX (Germany)":   dict(sigma=0.0085, n=8000),
    "MIBTel (Italy)":  dict(sigma=0.0072, n=6000),
}


def simulate_returns(n, sigma, seed=42):
    rng = np.random.default_rng(seed)
    returns, vol = [], sigma
    for _ in range(n):
        z = rng.standard_t(4) / np.sqrt(2)
        r = z * vol
        returns.append(r)
        vol = np.sqrt(1e-6 + 0.05 * r**2 + 0.94 * vol**2)
    return np.array(returns)


def get_drawdowns(returns):
    drawdowns = []
    i = 0
    while i < len(returns):
        if returns[i] < 0:
            depth = 0.0
            while i < len(returns) and returns[i] < 0:
                depth += returns[i]
                i += 1
            if abs(depth) > 0.001:
                drawdowns.append(abs(depth))
        else:
            i += 1
    return np.array(drawdowns)


def generate(output_dir: str | Path, **kwargs) -> dict:
    output_dir = Path(output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    before = _scan(output_dir)

    for idx, (name, cfg) in enumerate(MARKETS.items()):
        returns = simulate_returns(cfg["n"], cfg["sigma"], seed=idx)
        drawdowns = np.sort(get_drawdowns(returns))[::-1]
        n = len(drawdowns)
        log_cum = np.log10(np.arange(1, n + 1))

        lam = 1.0 / np.mean(drawdowns)
        x_line = np.linspace(0, drawdowns.max() * 1.05, 300)
        y_line = np.log10(n) - lam * x_line / np.log(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_line, y_line, color="black", lw=1.2, label="Null Hypothesis")
        ax.scatter(drawdowns, log_cum, s=8, color="black", marker=".", label="Draw Down")
        ax.invert_xaxis()
        ax.set_title(f"FIG. Drawdown Distribution — {name}", fontsize=10)
        ax.set_xlabel("Draw Down", fontsize=9)
        ax.set_ylabel("Log(Cumulative Number)", fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(False)
        fig.tight_layout()

        fname = output_dir / f"drawdown_{name.split()[0].lower()}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches="tight", facecolor="white")
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
