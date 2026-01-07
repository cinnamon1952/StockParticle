# Takens (Delay) Embedding Phase Space for Stock Prices / Log Returns (Python)

This project builds a **phase-space reconstruction** using **Takens' delay embedding** from a 1D stock time series:

- **Prices** \(p_t\)
- **Log returns** \(r_t = \log(p_t) - \log(p_{t-1})\)

It also includes simple, standard heuristics to choose:

- delay **\(\tau\)** via **AMI** (average mutual information) or **autocorrelation**
- embedding dimension **\(m\)** via **false nearest neighbors (FNN)**

## Setup (pyenv)

This repo is already configured to use a local `pyenv` virtualenv named `takens-stock` (see `.python-version`).

From the project folder:

```bash
pyenv local takens-stock
python --version
pip install -r requirements.txt
```

## Your CSV format

You need a CSV with at least a price column (default name: `Close`).

Example:

```text
Date,Close
2024-01-02,101.23
2024-01-03,100.80
...
```

If your columns differ, pass `--price-col` and optionally `--date-col`.

## Run: phase space (Takens embedding)

### Pull data from yfinance (no CSV needed)

```bash
python scripts/phase_space.py --ticker AAPL --start 2020-01-01 --interval 1d --series logret --tau-method ami --dims 3 --out outputs/aapl_logret.png
python scripts/phase_space.py --ticker AAPL --start 2020-01-01 --interval 1d --series price  --tau-method autocorr --dims 3 --out outputs/aapl_price.png
```

### Compare Apple vs Amazon with the SAME τ and m (apples-to-apples)

This makes the reconstruction parameters identical (same date range/interval, same standardization, shared τ and m).

```bash
python scripts/compare_phase_space.py --ticker1 AAPL --ticker2 AMZN --start 2024-01-01 --interval 1d --series logret --tau-method ami --dims 3 --out outputs/aapl_vs_amzn_logret.png
```

### Log returns (recommended for stocks)

```bash
python scripts/phase_space.py --csv path/to/your.csv --series logret --tau-method ami --dims 3
```

### Prices

```bash
python scripts/phase_space.py --csv path/to/your.csv --series price --tau-method autocorr --dims 3
```

### Save plot to file

```bash
python scripts/phase_space.py --csv path/to/your.csv --series logret --out outputs/embedding.png
```

### Override \(\tau\) and \(m\) manually

```bash
python scripts/phase_space.py --csv path/to/your.csv --series logret --tau 5 --m 6 --dims 3
```

## What “phase space” means here

Given a scalar series \(x_t\), Takens embedding constructs vectors:

\[
X_t = [x_t,\ x_{t+\tau},\ x_{t+2\tau},\ \dots,\ x_{t+(m-1)\tau}]
\]

Plotting \(X_t\) in 2D/3D gives a reconstructed attractor / phase portrait.


