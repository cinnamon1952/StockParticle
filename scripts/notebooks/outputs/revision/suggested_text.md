# Suggested replacement text (numbers are filled from this run; check each before pasting)

## Section 2, CL=F handling
Non-positive prices are removed before log returns are computed. For WTI crude (CL=F), the settlement of -37.63 on 2020-04-20 is dropped, so one return (2020-04-17 to 2020-04-21, -0.602) spans two trading days; CL=F has 4090 returns. Dropping that return changes the CL=F width from 0.666 to 0.595; excluding CL=F changes the commodity COVID-deletion change from -0.087 to +0.002 (mean width over all assets 0.548 vs 0.546).

## Section 2, sampling
Equities and ETFs use trading-day returns (4091 returns for 43 of 50 assets); the exceptions are META, TSLA, ABBV, SI=F, CL=F, BTC-USD, ETH-USD. Crypto series are sampled on all seven days of the week (BTC-USD starts 2014-09-18, ETH-USD 2017-11-10, when Yahoo history begins). Data end on 2026-04-10 (the download end date 2026-04-11 is exclusive). The 40 equities cover 8 of the 11 GICS sectors (no Materials, Real Estate, Utilities); counts: Communication Services 3, Consumer Discretionary 2, Consumer Staples 5, Energy 3, Financials 10, Health Care 5, Industrials 7, Information Technology 5.

## Section 3.5, block lengths and the benchmark
Episode lengths in the simulation are trading days: the 2020 window (2020-02-19 to 2020-06-30) contains 93 trading days (132 calendar days) and the 2008-09 window (2008-09-01 to 2009-06-30) 209 (302 calendar days). The mean width of i.i.d. Gaussian series is 0.218 (N = 4,091) and 0.218 (N = 4,600; B = 500), compared with 0.348 for i.i.d. Student-t(4) at N = 4,600 and 0.231 for the phase-randomized surrogates of the observed series. The phase-randomized mean preserves each asset's linear spectrum and is therefore not an i.i.d. Gaussian floor. Block position and length explain the two COVID-length x6 values (+0.178 random position, L = 133; +0.189 fixed late position, L = 132).

## Figure captions
Fig. 1: see tables/fig1_caption.txt.  Fig. 4: Shares are descriptive and depend on the order of the steps: across the six orderings that end at the observed width, the share of the SH step ranges from -2.0% to 68.1% (full sample). Brackets: 95% percentile bootstrap intervals of each share over assets (2,000 resamples); they do not include surrogate Monte Carlo error.
Fig. 5: n = 150 placebo windows; the minimum attainable p-value is 1/151.  Fig. 6: see tables/fig6_caption_numbers.txt.  Fig. 7: see tables/fig7_caption.txt.
Table 1 footnote: see tables/table1_footnote.txt.  Table 3 caption: see tables/scaling_variance_ratio_caption.txt.

## Data and code availability (fill the bracketed items after the upload)
Code, the frozen configuration (SHA-256 prefix 0aaa4807ab), the data manifest (SHA-256 of every cached price file; digest 548bf7e5c7d4e224), seeds (master seed 42) and run metadata are available at [REPOSITORY URL] and archived at Zenodo [DOI]. Prices are Yahoo Finance auto-adjusted closes retrieved with yfinance on the dates recorded in the manifest.
