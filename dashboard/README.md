# Plot Dashboard

This is a standalone local dashboard for the plot outputs in `scripts/trial`.

## What It Shows

- All images from `scripts/trial/mf_corr_plots`
- The 3 standalone attached images used for this request
- Curated LPPL-MFDFA highlights from `scripts/trial/lppl_mfdfa_v2_plots`
- Stock-selectable MFDFA plots from `scripts/trial/fractal_structure_results`

## Run

From the repository root:

```bash
python3 dashboard/app.py
```

The server starts at `http://127.0.0.1:8000` by default and opens in your browser automatically.

## Options

```bash
python3 dashboard/app.py --host 127.0.0.1 --port 8000 --no-browser
```

## Notes

- The dashboard reads the existing image files directly; it does not modify any notebook.
- The LPPL section prioritizes the `overview` and `rolling_trajectory` plots for each ticker, while still exposing the residual-surrogate plots.
- The fractal section now shows only the `mfdfa` output for the selected stock.
- Click any plot card to open a larger view with a short explanation.
