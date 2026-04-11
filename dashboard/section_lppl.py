from __future__ import annotations

from typing import Callable

from notebook_sources import LPPL_NOTEBOOK_SOURCE, source_manifest_entry
from plot_helpers import image_files, natural_sort_key, pretty_label, relative_to_project, TRIAL_DIR


DEFAULT_LPPL_CONFIG = {
    "tickers": ["^GSPC", "JPM", "XLF", "C", "GE"],
    "start_data": "2007-01-01",
    "end_data": "2009-06-01",
    "target_date": "2008-09-28",
    "window_start_offsets": [120, 100, 80, 60],
    "roll_days": 60,
    "roll_step": 5,
    "plot_duration": 150,
    "surrogate_n": 100,
}


def explain_lppl_plot(ticker: str, suffix: str) -> str:
    if suffix == "overview":
        return f"This is the main LPPL summary for `{ticker}` because it compares the fitted windows directly around the analysis date."
    if suffix == "rolling_trajectory":
        return f"This shows how the LPPL plus multifractal signal evolves as the analysis pivot rolls toward the target date."
    if suffix.endswith("residuals_surrogate"):
        window = suffix.split("_", 1)[0].upper()
        return f"This {window} residual-surrogate diagnostic checks whether the multifractal residual structure still looks stronger than a phase-randomized baseline."
    return f"This LPPL-MFDFA figure is part of the generated diagnostic set for `{ticker}`."


def build_lppl_section(register_asset: Callable[..., dict[str, str]]) -> dict[str, object]:
    lppl_dir = TRIAL_DIR / "lppl_mfdfa_v2_plots"
    grouped: dict[str, dict[str, object]] = {}

    for path in image_files(lppl_dir):
        stem = path.stem
        if "_" not in stem:
            continue
        ticker, suffix = stem.split("_", 1)
        entry = grouped.setdefault(ticker, {"highlights": [], "residuals": []})

        if suffix in {"overview", "rolling_trajectory"}:
            entry["highlights"].append(
                register_asset(
                    path,
                    title=f"{ticker} {pretty_label(suffix)}",
                    caption=relative_to_project(path),
                    explanation=explain_lppl_plot(ticker, suffix),
                )
            )
        elif suffix.endswith("residuals_surrogate"):
            window = suffix.split("_", 1)[0]
            entry["residuals"].append(
                register_asset(
                    path,
                    title=f"{ticker} Residual Diagnostics {window.upper()}",
                    caption=relative_to_project(path),
                    explanation=explain_lppl_plot(ticker, suffix),
                )
            )

    for ticker, data in grouped.items():
        highlights = data["highlights"]
        residuals = data["residuals"]
        if isinstance(highlights, list):
            highlights.sort(key=lambda item: 0 if "Overview" in item["title"] else 1)
        if isinstance(residuals, list):
            residuals.sort(key=lambda item: natural_sort_key(item["file_name"]))
        data["summary"] = f"{ticker} LPPL and residual diagnostics"

    return {
        "title": "LPPL-MFDFA",
        "description": "Run LPPL generation from the dashboard with explicit date and window inputs, then inspect the refreshed output files here.",
        "defaults": DEFAULT_LPPL_CONFIG,
        "source": source_manifest_entry(LPPL_NOTEBOOK_SOURCE),
        "tickers": dict(sorted(grouped.items(), key=lambda item: natural_sort_key(item[0]))),
    }
