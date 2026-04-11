from __future__ import annotations

from typing import Callable

from notebook_sources import DRAWDOWN_NOTEBOOK_SOURCE, source_manifest_entry
from plot_helpers import image_files, pretty_label, relative_to_project, TRIAL_DIR


MARKET_EXPLANATIONS = {
    "djia": "DJIA (USA) drawdown distribution against the exponential null hypothesis line.",
    "nasdaq": "NASDAQ (USA) drawdown distribution showing heavier tails than the null model predicts.",
    "ftse": "FTSE 100 (UK) drawdown distribution vs the exponential baseline.",
    "cac": "CAC 40 (France) drawdown distribution vs the exponential baseline.",
    "dax": "DAX (Germany) drawdown distribution vs the exponential baseline.",
    "mibtel": "MIBTel (Italy) drawdown distribution vs the exponential baseline.",
}


def explain_drawdown_plot(path_stem: str) -> str:
    for key, explanation in MARKET_EXPLANATIONS.items():
        if key in path_stem.lower():
            return explanation
    return "Drawdown distribution analysis replicating Sornette (2003) methodology with simulated Student-t shocks."


def build_drawdown_section(register_asset: Callable[..., dict[str, str]]) -> dict[str, object]:
    dd_images = [
        register_asset(
            path,
            title=pretty_label(path.stem),
            caption=relative_to_project(path),
            explanation=explain_drawdown_plot(path.stem),
        )
        for path in image_files(TRIAL_DIR)
        if path.name.startswith("drawdown_") and path.suffix.lower() == ".png"
    ]

    return {
        "title": "Drawdowns",
        "description": "Drawdown distribution analysis for major indices, comparing empirical tail behaviour against an exponential null model.",
        "source": source_manifest_entry(DRAWDOWN_NOTEBOOK_SOURCE),
        "images": dd_images,
    }
