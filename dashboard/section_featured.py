from __future__ import annotations

from pathlib import Path
from typing import Callable

from plot_helpers import resolve_featured_image


ATTACHED_IMAGE_DEFAULTS = [
    {
        "title": "2008 Crash Rolling Pairwise Correlation",
        "caption": "DJIA price and pairwise-correlation view around the 2008 bubble and crash.",
        "prefix": "fig5_bubble_correlation",
    },
    {
        "title": "NASDAQ Drawdown Distribution",
        "caption": "Drawdown distribution against the null-hypothesis line for NASDAQ (USA).",
        "prefix": "drawdown_nasdaq_usa_",
    },
    {
        "title": "DJIA Correlation Heatmaps",
        "caption": "Component correlation heatmaps across pre-bubble, pre-crash, and crash regimes.",
        "prefix": "fig5b_pairwise_heatmaps",
    },
]


def explain_featured_plot(title: str) -> str:
    if "2008 Crash" in title:
        return "This highlights how component correlations climb into the crash, which is the point where diversification tends to fail fastest."
    if "Drawdown" in title:
        return "The tail behavior is the useful part here because it shows whether real drawdowns exceed what a simpler null model would predict."
    if "Heatmaps" in title:
        return "More uniform and darker heatmaps usually mean the market is moving more as one block, which is a simple visual proxy for systemic stress."
    return "This attached reference plot is available directly in the dashboard."


def build_featured_section(register_asset: Callable[..., dict[str, str]]) -> dict[str, object]:
    images: list[dict[str, str]] = []

    for item in ATTACHED_IMAGE_DEFAULTS:
        image_path = resolve_featured_image(str(item["prefix"]))
        if image_path is None:
            continue
        images.append(
            register_asset(
                Path(image_path),
                title=str(item["title"]),
                caption=str(item["caption"]),
                explanation=explain_featured_plot(str(item["title"])),
            )
        )

    return {
        "title": "Featured Attached Plots",
        "description": "Reference figures bundled separately from the generated trial outputs.",
        "images": images,
    }
