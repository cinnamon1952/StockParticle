from __future__ import annotations

from pathlib import Path
from typing import Callable

from notebook_sources import MULTIFRACTAL_NOTEBOOK_SOURCES, source_manifest_entry
from plot_helpers import image_files, pretty_label, relative_to_project, TRIAL_DIR


def explain_multifractal_plot(path: Path) -> str:
    name = path.stem.lower()
    if path.parent.name == "mf_corr_plots":
        if "fullrange" in name:
            return "This keeps the full y-range so any correlation drop-offs across q-values stay visible instead of being visually compressed."
        if "sector" in name:
            return "This compares multifractal correlation behavior by sector, which is useful for spotting whether co-movement concentrates in one part of the market or broadens out."
        return "This multifractal correlation plot summarizes how positive and negative fluctuation functions move together across q-values."
    return "This MF-DFA output shows the per-ticker multifractal return structure saved from the trial notebook pipeline."


def build_group(
    register_asset: Callable[..., dict[str, str]],
    *,
    group_key: str,
    directory: Path,
    title: str,
    description: str,
) -> dict[str, object]:
    return {
        "title": title,
        "description": description,
        "source": source_manifest_entry(MULTIFRACTAL_NOTEBOOK_SOURCES[group_key]),
        "images": [
            register_asset(
                path,
                title=pretty_label(path.stem),
                caption=relative_to_project(path),
                explanation=explain_multifractal_plot(path),
            )
            for path in image_files(directory)
        ],
    }


def build_multifractal_section(register_asset: Callable[..., dict[str, str]]) -> dict[str, object]:
    mf_dir = TRIAL_DIR / "mf_plots"
    mf_corr_dir = TRIAL_DIR / "mf_corr_plots"

    return {
        "title": "Multifractal Plots",
        "description": "This tab now pulls only from the direct multifractal output folders under scripts/trial.",
        "groups": {
            "mf_plots": build_group(
                register_asset,
                group_key="mf_plots",
                directory=mf_dir,
                title="MF Plots",
                description="Direct outputs from `scripts/trial/mf_plots/`.",
            ),
            "mf_corr_plots": build_group(
                register_asset,
                group_key="mf_corr_plots",
                directory=mf_corr_dir,
                title="MF Correlation Plots",
                description="Direct outputs from `scripts/trial/mf_corr_plots/`.",
            ),
        },
    }
