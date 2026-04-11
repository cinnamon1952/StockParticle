from __future__ import annotations

from typing import Callable

from notebook_sources import CORRELATION_NOTEBOOK_SOURCE, source_manifest_entry
from plot_helpers import image_files, pretty_label, relative_to_project, TRIAL_DIR


FIGURE_EXPLANATIONS = {
    "fig1_distributions": "Empirical return distributions for DJIA and NASDAQ compared against a Gaussian baseline, replicating the Sornette (2003) tail-fatness observation.",
    "fig2_autocorrelation": "ACF of raw and absolute returns showing that linear correlations vanish quickly while volatility clustering persists over dozens of lags.",
    "fig3_scaling_law": "Volatility scaling test across daily, weekly, monthly, and quarterly horizons vs the theoretical sqrt(t) prediction.",
    "fig4_volatility_clustering": "Volatility clustering visualization across multiple resampling frequencies, with extreme events (>3 sigma) highlighted.",
    "fig5_bubble_correlation": "Rolling average pairwise correlation among 2008-era DJIA components overlaid on the index price, showing correlation spikes into the crash.",
    "fig5b_pairwise_heatmaps": "Pairwise correlation heatmaps across pre-bubble, pre-crash, and during-crash regimes for DJIA components.",
    "fig6_log_vs_simple": "Additivity comparison of log-returns vs simple returns for DJIA during the 2020 COVID drawdown.",
}


def explain_correlation_plot(path_stem: str) -> str:
    for key, explanation in FIGURE_EXPLANATIONS.items():
        if path_stem.startswith(key):
            return explanation
    return "Correlation and returns analysis plot from the trial notebook pipeline."


def build_correlation_section(register_asset: Callable[..., dict[str, str]]) -> dict[str, object]:
    corr_dir = TRIAL_DIR / "corr&returns_outputs"

    images = [
        register_asset(
            path,
            title=pretty_label(path.stem),
            caption=relative_to_project(path),
            explanation=explain_correlation_plot(path.stem),
        )
        for path in image_files(corr_dir)
    ]

    return {
        "title": "Correlation & Returns",
        "description": "Return distributions, autocorrelation, volatility scaling, clustering, and 2008 pairwise correlation analysis.",
        "source": source_manifest_entry(CORRELATION_NOTEBOOK_SOURCE),
        "images": images,
    }
