from __future__ import annotations

from plot_helpers import relative_to_project, TRIAL_DIR


MULTIFRACTAL_NOTEBOOK_SOURCES = {
    "mf_plots": {
        "notebook": TRIAL_DIR / "dfa_and_its_variants.ipynb",
        "cell_index": 0,
        "output_dir": TRIAL_DIR / "mf_plots",
        "label": "Multifractal plots",
    },
    "mf_corr_plots": {
        "notebook": TRIAL_DIR / "dfa_and_its_variants.ipynb",
        "cell_index": 1,
        "output_dir": TRIAL_DIR / "mf_corr_plots",
        "label": "Multifractal correlation plots",
    },
}

LPPL_NOTEBOOK_SOURCE = {
    "notebook": TRIAL_DIR / "dfa_and_its_variants.ipynb",
    "cell_index": 5,
    "output_dir": TRIAL_DIR / "lppl_mfdfa_v2_plots",
    "label": "LPPL-MFDFA v2 plots",
    "legacy_cell_index": 4,
}


def source_manifest_entry(source: dict[str, object]) -> dict[str, object]:
    notebook = source["notebook"]
    output_dir = source["output_dir"]
    return {
        "notebook": relative_to_project(notebook),
        "cell_index": source["cell_index"],
        "output_dir": relative_to_project(output_dir),
        "label": source["label"],
        **({"legacy_cell_index": source["legacy_cell_index"]} if "legacy_cell_index" in source else {}),
    }
