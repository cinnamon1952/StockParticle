from __future__ import annotations

from dashboard.tab_api import TabSpec
from dashboard.tabs.multifractal import TAB_SPEC as MULTIFRACTAL_TAB
from dashboard.tabs.lppl import TAB_SPEC as LPPL_TAB


def get_tabs() -> list[TabSpec]:
    return [
        MULTIFRACTAL_TAB,
        LPPL_TAB,
    ]
