from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TabSpec:
    slug: str
    title: str
    description: str
    render: Callable[[], None]
