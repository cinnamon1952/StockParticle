from __future__ import annotations

import re
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIAL_DIR = PROJECT_ROOT / "scripts" / "trial"


def natural_sort_key(value: str) -> list[Any]:
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", value)]


def pretty_label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: natural_sort_key(path.name),
    )


def cursor_asset_dir() -> Path:
    project_slug = str(PROJECT_ROOT.resolve()).strip("/").replace("/", "-")
    return Path.home() / ".cursor" / "projects" / project_slug / "assets"


def resolve_featured_image(prefix: str) -> Path | None:
    search_dirs = [
        PROJECT_ROOT / "dashboard" / "assets",
        PROJECT_ROOT / "assets",
        cursor_asset_dir(),
    ]

    for directory in search_dirs:
        if not directory.exists():
            continue
        matches = sorted(
            [path for path in directory.glob(f"{prefix}*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda path: natural_sort_key(path.name),
        )
        if matches:
            return matches[0]

    return None
