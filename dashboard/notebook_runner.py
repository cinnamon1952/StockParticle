from __future__ import annotations

import ast
import contextlib
from pathlib import Path
from pprint import pformat

from plot_helpers import IMAGE_SUFFIXES, PROJECT_ROOT, relative_to_project


@contextlib.contextmanager
def pushd(path: Path):
    import os

    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _literal(value: object) -> str:
    return pformat(value, width=88, sort_dicts=False)


def apply_assignment_overrides(source: str, overrides: dict[str, object]) -> str:
    if not overrides:
        return source

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements: list[tuple[int, int, str]] = []
    matched: set[str] = set()

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in overrides:
                replacement = f"{target.id} = {_literal(overrides[target.id])}\n"
                replacements.append((node.lineno - 1, node.end_lineno - 1, replacement))
                matched.add(target.id)
                break

    missing = sorted(set(overrides) - matched)
    if missing:
        raise ValueError(f"Could not override notebook assignments: {', '.join(missing)}")

    for start, end, replacement in reversed(replacements):
        lines[start : end + 1] = [replacement]

    return "".join(lines)


def patch_optional_imports(source: str) -> str:
    source = "import matplotlib\nmatplotlib.use('Agg')\n" + source
    numba_import = "from numba import njit\n"
    if numba_import in source:
        source = source.replace(
            numba_import,
            "try:\n"
            "    from numba import njit\n"
            "except ImportError:\n"
            "    def njit(func=None, *args, **kwargs):\n"
            "        if func is None:\n"
            "            return lambda inner: inner\n"
            "        return func\n",
        )
    yfinance_single_ticker = "    prices = data['Close'][ticker].dropna() if len(tickers) > 1 else data['Close'].dropna()\n"
    if yfinance_single_ticker in source:
        source = source.replace(
            yfinance_single_ticker,
            "    close_data = data['Close']\n"
            "    if hasattr(close_data, 'columns') and ticker in list(close_data.columns):\n"
            "        prices = close_data[ticker].dropna()\n"
            "    elif hasattr(close_data, 'ndim') and close_data.ndim == 2:\n"
            "        prices = close_data.iloc[:, 0].dropna()\n"
            "    else:\n"
            "        prices = close_data.dropna()\n",
        )
    return source


def _scan_output_dir(output_dir: Path) -> dict[str, tuple[int, int]]:
    if not output_dir.exists():
        return {}
    return {
        str(path.resolve()): (int(path.stat().st_mtime_ns), path.stat().st_size)
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def execute_notebook_cell(
    *,
    notebook_path: Path,
    cell_index: int,
    output_dir: Path,
    assignment_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    notebook_path = notebook_path.resolve()
    output_dir = output_dir.resolve()

    import json

    notebook = json.loads(notebook_path.read_text())
    cells = notebook.get("cells", [])
    if cell_index < 0 or cell_index >= len(cells):
        raise IndexError(f"Notebook cell {cell_index} does not exist in {notebook_path}")

    cell = cells[cell_index]
    if cell.get("cell_type") != "code":
        raise ValueError(f"Notebook cell {cell_index} is not a code cell")

    source = "".join(cell.get("source", []))
    source = apply_assignment_overrides(source, assignment_overrides or {})
    source = patch_optional_imports(source)

    before = _scan_output_dir(output_dir)
    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
    }
    with pushd(notebook_path.parent):
        exec(compile(source, f"{notebook_path.name}:cell_{cell_index}", "exec"), namespace, namespace)
    after = _scan_output_dir(output_dir)

    changed_files = [
        Path(path)
        for path, meta in after.items()
        if path not in before or before[path] != meta
    ]

    return {
        "notebook": relative_to_project(notebook_path),
        "cell_index": cell_index,
        "output_dir": relative_to_project(output_dir),
        "changed_files": [relative_to_project(path) for path in sorted(changed_files)],
        "changed_count": len(changed_files),
        "total_files": len(after),
    }
