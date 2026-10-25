"""Each script in ``examples/`` runs on the package as it is now, so an example cannot go stale unseen.

Nothing else runs these scripts, so a renamed or removed name would break them silently. A script
that needs a package which no extra of this project installs (a forecasting model, a graph
database) is skipped, but the ``process_improve`` names it imports must still exist.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
SCRIPTS = sorted(path for path in EXAMPLES.rglob("*.py") if path.name != "__init__.py")


def _name(script: Path) -> str:
    """Return the script's path in ``examples/``, which names its test."""
    return script.relative_to(EXAMPLES).as_posix()


def _imports(script: Path) -> list[tuple[str, list[str]]]:
    """Return each module the script imports, with the names it takes from it (none for a plain import)."""
    found = []
    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, [alias.name for alias in node.names]))
    return found


@pytest.mark.parametrize("script", SCRIPTS, ids=_name)
def test_the_package_names_an_example_imports_exist(script: Path) -> None:
    for module_name, names in _imports(script):
        if module_name.split(".")[0] != "process_improve":
            continue
        module = importlib.import_module(module_name)
        missing = [
            name
            for name in names
            if not hasattr(module, name) and importlib.util.find_spec(f"{module_name}.{name}") is None
        ]
        assert not missing, f"{_name(script)} imports {missing} from {module_name}, which has no such names"


@pytest.mark.dataset
@pytest.mark.parametrize("script", SCRIPTS, ids=_name)
def test_the_example_runs(script: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    absent = sorted({name.split(".")[0] for name, _ in _imports(script)} - set(sys.stdlib_module_names))
    absent = [name for name in absent if importlib.util.find_spec(name) is None]
    if absent:
        pytest.skip(f"needs {', '.join(absent)}, which no extra of this package installs")
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)  # it would open a browser
    monkeypatch.setattr(sys, "path", list(sys.path))  # a script may add to it
    monkeypatch.chdir(tmp_path)  # anything a script writes lands here
    runpy.run_path(str(script), run_name="__main__")
