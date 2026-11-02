"""Each script and notebook in ``examples/`` runs on the package as it is now, so an example cannot go stale unseen.

Nothing else runs these examples, so a renamed or removed name would break them silently. An example
that needs a package which no extra of this project installs (a forecasting model, a graph
database, a notebook widget) is skipped, but the ``process_improve`` names it imports must still exist.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import runpy
import shutil
import sys
from pathlib import Path

import nbclient
import nbformat
import plotly.graph_objects as go
import pytest
from IPython.core.inputtransformer2 import TransformerManager

EXAMPLES = Path(__file__).parents[1] / "examples"
SCRIPTS = sorted(path for path in EXAMPLES.rglob("*.py") if path.name != "__init__.py")
NOTEBOOKS = sorted(path for path in EXAMPLES.rglob("*.ipynb") if ".ipynb_checkpoints" not in path.parts)


def _name(example: Path) -> str:
    """Return the example's path in ``examples/``, which names its test."""
    return example.relative_to(EXAMPLES).as_posix()


def _source(example: Path) -> str:
    """Return the example's Python source: a notebook's code cells, with IPython's ``%`` and ``!`` lines as calls."""
    if example.suffix != ".ipynb":
        return example.read_text(encoding="utf-8")
    cells = nbformat.read(example, as_version=4).cells
    return "\n".join(TransformerManager().transform_cell(cell.source) for cell in cells if cell.cell_type == "code")


def _imports(example: Path) -> list[tuple[str, list[str]]]:
    """Return each module the example imports, with the names it takes from it (none for a plain import)."""
    found = []
    for node in ast.walk(ast.parse(_source(example))):
        if isinstance(node, ast.Import):
            found += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, [alias.name for alias in node.names]))
    return found


def _skip_unless_installed(example: Path) -> None:
    """Skip the test when the example imports a package that is not installed, and name it."""
    absent = sorted({name.split(".")[0] for name, _ in _imports(example)} - set(sys.stdlib_module_names))
    absent = [name for name in absent if importlib.util.find_spec(name) is None]
    if absent:
        pytest.skip(f"needs {', '.join(absent)}, which no extra of this package installs")


@pytest.mark.parametrize("example", SCRIPTS + NOTEBOOKS, ids=_name)
def test_the_package_names_an_example_imports_exist(example: Path) -> None:
    for module_name, names in _imports(example):
        if module_name.split(".")[0] != "process_improve":
            continue
        module = importlib.import_module(module_name)
        missing = [
            name
            for name in names
            if not hasattr(module, name) and importlib.util.find_spec(f"{module_name}.{name}") is None
        ]
        assert not missing, f"{_name(example)} imports {missing} from {module_name}, which has no such names"


@pytest.mark.dataset
@pytest.mark.parametrize("script", SCRIPTS, ids=_name)
def test_the_example_runs(script: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _skip_unless_installed(script)
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)  # it would open a browser
    monkeypatch.setattr(sys, "path", list(sys.path))  # a script may add to it
    monkeypatch.chdir(tmp_path)  # anything a script writes lands here
    runpy.run_path(str(script), run_name="__main__")


@pytest.mark.dataset
@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=_name)
def test_the_notebook_runs(notebook: Path, tmp_path: Path) -> None:
    _skip_unless_installed(notebook)
    for path in notebook.parent.iterdir():  # a notebook reads the files beside it, and writes to tmp_path
        if path.is_file():
            shutil.copy(path, tmp_path)
    # "python3" is the kernel ipykernel installs with this interpreter, whatever kernel the notebook was saved with.
    client = nbclient.NotebookClient(
        nbformat.read(notebook, as_version=4),
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
