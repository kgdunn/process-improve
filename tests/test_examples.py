"""Each script and notebook in ``examples/`` runs on the package as it is now, so an example cannot go stale unseen.

Nothing else runs these examples, so a renamed or removed name would break them silently. An example
that needs a package which no extra of this project installs (a forecasting model, a graph
database, a notebook widget) is skipped, but the ``process_improve`` names it imports must still exist.
A script that takes too long for CI has a quick mode, which it names ``--quick``: the test runs that.
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
QUICK = "--quick"


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


def _has_quick_mode(script: Path) -> bool:
    """Whether the script has a quick mode; such a script takes seconds even in it, so its run is tiered ``slow``."""
    return QUICK in _source(script)


def _silences_every_warning(call: ast.Call) -> bool:
    """Whether ``call`` ignores every warning: an ``"ignore"`` filter that names no category, message or module."""
    name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
    if name not in ("simplefilter", "filterwarnings", "catch_warnings"):
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    positional = call.args if name != "catch_warnings" else []  # catch_warnings takes keywords only
    action = positional[0] if positional else keywords.get("action")
    names_some = len(positional) > 1 or not keywords.keys().isdisjoint({"category", "message", "module"})
    return isinstance(action, ast.Constant) and action.value == "ignore" and not names_some


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


@pytest.mark.parametrize("example", SCRIPTS + NOTEBOOKS, ids=_name)
def test_an_example_silences_only_the_warnings_it_names(example: Path) -> None:
    # A filter that ignores every warning also hides the ones the package adds later, such as a fit that stopped early.
    calls = [node for node in ast.walk(ast.parse(_source(example))) if isinstance(node, ast.Call)]
    blanket = [call.lineno for call in calls if _silences_every_warning(call)]
    assert not blanket, f"{_name(example)} ignores every warning at line(s) {blanket}: name the category or the message"


@pytest.mark.dataset
@pytest.mark.parametrize(
    "script",
    [pytest.param(path, marks=pytest.mark.slow) if _has_quick_mode(path) else path for path in SCRIPTS],
    ids=_name,
)
def test_the_example_runs(
    script: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    # The modules beside it import as they do for ``python script.py``; and a script may add to sys.path.
    monkeypatch.setattr(sys, "path", [str(script.parent), *sys.path])
    _skip_unless_installed(script)
    monkeypatch.setattr(go.Figure, "show", lambda *_args, **_kwargs: None)  # it would open a browser
    monkeypatch.setattr(sys, "argv", [str(script), *([QUICK] if _has_quick_mode(script) else [])])
    monkeypatch.chdir(tmp_path)  # anything a script writes lands here
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:  # modules beside a script have plain names (``methods``, ``study``): do not leave them to another
        for name, module in list(sys.modules.items()):
            if Path(getattr(module, "__file__", None) or "").parent == script.parent:
                del sys.modules[name]
    # The package says that a name or a default will change with a warning that points at the caller's own line.
    notices = [
        f"{Path(w.filename).name}:{w.lineno}: {w.message}"
        for w in recwarn
        if issubclass(w.category, (DeprecationWarning, FutureWarning)) and EXAMPLES in Path(w.filename).parents
    ]
    assert not notices, f"{_name(script)} uses a name or a default that will change:\n" + "\n".join(notices)


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
