"""The package and its experiments subpackage import their public names when first asked for (#1235).

``import process_improve`` and ``from process_improve.experiments import generate_design`` load neither
scikit-learn nor statsmodels: making a design needs neither, and the browser app starts without both
(``fitting`` in ``web/runtime.json``). Every name stays importable as it was, and the packages look as they did
when they imported everything themselves: a module is an attribute, and several threads may ask at once.
These tests hold both halves: what is not loaded, in an interpreter of its own, and that nothing was lost.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

import process_improve
from process_improve import _lazy

experiments = process_improve.experiments  # a subpackage is an attribute of the package: one of the things tested

#: The packages a model is fitted with, which a design must not load.
FITTING = ("sklearn", "statsmodels")


def fresh(script: str) -> str:
    """Run ``script`` in an interpreter of its own and return what it printed; a failure there fails here."""
    result = subprocess.run(  # noqa: S603 - this interpreter, running the script given
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, encoding="locale"
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --------------------------------------------------------------------------- what is not loaded
@pytest.mark.slow
def test_importing_the_package_and_making_a_design_load_nothing_a_model_is_fitted_with():
    loaded = fresh(
        f"""
import sys
import process_improve
import process_improve.experiments
from process_improve.experiments import Factor, generate_design

factors = [Factor(name=name, low=0, high=1) for name in "ABC"]
for design_type in ("full_factorial", "box_behnken", "ccd", "d_optimal"):
    generate_design(factors, design_type=design_type)
print(sorted({{name.split(".")[0] for name in sys.modules}} & set({FITTING!r})))
"""
    )
    assert loaded == "[]"


@pytest.mark.slow
def test_a_name_that_fits_a_model_loads_what_it_needs_when_it_is_asked_for():
    loaded = fresh(
        """
import sys
from process_improve import experiments

before = "statsmodels" in sys.modules
experiments.analyze_experiment
print(before, "statsmodels" in sys.modules)
"""
    )
    assert loaded == "False True"


@pytest.mark.slow
def test_a_subpackage_is_an_attribute_of_the_package_without_an_import_of_its_own():
    """As when the package imported its subpackages itself: ``import process_improve as pi; pi.univariate``."""
    assert fresh("import process_improve as pi\nprint(pi.univariate.__name__, pi.PCA.__name__)") == (
        "process_improve.univariate PCA"
    )


# --------------------------------------------------------------------------- nothing public was lost
def test_every_public_name_of_the_experiments_package_is_the_object_its_module_defines():
    assert sorted(experiments.__all__) == sorted(experiments._MODULE_OF)
    for name, module in experiments._MODULE_OF.items():
        defined = getattr(importlib.import_module(f"process_improve.experiments.{module}"), name)
        assert getattr(experiments, name) is defined, name
        assert vars(experiments)[name] is defined, f"{name} is found in the package itself once asked for"


def test_the_lazy_names_are_the_names_a_type_checker_is_told_of():
    """The imports under ``TYPE_CHECKING`` and ``_WHERE`` say the same thing twice, for two readers."""
    tree = ast.parse(Path(experiments.__file__).read_text(encoding="utf-8"))
    (guard,) = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"]
    told = {}
    for node in guard.body:
        assert isinstance(node, ast.ImportFrom)
        assert node.module is not None
        assert all(alias.asname is None for alias in node.names)
        told[node.module.removeprefix("process_improve.experiments.")] = sorted(alias.name for alias in node.names)
    assert told == {module: sorted(names) for module, names in experiments._WHERE.items()}


@pytest.mark.parametrize("name", sorted(experiments._SAME_NAME))
def test_a_function_named_as_its_module_stays_the_function(name):
    """Whoever imports the module, and however: the package's attribute has always been the function."""
    module = importlib.import_module(f"process_improve.experiments.{name}")
    assert isinstance(module, types.ModuleType)
    function = getattr(experiments, name)
    assert callable(function)
    assert function is getattr(module, name)
    setattr(experiments, name, module)  # as the import system does when the module is first imported
    assert getattr(experiments, name) is function
    assert sys.modules[module.__name__] is module, "the module is still there for an import of it"


def test_the_names_a_function_shares_with_its_module_are_the_three_known():
    assert sorted(experiments._SAME_NAME) == ["least_squares_means", "run_sheet", "variance_dispersion"]


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(experiments._SAME_NAME))
def test_a_module_imported_before_its_function_is_asked_for_does_not_take_its_place(name):
    kind = fresh(
        f"""
import importlib
import importlib.util

module = importlib.import_module("process_improve.experiments.{name}")
from process_improve import experiments
from process_improve.experiments import {name}

print(type({name}).__name__, type(experiments.{name}).__name__, type(module).__name__)
"""
    )
    assert kind == "function function module"


def test_any_other_attribute_set_on_the_package_is_set_as_given(monkeypatch):
    marker = types.ModuleType("marker")
    monkeypatch.setattr(experiments, "a_test_s_own_attribute", marker, raising=False)
    assert experiments.a_test_s_own_attribute is marker


def test_the_packages_list_their_lazy_names():
    assert set(experiments.__all__) <= set(dir(experiments))
    assert {*process_improve.__all__, "experiments", "multivariate", "univariate"} <= set(dir(process_improve))


def test_star_imports_give_every_public_name():
    for package in (process_improve, experiments):
        names: dict[str, object] = {}
        exec(f"from {package.__name__} import *", names)  # noqa: S102 - the statement under test
        assert set(package.__all__) <= set(names), package.__name__


@pytest.mark.parametrize("package", [process_improve, experiments])
def test_a_name_the_package_does_not_have_is_an_attribute_error(package):
    with pytest.raises(AttributeError, match=f"module '{package.__name__}' has no attribute 'no_such_name'"):
        package.no_such_name  # noqa: B018 - the lookup is the test
    assert not hasattr(package, "no_such_name")


def test_the_package_s_own_lazy_names_are_the_objects_their_modules_define():
    for name, module in process_improve._LAZY_NAMES.items():
        assert process_improve.__getattr__(name) is getattr(importlib.import_module(module), name)
        assert getattr(process_improve, name) is vars(process_improve)[name]
    assert set(process_improve._LAZY_NAMES) | {"execute_tool_call", "get_tool_specs"} == set(process_improve.__all__)


# --------------------------------------------------------------------------- a module is an attribute
@pytest.mark.slow
def test_a_module_of_a_package_is_its_attribute_without_an_import_of_its_own():
    """As when the packages imported their modules themselves: ``doe.models.lm``, ``process_improve.config``."""
    found = fresh(
        """
import process_improve
import process_improve.experiments as doe

print(doe.models.lm.__name__, doe.designs.generate_design.__name__, doe._formula.__name__.rpartition(".")[2])
print(process_improve.config.__name__, process_improve.multivariate.methods.PCA.__name__)
"""
    )
    assert found.splitlines() == ["lm generate_design _formula", "process_improve.config PCA"]


@pytest.mark.parametrize("package", [process_improve, experiments])
def test_every_module_of_a_package_is_listed_and_is_what_importing_it_gives(package):
    root = Path(package.__file__).parent
    public = {path.stem for path in root.glob("[!_]*.py")} | {
        path.parent.name for path in root.glob("[!_]*/__init__.py")
    }
    assert public, "the package has modules"
    assert public <= set(dir(package))
    assert not {name for name in dir(package) if name.startswith("_")} - set(vars(package)), "no private module"
    for name in sorted(public - set(getattr(package, "_SAME_NAME", ()))):
        assert getattr(package, name) is importlib.import_module(f"{package.__name__}.{name}"), name


@pytest.mark.parametrize(
    ("name", "looked_for"), [("__wrapped__", 0), ("not a name", 0), ("no.such.name", 0), ("no_such_name", 1)]
)
def test_a_name_that_cannot_be_a_module_is_not_looked_for(monkeypatch, name, looked_for):
    """A dunder is what tools probe a module for; a name with a dot would import a parent that is not there."""
    asked = []
    load = _lazy.load
    monkeypatch.setattr(_lazy, "load", lambda module: asked.append(module) or load(module))
    for package in (process_improve, experiments):
        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(package, name)
        assert f"{package.__name__}.{name}" not in sys.modules
    assert len(asked) == 2 * looked_for, asked


def test_a_module_that_cannot_be_imported_is_no_attribute_and_the_error_says_why(tmp_path, monkeypatch):
    """``hasattr``, ``help`` and ``inspect.getmembers`` ask for every name ``dir`` lists, and expect this error only."""
    package = tmp_path / "a_lazy_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "whole.py").write_text("answer = 42\n", encoding="utf-8")
    (package / "lacking.py").write_text("import a_dependency_nobody_has\n", encoding="utf-8")
    (package / "refusing.py").write_text("raise ImportError('install the extra')\n", encoding="utf-8")
    (package / "broken.py").write_text("1 / 0\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        assert _lazy.submodule("a_lazy_package", "whole").answer == 42
        for name, cause, why in (
            ("lacking", ModuleNotFoundError, "No module named 'a_dependency_nobody_has'"),
            ("refusing", ImportError, "install the extra"),
        ):
            with pytest.raises(
                AttributeError, match=f"has no attribute '{name}': a_lazy_package.{name} could not be"
            ) as told:
                _lazy.submodule("a_lazy_package", name)
            assert why in str(told.value)
            assert type(told.value.__cause__) is cause, "the module's own error, for a reader of the traceback"
        with pytest.raises(AttributeError, match=r"^module 'a_lazy_package' has no attribute 'absent'$") as refused:
            _lazy.submodule("a_lazy_package", "absent")
        assert refused.value.__context__ is None, "an attribute that is not there, and no talk of imports"
        with pytest.raises(ZeroDivisionError):  # a module that is broken, not unavailable, says so itself
            _lazy.submodule("a_lazy_package", "broken")
        monkeypatch.setitem(sys.modules, "a_lazy_package.whole", None)  # how an import is refused, in a test say
        with pytest.raises(AttributeError, match=r"^module 'a_lazy_package' has no attribute 'whole'$"):
            _lazy.submodule("a_lazy_package", "whole")
    finally:
        for name in [name for name in sys.modules if name.partition(".")[0] == "a_lazy_package"]:
            del sys.modules[name]


@pytest.mark.slow
def test_a_package_can_be_looked_over_where_an_extra_is_not_installed():
    """Its server module needs the ``mcp`` extra. ``dir`` lists the module, and looking the package over asked for it.

    ``help(process_improve)`` and ``inspect.getmembers`` then failed for everyone who installed no extra.
    """
    found = fresh(
        """
import sys
sys.modules["mcp"] = None  # an import of it fails, as where the extra is not installed

import inspect
import pydoc

import process_improve

print("mcp_server" in dir(process_improve), hasattr(process_improve, "mcp_server"))
members = dict(inspect.getmembers(process_improve))
print("mcp_server" in members, members["experiments"].__name__, len(pydoc.render_doc(process_improve)) > 1000)
try:
    process_improve.mcp_server
except AttributeError as error:
    print(type(error.__cause__).__name__, "'mcp' extra" in str(error))
try:
    from process_improve import mcp_server
except ImportError as error:
    print(type(error).__name__, "'mcp' extra" in str(error))
"""
    )
    assert found.splitlines() == [
        "True False",
        "False process_improve.experiments True",
        "ImportError True",
        "ImportError True",
    ]


# --------------------------------------------------------------------------- several threads at once
#: Eight names that reach statsmodels and scikit-learn by different ways in.
AT_ONCE = (
    "analyze_experiment",
    "lm",
    "fit_gaussian_process",
    "analyze_functional_experiment",
    "optimize_responses",
    "evaluate_design",
    "compare_designs",
    "least_squares_means",
)


@pytest.mark.slow
def test_names_asked_for_by_several_threads_at_once_are_each_imported_whole():
    """Two threads that enter statsmodels at once, by two ways, fail: its modules import each other in a circle.

    A package that imported everything itself did so in one thread, under its import lock. One lock around the
    lazy imports keeps that: without it this fails nearly every time, with a module only half made.
    """
    failed = fresh(
        f"""
import threading
import process_improve
import process_improve.experiments as doe

names = {AT_ONCE!r}
failed, together = [], threading.Barrier(len(names) + 1)

def ask(package, name):
    together.wait()
    try:
        getattr(package, name)
    except BaseException as error:
        failed.append(f"{{name}}: {{type(error).__name__}}: {{error}}")

threads = [threading.Thread(target=ask, args=(doe, name)) for name in names]
threads.append(threading.Thread(target=ask, args=(process_improve, "PLS")))
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
print(failed)
"""
    )
    assert failed == "[]"


def at_import_time(tree: ast.Module) -> list[ast.ImportFrom]:
    """Every ``from ... import`` a module runs when it is imported: none inside a function or for a type checker."""
    found: list[ast.ImportFrom] = []
    waiting: list[ast.AST] = [tree]
    while waiting:
        node = waiting.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.If) and ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            waiting.extend(node.orelse)
            continue
        if isinstance(node, ast.ImportFrom):
            found.append(node)
        waiting.extend(ast.iter_child_nodes(node))
    return found


def test_no_module_of_the_library_asks_a_lazy_package_for_a_name_while_it_is_imported():
    """It would wait for the lock holding its own import lock, which the thread ahead of it may need.

    ``from process_improve.experiments.factor import Factor`` asks the module, not the package, and waits for
    nothing; so does an import inside a function, which runs when the function is called.
    """
    root = Path(process_improve.__file__).parent
    lazy = {"process_improve", "process_improve.experiments"}
    asked = []
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts).removesuffix(".__init__")
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in at_import_time(ast.parse(path.read_text(encoding="utf-8"))):
            source = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            names = {alias.name for alias in node.names} - {"_lazy"}  # bound by the package before anything else
            if source in lazy and names:
                asked.append(f"{path.relative_to(root)}:{node.lineno}: from {source} import {sorted(names)}")
    assert not asked, asked


def test_the_imports_a_module_runs_when_imported_are_told_from_the_others():
    tree = ast.parse(
        """
from a import at_the_top
if TYPE_CHECKING:
    from b import for_a_type_checker
else:
    from c import otherwise
class Holder:
    from d import in_a_class_body
    def method(self):
        from e import in_a_function
try:
    from f import in_a_try
except ImportError:
    from g import in_its_handler
later = lambda: __import__("h")
"""
    )
    assert sorted(node.module for node in at_import_time(tree)) == ["a", "c", "d", "f", "g"]
