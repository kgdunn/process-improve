"""The package and its experiments subpackage import their public names when first asked for (#1235).

``import process_improve`` and ``from process_improve.experiments import generate_design`` load neither
scikit-learn nor statsmodels: making a design needs neither, and the browser app starts without both
(``fitting`` in ``web/runtime.json``). Every name stays importable as it was. These tests hold both halves:
what is not loaded, in an interpreter of its own, and that nothing public was lost or changed its kind.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import types
from pathlib import Path

import pytest

import process_improve

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


def test_every_subpackage_named_is_one_and_every_one_is_named():
    root = Path(process_improve.__file__).parent
    found = {path.parent.name for path in root.glob("*/__init__.py")}
    assert found == process_improve._SUBPACKAGES
    for name in sorted(found):
        assert process_improve.__getattr__(name) is importlib.import_module(f"process_improve.{name}")
