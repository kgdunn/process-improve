"""Fixtures for the browser app's tests.

``web/bootstrap.py`` is loaded once per test session (once per xdist worker), and the worked
example is run once from it. A test that uses the worked example gets its own copy of each
part, so what one test changes cannot reach another.
"""

from __future__ import annotations

import copy
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from tests.web._support import BUILD_SCRIPT, WEB, Api, held_of, load_module

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture(scope="session")
def bootstrap() -> ModuleType:
    """Load ``web/bootstrap.py`` from its file, as ``web/worker.js`` loads it into Pyodide."""
    pytest.importorskip("openpyxl", reason="the page writes and reads .xlsx workbooks")
    return load_module("web_bootstrap", WEB / "bootstrap.py")


@pytest.fixture(scope="session")
def api(bootstrap: ModuleType) -> Api:
    """Call the entry points of ``web/bootstrap.py`` as the page calls them."""
    return Api(bootstrap)


@pytest.fixture(scope="session")
def build_script() -> ModuleType:
    """Load ``scripts/build_web_app.py``, which assembles the app for the documentation site."""
    return load_module("build_web_app", BUILD_SCRIPT)


@pytest.fixture(scope="session")
def node() -> str:
    """Find the Node.js that runs the page's JavaScript; a test that needs it is skipped without it."""
    found = shutil.which("node")
    if found is None:
        pytest.skip("Node.js runs the page's JavaScript")
    return found


@pytest.fixture(scope="session")
def worked_example(api: Api) -> dict[str, Any]:
    """Run the worked example as the page runs it live: its inputs, its design, and its simulated results.

    Shared by every test of the session: read it through the ``demo``, ``design``, ``held`` and
    ``simulated`` fixtures, which copy it.
    """
    demo = api.ok("api_catalogue")["demo"]
    design = api.ok("api_make_design", demo["design"])
    held = held_of(design)
    simulated = api.ok(
        "api_simulate", {**held, "scenario": demo["scenario"], "sigma": demo["sigma"], "seed": demo["seed"]}
    )
    return {"demo": demo, "design": design, "held": held, "simulated": simulated}


@pytest.fixture
def demo(worked_example: dict[str, Any]) -> dict[str, Any]:
    """Return the worked example's inputs, as ``api_catalogue`` gives them to the page."""
    return copy.deepcopy(worked_example["demo"])


@pytest.fixture
def design(worked_example: dict[str, Any]) -> dict[str, Any]:
    """Return the worked example's design: a Box-Behnken design in three factors, as ``api_make_design`` returns it."""
    return copy.deepcopy(worked_example["design"])


@pytest.fixture
def held(worked_example: dict[str, Any]) -> dict[str, Any]:
    """Return what the page holds after generating the worked example's design."""
    return copy.deepcopy(worked_example["held"])


@pytest.fixture
def simulated(worked_example: dict[str, Any]) -> dict[str, Any]:
    """Return the worked example's results, simulated from the reactor process with its noise and seed."""
    return copy.deepcopy(worked_example["simulated"])
