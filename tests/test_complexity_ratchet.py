"""The complexity-suppression ratchet (ENG-25, #307).

`tools/complexity_budget.py` counts how many functions in `src/process_improve`
breach a complexity threshold, ignoring in-line `# noqa` directives. These tests
make that count a CI gate: it may never rise, and when it falls the budget has
to follow it down, so a refactor cannot be quietly undone by the next one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "complexity_budget.py"

# The ratchet measures with ruff. Without it there is nothing to measure, so the
# whole module skips rather than each test deciding for itself.
pytest.importorskip("ruff", reason="run `uv sync --dev` to enable the complexity ratchet")


def _load_tool():
    """Import `tools/complexity_budget.py`, which is a script, not a package module."""
    if "complexity_budget" in sys.modules:
        return sys.modules["complexity_budget"]
    spec = importlib.util.spec_from_file_location("complexity_budget", _TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - only if the file is deleted
        msg = f"cannot load {_TOOL}, so the complexity ratchet cannot run"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the module uses `from __future__ import annotations`,
    # and dataclasses resolves those string annotations through ``sys.modules``.
    sys.modules["complexity_budget"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def budget():
    """Load the complexity-budget tool module."""
    return _load_tool()


@pytest.fixture(scope="module")
def offenders(budget):
    """Every complexity breach in the package, measured once for this module.

    A `RuntimeError` out of `measure()` is deliberately not caught: ruff is
    installed by this point, so a failure there means the ratchet itself is
    broken, and that should fail rather than quietly skip.
    """
    return budget.measure()


@pytest.mark.integration
def test_complexity_counts_match_their_budget(budget, offenders):
    """No rule may exceed its budget, and a budget below the count must be lowered."""
    problems = budget.check(offenders)
    assert not problems, "\n".join(["Complexity budget (#307) is out of date:", *problems])


@pytest.mark.integration
def test_budget_is_still_above_the_target(budget):
    """The v2.0 target must stay ahead of the budget, or it has been reached."""
    total = sum(budget.BUDGET.values())
    assert total >= budget.TARGET_TOTAL, (
        f"The #307 target of {budget.TARGET_TOTAL} by v{budget.TARGET_RELEASE} has been met "
        f"({total} breaches left). Set a new target in tools/complexity_budget.py."
    )


@pytest.mark.integration
def test_every_tracked_rule_has_a_budget(budget):
    """`RULES` and `BUDGET` cannot drift apart; a rule without a budget is unratcheted."""
    assert set(budget.BUDGET) == set(budget.RULES)


@pytest.mark.integration
def test_offenders_are_ranked_worst_first(budget, offenders):
    """The report must lead with the function most worth splitting next."""
    excesses = [off.excess for off in offenders]
    assert excesses == sorted(excesses, reverse=True)
    assert all(off.excess > 1.0 for off in offenders)


@pytest.mark.integration
def test_the_refactored_fit_methods_carry_no_suppression(offenders):
    """`MBPLS.fit` (#307, 1.95.1) and `MBPCA.fit` stay split; a regrowth fails here."""
    split = {
        ("multivariate/_mbpls.py", "fit"),
        ("multivariate/_mbpca.py", "fit"),
    }
    still_complex = {
        (off.path.as_posix().split("process_improve/")[-1], off.function)
        for off in offenders
        if off.code in {"C901", "PLR0912", "PLR0915"}
    }
    assert not (split & still_complex)


def test_the_tool_runs_as_a_script():
    """`python tools/complexity_budget.py` must work standalone, outside pytest."""
    module = _load_tool()
    assert module.main(["--worst", "1"]) == 0


def test_the_tool_reports_a_regression_without_measuring_again():
    """`check()` works on a supplied measurement, so callers pay for ruff once."""
    module = _load_tool()
    overflowing = [
        module.Offender(code="C901", path=_TOOL, row=1, function="f", value=99, threshold=10)
        for _ in range(module.BUDGET["C901"] + 1)
    ]
    problems = module.check(overflowing)
    assert any("C901" in problem and "budget is" in problem for problem in problems)
