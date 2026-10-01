"""Regression tests for the bugs found reviewing the constrained-DOE work (#624, #632-#635).

Each test reproduces a reviewer's failure scenario and checks the corrected behaviour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, designs_optimal, generate_design


def _continuous(k: int, low: float = 0, high: float = 1) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", low=low, high=high) for i in range(k)]


@pytest.fixture
def no_pyoptex(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the built-in candidate exchange, as on a process-improve[all] install."""
    monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)


class TestAutoSelectFallsBackWhenNoSupersaturatedDesignExists:
    @pytest.mark.parametrize(
        ("k", "budget"), [(3, 2), (3, 3), (4, 3), (5, 3), (5, 5), (6, 5), (7, 5), (8, 7), (10, 9), (10, 7), (12, 8)]
    )
    def test_previously_working_budgets_still_return_a_design(self, k: int, budget: int) -> None:
        """No supersaturated design exists for these (odd, too small, or aliased) budgets: fall back as before."""
        result = generate_design(_continuous(k), budget=budget)
        assert result.design_type in {"d_optimal", "plackett_burman"}

    @pytest.mark.parametrize(("k", "budget"), [(10, 6), (8, 6), (22, 12)])
    def test_supersaturated_is_chosen_when_an_unaliased_one_exists(self, k: int, budget: int) -> None:
        result = generate_design(_continuous(k), budget=budget)
        assert result.design_type == "supersaturated"
        assert result.metadata["n_fully_aliased_pairs"] == 0


@pytest.mark.usefixtures("no_pyoptex")
class TestFixedRunsThatCannotSupportTheModel:
    @pytest.mark.parametrize("design_type", ["d_optimal", "i_optimal", "a_optimal", "e_optimal"])
    def test_budget_rises_to_cover_the_missing_rank(self, design_type: str, caplog: pytest.LogCaptureFixture) -> None:
        """Three identical centre runs span 1 of 6 quadratic coefficients: 5 free runs are needed, not 3."""
        centre = pd.DataFrame({"X1": [0.0] * 3, "X2": [0.0] * 3})
        with caplog.at_level(logging.WARNING):
            result = generate_design(
                _continuous(2), design_type=design_type, model_type="quadratic", budget=6, fixed_runs=centre
            )
        assert result.n_runs == 8
        assert "span only 1 of the 6 model coefficients" in caplog.text
        x = result.design[["X1", "X2"]].to_numpy(dtype=float)
        model = np.column_stack([np.ones(len(x)), x, x[:, 0] * x[:, 1], x**2])
        assert np.linalg.matrix_rank(model) == 6


@pytest.mark.usefixtures("no_pyoptex")
def test_fixed_runs_stay_first_in_the_run_sheet() -> None:
    """Runs already performed are not shuffled into the new runs; the new runs are still randomised."""
    fixed = pd.DataFrame({"X1": [0.0, 1.0], "X2": [0.0, -1.0], "X3": [0.0, 1.0]})
    result = generate_design(_continuous(3, 0, 10), design_type="d_optimal", budget=10, fixed_runs=fixed)
    first_two = result.design[["X1", "X2", "X3"]].iloc[:2].to_numpy(dtype=float)
    np.testing.assert_array_equal(first_two, fixed.to_numpy())
    assert result.run_order[:2] == [1, 2]
    assert sorted(result.run_order[2:]) == list(range(3, 11))
