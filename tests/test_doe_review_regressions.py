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


class TestScheffeAnalysisTestsMeaningfulHypotheses:
    """A Scheffe model has no intercept, so 'beta_i = 0' and 'effect = 2 x coefficient' mean nothing."""

    @pytest.fixture
    def fitted(self) -> tuple[pd.DataFrame, pd.Series]:
        """Simplex-centroid design, replicated, with x1:x2 synergy and no other non-linear blending."""
        rng = np.random.default_rng(7)
        points = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)]
        x = pd.DataFrame(points * 2, columns=["x1", "x2", "x3"])
        y = 10 * x.x1 + 5 * x.x2 + 2 * x.x3 + 8 * x.x1 * x.x2 + rng.normal(0, 0.2, len(x))
        return x, pd.Series(y, name="y")

    def _analyse(self, fitted: tuple[pd.DataFrame, pd.Series], kind: str) -> dict:
        from process_improve.experiments.analysis import analyze_experiment

        x, y = fitted
        return analyze_experiment(x, y, model="scheffe_quadratic", analysis_type=kind)  # results merge flat

    def test_anova_tests_the_linear_block_jointly(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        rows = {r["source"]: r for r in self._analyse(fitted, "anova")["anova_table"]}
        assert set(rows) == {"Linear mixture", "x1:x2", "x1:x3", "x2:x3", "Residual"}
        assert rows["Linear mixture"]["df"] == 2
        assert rows["Linear mixture"]["p_value"] < 0.001
        model_ss = sum(r["sum_sq"] for name, r in rows.items() if name != "Residual")
        assert model_ss < ((fitted[1] - fitted[1].mean()) ** 2).sum()  # no SS beyond the corrected total

    def test_significance_lists_only_blending_terms(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "significance")
        assert result["significant_terms"] == ["x1:x2"]
        assert set(result["not_significant_terms"]) == {"x1:x3", "x2:x3"}
        assert result["linear_blending_differs"] is True

    def test_effects_are_in_the_cox_direction(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "effects")
        assert result["effect_direction"] == "cox"
        assert sum(result["effects"].values()) == pytest.approx(0.0, abs=1e-9)  # q/(q-1) times a centred vector
        assert max(result["effects"], key=result["effects"].get) == "x1"

    def test_lenth_is_refused_with_a_reason(self, fitted: tuple[pd.DataFrame, pd.Series]) -> None:
        result = self._analyse(fitted, "lenth_method")
        assert result["lenth_method"] is None
        assert "anova" in result["note"]
