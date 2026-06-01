"""SEC-14: end-to-end guards against patsy formula code-execution.

SEC-01 plugged ``fit_linear_model``; SEC-14 closes the same class of bug in
``lm()`` and the ``analyze_experiment`` / ``evaluate_design`` /
``augment_design`` paths. Patsy evaluates each formula term (and each ``I(...)``
or bare factor) as a Python expression, so an untrusted ``model`` /
``target_model`` string, ``response_column`` or column name is a remote
code-execution vector.

These tests assert that malicious input is rejected (and, at the tool boundary,
that no side effect runs) while legitimate models still fit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from process_improve.experiments.analysis import analyze_experiment
from process_improve.experiments.augment import augment_design
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.models import UnsafeFormulaError, lm
from process_improve.experiments.structures import c, gather
from process_improve.tool_spec import execute_tool_call

# A replicated 2^2 design with a response, reused across the analysis tests.
_DESIGN = pd.DataFrame(
    {
        "A": [-1, 1, -1, 1, -1, 1, -1, 1],
        "B": [-1, -1, 1, 1, -1, -1, 1, 1],
        "y": [28.0, 36.0, 18.0, 31.0, 27.5, 36.5, 17.5, 31.5],
    }
)

_RCE = "y ~ A + I(__import__('os').system('id'))"


# ---------------------------------------------------------------------------
# lm() - allows I()/np transforms, rejects code execution
# ---------------------------------------------------------------------------


class TestLmGuard:
    def _expt(self):
        A = c(-1, +1, -1, +1)
        B = c(-1, -1, +1, +1)
        y = c(41, 27, 35, 20, name="y")
        return gather(A=A, B=B, y=y, title="2x2")

    @pytest.mark.parametrize(
        "formula",
        [
            "y ~ I(__import__('os').system('id'))",
            "y ~ A + os.system(B)",
            "y ~ A + open('x')",
        ],
    )
    def test_malicious_formula_rejected(self, formula: str) -> None:
        with pytest.raises(UnsafeFormulaError):
            lm(formula, self._expt())

    def test_legitimate_transform_still_fits(self) -> None:
        # I() and curated numpy transforms remain available to the public API.
        model = lm("y ~ A + B + I(A ** 2)", self._expt())
        assert model is not None


# ---------------------------------------------------------------------------
# analyze_experiment
# ---------------------------------------------------------------------------


class TestAnalyzeExperimentGuard:
    def test_malicious_model_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError):
            analyze_experiment(_DESIGN, response_column="y", model=_RCE)

    def test_malicious_response_column_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError):
            analyze_experiment(_DESIGN, response_column="y); import os; (", model="interactions")

    def test_malicious_column_name_rejected(self) -> None:
        bad = _DESIGN.rename(columns={"A": "A + __import__('os')"})
        with pytest.raises(UnsafeFormulaError):
            analyze_experiment(bad, response_column="y", model="interactions")

    @pytest.mark.parametrize("model", ["main_effects", "interactions", "quadratic", "y ~ A*B"])
    def test_legitimate_models_fit(self, model: str) -> None:
        result = analyze_experiment(_DESIGN, response_column="y", model=model)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# evaluate_design
# ---------------------------------------------------------------------------


class TestEvaluateDesignGuard:
    _DM = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})

    def test_malicious_model_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError):
            evaluate_design(self._DM, model="~ A + I(__import__('os').system('id'))", metric="d_efficiency")

    @pytest.mark.parametrize("model", ["main_effects", "interactions", "quadratic"])
    def test_legitimate_models_evaluate(self, model: str) -> None:
        result = evaluate_design(self._DM, model=model, metric="d_efficiency")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# augment_design (D-optimal path reaches dmatrix via _build_model_rhs)
# ---------------------------------------------------------------------------


class TestAugmentDesignGuard:
    _DM = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})

    def test_malicious_target_model_rejected(self) -> None:
        with pytest.raises(UnsafeFormulaError):
            augment_design(
                existing_design=self._DM,
                augmentation_type="add_runs_optimal",
                target_model="I(__import__('os').system('id'))",
                n_additional_runs=2,
            )

    def test_legitimate_augmentation_runs(self) -> None:
        result = augment_design(
            existing_design=self._DM,
            augmentation_type="add_runs_optimal",
            target_model="quadratic",
            n_additional_runs=2,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tool boundary: malicious payload must error AND not execute (no side effect)
# ---------------------------------------------------------------------------


class TestToolBoundaryNoSideEffect:
    def test_analyze_experiment_tool_blocks_rce(self, tmp_path) -> None:
        sentinel = tmp_path / "pwned_analyze"
        malicious = f"y ~ A + I(__import__('os').system('touch {sentinel}'))"
        result = execute_tool_call(
            "analyze_experiment",
            {
                "design_matrix": [{"A": -1, "y": 1.0}, {"A": 1, "y": 2.0}],
                "response_column": "y",
                "model": malicious,
            },
        )
        assert "error" in result
        assert not sentinel.exists(), "formula was evaluated - RCE guard failed"

    def test_evaluate_design_tool_blocks_rce(self, tmp_path) -> None:
        sentinel = tmp_path / "pwned_evaluate"
        malicious = f"~ A + I(__import__('os').system('touch {sentinel}'))"
        result = execute_tool_call(
            "evaluate_design",
            {
                "design_matrix": [{"A": -1, "B": -1}, {"A": 1, "B": 1}],
                "model": malicious,
                "metric": "d_efficiency",
            },
        )
        assert "error" in result
        assert not sentinel.exists(), "formula was evaluated - RCE guard failed"

    def test_augment_design_tool_blocks_rce(self, tmp_path) -> None:
        sentinel = tmp_path / "pwned_augment"
        malicious = f"I(__import__('os').system('touch {sentinel}'))"
        result = execute_tool_call(
            "augment_design",
            {
                "existing_design": [
                    {"A": -1, "B": -1},
                    {"A": 1, "B": -1},
                    {"A": -1, "B": 1},
                    {"A": 1, "B": 1},
                ],
                "augmentation_type": "add_runs_optimal",
                "target_model": malicious,
                "n_additional_runs": 2,
            },
        )
        assert "error" in result
        assert not sentinel.exists(), "formula was evaluated - RCE guard failed"
