"""Tests for the pyoptex adapter layer in ``designs_optimal.py``.

These exercise the coordinate-exchange backend directly: factor
conversion (including the split-plot ``RandomEffect`` structure),
``_run_pyoptex`` metadata, the three optimality criteria, and the
run-order preservation branch in ``generate_design``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyoptex")

from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_optimal import (
    _convert_factors_to_pyoptex,
    _PyoptexOptions,
    _run_pyoptex,
    dispatch_a_optimal,
    dispatch_d_optimal,
    dispatch_i_optimal,
)
from process_improve.experiments.factor import Factor


def _continuous(n: int, names: str = "ABCDEF") -> list[Factor]:
    return [Factor(name=names[i], low=0, high=10) for i in range(n)]


# ---------------------------------------------------------------------------
# _convert_factors_to_pyoptex
# ---------------------------------------------------------------------------


class TestConvertFactors:
    """Translation of our Factor objects into pyoptex Factor objects."""

    def test_continuous_and_categorical_mapping(self) -> None:
        factors = [
            Factor(name="A", low=0, high=10),
            Factor(name="B", type="categorical", levels=["lo", "mid", "hi"]),
        ]
        converted = _convert_factors_to_pyoptex(factors)
        assert converted[0].name == "A"
        assert converted[0].type == "continuous"
        assert converted[0].re is None
        assert converted[1].name == "B"
        assert converted[1].type == "categorical"
        assert list(converted[1].levels) == ["lo", "mid", "hi"]
        assert converted[1].re is None

    def test_hard_to_change_builds_random_effect(self) -> None:
        """The htc factor gets a RandomEffect grouping runs into whole plots;
        easy-to-change factors do not.
        """
        factors = _continuous(2)
        converted = _convert_factors_to_pyoptex(factors, hard_to_change=["A"], n_runs=12)
        re_a = converted[0].re
        assert re_a is not None
        assert converted[1].re is None
        # 12 runs, default whole plots = max(4, 12 // 3) = 4, balanced blocks of 3.
        np.testing.assert_array_equal(re_a.Z, np.repeat(np.arange(4), 3))
        assert re_a.ratio == 0.5

    def test_multiple_hard_to_change_factors(self) -> None:
        """Every hard-to-change factor shares the whole-plot RandomEffect."""
        factors = _continuous(3)
        converted = _convert_factors_to_pyoptex(factors, hard_to_change=["A", "B"], n_runs=12)
        by_name = {f.name: f for f in converted}
        assert by_name["A"].re is not None
        assert by_name["B"].re is by_name["A"].re
        assert by_name["C"].re is None

    def test_whole_plot_padding_when_runs_do_not_divide(self) -> None:
        """When n_runs does not divide evenly, the last whole plot is padded."""
        factors = _continuous(2)
        converted = _convert_factors_to_pyoptex(
            factors,
            hard_to_change=["A"],
            n_runs=10,
            n_whole_plots=4,
        )
        np.testing.assert_array_equal(converted[0].re.Z, [0, 0, 1, 1, 2, 2, 3, 3, 3, 3])


# ---------------------------------------------------------------------------
# _run_pyoptex
# ---------------------------------------------------------------------------


class TestRunPyoptex:
    """Direct coordinate-exchange runs with a single restart for speed."""

    @pytest.mark.parametrize("model_type", ["main_effects", "quadratic"])
    def test_model_types_round_trip(self, model_type: str) -> None:
        design, meta = _run_pyoptex(
            _continuous(2),
            criterion="d_optimal",
            budget=8,
            options=_PyoptexOptions(model_type=model_type, n_tries=1),
        )
        assert design.shape == (8, 2)
        assert meta["model_type"] == model_type
        assert meta["backend"] == "pyoptex"

    def test_unknown_model_type_falls_back_to_interactions(self) -> None:
        """An unrecognized model_type maps to the 'tfi' pyoptex keyword."""
        design, meta = _run_pyoptex(
            _continuous(2),
            criterion="d_optimal",
            budget=8,
            options=_PyoptexOptions(model_type="banana", n_tries=1),
        )
        assert design.shape == (8, 2)
        assert meta["model_type"] == "banana"

    @pytest.mark.parametrize("criterion", ["d_optimal", "i_optimal", "a_optimal"])
    def test_all_criteria_report_metadata(self, criterion: str) -> None:
        _design, meta = _run_pyoptex(
            _continuous(2),
            criterion=criterion,
            budget=8,
            options=_PyoptexOptions(n_tries=1),
        )
        assert meta["optimality_criterion"] == criterion
        assert isinstance(meta["metric_value"], float)
        assert meta["backend"] == "pyoptex"
        assert "hard_to_change" not in meta

    def test_hard_to_change_recorded_in_metadata(self) -> None:
        _design, meta = _run_pyoptex(
            _continuous(2),
            criterion="d_optimal",
            budget=8,
            options=_PyoptexOptions(hard_to_change=["A"], n_tries=1),
        )
        assert meta["hard_to_change"] == ["A"]


# ---------------------------------------------------------------------------
# Dispatch defaults and mixed factor types
# ---------------------------------------------------------------------------


class TestDispatchDefaults:
    """Public dispatch functions with pyoptex available."""

    def test_i_optimal_default_budget(self) -> None:
        design, meta = dispatch_i_optimal(_continuous(3))
        assert design.shape[0] == 7  # 2k + 1
        assert meta["optimality_criterion"] == "i_optimal"

    def test_a_optimal_default_budget(self) -> None:
        design, meta = dispatch_a_optimal(_continuous(3))
        assert design.shape[0] == 7  # 2k + 1
        assert meta["optimality_criterion"] == "a_optimal"

    def test_two_hard_to_change_factors_dispatch(self) -> None:
        """Split-plot dispatch with two htc factors reports both in metadata."""
        design, meta = dispatch_d_optimal(_continuous(3), budget=12, hard_to_change=["A", "B"])
        assert design.shape[0] == 12
        assert meta["hard_to_change"] == ["A", "B"]
        assert meta["backend"] == "pyoptex"

    def test_mixed_factor_types_smoke(self) -> None:
        factors = [
            Factor(name="A", low=0, high=10),
            Factor(name="B", type="categorical", levels=["red", "green", "blue"]),
        ]
        design, meta = dispatch_d_optimal(factors, budget=9)
        assert design.shape == (9, 2)
        assert meta["backend"] == "pyoptex"


# ---------------------------------------------------------------------------
# generate_design integration: run order preserved for pyoptex backends
# ---------------------------------------------------------------------------


class TestRunOrderPreservation:
    """pyoptex designs keep their optimized ordering (vital for split-plot)."""

    def test_seed_does_not_randomize_pyoptex_design(self) -> None:
        result = generate_design(
            _continuous(2),
            design_type="i_optimal",
            budget=6,
            random_seed=999,
            center_points=0,
        )
        assert result.metadata.get("backend") == "pyoptex"
        assert result.run_order == list(range(1, 7))


# ---------------------------------------------------------------------------
# fixed_runs: seed the coordinate exchange with runs held fixed (pyoptex prior)
# ---------------------------------------------------------------------------


def _mixed_factors() -> list[Factor]:
    return [Factor(name="cat", type="categorical", levels=["A", "B", "C"]),
            Factor(name="x1", low=0, high=10),
            Factor(name="x2", low=0, high=10)]


class TestFixedRuns:
    """Design augmentation: hold given runs fixed and optimize the rest."""

    def test_fixed_centre_point_appears_first_and_budget_counts_it(self) -> None:

        centre = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0}])
        np.random.seed(1)  # noqa: NPY002 (pyoptex reads the legacy global RNG)
        result = generate_design(_mixed_factors(), design_type="i_optimal", budget=14,
                                 model_type="interactions", fixed_runs=centre)
        design = result.design
        assert len(design) == 14
        assert result.metadata.get("n_fixed_runs") == 1
        row0 = design.iloc[0]
        assert row0["cat"] == "A"
        assert abs(float(row0["x1"])) < 1e-9
        assert abs(float(row0["x2"])) < 1e-9

    def test_fixed_runs_supported_with_split_plot(self) -> None:

        centre = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0}])
        np.random.seed(2)  # noqa: NPY002
        result = generate_design(_mixed_factors(), design_type="i_optimal", budget=14,
                                 hard_to_change=["x1"], fixed_runs=centre)
        assert len(result.design) == 14
        assert result.metadata.get("n_fixed_runs") == 1

    def test_fixed_runs_rejected_for_non_optimal_design(self) -> None:

        centre = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0}])
        with pytest.raises(ValueError, match="only supported for the optimal"):
            generate_design(_mixed_factors(), design_type="full_factorial", fixed_runs=centre)

    @pytest.mark.parametrize(
        ("frame", "match"),
        [
            ([{"cat": "A", "x1": 0.0}], "missing columns"),
            ([{"cat": "Z", "x1": 0.0, "x2": 0.0}], "unknown levels"),
            ([{"cat": "A", "x1": 2.0, "x2": 0.0}], "coded"),
        ],
    )
    def test_fixed_runs_input_validation(self, frame: list, match: str) -> None:

        with pytest.raises(ValueError, match=match):
            generate_design(_mixed_factors(), design_type="i_optimal", budget=14,
                            fixed_runs=pd.DataFrame(frame))

    def test_fixed_runs_must_leave_room_to_optimize(self) -> None:

        centre = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0}])
        with pytest.raises(ValueError, match="budget must exceed"):
            generate_design(_mixed_factors(), design_type="i_optimal", budget=1, fixed_runs=centre)

    def test_fixed_runs_must_be_a_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            generate_design(_mixed_factors(), design_type="i_optimal", budget=14,
                            fixed_runs=[{"cat": "A", "x1": 0.0, "x2": 0.0}])  # type: ignore[arg-type]

    def test_fixed_runs_empty_frame_rejected(self) -> None:
        empty = pd.DataFrame({"cat": [], "x1": [], "x2": []})
        with pytest.raises(ValueError, match="empty"):
            generate_design(_mixed_factors(), design_type="i_optimal", budget=14, fixed_runs=empty)

    def test_fixed_runs_non_numeric_continuous_rejected(self) -> None:
        bad = pd.DataFrame([{"cat": "A", "x1": "middle", "x2": 0.0}])
        with pytest.raises(ValueError, match="non-numeric"):
            generate_design(_mixed_factors(), design_type="i_optimal", budget=14, fixed_runs=bad)

    def test_two_fixed_runs_both_appear_first(self) -> None:
        seed = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0},
                             {"cat": "B", "x1": 1.0, "x2": -1.0}])
        np.random.seed(3)  # noqa: NPY002
        result = generate_design(_mixed_factors(), design_type="i_optimal", budget=14, fixed_runs=seed)
        assert result.metadata.get("n_fixed_runs") == 2
        head = result.design.iloc[:2]
        assert list(head["cat"]) == ["A", "B"]
        assert float(head.iloc[1]["x1"]) == pytest.approx(1.0)
        assert float(head.iloc[1]["x2"]) == pytest.approx(-1.0)

    def test_fixed_runs_requires_pyoptex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from process_improve.experiments import designs_optimal

        monkeypatch.setattr(designs_optimal, "_PYOPTEX_AVAILABLE", False)
        centre = pd.DataFrame([{"cat": "A", "x1": 0.0, "x2": 0.0}])
        with pytest.raises(ImportError, match="requires pyoptex"):
            designs_optimal.dispatch_d_optimal(_mixed_factors(), budget=14, fixed_runs=centre)
