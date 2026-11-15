"""A blend design keeps the runs already made and chooses more beside them (``fixed_runs``)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, DesignRegion, DesignResult, Factor, generate_design
from process_improve.experiments.designs_constrained import ConstrainedOptions
from process_improve.experiments.designs_mixture_constrained import constrained_mixture_design, scheffe_matrix

NAMES = list("abcde")
COMPONENTS = [Factor(name=name, type="mixture", high=0.6) for name in NAMES]
RULES = ["a + b <= 0.7"]


def _design(budget: int, model: str = "scheffe_linear", seed: int = 1, **options: object) -> DesignResult:
    return generate_design(
        COMPONENTS,
        design_type="d_optimal",
        budget=budget,
        model_type=model,
        constraints=RULES,
        random_state=seed,
        **options,
    )


def _blends(result: DesignResult) -> np.ndarray:
    return result.design_actual[NAMES].to_numpy(dtype=float)


def _log_det(blends: np.ndarray, model: str) -> float:
    rows = scheffe_matrix(blends, model)
    return float(np.linalg.slogdet(rows.T @ rows)[1])


@pytest.fixture(scope="module")
def made() -> pd.DataFrame:
    """Eight blends already made, chosen for the linear model."""
    return _design(8).design_actual[NAMES]


class TestMoreRuns:
    """The runs already made are the first rows, as given, and the budget counts them."""

    def test_the_runs_already_made_come_first_and_are_kept(self, made: pd.DataFrame) -> None:
        more = _design(12, fixed_runs=made)
        blends = _blends(more)
        assert blends.shape == (12, len(NAMES))
        np.testing.assert_allclose(blends[:8], made.to_numpy(dtype=float))
        assert more.metadata["n_fixed_runs"] == 8
        assert "n_fixed_runs_outside_region" not in more.metadata
        assert DesignRegion(COMPONENTS, RULES).feasible(blends).all()

    def test_the_new_runs_are_chosen_for_what_the_first_ones_lack(self, made: pd.DataFrame) -> None:
        # Eight runs for the linear model cannot estimate the quadratic one; twelve more beside them can.
        model = "scheffe_quadratic"
        assert np.linalg.matrix_rank(scheffe_matrix(made.to_numpy(dtype=float), model)) < 15
        more = _blends(_design(20, model, fixed_runs=made))
        assert np.linalg.matrix_rank(scheffe_matrix(more, model)) == 15
        # Twelve runs chosen without regard to the eight are no better beside them than the twelve chosen for them.
        unaware = np.vstack([made.to_numpy(dtype=float), _blends(_design(15, model, seed=2))[:12]])
        assert _log_det(more, model) >= _log_det(unaware, model) - 1e-9

    @pytest.mark.parametrize("criterion", ["i_optimal", "a_optimal"])
    def test_another_criterion_keeps_them_too(self, made: pd.DataFrame, criterion: str) -> None:
        more = generate_design(
            COMPONENTS,
            design_type=criterion,
            budget=11,
            model_type="scheffe_linear",
            constraints=RULES,
            fixed_runs=made,
            random_state=1,
        )
        np.testing.assert_allclose(_blends(more)[:8], made.to_numpy(dtype=float))
        assert len(more.design_actual) == 11

    @pytest.mark.slow
    def test_the_rules_on_ingredients_left_out_hold_for_the_new_runs(self) -> None:
        names = [f"c{i}" for i in range(8)]
        components = [Factor(name=name, type="mixture", high=0.5, min_when_used=0.1) for name in names]
        rules = [f"3 <= count_nonzero({', '.join(names)}) <= 5", "c0 + c1 + c2 + c3 <= 0.6"]

        def design(budget: int, **options: object) -> np.ndarray:
            result = generate_design(
                components,
                design_type="d_optimal",
                budget=budget,
                model_type="scheffe_linear",
                constraints=rules,
                random_state=1,
                **options,
            )
            return result.design_actual[names].to_numpy(dtype=float)

        first = design(10)
        more = design(16, fixed_runs=pd.DataFrame(first, columns=names))
        np.testing.assert_allclose(more[:10], first)
        assert DesignRegion(components, rules).feasible(more).all()
        used = (more > 1e-9).sum(axis=1)
        assert used.min() >= 3
        assert used.max() <= 5

    def test_a_run_outside_the_region_is_kept_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        outside = pd.DataFrame([[0.5, 0.5, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2, 0.2]], columns=NAMES)
        with caplog.at_level(logging.WARNING):
            more = _design(9, fixed_runs=outside)
        np.testing.assert_allclose(_blends(more)[:2], outside.to_numpy())
        assert more.metadata["n_fixed_runs"] == 2
        assert more.metadata["n_fixed_runs_outside_region"] == 1  # a + b = 1 breaks a + b <= 0.7
        assert "1 fixed run(s) lie outside the constrained region; they are kept as given." in caplog.text

    def test_without_a_budget_the_design_is_a_few_runs_more_than_the_model(self, made: pd.DataFrame) -> None:
        rules = [Constraint(expression=rule) for rule in RULES]
        options = ConstrainedOptions(model_type="scheffe_linear", fixed_runs=made)
        design, meta = constrained_mixture_design(COMPONENTS, None, rules, options, 1)
        assert len(design) == 10  # the five terms and five more: two runs beside the eight
        assert meta["n_fixed_runs"] == 8
        many = pd.concat([made, made], ignore_index=True)  # sixteen runs made: the size chosen leaves none to choose
        options = ConstrainedOptions(model_type="scheffe_linear", fixed_runs=many)
        with pytest.raises(ValueError, match="fixed_runs has 16 runs but budget is 10"):
            constrained_mixture_design(COMPONENTS, None, rules, options, 1)


class TestRefusals:
    """What is not a set of blends already made, or leaves the design nothing to choose."""

    @pytest.mark.parametrize(
        ("runs", "message"),
        [
            (pd.DataFrame({"a": [0.5], "b": [0.5]}), r"missing columns for mixture components: \['c', 'd', 'e'\]"),
            (pd.DataFrame(columns=NAMES), "fixed_runs is empty"),
            (pd.DataFrame([[0.5, "x", 0.0, 0.0, 0.0]], columns=NAMES), "missing or non-numeric proportions"),
            (pd.DataFrame([[0.5, None, 0.0, 0.0, 0.0]], columns=NAMES), "missing or non-numeric proportions"),
            (pd.DataFrame([[50, 20, 10, 10, 10]], columns=NAMES), "must sum to 1"),
        ],
        ids=["no-column", "empty", "text", "missing", "amounts"],
    )
    def test_runs_that_are_not_blends(self, runs: pd.DataFrame, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            _design(10, fixed_runs=runs)

    def test_a_budget_that_leaves_no_run_to_choose(self, made: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="fixed_runs has 8 runs but budget is 8"):
            _design(8, fixed_runs=made)

    def test_a_budget_too_small_for_what_the_runs_lack(self) -> None:
        same = pd.DataFrame([[0.2] * 5] * 6, columns=NAMES)  # six runs of one blend span one coefficient of five
        with pytest.raises(ValueError, match="span only 1 of the 5 model coefficients"):
            _design(8, fixed_runs=same)

    def test_a_split_plot_blend_design_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="hard_to_change is not supported for mixture designs"):
            _design(10, hard_to_change=["a"])
