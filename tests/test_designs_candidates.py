"""Tests for optimal designs chosen from a user-supplied candidate set."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor, generate_design

TEMP = Factor(name="T", low=100, high=150)
DOSE = Factor(name="D", low=20, high=60)


@pytest.fixture
def history() -> pd.DataFrame:
    """Thirty operating points from plant records, labelled by batch."""
    rng = np.random.default_rng(0)
    points = pd.DataFrame({"T": rng.uniform(100, 150, 30).round(1), "D": rng.uniform(20, 60, 30).round(1)})
    points.index = [f"batch_{i:02d}" for i in range(30)]
    return points


def _rows_are_candidates(design: pd.DataFrame, candidates: pd.DataFrame, columns: list[str]) -> bool:
    supplied = {tuple(row) for row in candidates[columns].itertuples(index=False)}
    return all(tuple(row) in supplied for row in design[columns].round(9).itertuples(index=False))


class TestBox:
    def test_every_run_is_a_supplied_candidate(self, history: pd.DataFrame) -> None:
        result = generate_design([TEMP, DOSE], budget=8, candidates=history, model_type="interactions")
        assert result.design_type == "d_optimal"  # chosen because candidates were given
        assert result.n_runs == 8
        assert _rows_are_candidates(result.design_actual.round(9), history, ["T", "D"])
        meta = result.metadata
        assert meta["candidate_source"] == "user"
        assert meta["backend"] == "candidate_exchange"  # even when pyoptex is installed
        assert sum(meta["selected_candidates"].values()) == 8
        assert set(meta["selected_candidates"]) <= set(history.index)

    def test_d_optimal_picks_the_spread_out_points(self, history: pd.DataFrame) -> None:
        """For a main-effects model the chosen points should span the history's range in both factors."""
        result = generate_design([TEMP, DOSE], budget=4, candidates=history, model_type="main_effects")
        chosen = history.loc[list(result.metadata["selected_candidates"])]
        assert chosen["T"].max() - chosen["T"].min() > 0.8 * (history["T"].max() - history["T"].min())
        assert chosen["D"].max() - chosen["D"].min() > 0.8 * (history["D"].max() - history["D"].min())

    def test_constraints_drop_infeasible_candidates(self, history: pd.DataFrame) -> None:
        heat = Constraint(expression="3*T + 5*D <= 600")
        result = generate_design([TEMP, DOSE], budget=8, candidates=history, constraints=[heat])
        infeasible = int((3 * history["T"] + 5 * history["D"] > 600).sum())
        assert result.metadata["n_candidates_infeasible"] == infeasible
        assert (3 * result.design_actual["T"] + 5 * result.design_actual["D"] <= 600 + 1e-9).all()

    def test_i_optimal_over_the_candidates(self, history: pd.DataFrame) -> None:
        result = generate_design(
            [TEMP, DOSE], design_type="i_optimal", budget=10, candidates=history, model_type="quadratic"
        )
        assert result.metadata["optimality_criterion"] == "i_optimal"
        assert result.metadata["trace_criterion"] > 0

    def test_categorical_candidates(self) -> None:
        catalyst = Factor(name="Cat", type="categorical", levels=["A", "B"])
        rng = np.random.default_rng(1)
        pool = pd.DataFrame({"T": rng.uniform(100, 150, 20), "Cat": rng.choice(["A", "B"], 20)})
        result = generate_design([TEMP, catalyst], budget=6, candidates=pool, random_state=0)
        assert set(result.design_actual["Cat"]) == {"A", "B"}

    def test_candidates_outside_the_range_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        pool = pd.DataFrame({"T": [90.0, 110.0, 130.0, 150.0, 120.0], "D": [20.0, 60.0, 30.0, 50.0, 40.0]})
        with caplog.at_level(logging.WARNING):
            generate_design([TEMP, DOSE], budget=4, candidates=pool, model_type="main_effects")
        assert "outside the factors' low/high range" in caplog.text


class TestMixture:
    def test_blends_from_a_recipe_book(self) -> None:
        factors = [Factor(name=n, type="mixture", low=0.1, high=0.8) for n in ("a", "b", "c")]
        rng = np.random.default_rng(2)
        blends = pd.DataFrame(rng.dirichlet([2, 2, 2], 40), columns=["a", "b", "c"])
        result = generate_design(factors, design_type="d_optimal", budget=8, candidates=blends)
        meta = result.metadata
        assert meta["method"] == "d_optimal_user_candidates"
        in_bounds = ((blends >= 0.1 - 1e-9) & (blends <= 0.8 + 1e-9)).all(axis=1)
        assert set(meta["selected_candidates"]) <= {str(i) for i in blends.index[in_bounds]}
        np.testing.assert_allclose(result.design_actual[["a", "b", "c"]].sum(axis=1), 1.0)

    def test_amounts_instead_of_proportions_are_refused(self) -> None:
        factors = [Factor(name=n, type="mixture") for n in ("a", "b", "c")]
        amounts = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [5.0, 5.0, 5.0], "c": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="sum to 1"):
            generate_design(factors, design_type="d_optimal", budget=3, candidates=amounts)


class TestErrors:
    def test_non_optimal_design_type(self, history: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="only supported for the optimal design families"):
            generate_design([TEMP, DOSE], design_type="full_factorial", candidates=history)

    def test_missing_column(self, history: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            generate_design([TEMP, DOSE, Factor(name="P", low=1, high=5)], budget=8, candidates=history)

    def test_non_numeric_values(self) -> None:
        pool = pd.DataFrame({"T": [100, "hot", 150], "D": [20, 40, 60]})
        with pytest.raises(ValueError, match="non-numeric"):
            generate_design([TEMP, DOSE], budget=3, candidates=pool, model_type="main_effects")

    def test_unknown_categorical_level(self) -> None:
        catalyst = Factor(name="Cat", type="categorical", levels=["A", "B"])
        pool = pd.DataFrame({"T": [100.0, 150.0, 120.0], "Cat": ["A", "B", "Z"]})
        with pytest.raises(ValueError, match="unknown levels"):
            generate_design([TEMP, catalyst], budget=3, candidates=pool, model_type="main_effects")

    def test_too_few_distinct_candidates_for_the_model(self) -> None:
        pool = pd.DataFrame({"T": [100.0, 150.0], "D": [20.0, 60.0]})
        with pytest.raises(ValueError, match="cannot support"):
            generate_design([TEMP, DOSE], budget=6, candidates=pool, model_type="interactions")
