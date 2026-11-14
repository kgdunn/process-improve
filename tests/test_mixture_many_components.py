"""Mixture regions with many components, with groups held to a fixed total, and with components left out."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from process_improve.experiments import DesignRegion, Factor, generate_design, optimize_responses
from process_improve.experiments import designs_constrained as engine
from process_improve.experiments.designs_mixture_constrained import (
    TooManyVertices,
    _SampledCandidates,
    extreme_vertices,
    mixture_candidates,
    mixture_inequalities,
    sampled_vertices,
)
from process_improve.experiments.factor import as_constraints


def _buckets(per_bucket: int, cap: float, minimum: float | None = None) -> tuple[list[Factor], list[str], list[str]]:
    """Return components in 5 buckets, each bucket's share of the blend between 0.05 and ``cap``."""
    names = [[f"B{b + 1}_{i + 1}" for i in range(per_bucket)] for b in range(5)]
    flat = [name for bucket in names for name in bucket]
    factors = [Factor(name=name, type="mixture", low=0, high=cap, min_when_used=minimum) for name in flat]
    limits = [f"0.05 <= {' + '.join(bucket)} <= {cap}" for bucket in names]
    return factors, limits, flat


@pytest.fixture
def few_programmes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solve 128 linear programmes, not 1024: enough for these regions, and fast."""
    monkeypatch.setattr(engine, "_SAMPLED_VERTICES", 128)


@pytest.mark.usefixtures("few_programmes")
class TestSampledVertices:
    def test_a_region_with_too_many_vertices_to_list_is_sampled(self) -> None:
        factors, limits, _ = _buckets(3, 0.3)
        a_mat, b_vec = mixture_inequalities(factors, as_constraints(limits))
        with pytest.raises(TooManyVertices, match="Enumerating the vertices needs"):
            extreme_vertices(a_mat, b_vec)
        vertices = sampled_vertices(a_mat, b_vec)
        assert len(vertices) > 15
        np.testing.assert_allclose(vertices.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(vertices @ a_mat.T <= b_vec + 1e-9)
        assert len(np.unique(vertices.round(10), axis=0)) == len(vertices)
        assert np.linalg.matrix_rank(vertices) == 15  # the Scheffé linear model is supported

    def test_the_candidates_are_blends_of_those_vertices(self) -> None:
        factors, limits, _ = _buckets(3, 0.3)
        a_mat, b_vec = mixture_inequalities(factors, as_constraints(limits))
        candidates = mixture_candidates(a_mat, b_vec)
        assert isinstance(candidates, _SampledCandidates)
        assert candidates["face_centroid"].shape == (0, 15)
        for kind in ("vertex", "edge_midpoint", "edge_third", "plane_centroid", "centroid", "axial_blend"):
            blends = candidates[kind]
            assert len(blends) > 0
            np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
            assert np.all(blends @ a_mat.T <= b_vec + 1e-9)

    def test_a_region_whose_vertices_are_listed_is_not_sampled(self) -> None:
        factors, limits, _ = _buckets(2, 0.3)
        candidates = mixture_candidates(*mixture_inequalities(factors, as_constraints(limits)))
        assert not isinstance(candidates, _SampledCandidates)

    def test_an_empty_region_gives_no_vertices(self) -> None:
        factors, limits, _ = _buckets(3, 0.3)
        limits.append("B1_1 + B1_2 + B1_3 >= 0.9")  # above the bucket's own limit of 0.3
        a_mat, b_vec = mixture_inequalities(factors, as_constraints(limits))
        assert sampled_vertices(a_mat, b_vec).shape == (0, 15)
        with pytest.raises(ValueError, match="No mixture satisfies all the constraints"):
            mixture_candidates(a_mat, b_vec)

    def test_a_design_is_built_from_the_sample(self) -> None:
        factors, limits, flat = _buckets(3, 0.3)
        result = generate_design(factors, budget=24, model_type="main_effects", constraints=limits, random_state=1)
        assert result.metadata["vertices_sampled"] is True
        blends = result.design_actual[flat].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
        totals = blends.reshape(24, 5, 3).sum(axis=2)
        assert totals.min() >= 0.05 - 1e-9
        assert totals.max() <= 0.3 + 1e-9
        assert np.linalg.matrix_rank(blends) == 15

    def test_without_a_budget_the_design_is_not_every_sampled_vertex(self) -> None:
        factors, limits, _ = _buckets(3, 0.3)
        result = generate_design(factors, model_type="main_effects", constraints=limits, random_state=1)
        assert len(result.design_actual) == 20  # 15 terms and 5 more runs
        assert result.metadata["method"] == "d_optimal_extreme_vertices"


class TestGroupsHeldToATotal:
    """Five buckets that may each supply at most 20% of a blend supply exactly 20% each."""

    def test_the_columns_the_region_cannot_estimate_are_left_out(self, caplog: pytest.LogCaptureFixture) -> None:
        factors, limits, flat = _buckets(2, 0.2)
        with caplog.at_level(logging.WARNING):
            result = generate_design(factors, budget=16, model_type="main_effects", constraints=limits, random_state=1)
        assert result.metadata["inestimable_columns"] == ["B2_2", "B3_2", "B4_2", "B5_2"]
        assert "leaves 4 of the 10 columns of the scheffe_linear model inestimable" in caplog.text
        blends = result.design_actual[flat].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.reshape(16, 5, 2).sum(axis=2), 0.2, atol=1e-9)
        assert np.linalg.matrix_rank(blends) == 6  # the rank the region allows

    def test_without_a_budget_the_design_is_sized_to_the_columns_kept(self) -> None:
        factors, limits, _ = _buckets(2, 0.2)
        result = generate_design(factors, model_type="main_effects", constraints=limits, random_state=1)
        assert len(result.design_actual) == 11  # 6 columns and 5 more runs

    def test_a_budget_below_the_columns_kept_is_refused(self) -> None:
        factors, limits, _ = _buckets(2, 0.2)
        with pytest.raises(ValueError, match="budget"):
            generate_design(factors, budget=5, model_type="main_effects", constraints=limits, random_state=1)

    def test_a_region_that_supports_the_model_keeps_every_column(self) -> None:
        factors, limits, _ = _buckets(2, 0.3)
        result = generate_design(factors, budget=16, model_type="main_effects", constraints=limits, random_state=1)
        assert "inestimable_columns" not in result.metadata

    @pytest.mark.slow
    def test_thirty_five_components_in_buckets_of_seven(self) -> None:
        factors, limits, flat = _buckets(7, 0.2)
        result = generate_design(factors, budget=40, model_type="main_effects", constraints=limits, random_state=1)
        assert result.metadata["vertices_sampled"] is True
        assert len(result.metadata["inestimable_columns"]) == 4
        blends = result.design_actual[flat].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.reshape(40, 5, 7).sum(axis=2), 0.2, atol=1e-9)
        assert np.linalg.matrix_rank(blends) == 31


X1 = Factor(name="x1", type="mixture", min_when_used=0.1)
X2 = Factor(name="x2", type="mixture", min_when_used=0.3)
X3 = Factor(name="x3", type="mixture")
COMPONENTS = [X1, X2, X3]


def _traces(blends: np.ndarray) -> int:
    """Count the blends with x1 between 0 and 0.1 or x2 between 0 and 0.3."""
    low_x1 = (blends[:, 0] > 1e-9) & (blends[:, 0] < 0.1 - 1e-9)
    low_x2 = (blends[:, 1] > 1e-9) & (blends[:, 1] < 0.3 - 1e-9)
    return int((low_x1 | low_x2).sum())


class TestComponentsLeftOut:
    """A mixture component with a ``min_when_used`` is left out of a blend or makes up no less than that share."""

    @pytest.mark.parametrize(
        ("model_type", "budget"), [("scheffe_linear", 6), ("scheffe_quadratic", 10), ("scheffe_special_cubic", 12)]
    )
    def test_a_design_has_no_traces(self, model_type: str, budget: int) -> None:
        result = generate_design(COMPONENTS, budget=budget, model_type=model_type, random_state=1)
        blends = result.design_actual[["x1", "x2", "x3"]].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
        assert _traces(blends) == 0
        assert result.metadata["region"]["factors"][0]["min_when_used"] == 0.1

    def test_without_a_budget_the_design_is_d_optimal(self) -> None:
        # The design used otherwise ends with the centroid, a third of each: below x2's minimum.
        result = generate_design(COMPONENTS, model_type="scheffe_quadratic", random_state=1)
        assert result.metadata["method"] == "d_optimal_extreme_vertices"
        assert _traces(result.design_actual[["x1", "x2", "x3"]].to_numpy(dtype=float)) == 0

    def test_an_i_optimal_design_averages_over_its_candidates(self) -> None:
        result = generate_design(
            COMPONENTS, design_type="i_optimal", budget=10, model_type="scheffe_quadratic", random_state=1
        )
        assert _traces(result.design_actual[["x1", "x2", "x3"]].to_numpy(dtype=float)) == 0

    def test_candidates_that_all_hold_a_trace_are_refused(self) -> None:
        tight = [Factor(name="x1", type="mixture", high=0.2, min_when_used=0.15), Factor(name="x2", type="mixture")]
        with pytest.raises(ValueError, match="Every candidate blend has a component between 0 and its min_when_used"):
            generate_design(tight, budget=4, model_type="scheffe_linear", constraints=["0.05 <= x1 <= 0.1"])

    def test_a_trace_is_outside_the_region(self) -> None:
        region = DesignRegion(COMPONENTS)
        blends = np.array([[0.05, 0.45, 0.5], [0.1, 0.4, 0.5], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
        np.testing.assert_array_equal(region.feasible(blends), [False, True, False, True, True])
        assert region.gaps == [(0, 0.1), (1, 0.3)]
        assert region.feasible(region.support_points()).all()

    def test_the_region_is_not_sampled(self) -> None:
        with pytest.raises(ValueError, match="cannot be sampled"):
            DesignRegion(COMPONENTS).sample(5, np.random.default_rng(1))

    def test_the_optimum_leaves_out_a_component(self) -> None:
        # y = 4 x1 + 8 x2 + 5 x3 + 12 x1 x3: x2 alone gives 8, and the best blend of x1 and x3 gives 7.52.
        terms = {"x1": 4.0, "x2": 8.0, "x3": 5.0, "x1:x3": 12.0}
        model = {
            "response_name": "y",
            "factor_names": ["x1", "x2", "x3"],
            "coefficients": [{"term": term, "coefficient": value} for term, value in terms.items()],
        }
        goals = [{"response": "y", "goal": "maximize", "low": 0.0, "high": 12.0}]
        best = optimize_responses([model], goals=goals, region=DesignRegion(COMPONENTS))["desirability"]
        assert best["optimal_coded"] == pytest.approx({"x1": 0.0, "x2": 1.0, "x3": 0.0}, abs=1e-5)
        # x2 at most 0.2 is below its minimum of 0.3, so x2 is left out and the blend of x1 and x3 is the best.
        limited = optimize_responses([model], goals=goals, region=DesignRegion(COMPONENTS, ["x2 <= 0.2"]))
        found = limited["desirability"]
        assert found["optimal_coded"] == pytest.approx({"x1": 11 / 24, "x2": 0.0, "x3": 13 / 24}, abs=1e-4)
        assert found["predicted_responses"]["y"] == pytest.approx(7.5208, abs=1e-3)
        assert found["within_region"] is True

    def test_blends_beside_process_factors_have_no_traces(self) -> None:
        factors = [*COMPONENTS, Factor(name="T", low=0, high=100, min_when_used=40)]
        result = generate_design(factors, budget=14, model_type="main_effects", random_state=1)
        blends = result.design_actual[["x1", "x2", "x3"]].to_numpy(dtype=float)
        assert _traces(blends) == 0
        temperature = result.design_actual["T"].to_numpy(dtype=float)
        assert not np.any((temperature > 1e-9) & (temperature < 40 - 1e-9))
