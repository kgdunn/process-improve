"""Tests for constrained mixture designs, Scheffé models and the DesignRegion they share."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from process_improve.experiments import (
    Constraint,
    DesignRegion,
    Factor,
    analyze_experiment,
    evaluate_design,
    generate_design,
    optimize_responses,
)
from process_improve.experiments.analysis import build_formula
from process_improve.experiments.designs_constrained import ConstrainedOptions
from process_improve.experiments.designs_mixture_constrained import (
    constrained_mixture_design,
    extreme_vertices,
    mixture_candidates,
    mixture_inequalities,
    scheffe_matrix,
)

# A bounded three-component formulation, used throughout.
BOUNDED = [
    Factor(name="x1", type="mixture", low=0.1, high=0.5),
    Factor(name="x2", type="mixture", low=0.1, high=0.7),
    Factor(name="x3", type="mixture", low=0.05, high=0.3),
]
CAP = Constraint(expression="x1 + x2 <= 0.85")
FULL = [Factor(name=n, type="mixture") for n in "ABC"]


def _feasible(x: np.ndarray, factors: list[Factor], constraints: list[Constraint] | None = None) -> np.ndarray:
    return DesignRegion(factors, constraints).feasible(x, tol=1e-7)


# ---------------------------------------------------------------------------
# Vertex enumeration and candidates
# ---------------------------------------------------------------------------


class TestExtremeVertices:
    def test_full_simplex_has_the_pure_components_as_vertices(self) -> None:
        vertices = extreme_vertices(*mixture_inequalities(FULL, None))
        assert {tuple(v) for v in vertices} == {(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)}

    @pytest.mark.parametrize("constraints", [None, [CAP], [CAP, Constraint(expression="x1 >= 2*x3")]])
    def test_every_linear_optimum_is_attained_at_a_vertex(self, constraints: list[Constraint] | None) -> None:
        """A linear objective over a polytope is optimised at a vertex, so the LP optimum must be one of ours."""
        a_mat, b_vec = mixture_inequalities(BOUNDED, constraints)
        vertices = extreme_vertices(a_mat, b_vec)
        assert _feasible(vertices, BOUNDED, constraints).all()
        for c in np.random.default_rng(0).normal(size=(25, 3)):
            lp = linprog(c, A_ub=a_mat, b_ub=b_vec, A_eq=np.ones((1, 3)), b_eq=[1.0], bounds=(None, None))
            assert lp.status == 0
            assert (vertices @ c).min() == pytest.approx(lp.fun, abs=1e-9)

    def test_edge_midpoints_lie_on_the_boundary(self) -> None:
        a_mat, b_vec = mixture_inequalities(BOUNDED, [CAP])
        candidates = mixture_candidates(a_mat, b_vec)
        slack = candidates["edge_midpoint"] @ a_mat.T - b_vec
        assert np.all(np.isclose(slack, 0.0, atol=1e-9).any(axis=1))  # at least one constraint active
        assert len(candidates["edge_midpoint"]) == len(candidates["vertex"])  # a polygon: as many edges as vertices


class TestInputErrors:
    def test_bounds_that_cannot_sum_to_one(self) -> None:
        tight = [Factor(name=n, type="mixture", low=0.4, high=0.9) for n in "ABC"]
        with pytest.raises(ValueError, match=r"lower bounds sum to 1\.2"):
            mixture_inequalities(tight, None)

    def test_nonlinear_constraint_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not linear"):
            mixture_inequalities(BOUNDED, [Constraint(expression="x1 * x2 <= 0.1", type="nonlinear")])

    def test_conflicting_constraints(self) -> None:
        with pytest.raises(ValueError, match="No mixture satisfies"):
            constrained_mixture_design(BOUNDED, None, [Constraint(expression="x3 >= 0.5")])

    def test_mixing_mixture_and_process_factors(self) -> None:
        with pytest.raises(ValueError, match="all mixture components or none"):
            DesignRegion([*BOUNDED, Factor(name="T", low=0, high=1)])


# ---------------------------------------------------------------------------
# Designs
# ---------------------------------------------------------------------------


class TestDesigns:
    def test_extreme_vertices_design_on_the_full_simplex_is_the_simplex_centroid(self) -> None:
        """With no bounds, vertices + edge midpoints + centroid are the degree-2 simplex-centroid points."""
        design, meta = constrained_mixture_design(FULL, None, options=ConstrainedOptions("quadratic"))
        expected = {(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)}
        assert {tuple(np.round(r, 9)) for r in design} == {tuple(np.round(r, 9)) for r in expected}
        assert meta["method"] == "extreme_vertices"

    @pytest.mark.parametrize("model", ["scheffe_linear", "scheffe_quadratic", "scheffe_special_cubic"])
    def test_d_optimal_runs_are_feasible_blends(self, model: str) -> None:
        design, meta = constrained_mixture_design(BOUNDED, 14, [CAP], ConstrainedOptions(model), random_state=0)
        assert design.shape == (14, 3)
        np.testing.assert_allclose(design.sum(axis=1), 1.0)
        assert _feasible(design, BOUNDED, [CAP]).all()
        assert np.linalg.matrix_rank(scheffe_matrix(design, model)) == scheffe_matrix(design[:1], model).shape[1]
        assert meta["constraints_enforced"] is True

    def test_budget_below_the_model_is_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            design, _ = constrained_mixture_design(
                BOUNDED, 3, None, ConstrainedOptions("scheffe_quadratic"), random_state=0
            )
        assert len(design) == 6
        assert "raising the budget to 6" in caplog.text

    def test_generate_design_picks_the_constrained_path_for_bounded_components(self) -> None:
        result = generate_design(BOUNDED)
        assert result.design_type == "mixture"
        assert result.metadata["method"] == "extreme_vertices"
        assert result.metadata["region"]["kind"] == "mixture"
        np.testing.assert_allclose(result.design_actual[["x1", "x2", "x3"]].sum(axis=1), 1.0)

    def test_d_optimal_request_with_mixture_factors_uses_the_mixture_engine(self) -> None:
        result = generate_design(BOUNDED, design_type="d_optimal", budget=10, constraints=[CAP], random_seed=1)
        assert result.metadata["method"] == "d_optimal_extreme_vertices"
        assert _feasible(result.design[["x1", "x2", "x3"]].to_numpy(dtype=float), BOUNDED, [CAP]).all()

    def test_unbounded_mixture_keeps_the_simplex_centroid(self) -> None:
        assert generate_design(FULL).metadata["method"] == "simplex_centroid"


# ---------------------------------------------------------------------------
# Scheffé analysis
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_mixture() -> tuple:
    result = generate_design(BOUNDED, budget=12, constraints=[CAP], random_seed=3)
    x = result.design[["x1", "x2", "x3"]].astype(float)
    y = 10 * x.x1 + 6 * x.x2 + 4 * x.x3 + 12 * x.x1 * x.x2 + np.random.default_rng(1).normal(0, 0.05, len(x))
    return result, x, y.rename("y")


def test_build_formula_for_scheffe_models() -> None:
    assert build_formula("y", ["A", "B", "C"], "scheffe_linear") == "y ~ -1 + A + B + C"
    assert build_formula("y", ["A", "B", "C"], "scheffe_quadratic") == "y ~ -1 + (A + B + C) ** 2"
    assert build_formula("y", ["A", "B", "C"], "scheffe_special_cubic") == "y ~ -1 + (A + B + C) ** 3"


def test_scheffe_fit_reports_a_centred_r_squared(fitted_mixture: tuple) -> None:
    """No intercept is fitted, but sum(x) = 1 carries one implicitly: R2 and df_model must be centred."""
    _, x, y = fitted_mixture
    summary = analyze_experiment(x, y, model="scheffe_quadratic", analysis_type="coefficients")["model_summary"]
    assert summary["n_terms"] == 6
    assert summary["df_model"] == 5
    design = scheffe_matrix(x.to_numpy(), "scheffe_quadratic")
    residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert summary["r_squared"] == pytest.approx(1 - (residual**2).sum() / ((y - y.mean()) ** 2).sum())


# ---------------------------------------------------------------------------
# DesignRegion
# ---------------------------------------------------------------------------


class TestDesignRegion:
    def test_round_trip_through_metadata(self) -> None:
        region = DesignRegion(BOUNDED, [CAP])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.kind == "mixture"
        assert [c.expression for c in again.constraints] == ["x1 + x2 <= 0.85"]

    def test_mixture_samples_are_uniform_blends_inside_the_region(self) -> None:
        region = DesignRegion(BOUNDED, [CAP])
        points = region.sample(5000, np.random.default_rng(0))
        assert points.shape == (5000, 3)
        assert region.feasible(points).all()
        # Uniform in the polygon: the sample mean approaches the area centroid, which lies inside the hull.
        assert region.feasible(points.mean(axis=0, keepdims=True)).all()

    def test_box_region_codes_actual_constraints(self) -> None:
        region = DesignRegion(
            [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)],
            [Constraint(expression="3*T + 5*D <= 600")],
        )
        assert region.feasible(np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -0.5]])).tolist() == [False, True, True]
        assert region.is_constrained

    def test_a_region_too_thin_to_sample_is_reported(self) -> None:
        region = DesignRegion([Factor(name=n, low=0, high=1) for n in "AB"], [Constraint(expression="A + B <= 1e-6")])
        with pytest.raises(ValueError, match="too small a part"):
            region.sample(100, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Evaluation over the region
# ---------------------------------------------------------------------------


class TestEvaluateOverRegion:
    @pytest.fixture
    def heat_design(self) -> object:
        factors = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
        heat = Constraint(expression="3*T + 5*D <= 600")
        return generate_design(factors, budget=10, constraints=[heat], model_type="quadratic")

    def test_recorded_region_is_used_by_default(self, heat_design: object) -> None:
        """The cut-off corner dominates the cube's worst case; inside the region the design is far better."""
        kwargs = {"model": "quadratic", "metric": "g_efficiency", "n_samples": 20_000}
        inside = evaluate_design(heat_design, **kwargs)["g_efficiency"]
        cube = evaluate_design(heat_design, region="cuboidal", **kwargs)["g_efficiency"]
        assert inside > 5 * cube

    def test_fds_names_the_region(self, heat_design: object) -> None:
        assert evaluate_design(heat_design, model="quadratic", metric="fds", n_samples=2000)["fds"]["region"] == (
            "constrained"
        )

    def test_mixture_design_defaults_to_scheffe_quadratic(self) -> None:
        result = generate_design(BOUNDED, budget=10, random_seed=0)
        metrics = evaluate_design(result, metric=["d_efficiency", "i_efficiency"], n_samples=5000)
        assert metrics["d_efficiency"] > 0
        assert metrics["i_efficiency"] is not None

    def test_explicit_region_on_a_dataframe(self) -> None:
        design = pd.DataFrame(
            constrained_mixture_design(BOUNDED, 10, None, random_state=0)[0], columns=["x1", "x2", "x3"]
        )
        metrics = evaluate_design(
            design, model="scheffe_quadratic", metric="g_efficiency", region=DesignRegion(BOUNDED)
        )
        assert 0 < metrics["g_efficiency"] <= 100

    def test_region_with_unknown_factor(self) -> None:
        design = pd.DataFrame({"A": [0.2, 0.5, 0.8], "B": [0.8, 0.5, 0.2]})
        with pytest.raises(ValueError, match="not columns of the design"):
            evaluate_design(design, model="scheffe_linear", metric="i_efficiency", region=DesignRegion(BOUNDED))


# ---------------------------------------------------------------------------
# Optimisation inside the region
# ---------------------------------------------------------------------------

HEAT_FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT_REGION = DesignRegion(HEAT_FACTORS, [Constraint(expression="3*T + 5*D <= 600")])
# y = 60 + 8T + 6D - 2T^2 - 3D^2 in coded units: its unconstrained maximum is the forbidden corner (1, 1).
HEAT_MODEL = {
    "response_name": "y",
    "factor_names": ["T", "D"],
    "coefficients": [
        {"term": "Intercept", "coefficient": 60.0},
        {"term": "T", "coefficient": 8.0},
        {"term": "D", "coefficient": 6.0},
        {"term": "I(T ** 2)", "coefficient": -2.0},
        {"term": "I(D ** 2)", "coefficient": -3.0},
    ],
}
MAXIMISE = [{"response": "y", "goal": "maximize", "low": 40, "high": 75}]


def _heat_y(x: np.ndarray) -> np.ndarray:
    return 60 + 8 * x[:, 0] + 6 * x[:, 1] - 2 * x[:, 0] ** 2 - 3 * x[:, 1] ** 2


class TestOptimizeInRegion:
    def test_desirability_optimum_respects_the_constraint(self) -> None:
        free = optimize_responses([HEAT_MODEL], MAXIMISE)["desirability"]
        inside = optimize_responses([HEAT_MODEL], MAXIMISE, region=HEAT_REGION)["desirability"]
        assert not HEAT_REGION.feasible(np.array([list(free["optimal_coded"].values())]))[0]
        x = np.array([list(inside["optimal_coded"].values())])
        assert inside["within_region"] is True
        # No feasible point does better: compare with a dense sample of the region.
        best_sampled = _heat_y(HEAT_REGION.sample(20_000, np.random.default_rng(0))).max()
        assert _heat_y(x)[0] >= best_sampled - 1e-3

    def test_mixture_optimum_sums_to_one(self, fitted_mixture: tuple) -> None:
        result, x, y = fitted_mixture
        coefs = analyze_experiment(x, y, model="scheffe_quadratic", analysis_type="coefficients")["coefficients"]
        model = {"response_name": "y", "coefficients": coefs, "factor_names": ["x1", "x2", "x3"]}
        region = DesignRegion.from_dict(result.metadata["region"])
        goal = [{"response": "y", "goal": "maximize", "low": 5, "high": 12}]
        out = optimize_responses([model], goal, region=region)["desirability"]
        blend = np.array([list(out["optimal_coded"].values())])
        assert blend.sum() == pytest.approx(1.0)
        assert out["within_region"] is True

    def test_pareto_front_points_are_feasible(self) -> None:
        second = {**HEAT_MODEL, "response_name": "cost", "coefficients": [{"term": "T", "coefficient": 1.0}]}
        goals = [*MAXIMISE, {"response": "cost", "goal": "minimize"}]
        front = optimize_responses([HEAT_MODEL, second], goals, method="pareto_front", region=HEAT_REGION)
        points = np.array([[p["coded"]["T"], p["coded"]["D"]] for p in front["pareto_front"]["front"]])
        assert HEAT_REGION.feasible(points, tol=1e-6).all()

    def test_other_methods_warn_and_ignore_the_region(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            optimize_responses([HEAT_MODEL], method="stationary_point", region=HEAT_REGION)
        assert "ignored by 'stationary_point'" in caplog.text

    def test_region_must_match_the_model_factors(self) -> None:
        with pytest.raises(ValueError, match="do not match"):
            optimize_responses([HEAT_MODEL], MAXIMISE, region=DesignRegion(BOUNDED))
