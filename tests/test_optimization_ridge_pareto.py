"""Ridge analysis and Pareto front for response surfaces (#208).

Both replace stubs. The tests check them against independent ground truth rather
than against their own output: the ridge against a dense sample of each sphere,
the front against a dense grid of the design region.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments.optimization import (
    _extract_b_and_B,
    _non_dominated,
    _pareto_front,
    _ridge_analysis,
    _simplex_weights,
    optimize_responses,
)

TERMS_2D = ("Intercept", "A", "B", "I(A ** 2)", "I(B ** 2)", "A:B")


def coefficients(values: tuple[float, ...], terms: tuple[str, ...] = TERMS_2D) -> list[dict[str, object]]:
    """Build the coefficient list ``analyze_experiment`` returns, from bare numbers."""
    return [{"term": term, "coefficient": float(value)} for term, value in zip(terms, values, strict=True)]


def model(name: str, values: tuple[float, ...]) -> dict[str, object]:
    """Build the fitted-model dict ``optimize_responses`` consumes."""
    return {"response_name": name, "factor_names": ["A", "B"], "coefficients": coefficients(values)}


def best_on_sphere(
    coefs: list[dict[str, object]],
    names: list[str],
    radius: float,
    *,
    maximise: bool,
    n: int = 200_000,
) -> float:
    """Sample the sphere densely and return the best predicted response on it."""
    b0, b, B = _extract_b_and_B(coefs, names)
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(n, len(names)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    x = directions * radius
    y = b0 + x @ b + np.einsum("ij,jk,ik->i", x, B, x)
    return float(np.max(y) if maximise else np.min(y))


# ---------------------------------------------------------------------------
# Ridge analysis
# ---------------------------------------------------------------------------

# Each case names the shape of surface it covers; together they exercise the
# ordinary path, a saddle, three factors, and both hard cases.
RIDGE_CASES = {
    "interior maximum": (
        coefficients((40.0, 5.25, -2.0, -3.0, -1.5, 1.5)),
        ["A", "B"],
    ),
    "saddle": (
        coefficients((10.0, 1.0, 0.5, 2.0, -3.0, 0.4)),
        ["A", "B"],
    ),
    "three factors, rising ridge": (
        coefficients(
            (5.0, 2.0, -1.0, 0.3, -1.0, -1.05, -0.2, 1.9),
            ("Intercept", "A", "B", "C", "I(A ** 2)", "I(B ** 2)", "I(C ** 2)", "A:B"),
        ),
        ["A", "B", "C"],
    ),
    "hard case: stationary point at the centre": (
        coefficients((7.0, 0.0, 0.0, -1.0, -4.0, 0.0)),
        ["A", "B"],
    ),
    "hard case: b orthogonal to the leading eigenvector": (
        coefficients((7.0, 0.0, 3.0, -1.0, -4.0, 0.0)),
        ["A", "B"],
    ),
}


@pytest.mark.parametrize("case", sorted(RIDGE_CASES))
@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_ridge_point_is_the_optimum_on_its_sphere(case: str, direction: str) -> None:
    """Every point on the path beats 200,000 samples of the same sphere.

    This is the property that matters and the one a grid search over the Lagrange
    multiplier does not have: the returned point is *the* constrained optimum,
    not a good point rescaled onto the right radius.
    """
    coefs, names = RIDGE_CASES[case]
    maximise = direction == "maximize"
    result = _ridge_analysis(coefs, names, direction=direction, n_radii=6, max_radius=1.5)

    for point in result["path"][1:]:
        x = np.array([point["coded"][n] for n in names])
        assert np.linalg.norm(x) == pytest.approx(point["radius"], abs=1e-9), "the point must lie on its sphere"
        sampled = best_on_sphere(coefs, names, point["radius"], maximise=maximise)
        if maximise:
            assert point["predicted_response"] >= sampled - 1e-9
        else:
            assert point["predicted_response"] <= sampled + 1e-9


def test_ridge_radii_and_multiplier_are_monotone() -> None:
    """Radii ascend from the centre, and the multiplier falls as the sphere grows.

    The monotone multiplier is what makes the root-find well posed, so a break in
    it means the solver has wandered inside the spectrum and is returning saddle
    points rather than maxima.
    """
    coefs, names = RIDGE_CASES["interior maximum"]
    path = _ridge_analysis(coefs, names, n_radii=8, max_radius=1.2)["path"]

    radii = [point["radius"] for point in path]
    assert radii == sorted(radii)
    assert radii[0] == 0.0

    mus = [point["mu"] for point in path[1:]]
    assert all(later < earlier for earlier, later in itertools.pairwise(mus))

    largest_eigenvalue = max(_ridge_analysis(coefs, names, max_radius=1.0)["eigenvalues"])
    assert all(mu > largest_eigenvalue for mu in mus), "a constrained maximum needs mu above the spectrum"


def test_ridge_turns_over_at_the_stationary_point() -> None:
    """Past the stationary point the constraint stops helping, so the ridge falls.

    The reported ``stationary_point_radius`` is where that happens, and it is the
    practical read of the whole trace: moving further than that buys nothing.
    """
    coefs, names = RIDGE_CASES["interior maximum"]
    result = _ridge_analysis(coefs, names, n_radii=40, max_radius=1.5)
    radius_at_peak = max(result["path"], key=lambda point: point["predicted_response"])["radius"]

    assert result["stationary_point_radius"] == pytest.approx(0.8508, abs=1e-3)
    assert radius_at_peak == pytest.approx(result["stationary_point_radius"], abs=0.04)


def test_ridge_from_a_centred_stationary_point_follows_an_eigenvector() -> None:
    """With ``b = 0`` there is no interior multiplier; the ridge is an eigenvector.

    This is the hard case of the trust-region subproblem. The surface here is
    ``7 - A**2 - 4*B**2``, so the flattest direction is A and the ridge of maxima
    runs along it.
    """
    coefs, names = RIDGE_CASES["hard case: stationary point at the centre"]
    path = _ridge_analysis(coefs, names, direction="maximize", n_radii=4, max_radius=1.0)["path"]

    for point in path[1:]:
        assert abs(point["coded"]["A"]) == pytest.approx(point["radius"], abs=1e-9)
        assert point["coded"]["B"] == pytest.approx(0.0, abs=1e-9)
        assert point["predicted_response"] == pytest.approx(7.0 - point["radius"] ** 2, abs=1e-9)


def test_ridge_needs_curvature() -> None:
    """A first-order model has no ridge, and the message says what to use instead."""
    first_order = coefficients((10.0, 2.0, -1.0), ("Intercept", "A", "B"))
    result = _ridge_analysis(first_order, ["A", "B"])
    assert "steepest_ascent" in result["error"]
    assert "path" not in result


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"direction": "sideways"}, "must be 'maximize' or 'minimize'"),
        ({"n_radii": 0}, "at least 1"),
        ({"max_radius": 0.0}, "must be positive"),
    ],
)
def test_ridge_rejects_bad_arguments(kwargs: dict, match: str) -> None:
    coefs, names = RIDGE_CASES["interior maximum"]
    with pytest.raises(ValueError, match=match):
        _ridge_analysis(coefs, names, **kwargs)


def test_ridge_through_optimize_responses_reports_actual_units() -> None:
    """``search_bounds`` sets how far the trace goes, and ranges add actual units."""
    result = optimize_responses(
        fitted_models=[model("yield", (40.0, 5.25, -2.0, -3.0, -1.5, 1.5))],
        method="ridge_analysis",
        n_steps=4,
        search_bounds=(-1.41, 1.41),
        factor_ranges={"A": {"low": 150.0, "high": 200.0}, "B": {"low": 2.0, "high": 4.0}},
    )["ridge_analysis"]

    assert result["max_radius"] == pytest.approx(1.41)
    assert len(result["path"]) == 5
    assert result["path"][0]["actual"] == {"A": 175.0, "B": 3.0}

    furthest = result["path"][-1]
    assert furthest["actual"]["A"] == pytest.approx(175.0 + furthest["coded"]["A"] * 25.0)


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

# Found by sweeping random quadratic pairs for a front that a weighted sum cannot
# trace: only its two extreme points maximise any weighted sum of the responses.
NON_CONVEX_R1 = (1.75, -3.87, 3.19, 1.33, 0.77, -2.84)
NON_CONVEX_R2 = (-0.85, 3.79, -3.72, -2.87, 1.43, -3.71)

YIELD = (70.0, 10.0, 4.0, -6.0, -3.0, 2.0)
COST = (20.0, 8.0, -1.0, 3.0, 0.0, 0.0)
MAX_MIN_GOALS = [
    {"response": "yield", "goal": "maximize"},
    {"response": "cost", "goal": "minimize"},
]


def grid_responses(*coefs: tuple[float, ...], n: int = 601) -> tuple[np.ndarray, ...]:
    """Evaluate each coefficient set over an n-by-n grid of the coded cube."""
    axis = np.linspace(-1, 1, n)
    a, b = (grid.ravel() for grid in np.meshgrid(axis, axis))
    design = np.column_stack([np.ones_like(a), a, b, a**2, b**2, a * b])
    return tuple(design @ np.array(c) for c in coefs)


def test_front_points_are_not_dominated_by_the_whole_design_region() -> None:
    """No point in a 601x601 sweep beats a front point on one response for free."""
    front = _pareto_front([model("yield", YIELD), model("cost", COST)], MAX_MIN_GOALS, ["A", "B"])["front"]
    y_grid, c_grid = grid_responses(YIELD, COST)

    for point in front:
        y, c = point["responses"]["yield"], point["responses"]["cost"]
        beats = (y_grid >= y - 1e-9) & (c_grid <= c + 1e-9) & ((y_grid > y + 1e-4) | (c_grid < c - 1e-4))
        assert not beats.any(), f"({y:.4f}, {c:.4f}) is dominated"


def test_front_spans_the_ideal_point_and_trades_monotonically() -> None:
    """The front runs between the two single-objective optima, giving up cost for yield."""
    result = _pareto_front([model("yield", YIELD), model("cost", COST)], MAX_MIN_GOALS, ["A", "B"])
    front = result["front"]

    assert len(front) > 5
    assert result["ideal"]["yield"] == pytest.approx(max(p["responses"]["yield"] for p in front))
    assert result["ideal"]["cost"] == pytest.approx(min(p["responses"]["cost"] for p in front))

    yields = [p["responses"]["yield"] for p in front]
    costs = [p["responses"]["cost"] for p in front]
    assert yields == sorted(yields), "the front is reported in order of the first response"
    assert costs == sorted(costs), "and buying yield always costs more, which is what makes it a front"


def test_chebyshev_reaches_a_front_no_weighted_sum_can() -> None:
    """The reason for Chebyshev rather than a weighted sum, on a case that shows it.

    A weighted sum can only ever return points on the convex hull of the front.
    Here the hull touches the front at exactly two points, so weighting the two
    responses against each other, however finely, returns one of two answers and
    hides every compromise in between.
    """
    front = _pareto_front(
        [model("r1", NON_CONVEX_R1), model("r2", NON_CONVEX_R2)],
        [{"response": "r1", "goal": "maximize"}, {"response": "r2", "goal": "maximize"}],
        ["A", "B"],
    )["front"]
    r1_grid, r2_grid = grid_responses(NON_CONVEX_R1, NON_CONVEX_R2)

    reachable = {
        (round(float(r1_grid[i]), 6), round(float(r2_grid[i]), 6))
        for i in (int(np.argmax(w * r1_grid + (1 - w) * r2_grid)) for w in np.linspace(0, 1, 2001))
    }
    assert len(reachable) == 2, "the weighted-sum sweep must be the degenerate case this test is about"

    hull = np.array(sorted(reachable))
    found = np.array([[p["responses"]["r1"], p["responses"]["r2"]] for p in front])
    unsupported = sum(not bool((np.abs(hull - row).max(axis=1) < 1e-2).any()) for row in found)

    assert len(front) > 10
    assert unsupported >= len(front) - 2, "all but the two extremes are out of any weighted sum's reach"


def test_front_handles_a_target_goal() -> None:
    """A target goal joins the trade-off on the squared deviation from its target."""
    result = _pareto_front(
        [model("yield", YIELD), model("cost", COST)],
        [
            {"response": "yield", "goal": "maximize"},
            {"response": "cost", "goal": "target", "target": 25.0},
        ],
        ["A", "B"],
        n_points=15,
    )
    assert result["objectives"][1] == {"response": "cost", "goal": "target"}
    assert result["ideal"]["cost"] == pytest.approx(25.0, abs=1e-3), "the target is attainable, so it is the ideal"
    assert min(abs(p["responses"]["cost"] - 25.0) for p in result["front"]) < 0.01


def test_front_is_reproducible() -> None:
    """The random multi-starts are seeded, so two calls give the same front."""
    args = ([model("yield", YIELD), model("cost", COST)], MAX_MIN_GOALS, ["A", "B"])
    first = _pareto_front(*args)["front"]
    second = _pareto_front(*args)["front"]
    assert [p["responses"] for p in first] == [p["responses"] for p in second]


def test_front_needs_a_trade_off() -> None:
    """One response is not a multi-objective problem, and the message says so."""
    with pytest.raises(ValueError, match="at least two responses"):
        _pareto_front([model("yield", YIELD)], MAX_MIN_GOALS[:1], ["A", "B"])


def test_front_rejects_a_target_goal_with_no_target() -> None:
    with pytest.raises(ValueError, match="must supply 'target'"):
        _pareto_front(
            [model("yield", YIELD), model("cost", COST)],
            [{"response": "yield", "goal": "maximize"}, {"response": "cost", "goal": "target"}],
            ["A", "B"],
        )


def test_front_through_optimize_responses_reports_actual_units() -> None:
    result = optimize_responses(
        fitted_models=[model("yield", YIELD), model("cost", COST)],
        goals=MAX_MIN_GOALS,
        method="pareto_front",
        n_pareto_points=9,
        factor_ranges={"A": {"low": 150.0, "high": 200.0}, "B": {"low": 2.0, "high": 4.0}},
    )["pareto_front"]

    assert result["n_weights"] == 9
    for point in result["front"]:
        assert point["actual"]["A"] == pytest.approx(175.0 + point["coded"]["A"] * 25.0)
        assert point["actual"]["B"] == pytest.approx(3.0 + point["coded"]["B"] * 1.0)


# ---------------------------------------------------------------------------
# The pieces the two methods are built from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_objectives", "n_points"), [(2, 5), (3, 10), (4, 20), (5, 30)])
def test_simplex_weights_are_a_partition_of_unity(n_objectives: int, n_points: int) -> None:
    """Das-Dennis: every weight vector is non-negative and sums to one."""
    weights = _simplex_weights(n_objectives, n_points)
    assert weights.shape[1] == n_objectives
    assert len(weights) >= n_points
    assert (weights >= 0).all()
    assert weights.sum(axis=1) == pytest.approx(1.0)
    assert len({tuple(np.round(w, 12)) for w in weights}) == len(weights), "no duplicate weight vectors"


def test_simplex_weights_include_the_single_objective_corners() -> None:
    """The corners anchor the ends of the front, so they must be in the lattice."""
    weights = _simplex_weights(3, 10)
    for corner in np.eye(3):
        assert any(np.allclose(w, corner) for w in weights)


def test_non_dominated_keeps_the_frontier_and_drops_the_rest() -> None:
    """More is better in every column, so (2, 2) dominates (1, 1) but not (1, 3)."""
    utilities = np.array([[2.0, 2.0], [1.0, 1.0], [1.0, 3.0], [3.0, 0.5], [2.0, 2.0]])
    keep = _non_dominated(utilities)
    assert keep.tolist() == [True, False, True, True, True]
