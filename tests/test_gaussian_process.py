"""Gaussian process models for computer experiments (#789 B8).

Montgomery's Example 11.4 (Section 11.5, Table 11.17) prints JMP's Gaussian process fitted to ten
runs of a computational fluid dynamics model. JMP fitted the design points to the digits that
Table 11.18 shows; the catalogue holds them as Table 11.16 prints them, to three decimals. So the
estimates here agree with the printed ones only to within what that rounding moves them, and the
likelihood the search reaches is at least the likelihood at the printed theta, on the same runs.
"""

from __future__ import annotations

import doctest
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy import special

from process_improve.experiments import GaussianProcessFit, fit_gaussian_process, gaussian_process
from process_improve.experiments.reference_catalogue import reference_experiments

PRINTED_THETA = np.array([65.40254, 3603.2483])

WAVY = lambda d: np.sin(6 * d.a) * np.cos(4 * d.b) + d.c  # noqa: E731
QUADRATIC = lambda d: 1 + d.a + 2 * d.b - d.c + d.a * d.b + d.c**2  # noqa: E731
ADDITIVE = lambda d: d.a + 2 * d.b + 0.1 * d.c  # noqa: E731


@pytest.fixture(scope="module")
def plume() -> pd.DataFrame:
    [entry] = [e for e in reference_experiments("montgomery-2013") if e["name"] == "montgomery-cfd-plume-temperature"]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def plume_fit(plume: pd.DataFrame) -> GaussianProcessFit:
    return fit_gaussian_process(plume, "temperature")


def three_inputs(function: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
    """Twenty runs of a function of three inputs, at random points of the unit cube."""
    x = np.random.default_rng(3).uniform(0, 1, size=(20, 3))
    return pd.DataFrame(x, columns=["a", "b", "c"]).assign(y=function)


def one_input() -> GaussianProcessFit:
    return fit_gaussian_process(pd.DataFrame({"x": np.linspace(0, 1, 6)}).assign(y=lambda d: np.sin(3 * d.x)), "y")


def indices_by_quadrature(fit: GaussianProcessFit, ranges: dict, nodes: int = 40) -> pd.DataFrame:
    """Return the main and total indices from the prediction's values on a Gauss-Legendre grid over the box."""
    t, weights = np.polynomial.legendre.leggauss(nodes)
    lows, highs = fit.runs.min(axis=0), fit.runs.max(axis=0)
    for j, name in enumerate(fit.inputs):
        lows[j], highs[j] = ranges.get(name, (lows[j], highs[j]))
    axes = [low + (high - low) * (t + 1) / 2 for low, high in zip(lows, highs, strict=True)]
    grid = np.meshgrid(*axes, indexing="ij")
    points = pd.DataFrame({name: g.ravel() for name, g in zip(fit.inputs, grid, strict=True)})
    values = fit.predict(points)["predicted"].to_numpy().reshape(grid[0].shape)

    def average(f: np.ndarray, over: list[int]) -> np.ndarray:
        for axis in sorted(over, reverse=True):
            f = np.tensordot(f, weights / 2, axes=([axis], [0]))
        return f

    def variance(f: np.ndarray) -> float:
        every = list(range(f.ndim))
        return float(average(f**2, every) - average(f, every) ** 2)

    total, k = variance(values), len(fit.inputs)
    rows = {
        name: {
            "main_effect": variance(average(values, [i for i in range(k) if i != j])) / total,
            "total_effect": 1 - variance(average(values, [j])) / total,
        }
        for j, name in enumerate(fit.inputs)
    }
    return pd.DataFrame.from_dict(rows, orient="index")


@pytest.mark.dataset
class TestMontgomeryExample114:
    def test_the_estimates_are_jmp_s(self, plume_fit):
        theta = dict(zip(plume_fit.inputs, plume_fit.theta, strict=True))
        assert theta["x_axis"] == pytest.approx(65.40254, rel=0.01)
        assert theta["r_axis"] == pytest.approx(3603.2483, rel=0.1)  # the likelihood is flat along it: see below
        assert plume_fit.mu == pytest.approx(734.54584, rel=0.005)
        assert plume_fit.sigma2 == pytest.approx(212205.18, rel=0.02)
        assert plume_fit.minus_two_log_likelihood == pytest.approx(132.98004, abs=0.05)

    def test_the_search_reaches_the_printed_likelihood(self, plume, plume_fit):
        """At the printed theta, the rounded runs give a -2 log L only 0.013 above the search's."""
        x = plume[["x_axis", "r_axis"]].to_numpy()
        squared = gaussian_process._squared_differences(x, x)
        at_printed = gaussian_process._profile(PRINTED_THETA, squared, plume["temperature"].to_numpy())[3]
        assert 0 <= at_printed - plume_fit.minus_two_log_likelihood < 0.02

    def test_the_sensitivity_indices_are_jmp_s(self, plume_fit):
        """Table 11.17: main effects 0.0141129 and 0.9650018, totals 0.0349982 and 0.9858871."""
        found = plume_fit.sensitivity()
        assert found["main_effect"].tolist() == pytest.approx([0.0141129, 0.9650018], abs=0.003)
        assert found["total_effect"].tolist() == pytest.approx([0.0349982, 0.9858871], abs=0.003)
        interaction = found["total_effect"] - found["main_effect"]
        assert interaction.tolist() == pytest.approx([0.0208852, 0.0208852], abs=0.003)

    def test_the_model_passes_through_every_run(self, plume, plume_fit):
        at_runs = plume_fit.predict(plume)
        np.testing.assert_allclose(at_runs["predicted"], plume["temperature"], rtol=1e-8)
        assert (at_runs["std_error"] < 1e-4 * np.sqrt(plume_fit.sigma2)).all()


def test_the_standard_error_is_the_kriging_variance():
    """Sacks, Welch, Mitchell and Wynn (1989), Equation 7: sigma^2 [1 - (1, r') [[0, 1'], [1, R]]^-1 (1, r')']."""
    runs = three_inputs(WAVY)
    fit = fit_gaussian_process(runs, "y")
    points = pd.DataFrame(np.random.default_rng(5).uniform(0, 1, size=(6, 3)), columns=["a", "b", "c"])
    x, n = runs[["a", "b", "c"]].to_numpy(), len(runs)
    correlation = np.exp(-(((x[:, None] - x[None]) ** 2) @ fit.theta))
    bordered = np.block([[np.zeros((1, 1)), np.ones((1, n))], [np.ones((n, 1)), correlation]])
    v = np.hstack([np.ones((len(points), 1)), np.exp(-(((points.to_numpy()[:, None] - x[None]) ** 2) @ fit.theta))])
    variance = fit.sigma2 * (1 - np.einsum("mi,im->m", v, np.linalg.solve(bordered, v.T)))
    # To rounding in 1 - r'R^-1 r, which cancels near a run: so to a small share of sigma, not of the error.
    np.testing.assert_allclose(
        fit.predict(points)["std_error"], np.sqrt(variance), rtol=1e-6, atol=1e-7 * np.sqrt(fit.sigma2)
    )


@pytest.mark.parametrize(
    "function",
    [pytest.param(WAVY, id="wavy"), pytest.param(QUADRATIC, id="quadratic"), pytest.param(ADDITIVE, id="additive")],
)
def test_the_indices_are_shares_of_the_prediction_s_variance(function):
    """Against quadrature of the prediction; a smooth output has a small theta and large weights of both signs."""
    fit = fit_gaussian_process(three_inputs(function), "y")
    ranges = {"a": (0.2, 0.8)}
    np.testing.assert_allclose(fit.sensitivity(ranges), indices_by_quadrature(fit, ranges), atol=1e-6)


@pytest.mark.parametrize("theta", [1e-3, 1.0, 1e3, 1e5])
def test_the_box_averages_are_the_closed_form_integrals(theta):
    """The quadrature on panels a correlation length wide, against the error-function integrals, to rounding."""
    runs = np.random.default_rng(2).uniform(0, 1, size=8)
    means, covariances = gaussian_process._moments(runs, theta, 0.0, 1.0)

    def average(rate: float, centre: np.ndarray) -> np.ndarray:  # of exp(-rate (t - centre)^2) over [0, 1]
        root = np.sqrt(rate)
        return np.sqrt(np.pi) / (2 * root) * (special.erf(root * (1 - centre)) - special.erf(-root * centre))

    pairs = np.exp(-theta * np.subtract.outer(runs, runs) ** 2 / 2) * average(2 * theta, np.add.outer(runs, runs) / 2)
    np.testing.assert_allclose(means, average(theta, runs), rtol=1e-12)
    np.testing.assert_allclose(covariances + np.outer(means, means), pairs, rtol=1e-11, atol=1e-15)


def test_one_input_explains_all_the_variance():
    assert one_input().sensitivity().loc["x"].tolist() == pytest.approx([1.0, 1.0])


def test_the_likelihood_s_gradient_is_its_derivative():
    runs = three_inputs(WAVY)
    x, y = runs[["a", "b", "c"]].to_numpy(), runs["y"].to_numpy()
    squared, step = gaussian_process._squared_differences(x, x), 1e-3  # R is near singular at the middle theta
    for log_theta in np.log([[4.0, 3.0, 0.2], [0.5, 0.5, 0.5], [50.0, 10.0, 1.0]]):
        value, gradient = gaussian_process._criterion(log_theta, squared, y)
        central = [
            (
                gaussian_process._criterion(log_theta + e, squared, y)[0]
                - gaussian_process._criterion(log_theta - e, squared, y)[0]
            )
            / (2 * step)
            for e in step * np.eye(3)
        ]
        assert np.isfinite(value)
        np.testing.assert_allclose(gradient, central, rtol=1e-5)


def test_a_seed_gives_the_same_fit_and_the_inputs_can_be_named():
    runs = three_inputs(WAVY)
    first = fit_gaussian_process(runs, "y", ["a", "b"], random_state=11)
    again = fit_gaussian_process(runs, "y", ["a", "b"], random_state=np.random.default_rng(11))
    assert first.inputs == ("a", "b")
    np.testing.assert_array_equal(first.theta, again.theta)


@pytest.mark.parametrize(
    ("runs", "inputs", "match"),
    [
        (pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}), None, "at least three runs"),
        (pd.DataFrame({"x": [0.0, 0.5, 1.0], "y": [1.0, np.nan, 2.0]}), None, "finite number"),
        (pd.DataFrame({"y": [1.0, 3.0, 2.0]}), None, "and an input"),
        (pd.DataFrame({"x": [0.0, 0.5, 1.0], "z": [2.0, 2.0, 2.0], "y": [1.0, 3.0, 2.0]}), None, r"\['z'\] do not"),
        (pd.DataFrame({"x": [0.0, 0.5, 1.0], "y": [2.0, 2.0, 2.0]}), None, "same in every run"),
        (pd.DataFrame({"x": [0.0, 0.5, 0.5, 1.0], "y": [1.0, 3.0, 4.0, 2.0]}), None, "Two runs share their inputs"),
        (pd.DataFrame({"x": [0.0, 0.5, 1.0], "y": [1.0, 3.0, 2.0]}), ["w"], r"\['w'\] are not columns"),
    ],
)
def test_runs_the_model_cannot_interpolate_are_refused(runs, inputs, match):
    with pytest.raises(ValueError, match=match):
        fit_gaussian_process(runs, "y", inputs)


@pytest.mark.parametrize(
    ("ranges", "match"), [({"w": (0, 1)}, r"\['w'\] are not inputs"), ({"x": (1, 0)}, "low end to a higher one")]
)
def test_a_range_of_no_input_or_backwards_is_refused(ranges, match):
    with pytest.raises(ValueError, match=match):
        one_input().sensitivity(ranges)


def test_a_prediction_needs_every_input():
    with pytest.raises(ValueError, match=r"\['x'\] are not columns"):
        one_input().predict(pd.DataFrame({"z": [0.5]}))


def test_the_example_in_the_docstring_prints_what_it_shows():
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(gaussian_process, report=False)
    assert result.attempted > 0
    assert result.failed == 0
