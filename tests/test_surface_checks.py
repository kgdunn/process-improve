"""Checks on a fitted response surface before it is read (#797).

The note on a fit too imprecise to read, third-order contrasts, pooled pure error, and the confidence region of
the stationary point.
"""

from __future__ import annotations

import doctest
import importlib
import itertools
import math
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import (
    Factor,
    analyze_experiment,
    generate_design,
    optimize_responses,
    pooled_pure_error,
    stationary_point_region,
    third_order_contrasts,
)

#: Box, Hunter and Hunter, Table 15.4: the composite design in two blocks, the cube with two centre runs and then
#: the axial runs at plus or minus 1.4 (coded, as printed) with two more. The response is the yield, %.
_COMPOSITE = pd.DataFrame(
    [
        (1, -1, -1, 78.8), (1, 1, -1, 84.5), (1, -1, 1, 91.2), (1, 1, 1, 77.4), (1, 0, 0, 89.7), (1, 0, 0, 86.8),
        (2, -1.4, 0, 83.3), (2, 1.4, 0, 81.2), (2, 0, -1.4, 81.2), (2, 0, 1.4, 79.5), (2, 0, 0, 87.0), (2, 0, 0, 86.0),
    ],
    columns=["Block", "time", "temperature", "y"],
)  # fmt: skip
#: Box, Hunter and Hunter, Table 15.1: the first design of the investigation, a 2^2 with three centre runs.
_FIRST = pd.DataFrame(
    [(-1, -1, 54.3), (1, -1, 60.3), (-1, 1, 64.6), (1, 1, 68.0), (0, 0, 60.3), (0, 0, 64.3), (0, 0, 62.3)],
    columns=["time", "temperature", "y"],
)
_FACTORS = ["time", "temperature"]


class TestThePrecisionNote:
    """Box and Wetz: read a fitted surface only when its fitted values change by several standard errors (#797 R1)."""

    @staticmethod
    def _summary(runs: pd.DataFrame, **options: object) -> dict:
        return analyze_experiment(runs, response_column="y", analysis_type="coefficients", **options)["model_summary"]

    def test_bhh_exercise_15_8_the_plane_is_estimated_well_enough_to_read(self) -> None:
        """The fitted values span 2(2.35 + 4.50) = 13.7, which is 14 times their average standard error, sqrt(0.964)."""
        summary = self._summary(_FIRST, model="main_effects", sigma=1.5)
        assert summary["adequate_precision"] == pytest.approx(13.7 / math.sqrt(3 * 1.5**2 / 7))
        assert "model_note" not in summary

    def test_the_same_plane_against_ten_times_the_noise_is_noted(self) -> None:
        """With sigma = 10 the span of 13.7 is 2.09 average standard errors, below Design-Expert's 4."""
        note = self._summary(_FIRST, model="main_effects", sigma=10.0)["model_note"]
        assert note.startswith("Adequate precision is 2.09: the fitted values change over the runs by less than 4")
        assert "not estimated precisely enough to read its contours or its optimum" in note

    def test_without_a_sigma_the_ratio_is_read_against_the_residual(self) -> None:
        """The residual's s = 1.59 leaves the plane at 13.2 standard errors: no note."""
        summary = self._summary(_FIRST, model="main_effects")
        assert summary["adequate_precision"] == pytest.approx(13.16, abs=0.01)
        assert "model_note" not in summary

    def test_a_saturated_fit_has_no_standard_error_to_compare_with(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # statsmodels divides by the 0 residual df
            summary = self._summary(_FIRST.iloc[:4], model="interactions")
        assert math.isnan(summary["adequate_precision"])
        assert "model_note" not in summary

    def test_it_follows_the_note_on_the_default_model(self) -> None:
        """Twelve Plackett-Burman runs fall back to main effects; read against sigma = 100 they are noise as well."""
        factors = [Factor(name=f"X{i}", low=-1, high=1) for i in range(1, 9)]
        design = generate_design(factors, design_type="plackett_burman", n_center_points=0).design
        runs = design.drop(columns="RunOrder").assign(y=range(12))
        with pytest.warns(UserWarning, match="main-effects model was fitted instead"):
            note = self._summary(runs, sigma=100.0)["model_note"]
        fallback, _, precision = note.partition(" Adequate precision is ")
        assert fallback.endswith("to choose the interactions the data support.")
        assert precision.startswith("0.104: ")


class TestThirdOrderContrasts:
    def test_box_hunter_and_hunter_s_contrasts_at_alpha_root_two(self) -> None:
        """BHH (15.20, 15.21): b111 - b122 = 1.28 +- 1.06 and b222 - b112 = -1.93 +- 1.06, with sigma = 1.5."""
        runs = _COMPOSITE.replace({1.4: math.sqrt(2), -1.4: -math.sqrt(2)})
        out = third_order_contrasts(runs, "y", _FACTORS, sigma=1.5)["contrasts"]
        assert (round(out["time"]["contrast"], 2), round(out["temperature"]["contrast"], 2)) == (1.28, -1.93)
        assert round(out["time"]["std_error"], 2) == round(out["temperature"]["std_error"], 2) == 1.06

    def test_the_printed_runs_at_one_point_four_with_the_noise_from_the_centre_runs(self) -> None:
        """At the axial distance the table prints, 1.4, and with the pure error of the centre runs within blocks."""
        out = third_order_contrasts(_COMPOSITE, "y", _FACTORS)
        time, temperature = out["contrasts"]["time"], out["contrasts"]["temperature"]
        assert (time["axial_slope"], time["factorial_slope"]) == pytest.approx((-2.1 / 2.8, -2.025))
        assert (time["contrast"], temperature["contrast"]) == pytest.approx((1.275, -1.9321), abs=5e-5)
        # the pairs 89.7, 86.8 and 87.0, 86.0, each about its own block's mean: 4.705 on 2 df
        assert (out["sigma"], out["sigma_df"]) == (pytest.approx(math.sqrt(4.705 / 2)), 2)
        assert time["std_error"] == pytest.approx(out["sigma"] * math.sqrt(2 / 2.8**2 + 1 / 4))

    def test_without_a_block_column_the_centre_runs_are_one_set(self) -> None:
        out = third_order_contrasts(_COMPOSITE.rename(columns={"Block": "day"}), "y", _FACTORS)
        assert out["sigma_df"] == 3

    def test_under_a_second_order_model_the_contrasts_are_zero(self) -> None:
        runs = _composite_of(lambda a, b: 60 + 2 * a - b + 1.5 * a * b - 3 * a**2 - b**2)
        for contrast in third_order_contrasts(runs, "y", ["a", "b"], sigma=1.0)["contrasts"].values():
            assert contrast["contrast"] == pytest.approx(0.0, abs=1e-12)

    def test_a_contrast_is_alpha_squared_less_one_times_b_iii_less_the_b_ijj(self) -> None:
        """At alpha = sqrt(2), with b111 = 0.5 and b122 = 0.3: (2 - 1)(0.5) - 0.3 = 0.2 along a."""
        runs = _composite_of(lambda a, b: 10 + 2 * a + b + 0.5 * a**3 + 0.3 * a * b**2)
        out = third_order_contrasts(runs, "y", ["a", "b"], sigma=1.0)["contrasts"]
        assert out["a"]["contrast"] == pytest.approx(0.2)
        assert out["b"]["contrast"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("runs", "options", "message"),
        [
            (_COMPOSITE, {"factor_cols": ["time"]}, "two factors or more"),
            (_COMPOSITE.query("time != 1 and time != -1"), {}, "no factorial cube"),
            (_FIRST, {}, "'time' needs an axial run on each side"),
            (_COMPOSITE.drop_duplicates(_FACTORS), {}, "no replicated runs"),
            (_COMPOSITE, {"sigma": 0.0}, "sigma must be a positive number"),
        ],
    )
    def test_what_is_not_a_composite_design_is_refused(self, runs: pd.DataFrame, options: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            third_order_contrasts(runs, "y", **{"factor_cols": _FACTORS, **options})


def _composite_of(response: Callable[[float, float], float]) -> pd.DataFrame:
    """Return a two-factor composite design at alpha = sqrt(2), with two centre runs, and the response given."""
    a = math.sqrt(2)
    points = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-a, 0), (a, 0), (0, -a), (0, a), (0, 0), (0, 0)]
    return pd.DataFrame([(x, z, response(x, z)) for x, z in points], columns=["a", "b", "y"])


class TestPooledPureError:
    def test_box_hunter_and_hunter_pool_the_centre_runs_of_three_designs(self) -> None:
        """BHH (15.16): (2 x 4.00 + 1 x 4.21)/3 = 4.07; (15.24) with the axial block's pair, 3.18, s = 1.78."""
        blocks = [_COMPOSITE[_COMPOSITE["Block"] == block] for block in (1, 2)]
        two = pooled_pure_error([_FIRST, blocks[0]], "y", _FACTORS)
        assert round(two["ms_pure_error"], 2) == 4.07
        assert two["per_design"] == [{"ss": pytest.approx(8.0), "df": 2}, {"ss": pytest.approx(4.205), "df": 1}]
        three = pooled_pure_error([_FIRST, *blocks], "y", _FACTORS)
        assert (round(three["ms_pure_error"], 2), round(three["std_dev"], 2), three["df_pure_error"]) == (3.18, 1.78, 4)

    def test_runs_of_two_designs_at_one_setting_are_not_replicates(self) -> None:
        """Each design's centre runs are compared with their own mean, not with the other design's."""
        one = pd.DataFrame({"x": [0, 0], "y": [10.0, 12.0]})
        other = pd.DataFrame({"x": [0, 0], "y": [20.0, 22.0]})
        assert pooled_pure_error([one, other], "y", ["x"])["ms_pure_error"] == pytest.approx(2.0)

    @pytest.mark.parametrize(
        ("designs", "message"),
        [([], "at least one design"), ([_COMPOSITE.drop_duplicates(_FACTORS)], "No design has replicated runs")],
    )
    def test_nothing_to_pool_is_refused(self, designs: list, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            pooled_pure_error(designs, "y", _FACTORS)


def _composite_runs(k: int = 2, centre: int = 5) -> pd.DataFrame:
    """Return the factor settings of a rotatable composite design in k factors, with ``centre`` centre runs."""
    cube = list(itertools.product([-1.0, 1.0], repeat=k))
    reach = (2**k) ** 0.25
    axial = [tuple(sign * reach * (i == j) for j in range(k)) for i in range(k) for sign in (-1, 1)]
    return pd.DataFrame(cube + axial + [(0.0,) * k] * centre, columns=[f"x{i + 1}" for i in range(k)])


#: Second-order surfaces in x1 and x2 with a known stationary point: b0, b (linear) and B (b_ii on the diagonal, half
#: b_ij off it).
_MAXIMUM = (80.0, np.array([2.0, -1.5]), np.array([[-3.0, 0.5], [0.5, -2.0]]))
_SADDLE = (80.0, np.array([2.0, -1.5]), np.array([[-3.0, 0.5], [0.5, 2.0]]))
#: One draw of the noise, sigma = 1, for the 13 runs of the two-factor composite design.
_NOISE = np.random.default_rng(13).normal(0.0, 1.0, 13)


def _surface(x: np.ndarray, truth: tuple) -> np.ndarray:
    b0, b, quadratic = truth
    return b0 + x @ b + np.einsum("ni,ij,nj->n", x, quadratic, x)


def _true_point(truth: tuple) -> np.ndarray:
    return np.linalg.solve(2 * truth[2], -truth[1])


class TestStationaryPointRegion:
    """Box and Hunter (1954): where the fitted gradient does not differ significantly from zero (#797 R5)."""

    def test_the_gradient_is_zero_at_the_estimate_which_is_inside(self) -> None:
        runs = _composite_runs().assign(y=lambda d: _surface(d.to_numpy(), _MAXIMUM) + _NOISE)
        region = stationary_point_region(runs, "y", ["x1", "x2"])
        assert region["classification"] == "maximum"
        assert float(region["statistic"](region["stationary_point"])) == pytest.approx(0.0, abs=1e-20)
        assert bool(region["contains"](region["stationary_point"]))
        assert region["critical_value"] == pytest.approx(stats.f.ppf(0.95, 2, 7))
        assert (region["sigma_df"], region["confidence_level"]) == (7, 0.95)

    def test_box_hunter_and_hunter_s_composite_design_is_nearly_a_ridge(self) -> None:
        """The stationary point is 3.7 and 3.0 coded units out, along a ridge, so neither is known to within 15."""
        runs = _COMPOSITE.drop(columns="Block").replace({1.4: math.sqrt(2), -1.4: -math.sqrt(2)})
        region = stationary_point_region(runs, "y", _FACTORS)
        found = optimize_responses(
            [analyze_experiment(runs, response_column="y", model="quadratic", analysis_type="coefficients")],
            method="stationary_point",
        )["stationary_point"]
        assert region["stationary_point"] == pytest.approx(found["stationary_point_coded"])
        assert min(region["std_error"].values()) > 15
        # along the ridge, through the estimate and the centre, and on beyond the estimate
        ridge = np.outer([0.0, 1.0, 2.0], list(region["stationary_point"].values()))
        assert region["contains"](ridge).all()
        assert not region["contains"]([1.0, 0.0])

    @pytest.mark.slow
    @pytest.mark.parametrize("truth", [_MAXIMUM, _SADDLE], ids=["maximum", "saddle"])
    def test_the_region_covers_the_true_stationary_point_as_often_as_it_says(self, truth: tuple) -> None:
        """The F test is exact at the true point under normal errors, so 95% of regions cover it, for a saddle too."""
        runs = _composite_runs()
        mean, true_point = _surface(runs.to_numpy(), truth), _true_point(truth)
        rng = np.random.default_rng(797)
        regions = (
            stationary_point_region(runs.assign(y=mean + rng.normal(0, 1.0, len(runs))), "y", ["x1", "x2"])
            for _ in range(800)
        )
        covered = [bool(region["contains"](true_point)) for region in regions]
        assert np.mean(covered) == pytest.approx(0.95, abs=0.02)  # 800 regions: a standard error of 0.008

    @pytest.mark.slow
    def test_the_delta_method_standard_errors_are_the_spread_of_the_estimates(self) -> None:
        runs = _composite_runs()
        mean = _surface(runs.to_numpy(), _MAXIMUM)
        rng = np.random.default_rng(1954)
        fits = [
            stationary_point_region(runs.assign(y=mean + rng.normal(0, 0.5, len(runs))), "y", ["x1", "x2"])
            for _ in range(800)
        ]
        points = np.array([list(fit["stationary_point"].values()) for fit in fits])
        reported = np.array([list(fit["std_error"].values()) for fit in fits])
        assert points.std(axis=0, ddof=1) == pytest.approx(np.median(reported, axis=0), rel=0.1)

    def test_with_one_factor_the_region_is_fieller_s_interval(self) -> None:
        """The x at which (b1 + 2 b11 x)^2 = F s^2 var(b1 + 2 b11 x) / s^2: the roots of a quadratic in x."""
        x = np.repeat([-1.0, 0.0, 1.0], 3)
        runs = pd.DataFrame({"x": x, "y": 10 + x - 2 * x**2 + np.tile([0.4, -0.3, -0.1], 3)})
        region = stationary_point_region(runs, "y", ["x"])
        model = np.column_stack([np.ones_like(x), x, x**2])
        (_, b1, b11), *_ = np.linalg.lstsq(model, runs["y"], rcond=None)
        v = np.linalg.inv(model.T @ model)[1:, 1:] * region["sigma"] ** 2  # cov(b1, b11)
        f = region["critical_value"]
        # (b1 + 2 b11 x)^2 - f (v11 + 4 v12 x + 4 v22 x^2) = 0
        roots = np.sort(np.roots([4 * b11**2 - 4 * f * v[1, 1], 4 * b1 * b11 - 4 * f * v[0, 1], b1**2 - f * v[0, 0]]))
        assert region["statistic"](roots[:, None]) == pytest.approx([f, f])
        assert region["contains"]([[roots[0] + 1e-6], [roots[1] - 1e-6]]).all()
        assert not region["contains"]([[roots[0] - 1e-6], [roots[1] + 1e-6]]).any()
        assert region["stationary_point"]["x"] == pytest.approx(-b1 / (2 * b11))

    def test_along_a_stationary_ridge_the_point_is_not_determined(self) -> None:
        """80 - 1.5 (x1 - x2)^2: every point of the line x1 = x2 is a maximum."""
        runs = _composite_runs().assign(y=lambda d: 80 - 1.5 * (d["x1"] - d["x2"]) ** 2)
        region = stationary_point_region(runs, "y", ["x1", "x2"], sigma=1.0)
        assert region["classification"] == "stationary_ridge"
        assert region["std_error"] is None
        assert region["stationary_point"] == pytest.approx({"x1": 0.0, "x2": 0.0}, abs=1e-12)
        assert region["contains"]([[5.0, 5.0], [-20.0, -20.0]]).all()
        assert not region["contains"]([2.0, -2.0])

    def test_a_grid_gives_a_mask_of_its_shape_and_the_units_do_not_move_the_region(self) -> None:
        """A temperature in degC and a pressure in pascals, 1e5 from zero: the region is the coded one, mapped."""
        runs = _composite_runs().assign(y=lambda d: _surface(d.to_numpy(), _SADDLE) + _NOISE)
        actual = runs.assign(x1=150 + 10 * runs["x1"], x2=1e5 + 500 * runs["x2"])
        grid = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(-1, 1, 11))
        coded = stationary_point_region(runs, "y", ["x1", "x2"])
        mask = coded["contains"]({"x1": grid[0], "x2": grid[1]})
        assert mask.shape == (11, 21)
        assert 0 < mask.sum() < mask.size
        uncoded = stationary_point_region(actual, "y", ["x1", "x2"])
        assert uncoded["contains"]({"x1": 150 + 10 * grid[0], "x2": 1e5 + 500 * grid[1]}).tolist() == mask.tolist()
        assert uncoded["stationary_point"] == pytest.approx(
            {"x1": 150 + 10 * coded["stationary_point"]["x1"], "x2": 1e5 + 500 * coded["stationary_point"]["x2"]}
        )
        assert uncoded["std_error"] == pytest.approx(
            {"x1": 10 * coded["std_error"]["x1"], "x2": 500 * coded["std_error"]["x2"]}
        )
        assert coded["classification"] == uncoded["classification"] == "saddle_point"

    def test_blocks_are_fitted_and_cost_a_degree_of_freedom(self) -> None:
        """BHH's composite design blocks orthogonally at sqrt(2): the point does not move, the residual loses a df."""
        runs = _COMPOSITE.replace({1.4: math.sqrt(2), -1.4: -math.sqrt(2)})
        blocked = stationary_point_region(runs, "y", _FACTORS)
        plain = stationary_point_region(runs.drop(columns="Block"), "y", _FACTORS)
        assert blocked["stationary_point"] == pytest.approx(plain["stationary_point"])
        assert (blocked["sigma_df"], plain["sigma_df"]) == (5, 6)

    def test_a_given_sigma_reads_the_chi_square_and_needs_no_residual(self) -> None:
        """Six runs fit the six terms exactly: there is no residual, but a prior sigma still gives the region."""
        runs = pd.DataFrame(
            [(-1, -1, 5.0), (1, -1, 7.0), (-1, 1, 6.0), (1, 1, 6.5), (0, 0, 9.0), (1, 0, 8.0)],
            columns=["x1", "x2", "y"],
        )
        with pytest.raises(ValueError, match="no residual"):
            stationary_point_region(runs, "y", ["x1", "x2"])
        region = stationary_point_region(runs, "y", ["x1", "x2"], sigma=0.5, significance_level=0.1)
        assert region["critical_value"] == pytest.approx(stats.chi2.ppf(0.9, 2) / 2)
        assert (region["sigma"], region["sigma_df"], region["confidence_level"]) == (0.5, None, 0.9)

    @pytest.mark.parametrize(
        ("runs", "options", "message"),
        [
            (_FIRST, {"factor_cols": []}, "at least one factor"),
            (_FIRST, {}, "cannot estimate every term"),
            (_FIRST.assign(y=[*_FIRST["y"][:-1], None]), {}, "missing value"),
            (_COMPOSITE.assign(y=5.0), {}, "response does not vary"),
            (_COMPOSITE, {"sigma": -1.0}, "sigma must be a positive number"),
            (_COMPOSITE, {"significance_level": 1.0}, "between 0 and 1"),
        ],
    )
    def test_what_cannot_give_a_region_is_refused(self, runs: pd.DataFrame, options: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            stationary_point_region(runs, "y", **{"factor_cols": _FACTORS, **options})

    def test_a_setting_must_hold_every_factor(self) -> None:
        region = stationary_point_region(_COMPOSITE, "y", _FACTORS)
        with pytest.raises(ValueError, match="2 values in the last axis"):
            region["contains"]([0.0, 0.0, 0.0])


def test_the_examples_in_the_docstrings_print_what_they_show() -> None:
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(importlib.import_module("process_improve.experiments.surface_checks"), report=False)
    assert result.attempted > 0
    assert result.failed == 0
