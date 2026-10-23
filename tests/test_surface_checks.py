"""Checks on a fitted response surface before it is read (#797).

The note on a fit too imprecise to read, third-order contrasts and pooled pure error.
"""

from __future__ import annotations

import doctest
import importlib
import math
import warnings
from collections.abc import Callable

import pandas as pd
import pytest

from process_improve.experiments import (
    Factor,
    analyze_experiment,
    generate_design,
    pooled_pure_error,
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


def test_the_examples_in_the_docstrings_print_what_they_show() -> None:
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(importlib.import_module("process_improve.experiments.surface_checks"), report=False)
    assert result.attempted > 0
    assert result.failed == 0
