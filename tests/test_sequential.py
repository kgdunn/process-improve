"""The path of steepest ascent as sources set it out, and the next design on it (``sequential.py``, #798)."""

from __future__ import annotations

import doctest
import importlib
import math

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import NoSteepestDirectionError, analyze_experiment, next_design, steepest_path
from process_improve.experiments.optimization import optimize_responses

#: Davies (1956), Table 11-21: a 2^(5-2) with x4 = x1x2x3 and x5 = -x2x3, in coded units, and the yields (%).
_DAVIES_RUNS = [
    (-1, -1, -1, -1, -1, 34.4), (-1, -1, 1, 1, 1, 51.6), (-1, 1, -1, 1, 1, 31.2), (-1, 1, 1, -1, -1, 45.1),
    (1, -1, -1, 1, -1, 54.1), (1, -1, 1, -1, 1, 62.4), (1, 1, -1, -1, 1, 50.2), (1, 1, 1, 1, -1, 58.6),
]  # fmt: skip
#: Davies's ranges (Table 11-2): x1 solvent in c.c., x2 and x5 in mol per mol of A, x3 in %, x4 in hours.
_DAVIES_RANGES = {"x1": (200, 250), "x2": (4.0, 4.5), "x3": (90, 93), "x4": (1, 2), "x5": (3.0, 3.5)}


def _model(slopes: dict[str, float], ranges: dict[str, tuple[float, float]] | None = None, **terms: float) -> dict:
    """Return a fitted model written out by hand: an intercept of 50, ``slopes``, and any other ``terms``."""
    rows = [{"term": "Intercept", "coefficient": 50.0}, *({"term": t, "coefficient": b} for t, b in slopes.items())]
    rows += [{"term": t.replace("__", ":"), "coefficient": b} for t, b in terms.items()]
    model = {"coefficients": rows, "factor_names": list(slopes)}
    if ranges:
        model["factor_coding"] = {name: {"low": low, "high": high} for name, (low, high) in ranges.items()}
    return model


@pytest.fixture(scope="module")
def davies() -> dict:
    """Davies's first-order fit, from the runs in the factors' own units, coded from their ranges as by default."""
    columns = list(_DAVIES_RANGES)
    coded = pd.DataFrame(_DAVIES_RUNS, columns=[*columns, "y"])
    actual = coded.assign(
        **{name: (low + high) / 2 + coded[name] * (high - low) / 2 for name, (low, high) in _DAVIES_RANGES.items()}
    )
    return analyze_experiment(actual, response_column="y", model=" + ".join(columns), analysis_type="coefficients")


class TestSteepestPathAsSourcesSetItOut:
    def test_davies_steps_ten_cc_of_solvent(self, davies: dict) -> None:
        """Davies, Table 11-22: the change in each level per 10 c.c. of x1, and trials (9) and (10) on the path."""
        path = steepest_path(davies, step={"x1": 10}, n_steps=7)
        change = path["increment"]["actual"]
        assert path["lead"] == "x1"
        assert change["x1"] == pytest.approx(10.0)
        assert round(change["x2"], 3) == -0.028
        assert round(change["x5"], 3) == 0.005
        # The book prints 0.456 and 0.010, worked from slopes rounded to one decimal (9.0/197.5 and 0.2/197.5).
        assert change["x3"] == pytest.approx(0.4552, abs=5e-5)
        assert change["x4"] == pytest.approx(0.0108, abs=5e-5)
        trial_9, trial_10 = path["steps"][5]["actual"], path["steps"][7]["actual"]
        assert [trial_9["x1"], round(trial_9["x2"], 2), round(trial_9["x3"], 1), round(trial_9["x4"], 1)] == [
            275.0, 4.11, 93.8, 1.6,
        ]  # fmt: skip
        assert [trial_10["x1"], round(trial_10["x2"], 2), round(trial_10["x3"], 1), round(trial_10["x4"], 1)] == [
            295.0, 4.06, 94.7, 1.6,
        ]  # fmt: skip
        # The book adds its rounded step of 0.005 to x5, so prints 3.27 and 3.28.
        assert (trial_9["x5"], trial_10["x5"]) == pytest.approx((3.2754, 3.2856), abs=5e-5)

    def test_box_hunter_and_hunter_step_five_minutes_of_time(self) -> None:
        """BHH, Table 15.2: time up 5 minutes a step, the temperature 4.50/2.35 = 1.91 coded units a step."""
        model = _model({"time": 2.35, "temperature": 4.50}, {"time": (70, 80), "temperature": (127.5, 132.5)})
        steps = steepest_path(model, step={"time": 5}, n_steps=5)["steps"][1:]
        assert [s["actual"]["time"] for s in steps] == [80.0, 85.0, 90.0, 95.0, 100.0]
        assert [round(s["actual"]["temperature"], 1) for s in steps] == [134.8, 139.6, 144.4, 149.1, 153.9]
        assert [round(s["coded"]["temperature"], 2) for s in steps] == [1.91, 3.83, 5.74, 7.66, 9.57]

    def test_schmidt_and_launsby_lead_with_the_smaller_slope(self) -> None:
        """Schmidt and Launsby, pp. 7-11: g = (4.55, -1.45) over 1.45, so R1 is (3.14, -1) coded, f1 = 307."""
        model = _model({"x1": 4.55, "x2": -1.45}, {"x1": (100, 200), "x2": (125, 175)})
        r1 = steepest_path(model, step={"x2": 25}, n_steps=1)["steps"][1]
        assert (round(r1["coded"]["x1"], 2), r1["coded"]["x2"]) == (3.14, -1.0)
        assert round(r1["actual"]["x1"]) == 307

    def test_a_coded_lead_step_without_the_coding(self) -> None:
        """A model with no coding steps its lead in coded units, and reports no actual settings."""
        path = steepest_path(_model({"A": 2.0, "B": -1.0}), step={"A": 1.0}, n_steps=2)
        assert [s["coded"] for s in path["steps"]] == [
            {"A": 0.0, "B": 0.0},
            {"A": 1.0, "B": -0.5},
            {"A": 2.0, "B": -1.0},
        ]
        assert "actual" not in path["steps"][0]
        assert "actual" not in path["increment"]

    def test_a_distance_in_coded_units_is_what_optimize_responses_steps(self) -> None:
        model = _model({"A": 4.0, "B": 3.0}, {"A": (10, 20), "B": (1, 2)})
        path = steepest_path(model, 0.5, 4)
        assert path["lead"] is None
        assert path["step_size"] == pytest.approx(0.5)
        assert path["increment"]["coded"] == pytest.approx({"A": 0.4, "B": 0.3})
        assert path == optimize_responses([model], method="steepest_ascent", step_size=0.5, n_steps=4)["steepest_path"]

    def test_descent_goes_the_other_way(self) -> None:
        model = _model({"A": 4.0, "B": 3.0})
        up, down = (steepest_path(model, 1.0, 1, goal=goal) for goal in ("maximize", "minimize"))
        assert (up["direction"], down["direction"]) == ("ascent", "descent")
        assert down["steps"][1]["coded"] == pytest.approx({"A": -0.8, "B": -0.6})
        assert down["steps"][1]["predicted_response"] < down["steps"][0]["predicted_response"]

    def test_a_base_off_centre_follows_the_slopes_there(self) -> None:
        """The model 50 + 2A + B - A^2 has slopes (1, 1) at A = 0.5 (coded), so the path goes at 45 degrees."""
        model = _model({"A": 2.0, "B": 1.0}, **{"I(A ** 2)": -1.0})
        path = steepest_path(model, math.sqrt(2) / 2, 1, base={"A": 0.5})
        assert path["steps"][0]["coded"] == {"A": 0.5, "B": 0.0}
        assert path["steps"][1]["coded"] == pytest.approx({"A": 1.0, "B": 0.5})

    def test_the_base_is_in_the_factors_own_units(self) -> None:
        model = _model({"A": 2.0, "B": 1.0}, {"A": (100, 120), "B": (1, 3)})
        path = steepest_path(model, step={"A": 10}, n_steps=1, base={"A": 115})
        assert path["steps"][0]["actual"] == {"A": 115.0, "B": 2.0}
        assert path["steps"][1]["actual"] == pytest.approx({"A": 125.0, "B": 2.5})

    def test_each_appearance_of_a_factor_in_a_term_adds_to_its_slope(self) -> None:
        """The slopes of A:I(B ** 2) and A:B at a point are those a central difference measures."""
        model = _model({"A": 1.0, "B": -2.0, "C": 0.5}, **{"A__I(B ** 2)": 0.7, "A__B": -1.3, "I(C ** 2)": 0.4})
        at = {"A": 0.3, "B": -0.6, "C": 0.2}
        path = steepest_path(model, 1e-6, 1, base=at)
        steps = path["steps"]
        moved = np.array([steps[1]["coded"][f] - at[f] for f in at])
        slopes = []
        for name in at:
            ahead, behind = dict(at), dict(at)
            ahead[name] += 1e-5
            behind[name] -= 1e-5
            values = [
                steepest_path(model, 1.0, 1, base=point)["steps"][0]["predicted_response"] for point in (ahead, behind)
            ]
            slopes.append((values[0] - values[1]) / 2e-5)
        assert moved / np.linalg.norm(moved) == pytest.approx(np.array(slopes) / np.linalg.norm(slopes), abs=1e-6)


class TestSteepestPathRefusals:
    def test_a_flat_model_has_no_steepest_direction(self) -> None:
        with pytest.raises(NoSteepestDirectionError, match="flat at the base"):
            steepest_path(_model({"A": 0.0, "B": 0.0}))
        assert issubclass(NoSteepestDirectionError, ValueError)

    def test_a_slope_left_by_cancelling_terms_is_zero(self) -> None:
        """At the top of 50 + 0.2A - 0.3A^2, A = 1/3, the slope is rounding error, 3e-17: the model is flat there."""
        with pytest.raises(NoSteepestDirectionError):
            steepest_path(_model({"A": 0.2}, **{"I(A ** 2)": -0.3}), base={"A": 1 / 3})

    def test_a_response_in_small_units_still_has_a_path(self) -> None:
        """Slopes of 1e-9 are slopes: flatness is judged against the terms, not in the units of the response."""
        path = steepest_path(_model({"A": 2e-9, "B": 1e-9}), 1.0, 1)
        assert path["direction_vector"] == pytest.approx({"A": 2 / math.sqrt(5), "B": 1 / math.sqrt(5)})

    def test_optimize_responses_still_reports_a_flat_model_as_an_error(self) -> None:
        result = optimize_responses([_model({"A": 0.0, "B": 0.0})], method="steepest_ascent")
        assert result["steepest_path"] == {"error": "All linear coefficients are zero - no steepest direction."}

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"goal": "max"}, "goal must be"),
            ({"n_steps": 0}, "n_steps must be"),
            ({"n_steps": 2.5}, "n_steps must be"),
            ({"n_steps": True}, "n_steps must be"),
            ({"step": -1.0}, "step must be a positive number"),
            ({"step": 0.0}, "step must be a positive number"),
            ({"step": float("nan")}, "step must be a positive number"),
            ({"step": True}, "step must be a positive number"),
            ({"step": {"A": 1.0, "B": 1.0}}, "one lead factor"),
            ({"step": {"D": 1.0}}, "'D', which is not a factor"),
            ({"step": {"A": 0.0}}, "The step of 'A' must be a positive number"),
            ({"step": {"C": 1.0}}, r"does not move 'C'.*\['A', 'B'\]"),
            ({"base": {"D": 1.0}}, r"base names \['D'\]"),
        ],
    )
    def test_options_the_path_cannot_take_are_refused(self, options: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            steepest_path(_model({"A": 2.0, "B": 1.0, "C": 0.0}), **options)

    def test_a_model_without_its_terms_is_refused(self) -> None:
        with pytest.raises(ValueError, match="model has no"):
            steepest_path({"factor_names": ["A"]})


class TestNextDesign:
    def test_the_next_design_is_centred_on_a_step_with_its_ranges_kept_or_scaled(self) -> None:
        model = _model({"time": 2.35, "temperature": 4.50}, {"time": (70, 80), "temperature": (127.5, 132.5)})
        path = steepest_path(model, step={"time": 5}, n_steps=5)
        new = next_design(path, 5, scale={"temperature": 2.0}, design_type="full_factorial", n_center_points=1)
        settings = new.design_actual
        assert sorted(set(settings["time"])) == [95.0, 100.0, 105.0]
        centre = path["steps"][5]["actual"]["temperature"]
        assert sorted(set(settings["temperature"])) == pytest.approx([centre - 5.0, centre, centre + 5.0])
        assert new.n_runs == 5

    def test_a_path_without_the_coding_is_set_out_in_the_first_design_s_coded_units(self) -> None:
        path = steepest_path(_model({"A": 2.0, "B": -1.0}), step={"A": 1.0}, n_steps=2)
        new = next_design(path, 2, design_type="full_factorial", n_center_points=0)
        assert sorted(set(new.design_actual["A"])) == [1.0, 3.0]
        assert sorted(set(new.design_actual["B"])) == [-2.0, 0.0]

    @pytest.mark.parametrize(
        ("at_step", "scale", "message"),
        [
            (3, None, "at_step must be a step of the path, 0 to 2"),
            (-1, None, "at_step must be"),
            (True, None, "at_step must be"),
            (1, {"C": 2.0}, r"scale names \['C'\]"),
            (1, {"A": 0.0}, "positive, finite width ratio"),
            (1, {"A": float("inf")}, "positive, finite width ratio"),
        ],
    )
    def test_steps_and_scales_the_path_does_not_hold_are_refused(self, at_step: int, scale: dict, message: str) -> None:
        path = steepest_path(_model({"A": 2.0, "B": -1.0}), step={"A": 1.0}, n_steps=2)
        with pytest.raises(ValueError, match=message):
            next_design(path, at_step, scale=scale, design_type="full_factorial")


def test_the_examples_in_the_docstrings_print_what_they_show() -> None:
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(importlib.import_module("process_improve.experiments.sequential"), report=False)
    assert result.attempted > 0
    assert result.failed == 0
