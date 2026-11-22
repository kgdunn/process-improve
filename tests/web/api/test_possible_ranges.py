"""Each numeric factor's possible range: the lowest and highest settings the process can take (#1322).

The worked example's path of steepest ascent takes the pressure P below zero (its third and fourth runs). With P's
lowest possible setting typed, the path stops before the first run past it, and the next study's range of P stays at
or above it. A design whose runs pass a possible range, as a rotatable design's axial runs can, says which.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS


def step(api, held: dict, simulated: dict, possible: dict) -> dict:
    """Return the step the worked example's results call for, with these possible ranges."""
    spec = {**held["spec"], "possible": possible}
    return api.ok("api_next_runs", {**held, "spec": spec, "responses": simulated["responses"]})["recommendation"]


def test_without_a_possible_range_the_path_goes_on_below_zero(api, held, simulated):
    called = step(api, held, simulated, {})
    assert [run["settings"]["P"] for run in called["path"]] == [0.62, 0.24, -0.14, -0.51]
    assert called["path_stopped"] is False


@pytest.mark.parametrize(("lowest", "kept"), [(0, [0.62, 0.24]), (0.5, [0.62])])
def test_the_path_stops_before_its_first_run_past_the_lowest_possible_setting(api, held, simulated, lowest, kept):
    called = step(api, held, simulated, {"P": [lowest, None]})
    assert [run["settings"]["P"] for run in called["path"]] == kept
    assert called["path_stopped"] is True
    assert len(called["path"][0]["settings"]) == 3, "each run keeps every factor's setting"


def test_a_path_whose_first_run_is_past_a_possible_range_lists_no_runs(api, held, simulated):
    called = step(api, held, simulated, {"T": [None, 210]})
    assert called["path"] is None
    assert called["path_stopped"] is False


def test_the_next_study_s_range_is_moved_back_inside_the_possible_range(api, held, simulated):
    ranges = {r["name"]: (r["low"], r["high"]) for r in step(api, held, simulated, {"P": [0.5, 10]})["ranges"]}
    assert ranges["P"] == (0.5, 2.5), "moved to 0.24 to 2.24, then back up whole to start at 0.5"
    assert ranges["T"] == (175.0, 225.0), "a factor with no possible range moves as before"


def test_a_design_keeps_its_possible_ranges_and_notes_the_runs_past_them(api):
    made = api.ok(
        "api_make_design",
        {"design_type": "ccd", "alpha": "rotatable", "factors": FACTORS, "possible": {"P": [0.5, None]}},
    )
    assert made["spec"]["possible"] == {"P": [0.5, None]}
    note = next(n for n in made["facts"]["notes"] if " past what it can be, " in n)
    assert note == (
        "Run 5 sets P past what it can be, at least 0.5 bar. Choose another design, or narrower ranges, so that every "
        "run can be made."
    ), "the low axial run of P, at 0.318 bar"
    face = api.ok(
        "api_make_design",
        {"design_type": "ccd", "alpha": "face_centered", "factors": FACTORS, "possible": {"P": [0.5, None]}},
    )
    assert not any(" past what it can be, " in n for n in face["facts"]["notes"]), "a face-centered design stays inside"


def test_a_highest_possible_setting_is_noted_as_at_most(api):
    asked = {"design_type": "ccd", "alpha": "rotatable", "factors": FACTORS, "possible": {"T": [None, 210]}}
    notes = api.ok("api_make_design", asked)["facts"]["notes"]
    assert any(" T past what it can be, at most 210 degC." in n for n in notes), "the high axial run of T, at 217 degC"


def test_a_blank_possible_range_is_none(api):
    made = api.ok(
        "api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "possible": {"P": [None, None]}}
    )
    assert "possible" not in made["spec"]


@pytest.mark.parametrize(
    ("possible", "refusal"),
    [
        ({"P": [1.5, None]}, "Factor P: its range starts at 1, below its lowest possible setting."),
        ({"T": [None, 180]}, "Factor T: its range ends at 200, above its highest possible setting."),
        ({"Z": [0, 1]}, "Only a numeric factor of the study has a possible range, and Z is none."),
        ({"P": ["low", None]}, "Factor P: its lowest and highest possible settings are numbers, or blank."),
        ({"P": [0]}, "Only a numeric factor of the study has a possible range, and P is none."),
    ],
)
def test_a_possible_range_the_factor_s_range_does_not_lie_inside_is_refused(api, possible, refusal):
    assert (
        api.error("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "possible": possible})
        == refusal
    )
