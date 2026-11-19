"""The study's significance level: every request carries it, and every test and interval follows it.

The page offers 0.01, 0.05 and 0.1 in its Settings. A request without one runs at 0.05, as every study did before
the setting; the confidence of an interval is one minus the level.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS
from tests.web.api.test_best_settings import aimed
from tests.web.api.test_compare import generate_all

LEVELS = (0.1, 0.05, 0.01)


def widths(api, held: dict, simulated: dict, **level: float) -> dict[str, float]:
    """Return the width of each coefficient's interval, by its label, with the worked example analysed at ``level``."""
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"], **level})
    return {c["label"]: c["ci_high"] - c["ci_low"] for c in analysis["coefficients"] if c["ci_low"] is not None}


def test_a_stricter_level_widens_every_interval(api, held, simulated):
    loose, usual, strict = (widths(api, held, simulated, significance_level=level) for level in LEVELS)
    assert usual
    assert all(loose[term] < usual[term] < strict[term] for term in usual)


def test_a_request_without_a_level_runs_at_0_05_whatever_ran_before(api, held, simulated):
    usual = widths(api, held, simulated, significance_level=0.05)
    widths(api, held, simulated, significance_level=0.01)
    assert widths(api, held, simulated) == usual


@pytest.mark.parametrize("level", [0.2, 0, True, "0.05", None])
def test_a_level_the_page_does_not_offer_is_refused(api, held, simulated, level):
    reply = api.reply("api_analyze_runs", {**held, "responses": simulated["responses"], "significance_level": level})
    assert not reply["ok"]
    assert "Unknown significance level" in reply["error"]


@pytest.mark.parametrize(("level", "runs"), [(0.1, 10), (0.05, 20), (0.01, 100)])
def test_the_best_settings_say_their_confidence_and_a_prediction_is_one_run_for_each_equal_share(
    api, held, simulated, level, runs
):
    request = {**aimed(held, "maximize"), "responses": simulated["responses"], "significance_level": level}
    assert api.ok("api_best_settings", request)["level"] == pytest.approx(1 - level)
    [fit] = api.ok("api_predictors", {**request, "models": {"Yield": "main_effects"}})["responses"]
    assert len(fit["run_quantiles"]) == runs


def test_the_power_of_a_design_is_worked_out_at_the_study_s_level(api):
    _, designs = generate_all(api, FACTORS, "main_quadratic")
    request = {"model": "main_quadratic", "designs": designs}
    usual, strict = (api.ok("api_compare", {**request, "significance_level": level}) for level in (0.05, 0.01))
    assert (usual["significance_level"], strict["significance_level"]) == (0.05, 0.01)
    for label, at_usual in usual["values"].items():
        if at_usual["power_main"] is not None:
            assert strict["values"][label]["power_main"] < at_usual["power_main"], label
