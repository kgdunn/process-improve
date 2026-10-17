"""A space-filling design: ``api_make_design`` with ``design_type="maximin_lhs"``, and what the page does with it.

A maximin Latin hypercube spreads its runs through the factors' ranges for no model in particular: each
factor's range is cut into as many equal steps as there are runs, one run sits in each step, and the
runs are placed so the nearest two are as far apart as the search could make them. The page rounds each
setting to a tenth of its step, so it can be set and stays in its step, opens on ten runs a factor, and
analyses the runs with the model the study aims to learn.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.web._support import FACTORS, held_of, run_sheet

NAMES = [f["name"] for f in FACTORS]


def make(api, **options: object) -> dict:
    return api.ok("api_make_design", {"design_type": "maximin_lhs", "factors": FACTORS, **options})


def steps_taken(design: dict, runs: int) -> dict[str, list[int]]:
    """Return, for each factor, which of its ``runs`` equal steps each run's setting falls in, sorted."""
    sheet = run_sheet(design)
    taken = {}
    for f in FACTORS:
        step = (f["high"] - f["low"]) / runs
        taken[f["name"]] = sorted(np.floor((sheet[f["name"]] - f["low"]) / step).astype(int).tolist())
    return taken


def test_ten_runs_a_factor_one_in_each_step_of_every_factor_s_range(api, bootstrap):
    design = make(api)
    runs = bootstrap.SPACE_FILLING_RUNS * len(FACTORS)
    assert design["n_runs"] == runs == 30
    assert design["spec"]["options"] == {"random_state": bootstrap.RUN_ORDER_SEED, "budget": runs}
    assert steps_taken(design, runs) == {name: list(range(runs)) for name in NAMES}, "a Latin hypercube"
    assert design["facts"]["notes"][0] == (
        "Each factor takes 30 different settings, evenly spaced across its range, and the runs are placed so the "
        "nearest two are as far apart as the search could make them."
    )


def test_the_reader_gives_the_runs(api):
    design = make(api, budget=12)
    assert design["n_runs"] == 12
    assert steps_taken(design, 12) == {name: list(range(12)) for name in NAMES}


def test_each_setting_is_rounded_to_a_tenth_of_its_step_and_stays_in_it(api):
    """T steps 50/30 degC, so one decimal; P steps 2/30 bar, so three; F steps 10/30 L/min, so two."""
    sheet = run_sheet(make(api))
    for name, decimals in (("T", 1), ("P", 3), ("F", 2)):
        np.testing.assert_array_equal(sheet[name], sheet[name].round(decimals), err_msg=name)
        assert not np.array_equal(sheet[name], sheet[name].round(decimals - 1)), f"{name} needs its {decimals}"


def test_a_space_filling_design_s_runs_spread_further_apart_than_a_d_optimal_design_s(api):
    """The point of it: no two runs close, where a D-optimal design repeats its corners and centres."""

    def nearest(design: dict) -> float:
        x = run_sheet(design)[NAMES].to_numpy(float)
        x = (x - [f["low"] for f in FACTORS]) / [f["high"] - f["low"] for f in FACTORS]
        d = np.linalg.norm(x[:, None] - x[None], axis=-1) + np.eye(len(x)) * 9
        return d.min()

    optimal = api.ok("api_make_design", {"design_type": "d_optimal", "factors": FACTORS, "budget": 30})
    assert nearest(make(api)) > 0.15 > nearest(optimal) == 0.0


@pytest.mark.parametrize(
    ("aim", "model"),
    [
        ("main_effects", "main_effects"),
        ("interactions", "interactions"),
        ("main_quadratic", "quadratic"),
        (None, "quadratic"),
    ],
)
def test_learn_fits_the_model_the_study_aims_to_learn(api, aim, model):
    design = make(api, **({"aim": aim} if aim else {}))
    assert design["model"] == design["spec"]["model"] == model


def test_its_simulated_results_are_analysed_with_that_model(api):
    design = make(api, aim="interactions", budget=12)
    held = held_of(design)
    simulated = api.ok("api_simulate", {**held, "scenario": "reactor", "seed": 2})
    analysis = api.ok("api_analyze_runs", {**held, "responses": simulated["responses"]})
    assert [c["label"] for c in analysis["coefficients"]] == ["Intercept", "T", "P", "F", "T·P", "T·F", "P·F"]
    assert analysis["summary"]["df_residual"] == 12 - 7


def test_replicates_make_each_run_again_and_blocks_split_them(api):
    design = make(api, budget=8, n_replicates=2, n_blocks=2)
    sheet = run_sheet(design)
    assert design["n_runs"] == 16
    assert sheet.groupby(NAMES).size().tolist() == [2] * 8, "each of the 8 runs twice"
    assert sorted(set(sheet["Block"])) == [1, 2]


def test_it_needs_numeric_factors_and_no_limits_on_the_settings(api):
    labelled = [*FACTORS[:2], {"name": "S", "low": "A", "high": "B", "units": ""}]
    said = api.error("api_make_design", {"design_type": "maximin_lhs", "factors": labelled})
    assert said.startswith("The Space-filling (Latin hypercube) design needs numeric factors, and S is categorical")
    limited = {"design_type": "maximin_lhs", "factors": FACTORS, "constraints": [{"expression": "T + 25*P <= 250"}]}
    assert api.error("api_make_design", limited).startswith(
        "The Space-filling (Latin hypercube) design cannot keep its runs inside limits on the settings"
    )
