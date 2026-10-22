"""Runs as given: ``api_make_design`` with ``design_type="given"``, the design a run sheet holds as it was handed over.

The design catalog opens a published design this way when the page does not make its kind: the runs
are the source's, in its order and its factors' own units. The page makes none of them, works out
the largest model they estimate, and, with limits on the settings, says which runs the limits rule
out, so the reader can make a design inside them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.web._support import FACTORS, coded, factors_named, fill_workbook, held_of, quadratic, run_sheet

CUBE = [list(p) for p in itertools.product([-1, 1], repeat=3)]
AXIAL = 1.682
CCD = [
    *CUBE,
    *([s * AXIAL if j == i else 0 for j in range(3)] for i in range(3) for s in (-1, 1)),
    [0, 0, 0],
    [0, 0, 0],
]


def given(runs: list[list], factors: list[dict] | None = None, **more) -> dict:
    """Ask for runs as given, in factors from -1 to +1 named A, B and C unless others are named."""
    return {"design_type": "given", "factors": factors or factors_named("A", "B", "C"), "runs": runs, **more}


def test_the_runs_are_kept_as_given_and_in_their_order(api):
    reversed_cube = CUBE[::-1]
    result = api.ok("api_make_design", given(reversed_cube))
    sheet = run_sheet(result)
    assert sheet[["A", "B", "C"]].to_numpy().tolist() == reversed_cube
    assert sheet["Std order"].tolist() == sheet["Run order"].tolist() == list(range(1, 9))
    assert result["spec"]["design_type"] == "given"
    assert result["spec"]["options"] == {}
    assert result["facts"]["notes"][0].startswith("These runs are taken as they were given")


@pytest.mark.parametrize(
    ("runs", "model"),
    [(CCD, "quadratic"), (CUBE, "interactions"), (CUBE[:4], "main_effects")],
    ids=["central-composite", "full-factorial", "four-runs"],
)
def test_the_model_is_the_largest_the_runs_estimate_in_full(api, runs, model):
    assert api.ok("api_make_design", given(runs))["model"] == model


def test_settings_in_the_factors_own_units_and_labels_are_taken(api):
    factors = [FACTORS[0], {"name": "S", "low": "old", "high": "new", "units": ""}]
    runs = [[150, "old"], [200, "old"], [150, "new"], [200, "new"]]
    result = api.ok("api_make_design", given(runs, factors))
    sheet = run_sheet(result)
    assert coded(sheet, result["spec"]["factors"]).to_numpy().tolist() == [[-1, -1], [1, -1], [-1, 1], [1, 1]]
    assert result["model"] == "interactions"


def test_a_run_beyond_a_factor_s_range_is_kept_and_named(api):
    result = api.ok("api_make_design", given(CCD))
    assert result["outside_range"] == ["A", "B", "C"], "the axial runs, as a rotatable CCD's"


@pytest.mark.parametrize(
    ("asked", "said"),
    [
        (given([]), "The runs as given are not there"),
        (given([[1, 1]]), "Run 1 does not give one setting for each of the 3 factors"),
        (given([[1, 1, "high"]]), "Factor C: each run's setting must be a number"),
        (given([[1, 1, 1]] * 2001), "at most 2000 runs"),
        (
            given(
                [[1, "fresh"], [-1, "old"]],
                [{"name": "T", "low": -1, "high": 1}, {"name": "S", "low": "old", "high": "new"}],
            ),
            "not 'fresh'",
        ),
    ],
    ids=["no-runs", "short-run", "a-word-for-a-number", "too-many", "unknown-label"],
)
def test_runs_that_do_not_fit_the_factors_are_refused(api, asked, said):
    assert said in api.error("api_make_design", asked)


def test_the_limits_name_the_runs_they_rule_out_and_not_the_runs_beyond_a_range(api):
    """A + B <= 1 rules out the corners with A and B high and the axial runs past 1; the other axial runs meet it."""
    result = api.ok("api_make_design", given(CCD, constraints=[{"expression": "A + B <= 1"}]))
    assert result["facts"]["notes"][-1] == (
        "Runs 7, 8, 10 and 12 are outside your limits on the settings: choose the D-optimal or I-optimal design "
        "to make runs inside them."
    )
    assert result["spec"]["constraints"] == [{"expression": "A + B <= 1", "type": "linear"}]
    inside = api.ok("api_make_design", given(CUBE, constraints=[{"expression": "A + B + C <= 3"}]))
    assert inside["facts"]["notes"][-1] == "Every run is inside your limits on the settings."


def test_a_design_of_given_runs_is_analysed_and_read_back_from_its_workbook(api):
    factors = factors_named("T", "P", "F")
    result = api.ok("api_make_design", given(CCD, factors))
    analysis = api.ok("api_analyze", {"xlsx_base64": fill_workbook(result["xlsx_base64"], quadratic), "model": ""})
    assert analysis["model"] == "quadratic"
    read = api.ok("api_read_workbook", {"xlsx_base64": result["xlsx_base64"]})
    assert read["spec"]["design_type"] == "given"
    assert len(read["rows"]) == len(CCD)


def test_a_design_of_given_runs_is_renamed_as_any_other(api):
    result = api.ok("api_make_design", given(CUBE, constraints=[{"expression": "A + B <= 1"}]))
    renamed = api.ok(
        "api_relabel", {"design": held_of(result) | {"facts": result["facts"]}, "factors": factors_named("X", "B", "C")}
    )
    assert renamed["spec"]["constraints"][0]["expression"] == "X + B <= 1"
    assert renamed["columns"][2] == "X"


def test_the_factor_count_is_the_page_s(api):
    many = factors_named(*"ABCDEFGHJKLMN")
    assert "1 to 12 factors" in api.error("api_make_design", given([[0] * 13], many))
    assert np.array(
        api.ok("api_make_design", given([[0, 1], [1, 0], [1, 1]], factors_named("A", "B")))["rows"]
    ).shape == (3, 5)
