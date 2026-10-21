"""A study of one factor: two treatments compared, such as two suppliers, or a temperature at its low and its high.

The page makes a full factorial in one factor, its two levels each run several times in random order,
and takes one factor's runs as given. The design's notes say how many runs sit at each level, as an
alias sentence would mean nothing with one factor, and the design is analysed for its main effect.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS, run_sheet

SUPPLIER = {"name": "Supplier", "low": "A", "high": "B", "units": ""}


def comparison(factor: dict, **options: object) -> dict:
    """Ask for a full factorial in one factor, with the options given."""
    return {"design_type": "full_factorial", "factors": [factor], "response": "y", **options}


def test_a_full_factorial_in_one_factor_compares_its_two_levels(api):
    result = api.ok("api_make_design", comparison(SUPPLIER, n_replicates=4, n_center_points=0))
    sheet = run_sheet(result)
    assert sorted(sheet["Supplier"]) == ["A"] * 4 + ["B"] * 4
    assert result["model"] == result["spec"]["model"] == "main_effects", "one factor has no interaction"
    assert result["facts"]["notes"] == ["The runs compare Supplier at A and at B: 4 runs at each, in random order."]


def test_a_numeric_factor_s_center_runs_are_said_with_their_setting(api):
    result = api.ok("api_make_design", comparison(FACTORS[0], n_replicates=2, n_center_points=3))
    assert sorted(run_sheet(result)["T"]) == [150] * 2 + [175] * 3 + [200] * 2
    assert result["facts"]["notes"][0] == (
        "The runs compare T at 150 and at 200 degC: 2 runs at each, and 3 runs at its center, 175 degC, "
        "in random order."
    )


def test_one_factor_s_runs_as_given_keep_their_order_and_are_counted_at_each_level(api):
    runs = [["A"], ["B"], ["B"], ["A"], ["A"]]
    result = api.ok("api_make_design", {"design_type": "given", "factors": [SUPPLIER], "runs": runs})
    assert run_sheet(result)["Supplier"].tolist() == ["A", "B", "B", "A", "A"]
    assert result["model"] == "main_effects"
    assert result["facts"]["notes"][1] == "The runs compare Supplier at A and at B: 3 runs at A and 2 at B."


def test_one_factor_s_runs_as_given_at_several_settings_are_said_as_levels(api):
    runs = [[150], [160], [175], [190], [200]]
    result = api.ok("api_make_design", {"design_type": "given", "factors": [FACTORS[0]], "runs": runs})
    assert result["facts"]["notes"][1] == "The runs set T at 5 levels, from 150 to 200 degC."


@pytest.mark.parametrize(
    ("design_type", "counts"),
    [("plackett_burman", "2 to 11"), ("ccd", "2 to 7"), ("box_behnken", "3 to 7")],
)
def test_the_other_designs_still_say_the_factors_they_take(api, design_type, counts):
    error = api.error("api_make_design", {"design_type": design_type, "factors": FACTORS[:1]})
    assert error.endswith(f"takes {counts} factors on this page; you have 1.")
