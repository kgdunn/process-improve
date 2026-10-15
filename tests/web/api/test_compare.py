"""Comparing designs: ``api_compare_plan`` lists the designs and ``api_compare`` scores them, for ``api_candidates``.

Every design the factor count allows is generated as Design would make it, then all of them
are scored for one model on the nine measures of ``compare_designs``, with an FDS curve for
each design that can fit the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import compare_designs
from tests.web._support import FACTORS, coded, factors_named, generate_all, held_of, run_sheet


# --------------------------------------------------------------------------- the plan
@pytest.mark.parametrize(
    ("n_factors", "expected"),
    [
        (2, ["full_factorial", "plackett_burman", "ccd", "d_optimal", "i_optimal"]),
        (
            3,
            [
                "full_factorial",
                "fractional_factorial",
                "plackett_burman",
                "dsd",
                "box_behnken",
                "ccd",
                "d_optimal",
                "i_optimal",
            ],
        ),
        (8, ["fractional_factorial", "plackett_burman", "dsd", "d_optimal"]),
    ],
)
def test_the_comparison_includes_every_design_the_factor_count_allows(api, n_factors, expected):
    factors = [{"name": f"X{i}", "low": 0, "high": 1} for i in range(n_factors)]
    plan = api.ok("api_compare_plan", {"factors": factors})
    assert [d["design_type"] for d in plan["designs"]] == expected


def test_each_design_is_compared_as_step_1_would_generate_it(api, bootstrap):
    plan = api.ok("api_compare_plan", {"factors": FACTORS, "model": "interactions"})
    for item in plan["designs"]:
        options = dict(bootstrap.COMPARE_OPTIONS[item["design_type"]])
        if item["design_type"].endswith("_optimal"):
            options["model_type"] = "interactions"
        expected = {
            "design_type": item["design_type"],
            "factors": FACTORS,
            "response": "y",
            **options,
            "aim": "interactions",
        }
        assert item["spec"] == expected
        assert item["label"] == bootstrap.COMPARE_LABELS[item["design_type"]]


def test_the_comparison_s_designs_keep_the_reader_s_response(api):
    """A design taken from the comparison into the plan keeps the response named in the brief."""
    plan = api.ok("api_compare_plan", {"factors": FACTORS, "response": "Yield"})
    assert {item["spec"]["response"] for item in plan["designs"]} == {"Yield"}


@pytest.mark.parametrize("response", ["", "P"], ids=["no response", "response named as a factor"])
def test_a_factor_named_y_can_be_compared(api, response):
    """The comparison has no results, so its response takes a name no factor has when the reader's cannot be used."""
    plan = api.ok("api_compare_plan", {"factors": factors_named("y", "P", "F"), "response": response})
    assert {item["spec"]["response"] for item in plan["designs"]} == {"y1"}
    made, designs = generate_all(api, factors_named("y", "P", "F"), "main_quadratic")
    compared = api.ok("api_compare", {"model": "main_quadratic", "designs": designs})
    assert len(compared["designs"]) == len(made) == len(plan["designs"]), "every design was made and compared"


def test_the_comparison_is_on_the_book_s_model_unless_another_is_chosen(api):
    assert api.ok("api_compare_plan", {"factors": FACTORS})["model"] == "main_quadratic"


def test_optimal_designs_are_built_for_the_smallest_model_containing_the_comparison_s(api):
    plan = api.ok("api_compare_plan", {"factors": FACTORS, "model": "main_quadratic"})
    optimal = [d for d in plan["designs"] if d["design_type"].endswith("_optimal")]
    assert {d["spec"]["model_type"] for d in optimal} == {"quadratic"}
    assert "built for the quadratic model" in plan["notes"][0]
    same = api.ok("api_compare_plan", {"factors": FACTORS, "model": "interactions"})
    assert same["notes"] == []


def test_without_an_optimal_design_there_is_no_note_about_its_model(api):
    """Nine factors are more than the optimal designs take on this page, so no note explains their model."""
    factors = [{"name": f"X{i}", "low": 0, "high": 1} for i in range(9)]
    plan = api.ok("api_compare_plan", {"factors": factors, "model": "main_quadratic"})
    assert not any(d["design_type"].endswith("_optimal") for d in plan["designs"])
    assert plan["notes"] == []


def test_a_model_the_comparison_does_not_define_is_refused(api):
    assert "Model must be one of" in api.error("api_compare_plan", {"factors": FACTORS, "model": "cubic"})


# --------------------------------------------------------------------------- the comparison
def test_the_page_s_comparison_is_compare_designs_on_the_page_s_designs(api):
    made, held = generate_all(api, FACTORS, "main_quadratic")
    result = api.ok("api_compare", {"model": "main_quadratic", "designs": held})
    direct = compare_designs({m["label"]: coded(run_sheet(m["design"]), FACTORS) for m in made}, "main_quadratic")
    assert result["designs"] == list(direct.designs)
    for label in direct.designs:
        for key, value in direct.values[label].items():
            if isinstance(value, float):
                assert result["values"][label][key] == pytest.approx(value, rel=1e-9), (label, key)
        assert result["display"][label] == direct.table[label].tolist()
    assert result["values"]["Box-Behnken [n=15]"]["a"] == pytest.approx(0.2173, abs=5e-5), "the book's value"


def test_fds_curves_belong_to_the_designs_that_fit_the_model(api):
    _, held = generate_all(api, FACTORS, "main_quadratic")
    result = api.ok("api_compare", {"model": "main_quadratic", "designs": held})
    fits = {label for label in result["designs"] if result["values"][label]["fits"]}
    assert set(result["fds"]) == fits
    for curve in result["fds"].values():
        assert len(curve["fraction"]) == len(curve["variance"]) == 101
        assert np.all(np.diff(curve["variance"]) >= -1e-12), "sorted"


def test_designs_sent_back_are_checked_as_data(api):
    _, held = generate_all(api, FACTORS, "main_effects")
    held[0]["spec"] = {**held[0]["spec"], "response": "__import__('os')"}
    api.error("api_compare", {"model": "main_effects", "designs": held})


def test_each_design_is_coded_from_its_own_ranges(api):
    """The same design over other ranges is the same design in coded units, so it scores the same."""
    narrow = [{**f, "high": f["low"] + 1} for f in FACTORS]
    designs = [
        {"label": label, **held_of(api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors}))}
        for label, factors in (("wide", FACTORS), ("narrow", narrow))
    ]
    result = api.ok("api_compare", {"model": "main_effects", "designs": designs})
    wide, narrowed = (result["display"][label] for label in result["designs"])
    assert wide == narrowed


def test_a_comparison_with_a_categorical_factor_offers_the_two_level_designs(api):
    factors = [*FACTORS, {"name": "Supplier", "low": "A", "high": "B", "units": ""}]
    plan = api.ok("api_compare_plan", {"factors": factors, "model": "main_effects"})
    assert [d["design_type"] for d in plan["designs"]] == ["full_factorial", "fractional_factorial", "plackett_burman"]
