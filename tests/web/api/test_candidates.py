"""The designs that can learn the brief: ``api_candidates`` makes and scores them in one call, for Design.

It is ``api_compare_plan`` and ``api_make_design`` for each design, then ``api_compare`` on
them all, at the power for the response's smallest change worth finding. With five factors or
more and a model beyond main effects, the optimal designs' slow search waits to be asked for.
"""

from __future__ import annotations

import pytest

from tests.web._support import FACTORS, factors_named, held_of


def test_every_design_is_made_as_plan_makes_it_and_all_are_scored(api):
    plan = api.ok("api_compare_plan", {"factors": FACTORS, "model": "main_quadratic"})
    listed = api.ok("api_candidates", {"factors": FACTORS, "model": "main_quadratic"})
    assert [m["design_type"] for m in listed["made"]] == [d["design_type"] for d in plan["designs"]]
    assert listed["waiting"] == []
    assert listed["notes"] == plan["notes"]
    for item in listed["made"]:
        again = api.ok("api_make_design", item["spec"])
        assert item["design"]["rows"] == again["rows"], item["design_type"]
        assert "xlsx_base64" not in item["design"], "the workbook is written when it is asked for"
        assert isinstance(item["warnings"], list)
    scored = api.ok(
        "api_compare",
        {"model": "main_quadratic", "designs": [{"label": m["label"], **held_of(m["design"])} for m in listed["made"]]},
    )
    assert listed["result"]["values"] == scored["values"]


def test_a_design_s_warnings_stay_with_it(api):
    """An even number of factors and the DSD's own center run leave its staged analysis no error df."""
    listed = api.ok("api_candidates", {"factors": factors_named(*"ABCD"), "model": "main_effects"})
    dsd = next(m for m in listed["made"] if m["design_type"] == "dsd")
    others = [m for m in listed["made"] if m["design_type"] != "dsd"]
    assert dsd["warnings"] == api.reply("api_make_design", dsd["spec"])["warnings"]
    assert all(m["warnings"] == api.reply("api_make_design", m["spec"])["warnings"] for m in others)


def test_power_is_for_the_response_s_smallest_change_worth_finding(api):
    described = {"label": "Purity", "units": "%", "noise": 1.5, "smallest_change": 6.0}
    listed = api.ok("api_candidates", {"factors": FACTORS, "model": "main_effects", "response_info": described})
    assert listed["result"]["effect_size"] == 2.0, "a change of 6 is a coefficient of 3, two noise standard deviations"
    for item in listed["made"]:
        assert item["design"]["spec"]["response_info"] == described, "a design taken into the plan keeps it"
        assert item["design"]["spec"]["aim"] == "main_effects"


@pytest.mark.parametrize("model", ["main_quadratic", "quadratic", "interactions"])
def test_with_five_factors_the_optimal_designs_wait_for_the_reader(api, model):
    factors = factors_named(*"ABCDE")
    listed = api.ok("api_candidates", {"factors": factors, "model": model})
    assert [w["design_type"] for w in listed["waiting"]] == ["d_optimal", "i_optimal"]
    assert not {m["design_type"] for m in listed["made"]} & {"d_optimal", "i_optimal"}


def test_a_screening_aim_needs_no_wait_for_the_optimal_designs(api):
    listed = api.ok("api_candidates", {"factors": factors_named(*"ABCDE"), "model": "main_effects"})
    assert listed["waiting"] == []
    assert {"d_optimal", "i_optimal"} <= {m["design_type"] for m in listed["made"]}


@pytest.mark.slow
def test_searching_makes_the_optimal_designs_too(api):
    listed = api.ok("api_candidates", {"factors": factors_named(*"ABCDE"), "model": "main_quadratic", "search": True})
    assert listed["waiting"] == []
    assert {"d_optimal", "i_optimal"} <= {m["design_type"] for m in listed["made"]}


def test_a_design_that_cannot_be_made_is_left_out_with_a_note(api, bootstrap, monkeypatch):
    real = bootstrap.make_design

    def refusing(spec: dict) -> dict:
        if spec["design_type"] == "ccd":
            raise ValueError("the axial runs could not be placed")
        return real(spec)

    monkeypatch.setattr(bootstrap, "make_design", refusing)
    listed = api.ok("api_candidates", {"factors": FACTORS, "model": "main_quadratic"})
    assert "ccd" not in [m["design_type"] for m in listed["made"]]
    assert "CCD could not be made: the axial runs could not be placed" in listed["notes"]
