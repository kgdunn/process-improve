"""Random effects: REML variance components (bounded and JMP's unbounded), expected mean squares, intervals.

The gauge capability study is Montgomery's Example 13.1 (Table 13.1): 20 parts, 3 operators, 2
measurements each. JMP fits it by REML with unbounded variance components.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import (
    expected_mean_squares,
    intraclass_correlation_interval,
    reference_catalogue,
    variance_components,
)


def _runs(name: str) -> pd.DataFrame:
    source = json.loads((reference_catalogue._CATALOGUE_DIR / "montgomery-2013.json").read_text(encoding="utf-8"))
    entry = next(e for e in source["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def gauge() -> pd.DataFrame:
    return _runs("montgomery-gauge-rr-random-factors")


def test_unbounded_reml_reproduces_jmp_negative_interaction_component(gauge):
    found = variance_components(gauge, "y ~ 1", random=["part", "operator", "part:operator"], bounded=False)
    components = found["components"]
    assert components["part"] == pytest.approx(10.279825, abs=5e-7)
    assert components["operator"] == pytest.approx(0.0149123, abs=5e-8)
    assert components["part:operator"] == pytest.approx(-0.139912, abs=5e-7)
    assert components["residual"] == pytest.approx(0.9916667, abs=5e-8)
    assert found["reml_criterion"] == pytest.approx(408.14904346, abs=5e-8)
    assert found["r_squared"] == pytest.approx(0.910717, abs=5e-7)


def test_bounded_reml_keeps_every_component_at_or_above_zero(gauge):
    found = variance_components(gauge, "y ~ 1", random=["part", "operator", "part:operator"])
    assert found["components"]["part:operator"] == 0
    assert all(value >= 0 for value in found["components"].values())
    assert sum(found["percent"].values()) == pytest.approx(100)
    assert "part:operator" not in found["std_errors"]  # on the boundary: no asymptotic standard error


def test_fixed_operators_are_tested_on_satterthwaite_degrees_of_freedom(gauge):
    """JMP's mixed model, operators fixed: F = 1.8380, R-square 0.911896, adjusted 0.91039."""
    found = variance_components(gauge, "y ~ operator", random=["part", "part:operator"], bounded=False)
    test = found["tests"][0]
    assert (test["term"], test["df_num"]) == ("operator", 2)
    assert test["f_value"] == pytest.approx(1.8380, abs=5e-5)
    assert (found["r_squared"], found["r_squared_adj"]) == pytest.approx((0.911896, 0.91039), abs=5e-6)
    assert [row["term"] for row in found["fixed_effects"]] == ["Intercept", "operator_1", "operator_2"]


def test_blups_are_the_random_effects_predicted_level_by_level():
    looms = _runs("montgomery-looms-random")
    found = variance_components(looms, "y ~ 1", random=["loom"])
    assert set(found["blups"]["loom"]) == {"1", "2", "3", "4"}
    assert sum(found["blups"]["loom"].values()) == pytest.approx(0, abs=1e-9)  # balanced: they sum to zero
    assert found["r_squared"] == found["r_squared_adj"] == pytest.approx(0.793521, abs=5e-7)
    low, high = found["intervals"]["residual"]
    assert low == pytest.approx(0.97486, abs=5e-5)  # the chi-squared interval JMP prints
    assert low < found["components"]["residual"] < high


def test_a_nested_random_factor_is_grouped_within_its_outer_factor():
    purity = _runs("montgomery-purity-nested")
    by_name = variance_components(purity, "purity ~ supplier", random=["batch(supplier)"])
    by_pair = variance_components(purity, "purity ~ supplier", random=["supplier:batch"])
    assert list(by_name["components"].values()) == pytest.approx(list(by_pair["components"].values()))
    assert list(by_name["components"]) == ["batch(supplier)", "residual"]


@pytest.mark.parametrize(
    ("restricted", "part_row"),
    [(True, {"part": 6, "Error": 1}), (False, {"part": 6, "part:operator": 2, "Error": 1})],
)
def test_the_restricted_model_drops_a_random_interaction_with_a_fixed_factor(restricted, part_row):
    """Montgomery's Tables 13.4 and 13.5: parts random, operators fixed."""
    found = expected_mean_squares({"part": 20, "operator": 3}, random=["part"], replicates=2, restricted=restricted)
    assert found["ems"]["part"] == part_row
    assert found["ems"]["operator"] == {"operator": 40, "part:operator": 2, "Error": 1}
    assert found["error_term"]["operator"] == "part:operator"
    assert found["error_term"]["part"] == ("Error" if restricted else "part:operator")


def test_nested_and_crossed_factors_in_one_design():
    """Montgomery's Example 14.2: fixtures and layouts fixed, operators random within layouts."""
    levels = {"layout": 2, "operator(layout)": 4, "fixture": 3}
    found = expected_mean_squares(levels, random=["operator"], replicates=2)
    assert found["terms"] == [
        "layout",
        "operator(layout)",
        "fixture",
        "layout:fixture",
        "operator:fixture(layout)",
        "Error",
    ]
    assert found["ems"]["fixture"] == {"fixture": 16, "operator:fixture(layout)": 2, "Error": 1}
    assert found["df"] == {
        "layout": 1,
        "operator(layout)": 6,
        "fixture": 2,
        "layout:fixture": 2,
        "operator:fixture(layout)": 12,
    }
    unrestricted = expected_mean_squares(levels, random=["operator"], replicates=2, restricted=False)
    assert unrestricted["fixed_forms"]["layout"] == ["layout", "layout:fixture"]  # Minitab's Q[1,4]
    assert unrestricted["error_term"]["operator(layout)"] == "operator:fixture(layout)"


def test_a_reduced_model_pools_the_left_out_terms_into_the_error():
    found = expected_mean_squares(
        {"part": 20, "operator": 3}, random=["part", "operator"], replicates=2, terms=["part", "operator"]
    )
    assert found["ems"]["operator"] == {"operator": 40, "Error": 1}
    with pytest.raises(ValueError, match="No such terms"):
        expected_mean_squares({"a": 2}, terms=["b"])


def test_a_term_with_no_exact_error_term_says_so():
    """Three random factors: no single mean square tests a main effect (Montgomery, section 13.6)."""
    found = expected_mean_squares({"a": 3, "b": 4, "c": 3}, random=["a", "b", "c"], replicates=2)
    assert found["error_term"]["a"] is None
    assert found["error_term"]["a:b:c"] == "Error"


def test_the_intraclass_correlation_interval_of_a_one_way_random_model():
    """The looms: L and U from the F points, the intraclass correlation between L/(1+L) and U/(1+U)."""
    found = intraclass_correlation_interval(29.73, 1.90, 3, 4)
    f0 = 29.73 / 1.90
    assert found["ratio_low"] == pytest.approx((f0 / 4.4742 - 1) / 4, abs=5e-4)
    assert found["icc_low"] == pytest.approx(found["ratio_low"] / (1 + found["ratio_low"]))
    assert 0 < found["icc_low"] < found["icc_high"] < 1
    assert found["error_low"] == pytest.approx(12 * 1.90 / 23.3367, abs=5e-4)
    assert np.isfinite(found["ratio_high"])
