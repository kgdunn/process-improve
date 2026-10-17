"""Least-squares means: adjusted means of a blocked design, and Tukey's, Dunnett's and unadjusted comparisons.

The catalyst experiment is Montgomery's Example 4.5, a balanced incomplete block design that
Minitab analyses with Tukey's comparisons of the adjusted means; the wear test is Box, Hunter and
Hunter's Youden square, whose adjusted averages they print.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from process_improve.experiments import Comparisons, least_squares_means, reference_catalogue


def _runs(source: str, name: str) -> pd.DataFrame:
    path = reference_catalogue._CATALOGUE_DIR / f"{source}.json"
    entry = next(e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def catalyst() -> pd.DataFrame:
    return _runs("montgomery-2013", "montgomery-catalyst-bibd")


def test_adjusted_means_of_an_incomplete_block_design_differ_from_the_raw_averages(catalyst):
    found = least_squares_means(catalyst, "y ~ catalyst + block", "catalyst")
    assert [row["mean"] for row in found["means"]] == pytest.approx([71.375, 71.625, 72.0, 75.0])
    raw = catalyst.groupby("catalyst")["y"].mean()
    assert raw["1"] != pytest.approx(found["by_level"]["1"]["mean"])  # the raw average carries its blocks' effects


def test_tukey_p_values_match_minitab_on_the_adjusted_means(catalyst):
    found = least_squares_means(catalyst, "y ~ catalyst + block", "catalyst")
    p_values = [found["pairs"][pair]["p_value"] for pair in ("2 - 1", "3 - 1", "4 - 1", "3 - 2", "4 - 2", "4 - 3")]
    assert p_values == pytest.approx([0.9825, 0.8085, 0.0130, 0.9462, 0.0175, 0.0281], abs=5e-5)
    assert found["studentized_range"] == pytest.approx(5.2183, abs=5e-5)  # q0.05(4, 5)
    assert found["letters"] == {"1": "B", "2": "B", "3": "B", "4": "A"}


def test_youden_square_adjusted_averages():
    wear = _runs("box-hunter-hunter-1978", "bhh-wear-youden-square")
    found = least_squares_means(wear, "y ~ run + treatment", "treatment")
    means = {level: round(row["mean"], 2) for level, row in found["by_level"].items()}
    assert means == {"A": 367.43, "B": 558.79, "C": 255.86, "D": 219.79, "E": 182.93, "F": 555.86, "G": 279.86}


def test_dunnett_against_a_numeric_control_level():
    """Montgomery's Example 3.9: d0.05(3, 16) = 2.59, from the multivariate t."""
    plasma = _runs("montgomery-2013", "montgomery-plasma-etching")
    found = least_squares_means(plasma, "y ~ power", "power", Comparisons("dunnett", control=220))
    assert list(found["pairs"]) == ["160 - 220", "180 - 220", "200 - 220"]
    assert found["critical_value"] == pytest.approx(2.5923, abs=5e-4)
    assert found["family_confidence"] == 0.95
    assert "letters" not in found  # comparisons with one control do not group the levels
    with pytest.raises(ValueError, match="no level"):
        least_squares_means(plasma, "y ~ power", "power", Comparisons("dunnett", control=230))


def test_unadjusted_intervals_report_their_lower_family_confidence():
    """Montgomery's section 3.8.1: three 95% Fisher intervals hold together with 88.02% confidence."""
    chocolate = _runs("montgomery-2013", "montgomery-chocolate-antioxidant")
    found = least_squares_means(chocolate, "y ~ diet", "diet", "none")
    assert found["individual_confidence"] == pytest.approx(0.95)
    assert found["family_confidence"] == pytest.approx(0.8802, abs=5e-5)
    tukey = least_squares_means(chocolate, "y ~ diet", "diet")
    assert tukey["individual_confidence"] > 0.95 > found["family_confidence"]


def test_comparisons_within_one_level_of_another_factor():
    """Montgomery's Example 5.1: the materials compared at 70 F on the error of the whole factorial."""
    battery = _runs("montgomery-2013", "montgomery-battery-life")
    found = least_squares_means(battery, "y ~ material * C(temperature)", "material", at={"temperature": 70})
    assert [row["mean"] for row in found["means"]] == pytest.approx([57.25, 119.75, 145.75])
    assert found["df"] == 27
    assert found["studentized_range"] == pytest.approx(3.5064, abs=5e-5)
    assert found["pairs"]["2 - 1"]["half_width"] == pytest.approx(45.56, abs=5e-3)
    with pytest.raises(ValueError, match="not other factors"):
        least_squares_means(battery, "y ~ material * temperature", "material", at={"material": "1"})


def test_a_numeric_covariate_sits_at_its_average():
    runs = pd.DataFrame({"t": list("AAABBB"), "x": [0, 1, 2, 1, 2, 3.0], "y": [1.0, 2.5, 3.0, 4.0, 4.6, 6.1]})
    found = least_squares_means(runs, "y ~ t + x", "t", "none")
    fit = smf.ols("y ~ t + x", runs).fit()
    at_average = fit.predict(pd.DataFrame({"t": ["A", "B"], "x": [runs["x"].mean()] * 2}))
    assert [row["mean"] for row in found["means"]] == pytest.approx(list(at_average))


def test_connecting_letters_overlap_when_a_level_sits_between_two_that_differ():
    """A and B do not differ, nor B and C, but A and C do: B shares a letter with each."""
    runs = pd.DataFrame({"t": list("AAABBBCCC"), "y": [-0.6, 0.0, 0.6, 0.4, 1.0, 1.6, 1.4, 2.0, 2.6]})
    found = least_squares_means(runs, "y ~ t", "t", "none")
    assert found["pairs"]["B - A"]["p_value"] > 0.05 > found["pairs"]["C - A"]["p_value"]
    assert found["letters"] == {"A": "B", "B": "AB", "C": "A"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"comparisons": "scheffe"}, "comparisons must be"),
        ({"factor": "z"}, "does not hold"),
    ],
)
def test_bad_arguments_are_refused(kwargs, match):
    runs = pd.DataFrame({"t": list("AABB"), "z": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match=match):
        least_squares_means(runs, "y ~ t", kwargs.pop("factor", "t"), **kwargs)
