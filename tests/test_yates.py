"""Yates's algorithm and its reverse (#803 H1), against Box, Hunter and Hunter's pilot-plant example.

The pilot plant is a 2^3 factorial in temperature (T), concentration (C) and catalyst (K): Table
10.5 works the algorithm, and Tables 10A.1 and 10A.2 run it in reverse, keeping T, C and TK.
"""

from __future__ import annotations

import doctest
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from process_improve.experiments import reference_catalogue, reverse_yates, yates, yates_table


@pytest.fixture(scope="module")
def pilot() -> pd.DataFrame:
    """Return the pilot plant's runs as the catalogue holds them: coded, in standard order, each an average."""
    path = reference_catalogue._CATALOGUE_DIR / "box-hunter-hunter-1978.json"
    entry = next(
        e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == "bhh-pilot-plant-2-3"
    )
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])[["T", "C", "K", "y"]]


def test_the_pilot_plant_s_columns_effects_and_checks(pilot):
    """Table 10.5, and the checks on its columns: the crude sum of squares 34,342 times 1, 2, 4 and 8."""
    table = yates_table(pilot, "y")
    assert table["(1)"].tolist() == [132, 122, 135, 125, 12, 14, 31, 35]
    assert table["(2)"].tolist() == [254, 260, 26, 66, -10, -10, 2, 4]
    assert table["(3)"].tolist() == [514, 92, -20, 6, 6, 40, 0, 2]
    assert table["effect"].tolist() == ["mean", "T", "C", "T:C", "K", "T:K", "C:K", "T:C:K"]
    assert table["estimate"].tolist() == [64.25, 23.0, -5.0, 1.5, 1.5, 10.0, 0.0, 0.5]
    assert table.attrs["sum_of_squares"] == {"y": 34342, "(1)": 68684, "(2)": 137368, "(3)": 274736}
    assert table["sum_sq"].iloc[1:].sum() == pytest.approx(34342 - 514**2 / 8), "the effects split the corrected total"


def test_the_runs_may_come_in_any_order(pilot):
    shuffled = pilot.sample(frac=1, random_state=7).reset_index(drop=True)
    pd.testing.assert_frame_equal(yates_table(shuffled, "y"), yates_table(pilot, "y"))


def test_the_reverse_algorithm_gives_the_fitted_values_of_the_kept_effects(pilot):
    """Tables 10A.1 and 10A.2: K, TC, CK and TCK set to zero."""
    reverse = reverse_yates(yates_table(pilot, "y"), keep=["T", "C", "T:K"])
    assert reverse["kept"].tolist() == [514, 92, -20, 0, 0, 40, 0, 0]
    assert reverse["fitted"].tolist() == [60.25, 73.25, 55.25, 68.25, 50.25, 83.25, 45.25, 78.25]
    assert reverse["residual"].tolist() == [-0.25, -1.25, -1.25, -0.25, 1.75, -0.25, -0.25, 1.75]
    assert reverse.attrs["sum_of_squares"] == {
        "kept": 274660,
        "(1)": 549320,
        "(2)": 1098640,
        "(3)": 2197280,
        "fitted": 34332.5,
        "residual": 9.5,
    }


def test_the_reverse_fit_is_the_least_squares_fit():
    """On any 2^4 factorial, the reverse algorithm's fitted values are those of the model the kept effects make."""
    rng = np.random.default_rng(20261019)
    signs = np.array([[1 if i >> j & 1 else -1 for j in range(4)] for i in range(16)])
    runs = pd.DataFrame(signs, columns=list("ABCD")).assign(y=rng.normal(50, 5, 16))
    fitted = reverse_yates(yates_table(runs, "y"), keep=["A", "C", "A:B", "B:C:D"])["fitted"]
    expected = smf.ols("y ~ A + C + A:B + B:C:D", runs).fit().fittedvalues
    assert fitted.to_numpy() == pytest.approx(expected.to_numpy())


def test_keeping_every_effect_gives_the_runs_back(pilot):
    table = yates_table(pilot, "y")
    reverse = reverse_yates(table, keep=table["effect"].tolist())
    assert reverse["fitted"].tolist() == pilot["y"].tolist()
    assert reverse.attrs["sum_of_squares"]["residual"] == 0


def test_the_factors_can_be_named_and_ordered(pilot):
    """A fraction's basic factors, or a different standard order: the first factor named changes fastest."""
    table = yates_table(pilot, "y", factors=["K", "T", "C"])
    assert table["effect"].tolist() == ["mean", "K", "T", "K:T", "C", "K:C", "T:C", "K:T:C"]
    assert dict(zip(table["effect"], table["estimate"], strict=True))["K:T"] == 10.0


def test_a_half_fraction_is_worked_on_its_basic_factors(pilot):
    """The pilot plant's half with K = TC, on T and C alone: C estimates C + TK (-5 + 10), and T:C TC + K."""
    half = pilot[pilot["K"] == pilot["T"] * pilot["C"]]
    with pytest.raises(ValueError, match="give a fraction's basic factors"):
        yates_table(half, "y")
    table = yates_table(half, "y", factors=["T", "C"])
    assert table.columns.tolist()[:3] == ["T", "C", "y"], "the generated factor is left out"
    assert dict(zip(table["effect"], table["estimate"], strict=True)) == {"mean": 64.5, "T": 23.0, "C": 5.0, "T:C": 3.0}


def test_one_factor_is_one_pass():
    table = yates_table(pd.DataFrame({"A": [1, -1], "y": [7.0, 3.0]}), "y")
    assert table[["(1)", "effect", "estimate"]].to_numpy().tolist() == [[10.0, "mean", 5.0], [4.0, "A", 4.0]]
    assert reverse_yates(table, keep=[])["fitted"].tolist() == [5.0, 5.0]


def test_the_docstring_examples_print_what_they_show():
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    result = doctest.testmod(yates, report=False)
    assert result.attempted > 0
    assert result.failed == 0


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda runs: runs.assign(T=runs["T"] * 2), "coded -1 and \\+1"),
        (lambda runs: runs.iloc[:7], "once; the design has 7 runs"),
        (lambda runs: pd.concat([runs, runs.iloc[[0]]]), "the design has 9 runs, 8 of them different"),
        (lambda runs: runs.assign(y=runs["y"].where(runs.index != 3)), "Every run needs its response"),
        (lambda runs: runs.rename(columns={"C": "(1)"}), "are columns the table adds"),
    ],
)
def test_what_is_not_a_full_factorial_is_refused(pilot, change, reason):
    with pytest.raises(ValueError, match=reason):
        yates_table(change(pilot), "y")


def test_the_other_refusals(pilot):
    with pytest.raises(ValueError, match="not a column"):
        yates_table(pilot, "yield")
    with pytest.raises(ValueError, match="at least one factor"):
        yates_table(pilot[["y"]], "y")
    with pytest.raises(ValueError, match="has no effect \\['TK'\\]"):
        reverse_yates(yates_table(pilot, "y"), keep=["T", "TK"])
    with pytest.raises(ValueError, match="Pass a table from yates_table"):
        reverse_yates(pilot, keep=["T"])
