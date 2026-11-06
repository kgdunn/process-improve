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
from scipy import stats

from process_improve.experiments import reference_catalogue, reverse_yates, yates, yates_table, yates_table_3k


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
    with pytest.raises(ValueError, match="'y' cannot also be a factor"):
        yates_table(pilot, "y", factors=["T", "y"])
    with pytest.raises(ValueError, match="has no effect \\['TK'\\]"):
        reverse_yates(yates_table(pilot, "y"), keep=["T", "TK"])
    with pytest.raises(ValueError, match="Pass a table from yates_table"):
        reverse_yates(pilot, keep=["T"])


# Yates's algorithm for a 3^k factorial (#803 H4), against Box and Draper's textile experiment: a 3^3 in the
# length, amplitude and load of a test of worsted yarn, cycles to failure Y and y = log10 Y. Appendix 7G works the
# algorithm in Tables 7G.1 (Y) and 7G.2 (y); the book numbers the response (1), so its columns (2) to (4) are the
# table's (1) to (3), and its (6), (7) and (8) are ``divisor``, ``estimate`` and ``sum_sq``.

#: Table 7G.1, columns (2) to (4). The book prints -1.560 for -1,560, the z'y of x2^2:x3^2; its columns (7) and (8)
#: use -1,560.
_TEXTILE_COLUMNS = {
    "(1)": [
        *(5724, 2928, 1752, 4752, 1956, 1334, 2926, 1214, 670, 2962, 1230, 970, 2814, 804),
        *(766, 1708, 356, 270, 1482, -138, 426, 1158, 96, 338, 1024, -100, 10),
    ],
    "(2)": [
        *(10404, 8042, 4810, 5162, 4384, 2334, 1770, 1592, 934, -3972, -3418, -2256, -1992, -2048),
        *(-1438, -1056, -820, -1014, 1620, 2174, 1168, 1472, 1972, 1266, 2184, 1304, 1234),
    ],
    "(3)": [
        *(23256, 11880, 4296, -9646, -5478, -2890, 4962, 4710, 4722, -5594, -2828, -836, 1716, 554),
        *(42, -452, -206, -950, -870, -1272, -480, 608, 666, -430, -1560, -1206, 810),
    ],
}


@pytest.fixture(scope="module")
def textile() -> pd.DataFrame:
    """Return the textile experiment's runs as the catalogue holds them: coded, in standard order, x1 fastest."""
    path = reference_catalogue._CATALOGUE_DIR / "box-draper-2007.json"
    entry = next(
        e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == "box-draper-7-textile"
    )
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])[["x1", "x2", "x3", "Y", "y"]]


def test_the_textile_experiment_s_columns_and_effects(textile):
    """Table 7G.1: the columns, the effects' names, divisors, orthogonal effects, mean squares and their roots."""
    table = yates_table_3k(textile[["x1", "x2", "x3", "Y"]], "Y")
    for column, printed in _TEXTILE_COLUMNS.items():
        assert table[column].tolist() == printed, column
    assert table["effect"].tolist()[:10] == [
        *("mean", "x1", "x1^2", "x2", "x1:x2", "x1^2:x2", "x2^2", "x1:x2^2", "x1^2:x2^2", "x3"),
    ]
    assert table["effect"].iloc[-1] == "x1^2:x2^2:x3^2"
    assert table["divisor"].tolist() == [
        *(27, 18, 54, 18, 12, 36, 54, 36, 108, 18, 12, 36, 12, 8),
        *(24, 36, 24, 72, 54, 36, 108, 36, 24, 72, 108, 72, 216),
    ]
    assert table["estimate"].round(2).tolist() == [
        *(861.33, 660.0, 79.56, -535.89, -456.5, -80.28, 91.89, 130.83, 43.72, -310.78, -235.67, -23.22, 143.0),
        *(69.25, 1.75, -12.56, -8.58, -13.19, -16.11, -35.33, -4.44, 16.89, 27.75, -5.97, -14.44, -16.75, 3.75),
    ]
    # The book rounds half up: 38,365 for 554^2 / 8 = 38,364.5.
    assert table["sum_sq"].tolist() == pytest.approx(
        [
            *(20031168, 7840800, 341771, 5169184, 2500707, 232003, 455953, 616225, 206456, 1738491, 666465, 19414),
            *(245388, 38365, 74, 5675, 1768, 12535, 14017, 44944, 2133, 10268, 18482, 2568, 22533, 20201, 3038),
        ],
        abs=0.5,
    )
    # The book prints 136.0 for 666 / 24^(1/2) = 135.947 of x1:x2:x3^2, as if rounded to 135.95 first.
    assert table["standardized"].abs().tolist()[1:] == pytest.approx(
        [
            *(2800.1, 584.6, 2273.6, 1581.4, 481.7, 675.2, 785.0, 454.4, 1318.5, 816.4, 139.3, 495.4, 195.9),
            *(8.6, 75.3, 42.0, 112.0, 118.4, 212.0, 46.2, 101.3, 136.0, 50.7, 150.1, 142.1, 55.1),
        ],
        abs=0.06,
    )
    y = textile["Y"].to_numpy(dtype=float)
    assert table["sum_sq"].iloc[1:].sum() == pytest.approx(((y - y.mean()) ** 2).sum()), "26 components, one df each"


def test_the_logged_textile_experiment_s_columns(textile):
    """Table 7G.2, columns (2) and (3), on y as printed to two decimals."""
    table = yates_table_3k(textile[["x1", "x2", "x3", "y"]], "y")
    assert table["(1)"].round(2).tolist()[:9] == [9.54, 8.73, 7.94, 9.15, 8.24, 7.54, 8.57, 7.71, 6.85]
    assert table["(2)"].round(2).tolist()[-9:] == [0.02, 0.21, 0.0, 0.24, 0.59, 0.58, 0.68, 0.15, 0.42]


def test_the_estimates_are_the_least_squares_fit_on_the_orthogonal_polynomials():
    """On any 3^2 factorial, each estimate is the coefficient of its column: x, 3x^2 - 2, and their products."""
    rng = np.random.default_rng(20261106)
    levels = np.array([(a, b) for b in (-1, 0, 1) for a in (-1, 0, 1)])
    runs = pd.DataFrame(levels, columns=["A", "B"]).assign(y=rng.normal(50, 5, 9))
    table = yates_table_3k(runs.sample(frac=1, random_state=3), "y")
    # The constant, linear and quadratic orthogonal polynomials of a factor at -1, 0 and +1.
    polynomials = (np.ones_like, lambda x: x, lambda x: 3 * x**2 - 2)
    columns = [polynomials[i % 3](levels[:, 0]) * polynomials[i // 3](levels[:, 1]) for i in range(9)]
    coefficients = np.linalg.lstsq(np.column_stack(columns), runs["y"].to_numpy(), rcond=None)[0]
    assert table["estimate"].to_numpy() == pytest.approx(coefficients)
    assert table["y"].tolist() == runs["y"].tolist(), "the runs come back in standard order"


def test_the_normal_scores_are_those_of_the_components_ranks(textile):
    """Each component but the mean at P = (i - 1/2)/26 of its rank i, as Box and Draper plot them (Figure 7F.1)."""
    table = yates_table_3k(textile[["x1", "x2", "x3", "Y"]], "Y")
    assert np.isnan(table["normal_score"].iloc[0])
    components = table.iloc[1:].sort_values("standardized")
    assert components["normal_score"].to_numpy() == pytest.approx(stats.norm.ppf((np.arange(26) + 0.5) / 26))
    assert components["effect"].tolist()[-1] == "x1", "the largest component, the length of the specimen"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda runs: runs.assign(x1=runs["x1"] * 2), "coded -1, 0 and \\+1"),
        (lambda runs: runs.iloc[:26], "each of the 27 runs of the full 3\\^3 factorial"),
        (lambda runs: runs.assign(Y=runs["Y"].where(runs.index != 3)), "Every run needs its response"),
        (lambda runs: runs.rename(columns={"x3": "divisor"}), "are columns the table adds"),
    ],
)
def test_what_is_not_a_full_three_level_factorial_is_refused(textile, change, reason):
    with pytest.raises(ValueError, match=reason):
        yates_table_3k(change(textile[["x1", "x2", "x3", "Y"]]), "Y")


def test_the_other_three_level_refusals(textile):
    with pytest.raises(ValueError, match="not a column"):
        yates_table_3k(textile, "cycles")
    with pytest.raises(ValueError, match="at least one factor"):
        yates_table_3k(textile[["Y"]], "Y")
