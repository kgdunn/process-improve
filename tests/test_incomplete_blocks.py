"""Balanced incomplete block designs: intrablock, interblock and combined estimates (#789 B7).

Montgomery's Example 4.5 works the adjusted totals, Yates's interblock estimates and the combined
estimates of four catalysts in four batches of three. Davies's tyre test and rubber Youden square,
and Box, Hunter and Hunter's wear Youden square analysed as a BIBD (Table 8D.4), print further sums
and totals. Each is read from the reference catalogue.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from process_improve.experiments import bibd_analysis
from process_improve.experiments.reference_catalogue import reference_experiments


def runs(source: str, name: str) -> pd.DataFrame:
    [entry] = [e for e in reference_experiments(source=source) if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


@pytest.fixture(scope="module")
def catalyst() -> dict:
    return bibd_analysis(runs("montgomery-2013", "montgomery-catalyst-bibd"), "catalyst", "block", "y")


def as_fractions(values: dict[str, float], denominator: int) -> list[Fraction]:
    return [Fraction(round(v * denominator), denominator) for v in values.values()]


# --------------------------------------------------------------------------- Montgomery's Example 4.5
def test_the_design_is_montgomery_s(catalyst):
    """Four catalysts in four blocks of three: a = 4, b = 4, k = 3, r = 3, lambda = 2 and N = 12."""
    keys = ("n_treatments", "n_blocks", "block_size", "replicates", "concurrences", "n_runs")
    assert [catalyst[key] for key in keys] == [4, 4, 3, 3, 2, 12]


def test_the_adjusted_totals_and_the_intrablock_estimates(catalyst):
    """Q = -9/3, -7/3, -4/3, 20/3 and tau-hat = 3Q/8 = -9/8, -7/8, -4/8, 20/8; Q' = 7/3, 24/3, -31/3, 0."""
    assert as_fractions(catalyst["adjusted_treatment_totals"], 3) == [Fraction(q, 3) for q in (-9, -7, -4, 20)]
    assert as_fractions(catalyst["intrablock_effects"], 8) == [Fraction(t, 8) for t in (-9, -7, -4, 20)]
    assert as_fractions(catalyst["adjusted_block_totals"], 3) == [Fraction(q, 3) for q in (7, 24, -31, 0)]


def test_the_interblock_and_the_combined_estimates(catalyst):
    """Yates's tau-tilde_1 = [663 - (3)(3)(72.50)] / (3 - 2) = 10.50; the combined estimates -1.09 to 2.47."""
    assert catalyst["totals_of_blocks_with"]["1"] == 663
    assert list(catalyst["interblock_effects"].values()) == pytest.approx([10.50, -3.50, -0.50, -6.50])
    assert [round(v, 2) for v in catalyst["combined_effects"].values()] == [-1.09, -0.88, -0.50, 2.47]
    assert catalyst["ms_error"] == pytest.approx(0.65)
    assert catalyst["sigma2_blocks"] == pytest.approx((66.0833 / 3 - 0.65) * 3 / 8, abs=1e-4)


def test_the_sums_of_squares_are_montgomery_s_and_the_library_s_type_ii(catalyst):
    """Treatments 22.75 adjusted, 11.67 not; blocks 66.08 adjusted, 55.00 not; error 3.25; total 81.00."""
    assert catalyst["sums_of_squares"] == pytest.approx(
        {
            "total": 81.0,
            "treatments_unadjusted": 35 / 3,
            "treatments_adjusted": 22.75,
            "blocks_unadjusted": 55.0,
            "blocks_adjusted": 66.0833,
            "error": 3.25,
        },
        abs=5e-5,
    )
    table = runs("montgomery-2013", "montgomery-catalyst-bibd").astype({"y": float})
    type_ii = anova_lm(smf.ols("y ~ C(catalyst) + C(block)", data=table).fit(), typ=2)
    assert type_ii.loc["C(catalyst)", "sum_sq"] == pytest.approx(catalyst["sums_of_squares"]["treatments_adjusted"])
    assert type_ii.loc["C(block)", "sum_sq"] == pytest.approx(catalyst["sums_of_squares"]["blocks_adjusted"])
    assert type_ii.loc["Residual", "df"] == catalyst["df_error"] == 5


def test_the_intrablock_estimates_are_the_sum_coded_coefficients(catalyst):
    table = runs("montgomery-2013", "montgomery-catalyst-bibd").astype({"y": float})
    fit = smf.ols("y ~ C(catalyst, Sum) + C(block, Sum)", data=table).fit()
    coefficients = [fit.params[f"C(catalyst, Sum)[S.{level}]"] for level in "123"]
    effects = list(catalyst["intrablock_effects"].values())
    assert effects[:3] == pytest.approx(coefficients)
    assert sum(effects) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- Davies and Box, Hunter and Hunter
def test_davies_tyre_road_test():
    """Four compounds on four tyres of three: the total 61,603 and the tyres unadjusted 39,123; C's effect 30.9."""
    found = bibd_analysis(runs("davies-1956", "davies-tyre-road-test"), "Compound", "Tyre", "Wear")
    assert (found["block_size"], found["replicates"], found["concurrences"]) == (3, 3, 2)
    assert round(found["sums_of_squares"]["total"]) == 61603
    assert round(found["sums_of_squares"]["blocks_unadjusted"]) == 39123
    assert round(found["intrablock_effects"]["C"], 1) == 30.9


def test_davies_rubber_youden_square_as_a_bibd():
    """Five materials in five runs of four (lambda = 3); the total 8478.0."""
    found = bibd_analysis(runs("davies-1956", "davies-rubber-youden-square"), "Material", "Run", "Loss")
    assert (found["n_treatments"], found["block_size"], found["concurrences"]) == (5, 4, 3)
    assert found["sums_of_squares"]["total"] == pytest.approx(8478.0)


def test_box_hunter_and_hunter_s_wear_youden_square_as_a_bibd():
    """Table 8D.4: for each treatment T, B (the total of the blocks holding it) and Q = 4T - B."""
    found = bibd_analysis(runs("box-hunter-hunter-1978", "bhh-wear-youden-square"), "treatment", "run", "y")
    printed = {
        "A": (1446, 5481, 303),
        "B": (2286, 6162, 2982),
        "C": (1002, 5267, -1259),
        "D": (928, 5476, -1764),
        "E": (739, 5236, -2280),
        "F": (2206, 5883, 2941),
        "G": (1075, 5223, -923),
    }
    for label, (t, b, q) in printed.items():
        assert found["treatment_totals"][label] == t
        assert found["totals_of_blocks_with"][label] == b
        assert 4 * found["adjusted_treatment_totals"][label] == pytest.approx(q)


# --------------------------------------------------------------------------- the estimates' behaviour
def test_large_block_differences_leave_the_combined_estimates_at_the_intrablock_ones():
    """With the blocks far apart, the block totals carry little information about the treatments."""
    table = runs("montgomery-2013", "montgomery-catalyst-bibd").astype({"y": float})
    shifted = table.assign(y=table["y"] + table["block"].astype(int) * 1000.0)
    found = bibd_analysis(shifted, "catalyst", "block", "y")
    assert list(found["combined_effects"].values()) == pytest.approx(
        list(found["intrablock_effects"].values()), abs=1e-4
    )


def test_the_symmetric_design_of_seven_treatments_in_blocks_of_three():
    """The Fano plane: each pair of the 7 treatments shares exactly 1 of the 7 blocks; no block effect."""
    blocks = [(1, 2, 3), (1, 4, 5), (1, 6, 7), (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 5, 6)]
    effects = {t: float(t - 4) for t in range(1, 8)}
    table = pd.DataFrame(
        [(t, j, 50.0 + effects[t]) for j, members in enumerate(blocks, start=1) for t in members],
        columns=["treatment", "block", "y"],
    )
    found = bibd_analysis(table, "treatment", "block", "y")
    design = ("n_treatments", "n_blocks", "block_size", "replicates", "concurrences")
    assert tuple(found[key] for key in design) == (7, 7, 3, 3, 1)
    assert list(found["intrablock_effects"].values()) == pytest.approx(list(effects.values()))
    assert list(found["interblock_effects"].values()) == pytest.approx(list(effects.values()))
    assert found["sums_of_squares"]["error"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(1, 1), (2, 1), (1, 2), (2, 2), (3, 2)], "the blocks must hold the same number of treatments"),
        ([(1, 1), (2, 1), (1, 2), (2, 2)], "fewer than all 2"),
        ([(1, 1), (2, 1), (3, 2), (4, 2), (1, 3), (2, 3), (3, 4), (4, 4)], "pairs of treatments share different"),
        ([(1, 1), (1, 1), (2, 2), (3, 2)], "appears more than once in a block"),
    ],
)
def test_a_layout_that_is_not_a_bibd_is_refused(rows, message):
    table = pd.DataFrame([(t, b, float(i)) for i, (t, b) in enumerate(rows)], columns=["t", "b", "y"])
    with pytest.raises(ValueError, match=message):
        bibd_analysis(table, "t", "b", "y")


def test_a_missing_column_or_response_is_refused():
    table = pd.DataFrame({"t": [1, 2, 3], "b": [1, 1, 2], "y": [1.0, np.nan, 2.0]})
    with pytest.raises(ValueError, match="has no column"):
        bibd_analysis(table, "t", "block", "y")
    with pytest.raises(ValueError, match="must be a finite number"):
        bibd_analysis(table, "t", "b", "y")
