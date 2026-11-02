"""Three calculations the sources do by hand (#802): a cumulative sum about a target, the variance of a
sampling scheme, and the defects per million of a capability index.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import reference_catalogue, sampling_scheme_variance
from process_improve.monitoring import calculate_cpk, cusum_about_target


def _runs(name: str) -> pd.DataFrame:
    path = reference_catalogue._CATALOGUE_DIR / "box-hunter-hunter-1978.json"
    entry = next(e for e in json.loads(path.read_text(encoding="utf-8"))["experiments"] if e["name"] == name)
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


# --------------------------------------------------------------------------- G2
def test_the_cumulative_sum_about_a_target_of_bhh_table_17_1():
    """Twenty observations of one quantity, target 60: the deviations and their cumulative sum."""
    found = cusum_about_target(_runs("bhh-cusum-quality")["y"], target=60)
    deviations = [3.1, -1.0, -1.6, 2.4, -2.4, 0.9, -0.1, -0.3, 1.1, 0.0]
    deviations += [2.0, 2.6, 4.6, 2.9, 2.5, -1.4, 3.6, -1.1, 3.6, -0.6]
    sums = [3.1, 2.1, 0.5, 2.9, 0.5, 1.4, 1.3, 1.0, 2.1, 2.1, 4.1, 6.7, 11.3, 14.2, 16.7, 15.3, 18.9, 17.8, 21.4, 20.8]
    assert found["deviation"].round(1).tolist() == deviations
    assert found["cusum"].round(1).tolist() == sums


def test_whole_numbers_give_columns_of_floats():
    found = cusum_about_target([61, 59, 62], target=60)
    assert list(found.columns) == ["y", "deviation", "cusum"]
    assert (found.dtypes == np.float64).all()
    assert found["y"].tolist() == [61.0, 59.0, 62.0]
    assert found["cusum"].tolist() == [1.0, 0.0, 2.0]


def test_a_missing_observation_holds_the_sum():
    values = pd.Series([61.0, np.nan, 62.0], index=["a", "b", "c"])
    found = cusum_about_target(values, target=60.0)
    assert list(found.index) == ["a", "b", "c"], "the index of the observations"
    assert found["cusum"].tolist() == [1.0, 1.0, 3.0]
    assert np.isnan(found.loc["b", "deviation"])


# --------------------------------------------------------------------------- G4
def test_the_sampling_schemes_of_bhh_table_17_4():
    """V = 16.0/U + 1.3/A, from a second study's components, for U samples mixed and A tests on the mix."""
    schemes = pd.DataFrame([(u, a) for u in (1, 10, 100) for a in (1, 10, 100)], columns=["U", "A"])
    found = sampling_scheme_variance({"U": 16.0, "A": 1.3}, schemes)
    assert found["variance"].round(1).tolist() == [17.3, 16.1, 16.0, 2.9, 1.7, 1.6, 1.5, 0.3, 0.2]
    assert found["std_error"].tolist() == pytest.approx(np.sqrt(found["variance"]).tolist())
    assert "cost" not in found


def test_a_nested_scheme_counts_the_units_drawn_at_each_level():
    """Two batches, three samples from each and two tests on each sample: 2, 6 and 12 units."""
    nested = pd.DataFrame({"batch": [2], "sample": [6], "test": [12]})
    found = sampling_scheme_variance({"batch": 4.3, "sample": 16.0, "test": 1.3}, nested)
    assert found["variance"].iloc[0] == pytest.approx(4.3 / 2 + 16.0 / 6 + 1.3 / 12)


def test_the_cheapest_scheme_that_reaches_a_target():
    schemes = pd.DataFrame({"U": [4, 8, 16, 16], "A": [2, 1, 1, 4]}, index=["w", "x", "y", "z"])
    found = sampling_scheme_variance({"U": 16.0, "A": 1.3}, schemes, costs={"U": 1.0, "A": 5.0}, target=2.5)
    assert found["cost"].tolist() == [14.0, 13.0, 21.0, 36.0]
    assert found["variance"].round(3).tolist() == [4.65, 3.3, 2.3, 1.325]
    assert found.attrs["cheapest"] == "y"
    assert sampling_scheme_variance({"U": 16.0, "A": 1.3}, schemes, {"U": 1.0, "A": 5.0}, 0.1).attrs["cheapest"] is None


@pytest.mark.parametrize(
    ("components", "schemes", "costs", "target", "reason"),
    [
        ({"U": -1.0, "A": 1.3}, {"U": [1], "A": [1]}, None, None, "cannot be negative"),
        ({"U": 16.0, "A": 1.3}, {"U": [1]}, None, None, "no column \\['A'\\]"),
        ({"U": 16.0, "A": 1.3}, {"U": [0], "A": [1]}, None, None, "at least one unit"),
        ({"U": 16.0, "A": 1.3}, {"U": [1], "A": [1]}, {"U": 1.0}, None, "\\['A'\\] have no cost"),
        ({"U": 16.0, "A": 1.3}, {"U": [1], "A": [1]}, None, 1.0, "needs the costs"),
    ],
)
def test_the_refusals(components, schemes, costs, target, reason):
    with pytest.raises(ValueError, match=reason):
        sampling_scheme_variance(components, pd.DataFrame(schemes), costs, target)


# --------------------------------------------------------------------------- G5
def _process(mean: float, std: float, n: int = 4001) -> pd.DataFrame:
    """Return a column with exactly this mean and standard deviation (n - 1 in the divisor)."""
    z = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return pd.DataFrame({"x": mean + std * (z - z.mean()) / z.std(ddof=1)})


@pytest.mark.parametrize(
    ("cp", "printed", "formula"),
    [
        pytest.param(0.50, 133_600, 133_614, id="Cp-0.50"),
        pytest.param(0.77, 22_000, 20_888, id="Cpk-0.77"),
    ],
)
def test_the_defects_per_million_of_a_centred_process(cp, printed, formula):
    """Schmidt and Launsby's Table 1.3 rows: a centred process has 2 Phi(-3 Cp) x 10^6 outside its limits.

    The book's 22,000 at Cpk 0.77 is read off a table, so it is not the formula's 20,888.
    """
    found = calculate_cpk(_process(0.5, 0.01 / (3 * cp)), "x", specifications=(0.49, 0.51), trim_percentile=0)
    assert found.cpk == pytest.approx(cp)
    assert found.defects_per_million == pytest.approx(formula, abs=1)
    assert found.defects_per_million_nearer == pytest.approx(formula / 2, abs=1)
    assert abs(found.defects_per_million - printed) / printed < 0.06, printed


def test_an_off_centre_process_adds_its_two_sides():
    """Mean 0.505 with sd 0.0212: Cpk 0.079 on the upper side, and 3.5 sd from the lower limit."""
    found = calculate_cpk(_process(0.505, 0.0212), "x", specifications=(0.49, 0.51), trim_percentile=0)
    upper, lower = stats.norm.sf(0.005 / 0.0212), stats.norm.cdf(-0.015 / 0.0212)
    assert found.defects_per_million == pytest.approx((upper + lower) * 1e6)
    assert found.defects_per_million_nearer == pytest.approx(upper * 1e6)


def test_a_one_sided_specification_counts_its_one_side():
    with pytest.warns(RuntimeWarning, match="Cpk_lower"):  # no lower limit, so no lower capability
        found = calculate_cpk(_process(0.5, 0.01), "x", specifications=(np.nan, 0.53), trim_percentile=0)
    assert found.defects_per_million == pytest.approx(stats.norm.sf(3.0) * 1e6)
