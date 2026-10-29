"""The evolutionary-operation calculation sheet of a 2^2 design with a centre point (#802 G1).

Montgomery's Example 11.7 works three cycles by hand (Tables 11.23 to 11.26), and his Table 11.27
prints the factor f_{k,n} that turns the range of a cycle's differences into a standard deviation.
These tests hold the sheet to every printed number, read from the reference catalogue's record, and
the error limits to a simulation of a process whose noise is known.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.monitoring import evop_f, evop_worksheet, range_d2
from process_improve.monitoring.evop import CHANGE_IN_MEAN_LIMIT, POINTS

# The two entries of Table 11.27 that sit on a rounding boundary: f is 0.4054 and 0.2653.
BOUNDARY = {(5, 9), (10, 3)}


@pytest.fixture(scope="module")
def record() -> dict:
    """Return the catalogue's record of Montgomery's Example 11.7."""
    [entry] = [e for e in reference_experiments(source="montgomery-2013") if e["name"] == "montgomery-evop-yield"]
    return entry


@pytest.fixture(scope="module")
def cycles(record) -> list[list[float]]:
    """Return the yields of Example 11.7, a row per cycle, points (1) to (5) in each."""
    table: dict[int, dict[int, float]] = {}
    for cycle, point, _temperature, _pressure, y in record["runs"]["rows"]:
        table.setdefault(cycle, {})[point] = y
    return [[row[point] for point in range(1, 6)] for _, row in sorted(table.items())]


@pytest.fixture
def sheets(cycles) -> list[dict]:
    """Work the sheet as Montgomery does, with f_{k,n} rounded to two decimals as Table 11.27 prints it.

    Worked again for each test, not once per module: mutmut reruns only the tests it saw call the
    sheet, and a test that read a cached sheet would not be among them.
    """
    return evop_worksheet(cycles, factors=("T", "P"), range_factor_decimals=2)


@pytest.fixture(scope="module")
def table_11_27(record) -> dict[int, list[float]]:
    """Return Table 11.27's rows of f_{k,n}, n = 2 to 10, for k = 5, 9 and 10 points."""
    rows = {}
    for entry in record["expected"]:
        if entry["what"] == "f_k,n":
            k = int(re.match(r"k = (\d+)", entry["term"]).group(1))
            rows[k] = [float(value) for value in entry["value"].split()]
    return rows


# --------------------------------------------------------------------------- Montgomery's Example 11.7
def test_the_points_are_numbered_as_montgomery_numbers_them(record):
    """Figure 11.49: point (1) is the centre, 250 F and 145 psi; the corners are 5 F and 5 psi from it."""
    for _cycle, point, temperature, pressure, _y in record["runs"]["rows"]:
        assert ((temperature - 250) / 5, (pressure - 145) / 5) == POINTS[point - 1]


def test_cycle_1_has_averages_and_effects_but_no_estimate_of_the_noise(sheets, cycles):
    """Table 11.23. The change in mean is -0.02: the table prints its size, 0.02."""
    first = sheets[0]
    assert first["cycle"] == 1
    assert first["readings"] == cycles[0]
    assert first["averages"] == pytest.approx(cycles[0])
    assert first["effects"] == pytest.approx({"T": 0.45, "P": 0.25, "T:P": 0.15, "change_in_mean": -0.02})
    assert all(first[key] is None for key in ("differences", "range", "s", "sum_s", "average_s", "limits"))


def test_cycle_2_estimates_the_noise_from_the_range_of_the_differences(sheets):
    """Table 11.24, which prints the effects 0.425, 0.575 and 0.825 rounded half up: 0.43, 0.58, 0.83."""
    second = sheets[1]
    assert second["sums"] == pytest.approx([169.4, 168.8, 170.8, 168.0, 168.3])
    assert second["averages"] == pytest.approx([84.70, 84.40, 85.40, 84.00, 84.15])
    assert second["effects"] == pytest.approx({"T": 0.425, "P": 0.575, "T:P": 0.825, "change_in_mean": -0.17})
    assert second["differences"] == pytest.approx([-0.4, -0.4, -1.0, 1.0, 0.3])
    assert second["range"] == pytest.approx(2.0)
    assert second["s"] == pytest.approx(0.60), "2.0 x f_{5,2} = 2.0 x 0.30"
    assert (second["sum_s"], second["average_s"]) == pytest.approx((0.60, 0.60))
    assert second["limits"] == pytest.approx({"averages": 0.85, "effects": 0.85, "change_in_mean": 0.76}, abs=0.005)


def test_cycle_3_is_the_information_board(sheets):
    """Tables 11.25 and 11.26: S 0.56, its average 0.58, and limits 0.67 and 0.60."""
    board = sheets[-1]
    assert board["cycle"] == 3
    assert board["sums"] == pytest.approx([254.4, 252.8, 257.4, 252.9, 253.5])
    assert board["averages"] == pytest.approx([84.80, 84.27, 85.80, 84.30, 84.50], abs=0.005)
    assert board["effects"] == pytest.approx({"T": 2 / 3, "P": 13 / 15, "T:P": 19 / 30, "change_in_mean": -1 / 15})
    assert board["differences"] == pytest.approx([-0.30, 0.40, -1.20, -0.90, -1.05])
    assert board["range"] == pytest.approx(1.60)
    assert board["s"] == pytest.approx(0.56), "1.60 x f_{5,3} = 1.60 x 0.35"
    assert (board["sum_s"], board["average_s"]) == pytest.approx((1.16, 0.58))
    assert board["limits"] == pytest.approx({"averages": 0.67, "effects": 0.67, "change_in_mean": 0.60}, abs=0.005)


def test_the_board_s_interaction_of_0_64_is_worked_from_the_rounded_averages(sheets):
    """From the averages the sheet prints, 84.27 and so on, it is 0.635; from the exact ones, 0.633."""
    _y1, y2, y3, y4, y5 = (round(y, 2) for y in sheets[-1]["averages"])
    assert (y2 + y3 - y4 - y5) / 2 == pytest.approx(0.635)
    assert round(sheets[-1]["effects"]["T:P"], 3) == 0.633


def test_with_f_unrounded_the_limits_are_a_little_wider(cycles):
    """f_{5,2} = 0.304 and f_{5,3} = 0.351, so S averages 0.585 and the limit for the effects is 0.675."""
    board = evop_worksheet(cycles)[-1]
    assert board["average_s"] == pytest.approx((2.0 * evop_f(5, 2) + 1.6 * evop_f(5, 3)) / 2)
    assert board["average_s"] == pytest.approx(0.585, abs=5e-4)
    assert board["limits"]["effects"] == pytest.approx(0.675, abs=5e-4)


def test_the_factor_for_the_change_in_mean_is_montgomery_s_1_78():
    """Two standard deviations of (y2 + y3 + y4 + y5 - 4 y1) / 5, whose variance is 20 sigma^2 / 25 n."""
    assert pytest.approx(1.789, abs=5e-4) == CHANGE_IN_MEAN_LIMIT
    assert math.floor(CHANGE_IN_MEAN_LIMIT * 100) / 100 == 1.78, "the book truncates it"


# --------------------------------------------------------------------------- f_{k,n} and d2
def test_the_f_factors_of_montgomery_s_table_11_27(table_11_27):
    """25 of the 27 entries are f rounded to two decimals; the other two are 0.4054 and 0.2653, printed 0.40, 0.26."""
    assert table_11_27.keys() == {5, 9, 10}
    for k, row in table_11_27.items():
        for n, printed in enumerate(row, start=2):
            if (k, n) in BOUNDARY:
                assert evop_f(k, n) == pytest.approx(printed + 0.005, abs=5e-4)
                assert evop_f(k, n, decimals=2) == pytest.approx(printed + 0.01)
            else:
                assert evop_f(k, n, decimals=2) == printed, (k, n)


def test_the_table_is_reproduced_with_d2_to_two_decimals_and_the_root_to_three(table_11_27):
    """The rounding that evop_f's docstring names gives every one of the 27 printed entries."""
    for k, row in table_11_27.items():
        for n, printed in enumerate(row, start=2):
            assert round(round(math.sqrt((n - 1) / n), 3) / round(range_d2(k), 2), 2) == printed, (k, n)


@pytest.mark.parametrize(
    ("k", "d2"),
    [(2, 1.128), (3, 1.693), (4, 2.059), (5, 2.326), (6, 2.534), (7, 2.704), (8, 2.847), (9, 2.970), (10, 3.078)],
)
def test_d2_is_the_tabulated_mean_range_of_standard_normal_observations(k, d2):
    assert range_d2(k) == pytest.approx(d2, abs=5e-4)
    assert range_d2(np.int64(k)) == range_d2(k)


def test_d2_has_a_closed_form_for_two_and_three_observations():
    assert range_d2(2) == pytest.approx(2 / math.sqrt(math.pi))
    assert range_d2(3) == pytest.approx(3 / math.sqrt(math.pi))


# --------------------------------------------------------------------------- what the sheet estimates
def test_the_effects_are_those_of_the_factors_that_moved_the_readings():
    """A noiseless response 80 + 1.5 A - 0.5 B + 0.25 AB + 0.5 (A^2 + B^2).

    The effects are twice the coefficients, 3, -1 and 0.5. The corners sit 1 above the centre, so
    the change in mean is 4 x 1 / 5 = 0.8. Two identical cycles differ by nothing: S is 0.
    """
    readings = [80 + 1.5 * a - 0.5 * b + 0.25 * a * b + 0.5 * (a**2 + b**2) for a, b in POINTS]
    board = evop_worksheet([readings, readings], factors=("feed", "speed"))[-1]
    assert board["effects"] == pytest.approx({"feed": 3.0, "speed": -1.0, "feed:speed": 0.5, "change_in_mean": 0.8})
    assert (board["range"], board["s"]) == (0.0, 0.0)


def test_a_shift_of_every_reading_moves_the_averages_and_nothing_else(cycles):
    shifted = evop_worksheet(np.asarray(cycles) + 10.0)[-1]
    board = evop_worksheet(cycles)[-1]
    assert shifted["averages"] == pytest.approx([y + 10.0 for y in board["averages"]])
    assert shifted["effects"] == pytest.approx(board["effects"])
    assert (shifted["range"], shifted["average_s"]) == pytest.approx((board["range"], board["average_s"]))


@pytest.mark.parametrize("n", [2, 5])
def test_s_estimates_sigma_and_each_limit_is_two_standard_errors(n):
    """On a process with sigma 2 and no effects, over 4000 phases of n cycles each.

    S has mean sigma. The standard deviation of an average, of each effect and of the change in
    mean is half the limit the sheet gives for it at the true sigma: 2 sigma / sqrt(n), and
    1.79 sigma / sqrt(n) for the change in mean.
    """
    sigma = 2.0
    rng = np.random.default_rng(20261019 + n)
    boards = [evop_worksheet(rng.normal(80.0, sigma, (n, len(POINTS))))[-1] for _ in range(4000)]
    assert np.mean([board["average_s"] for board in boards]) == pytest.approx(sigma, rel=0.03)

    def two_sd(values: list[float]) -> float:
        return 2 * float(np.std(values, ddof=1))

    at_sigma = {name: limit * sigma / boards[0]["average_s"] for name, limit in boards[0]["limits"].items()}
    assert at_sigma["effects"] == pytest.approx(2 * sigma / math.sqrt(n))
    assert two_sd([board["averages"][0] for board in boards]) == pytest.approx(at_sigma["averages"], rel=0.05)
    for effect in ("A", "B", "A:B"):
        assert two_sd([board["effects"][effect] for board in boards]) == pytest.approx(at_sigma["effects"], rel=0.05)
    change = [board["effects"]["change_in_mean"] for board in boards]
    assert two_sd(change) == pytest.approx(at_sigma["change_in_mean"], rel=0.05)


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("cycles", "message"),
    [
        ([], "a row of 5 readings per cycle, one cycle at least; got shape"),
        ([[84.5, 84.2, 84.9, 84.5]], "a row of 5 readings per cycle, one cycle at least; got shape"),
        ([[84.5, 84.2, 84.9, 84.5, 84.3], [84.9, 84.6]], "each reading a number"),
        ([["high", 84.2, 84.9, 84.5, 84.3]], "each reading a number"),
        ([[84.5, 84.2, math.nan, 84.5, 84.3]], r"^Each reading must be a finite number\.$"),
        # A single cycle not wrapped in a list of cycles, and a table of five columns with no cycle in it.
        ([84.5, 84.2, 84.9, 84.5, 84.3], r"one cycle at least; got shape \(5,\)\.$"),
        (np.empty((0, 5)), r"one cycle at least; got shape \(0, 5\)\.$"),
    ],
)
def test_cycles_that_do_not_fill_the_sheet_are_refused(cycles, message):
    with pytest.raises(ValueError, match=message):
        evop_worksheet(cycles)


@pytest.mark.parametrize("factors", [("A", "A"), ("A", "change_in_mean"), ("A", "B", "C")])
def test_factor_names_that_would_overwrite_an_effect_are_refused(factors):
    with pytest.raises(ValueError, match="two different names"):
        evop_worksheet([[1.0, 2.0, 3.0, 4.0, 5.0]], factors=factors)


@pytest.mark.parametrize("k", [1, 0, True, 2.0])
def test_a_range_of_fewer_than_two_whole_observations_is_refused(k):
    """2.0 is refused even after d2(2) is cached: the cache sits behind the check."""
    range_d2(2)
    with pytest.raises(ValueError, match="a whole number of at least 2"):
        range_d2(k)


@pytest.mark.parametrize("n", [1, 0, 2.5, True])
def test_f_for_a_cycle_without_a_previous_one_is_refused(n):
    with pytest.raises(ValueError, match="the cycle, must be a whole number of at least 2"):
        evop_f(5, n)
