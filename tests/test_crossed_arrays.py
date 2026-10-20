"""Crossed inner and outer arrays for robust design (#788 A6).

Ullmann (Table 8) crosses a 2^(4-1) inner array with three centre runs and a 2^2 outer array: 11 x 4
= 44 runs. Montgomery crosses a 2^(4-1) with a noise factor read three times at each level (the leaf
springs, Table 12.1), a 3^(4-2) with a 2^3 (the connector pull-off force, Table 12.2) and a 2^(5-2)
with a 2^(3-1) (Problem 12.18).
"""

from __future__ import annotations

import doctest
import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, cross_arrays, generate_design
from process_improve.experiments import crossed_arrays as crossed_arrays_module
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments


def experiment(source: str, name: str) -> pd.DataFrame:
    [entry] = [e for e in reference_experiments(source) if e["name"] == name]
    return pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])


def coded(names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def settings(frame: pd.DataFrame, columns: str) -> list[tuple[int, ...]]:
    """Return the runs of a table as sorted tuples of integers, to compare runs whatever their order."""
    return sorted(map(tuple, frame[list(columns)].astype(int).to_numpy().tolist()))


# --------------------------------------------------------------------------- the sources
@pytest.mark.dataset
def test_ullmann_s_table_8_crosses_eleven_inner_runs_with_four_outer_runs():
    [entry] = [e for e in reference_designs("ullmann-doe-2012") if e["name"] == "ullmann-robust-inner-outer"]
    printed = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    inner = generate_design(coded("ABCD"), "fractional_factorial", n_center_points=3, random_state=1)
    outer = generate_design(coded("HT"), "full_factorial", n_center_points=0, random_state=1)
    crossed = cross_arrays(inner, outer)
    assert len(crossed) == 44
    layout = crossed.pivot(index="inner_run", columns="outer_run", values="A")  # noqa: PD010 - one value a cell
    assert layout.shape == (11, 4)
    # At each outer run, the inner runs are Table 8's eight fraction runs and three centre runs.
    for _, at_outer_run in crossed.groupby("outer_run"):
        assert settings(at_outer_run, "ABCD") == settings(printed, "ABCD")
    # And each inner run is made at all four settings of humidity and temperature.
    for _, of_inner_run in crossed.groupby("inner_run"):
        assert settings(of_inner_run, "HT") == [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@pytest.mark.dataset
def test_montgomery_s_leaf_springs_give_table_12_1_s_means_and_variances():
    springs = experiment("montgomery-2013", "montgomery-leaf-spring-crossed-array")
    noise = pd.DataFrame({"E": [-1, -1, -1, 1, 1, 1]})  # three readings at each quench oil temperature
    crossed = cross_arrays(springs[list("ABCD")], noise)
    crossed["height"] = springs[[f"e{i}" for i in range(1, 7)]].to_numpy().ravel()
    summary = crossed.groupby("inner_run")["height"].agg(["mean", "var"])
    # The book prints two decimals of each mean and three of each variance.
    np.testing.assert_allclose(summary["mean"], springs["ybar"], atol=0.005 + 1e-9)
    np.testing.assert_allclose(summary["var"], springs["s2"], atol=0.0005 + 1e-9)


@pytest.mark.dataset
def test_the_leaf_spring_crossed_array_is_problem_8_10_s_replicated_fraction():
    """Crossing a 2^(4-1) with D = ABC and a factor E gives a 2^(5-1), here read three times."""
    springs = experiment("montgomery-2013", "montgomery-leaf-spring-crossed-array")
    crossed = cross_arrays(springs[list("ABCD")], pd.DataFrame({"E": [-1, -1, -1, 1, 1, 1]}))
    crossed["height"] = springs[[f"e{i}" for i in range(1, 7)]].to_numpy().ravel()
    fraction = experiment("montgomery-2013", "montgomery-problem-8-10").melt(id_vars=list("ABCDE"), value_name="height")
    readings = crossed.groupby(list("ABCDE"))["height"].apply(sorted)
    assert readings.equals(fraction.groupby(list("ABCDE"))["height"].apply(sorted))
    assert len(readings) == 16


@pytest.mark.dataset
def test_montgomery_s_problem_12_18_from_its_generators():
    """D = -AC and E = -BC for the inner array, H = -FG for the outer: 8 x 4 = 32 runs, as printed."""
    table = experiment("montgomery-2013", "montgomery-problem-12-18")
    inner = generate_design(coded("ABCDE"), "fractional_factorial", generators=["D=-AC", "E=-BC"], n_center_points=0)
    outer = generate_design(coded("FGH"), "fractional_factorial", generators=["H=-FG"], n_center_points=0)
    crossed = cross_arrays(inner, outer)
    assert len(crossed[list("ABCDEFGH")].drop_duplicates()) == 32
    assert settings(crossed.drop_duplicates("inner_run"), "ABCDE") == settings(table, "ABCDE")
    printed_outer = pd.DataFrame([(-1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)], columns=list("FGH"))
    assert settings(crossed.drop_duplicates("outer_run"), "FGH") == settings(printed_outer, "FGH")


@pytest.mark.dataset
def test_the_readings_of_problem_12_18_spread_out_as_printed():
    """Readings given in the order of the crossed runs pivot back to the book's table."""
    table = experiment("montgomery-2013", "montgomery-problem-12-18")
    printed_outer = pd.DataFrame([(-1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)], columns=list("FGH"))
    crossed = cross_arrays(table[list("ABCDE")], printed_outer)
    readings = ["o1", "o2", "o3", "o4"]
    crossed["defects"] = table[readings].to_numpy().ravel()
    wide = crossed.pivot(index="inner_run", columns="outer_run", values="defects")  # noqa: PD010 - pivot_table would average
    np.testing.assert_array_equal(wide.to_numpy(), table[readings].to_numpy())


@pytest.mark.dataset
def test_montgomery_s_connector_pull_off_force_has_seventy_two_readings():
    """A 3^(4-2) inner array crossed with a 2^3 outer array: each of the 72 readings has its own setting."""
    table = experiment("montgomery-2013", "montgomery-connector-pull-off-crossed-array")
    outer = pd.DataFrame(list(itertools.product((-1, 1), repeat=3)), columns=list("EFG"))
    crossed = cross_arrays(table[list("ABCD")], outer)
    assert len(crossed[list("ABCDEFG")].drop_duplicates()) == 72
    assert crossed.groupby("inner_run").size().tolist() == [8] * 9


# --------------------------------------------------------------------------- what the crossing is
def test_every_inner_run_meets_every_outer_run_in_order():
    inner = pd.DataFrame({"A": [-1, 1, 0]}, index=[10, 20, 30])
    outer = pd.DataFrame({"N": ["dry", "wet"]})
    crossed = cross_arrays(inner, outer)
    assert crossed.to_dict("list") == {
        "inner_run": [1, 1, 2, 2, 3, 3],
        "outer_run": [1, 2, 1, 2, 1, 2],
        "A": [-1, -1, 1, 1, 0, 0],
        "N": ["dry", "wet", "dry", "wet", "dry", "wet"],
    }
    assert list(crossed.index) == list(range(6))


def test_a_design_result_is_crossed_in_its_run_order_and_in_the_units_asked_for():
    inner = generate_design([Factor(name="T", low=150, high=200)], "full_factorial", n_center_points=1, random_state=3)
    outer = generate_design([Factor(name="H", low=20, high=80)], "full_factorial", n_center_points=0, random_state=3)
    in_coded = cross_arrays(inner, outer)
    in_actual = cross_arrays(inner, outer, units="actual")
    assert list(in_actual.columns) == ["inner_run", "outer_run", "T", "H"]
    assert set(in_coded["T"]) == {-1, 0, 1}
    assert set(in_actual["T"]) == {150, 175, 200}
    assert set(in_actual["H"]) == {20, 80}
    assert in_actual.loc[in_actual["outer_run"] == 1, "T"].tolist() == inner.design_actual["T"].tolist()


def test_the_docstring_example_prints_what_it_shows():
    result = doctest.testmod(crossed_arrays_module, report=False)
    assert result.attempted > 0
    assert result.failed == 0


# --------------------------------------------------------------------------- what is refused
@pytest.mark.parametrize(
    ("inner", "outer", "message"),
    [
        (pd.DataFrame({"A": [-1, 1]}), pd.DataFrame({"A": [-1, 1]}), r"both have the factors \['A'\]"),
        (pd.DataFrame({"A": []}), pd.DataFrame({"N": [-1, 1]}), "inner array has no runs or no factors"),
        (pd.DataFrame({"A": [-1, 1]}), pd.DataFrame({"RunOrder": [1, 2]}), "outer array has no runs or no factors"),
        (pd.DataFrame({"A": [-1, 1], "Block": [1, 2]}), pd.DataFrame({"N": [-1, 1]}), "inner array has a Block"),
        (
            pd.DataFrame({"A": [-1, 1]}),
            pd.DataFrame({"N": [-1, 1], "WholePlot": [1, 1]}),
            "outer array has a WholePlot",
        ),
    ],
)
def test_arrays_that_cannot_be_crossed_are_refused(inner, outer, message):
    with pytest.raises(ValueError, match=message):
        cross_arrays(inner, outer)


def test_units_must_be_coded_or_actual():
    with pytest.raises(ValueError, match="units must be 'coded' or 'actual'"):
        cross_arrays(pd.DataFrame({"A": [-1, 1]}), pd.DataFrame({"N": [-1, 1]}), units="natural")  # type: ignore[arg-type]
