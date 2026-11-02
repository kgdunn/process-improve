"""ORTH-augmented definitive screening designs (Jones and Nachtsheim 2013, Section 3; #788 A4).

The four designs printed in the paper's Figure 1(b), m = 4 continuous factors with c = 1 to 4
two-level categorical factors, come from the catalogue. The paper's Table 4 gives the run sizes
and the D-efficiency of each DSD-augmented design relative to the ORTH-augmented one, and Table 6
the correlations left among more than four categorical factors.
"""

from __future__ import annotations

import doctest

import numpy as np
import pytest

from process_improve.experiments import designs_response_surface, generate_design
from process_improve.experiments.designs_response_surface import dispatch_dsd, dsd_run_count
from process_improve.experiments.factor import Factor
from process_improve.experiments.reference_catalogue import reference_designs
from process_improve.tool_spec import execute_tool_call


def factors(m: int, c: int) -> list[Factor]:
    """``m`` continuous factors and ``c`` two-level categorical ones, in that order, as the paper lists them."""
    continuous = [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(m)]
    return continuous + [Factor(name=f"z{i + 1}", type="categorical", levels=["a", "b"]) for i in range(c)]


def information(design: np.ndarray) -> np.ndarray:
    """Return X'X of the main-effects model."""
    x = np.column_stack([np.ones(len(design)), design])
    return x.T @ x


def d_efficiency(design: np.ndarray) -> float:
    """Return |X'X|^(1/p) / n for the main-effects model, as Table 4 normalises it."""
    return float(np.linalg.det(information(design)) ** (1 / (design.shape[1] + 1)) / len(design))


def off_diagonal(design: np.ndarray) -> int:
    """Count the nonzero off-diagonal entries of the main-effects X'X, both triangles."""
    m = information(design)
    return int(np.count_nonzero(np.abs(m - np.diag(np.diag(m))) > 1e-9))


@pytest.mark.dataset
@pytest.mark.parametrize("c", [1, 2, 3, 4])
def test_the_printed_designs_are_built(c):
    """Figure 1(b): the same runs, in another order, for each number of categorical factors."""
    [entry] = [e for e in reference_designs("jones-nachtsheim-2013") if e["name"] == f"jn2013-orth-augment-m4-c{c}"]
    printed = np.array([row[1:] for row in entry["runs"]["rows"]], dtype=float)
    built, meta = dispatch_dsd(factors(4, c), categorical_method="orth")
    assert sorted(map(tuple, built)) == sorted(map(tuple, printed))
    assert meta["categorical_method"] == "orth"


@pytest.mark.parametrize(("m", "c"), [(m, c) for m in (4, 5, 6, 8, 12) for c in (1, 2, 3, 4)])
def test_up_to_four_categorical_factors_every_main_effect_is_orthogonal(m, c):
    design, _ = dispatch_dsd(factors(m, c), categorical_method="orth")
    assert off_diagonal(design) == 0
    assert len(design) == dsd_run_count(m + c, c, "orth")


@pytest.mark.parametrize(("m", "c", "runs"), [(4, 1, 14), (4, 2, 16), (4, 3, 20), (4, 4, 20), (5, 1, 14), (12, 4, 36)])
def test_the_run_sizes_are_table_4_s(m, c, runs):
    """n_ORTH: the DSD-augmented size for one categorical factor, two more runs for two to four."""
    assert dsd_run_count(m + c, c, "orth") == runs
    assert len(dispatch_dsd(factors(m, c), categorical_method="orth")[0]) == runs


@pytest.mark.parametrize(("c", "printed"), [(1, 0.98), (2, 1.04), (3, 1.03), (4, 1.01)])
def test_dsd_augment_against_orth_augment_is_table_4_s_ratio(c, printed):
    """D-efficiency of the DSD-augmented design over the ORTH-augmented one, m = 4, to the two printed decimals."""
    dsd = dispatch_dsd(factors(4, c))[0]
    orth = dispatch_dsd(factors(4, c), categorical_method="orth")[0]
    assert d_efficiency(dsd) / d_efficiency(orth) == pytest.approx(printed, abs=0.005)


@pytest.mark.parametrize(("m", "c"), [(m, c) for m in (4, 8, 12) for c in (6, 8, 10)])
def test_more_than_four_categorical_factors_leave_table_6_s_correlations(m, c):
    """n_nz = 2 n_B (c - 2 n_B - 2), n_B = c // 4: only factors sharing a column of the centre block correlate."""
    design, _ = dispatch_dsd(factors(m, c), categorical_method="orth")
    n_b = c // 4
    assert off_diagonal(design) == 2 * n_b * (c - 2 * n_b - 2)


@pytest.mark.parametrize(("c", "fraction"), [(6, 0.0364), (8, 0.0513), (10, 0.0762)])
def test_table_6_s_fractions_for_four_continuous_factors(c, fraction):
    design, _ = dispatch_dsd(factors(4, c), categorical_method="orth")
    p = 4 + c + 1
    assert off_diagonal(design) / (p * (p - 1)) == pytest.approx(fraction, abs=5e-5)


def test_the_price_is_aliasing_with_second_order_effects():
    """DSD-augment keeps every main effect clear of the second-order effects; ORTH-augment does not."""
    columns = range(6)
    second = [(i, j) for i in columns for j in columns if i <= j and not (i == j and i >= 4)]
    for method, clear in (("dsd", True), ("orth", False)):
        design, _ = dispatch_dsd(factors(4, 2), categorical_method=method)
        products = np.column_stack([design[:, i] * design[:, j] for i, j in second])
        assert bool(np.allclose(design.T @ products, 0)) is clear


def test_generate_design_takes_the_method():
    result = generate_design(factors(4, 2), "dsd", categorical_method="orth")
    assert result.n_runs == 16
    assert set(result.design_actual["z1"]) == {"a", "b"}
    assert result.metadata["categorical_method"] == "orth"


def test_a_budget_adds_fake_factors_as_before():
    """22 runs fit the order-8 conference matrix (2 x 8 + 4), whose two extra columns are left out."""
    design, meta = dispatch_dsd(factors(4, 2), budget=22, categorical_method="orth")
    assert (len(design), meta["conference_order"], meta["fake_factors"]) == (20, 8, 2)
    assert off_diagonal(design) == 0


def test_without_categorical_factors_the_method_changes_nothing():
    plain = dispatch_dsd(factors(6, 0))[0]
    np.testing.assert_array_equal(dispatch_dsd(factors(6, 0), categorical_method="orth")[0], plain)


def test_the_tool_takes_the_method():
    spec = {
        "factors": [{"name": f"x{i}", "low": -1, "high": 1} for i in range(4)]
        + [{"name": "z", "type": "categorical", "levels": ["a", "b"]}],
        "design_type": "dsd",
        "categorical_method": "orth",
    }
    assert execute_tool_call("generate_design", spec)["n_runs"] == 14


def test_another_design_type_refuses_orth():
    with pytest.raises(ValueError, match="categorical_method='orth' chooses how a DSD"):
        generate_design(factors(3, 1), "fractional_factorial", categorical_method="orth")


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="categorical_method must be one of"):
        dispatch_dsd(factors(4, 1), categorical_method="orthogonal")


def test_the_run_count_example_prints_what_it_shows():
    """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner()
    for test in finder.find(dsd_run_count, "dsd_run_count", globs=vars(designs_response_surface)):
        runner.run(test)
    assert runner.failures == 0
    assert runner.tries > 0
