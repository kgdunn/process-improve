"""The two-way split-plot (strip-plot) search: rows and columns as crossed random effects (#788 A7).

Goos and Jones (Section 11.4, Tables 11.8 and 11.9) give twelve runs in four rows (x1, x2) and four
columns (x3), three runs in each, as D-optimal for the main effects and robust to the variance
ratios from 0.1 to 10. On its own cells the best design, found by the search and confirmed by
enumerating every setting of the rows and columns, is a little better: it assigns the four row
settings in another order and makes x3 orthogonal to x1. Their 48-run battery cell design (16 lots
by 6 curing cycles) is reached by the search.
"""

from __future__ import annotations

import doctest
import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, evaluate_design
from process_improve.experiments import designs_split_plot as split_plot_module
from process_improve.experiments.designs_split_plot import StripPlotLayout, StripPlotOptions, strip_plot_design
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments

CODED = [Factor(name=n, low=-1, high=1) for n in ("x1", "x2", "x3")]
RATIOS = [(1.0, 1.0), (0.1, 0.1), (10.0, 10.0), (0.1, 10.0), (10.0, 0.1)]


@pytest.fixture(scope="module")
def twelve_runs() -> pd.DataFrame:
    [entry] = [
        e for e in reference_designs("goos-jones-2011") if e["name"] == "goos-jones-two-way-split-plot-twelve-runs"
    ]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    return runs.assign(row=runs["row"] - 1, column=runs["column"] - 1)


def log_det(cells: np.ndarray, settings: np.ndarray, ratios: tuple[float, float]) -> np.ndarray:
    """Return log|X'V^-1 X| of the main-effects model, for one design or a stack of them."""
    same_row = cells[:, 0][:, None] == cells[:, 0][None, :]
    same_column = cells[:, 1][:, None] == cells[:, 1][None, :]
    weight = np.linalg.inv(np.eye(len(cells)) + ratios[0] * same_row + ratios[1] * same_column)
    x = np.concatenate([np.ones((*settings.shape[:-1], 1)), settings], axis=-1)
    return np.linalg.slogdet(np.einsum("...ri,rs,...sj->...ij", x, weight, x))[1]


def best_by_enumeration(cells: np.ndarray, ratios: tuple[float, float]) -> float:
    """Return the largest log|M| over all 2^12 settings of x1 and x2 by row and x3 by column."""
    bits = np.array(list(itertools.product((-1.0, 1.0), repeat=12)))
    settings = np.stack([bits[:, 0:4][:, cells[:, 0]], bits[:, 4:8][:, cells[:, 0]], bits[:, 8:12][:, cells[:, 1]]], -1)
    return float(log_det(cells, settings, ratios).max())


def search(cells: np.ndarray, ratios: tuple[float, float]) -> tuple[np.ndarray, dict]:
    layout = StripPlotLayout(("x1", "x2"), ("x3",), tuple(map(tuple, cells.tolist())))
    options = StripPlotOptions(row_variance_ratio=ratios[0], column_variance_ratio=ratios[1])
    return strip_plot_design(CODED, layout, options, random_state=1)


# --------------------------------------------------------------------------- the sources
@pytest.mark.dataset
@pytest.mark.parametrize("ratios", RATIOS)
def test_the_search_reaches_the_best_twelve_run_design_on_the_printed_cells(twelve_runs, ratios):
    cells = twelve_runs[["row", "column"]].to_numpy()
    printed = log_det(cells, twelve_runs[["x1", "x2", "x3"]].to_numpy(dtype=float), ratios)
    _, meta = search(cells, ratios)
    assert meta["log_det_information"] == pytest.approx(best_by_enumeration(cells, ratios), abs=1e-9)
    assert meta["log_det_information"] > printed


@pytest.mark.dataset
def test_the_printed_twelve_run_design_is_99_8_percent_d_efficient(twelve_runs):
    """At ratios of 1, (|M| / |M_best|)^(1/4) = 0.998; the best design makes x3 orthogonal to x1."""
    cells = twelve_runs[["row", "column"]].to_numpy()
    printed = twelve_runs[["x1", "x2", "x3"]].to_numpy(dtype=float)
    best, meta = search(cells, (1.0, 1.0))
    assert round(float(np.exp((log_det(cells, printed, (1.0, 1.0)) - meta["log_det_information"]) / 4)), 3) == 0.998
    assert (printed[:, 0] @ printed[:, 2], best[:, 0] @ best[:, 2]) == (-4.0, 0.0)


@pytest.mark.dataset
def test_the_printed_incidence_is_as_good_as_any_with_three_runs_in_each_row_and_column(twelve_runs):
    """Every 4 x 4 grid with three runs in each row and column leaves out one cell per row: 24 of them."""
    cells = twelve_runs[["row", "column"]].to_numpy()
    on_printed = best_by_enumeration(cells, (1.0, 1.0))
    others = [
        np.array([(r, c) for r in range(4) for c in range(4) if c != left_out[r]])
        for left_out in itertools.permutations(range(4))
    ]
    assert max(best_by_enumeration(grid, (1.0, 1.0)) for grid in others) == pytest.approx(on_printed, abs=1e-9)


@pytest.mark.dataset
def test_the_search_reaches_the_battery_cell_design():
    """Sixteen lots by six curing cycles, 48 runs, D-optimal for the main effects and two-factor interactions."""
    [entry] = [e for e in reference_experiments("goos-jones-2011") if e["name"] == "goos-jones-battery-cell"]
    runs = pd.DataFrame(entry["runs"]["rows"], columns=entry["runs"]["columns"])
    names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    cells = tuple(zip((runs["row"] - 1).tolist(), (runs["column"] - 1).tolist(), strict=True))
    layout = StripPlotLayout(("x1", "x2", "x3", "x4"), ("x5", "x6"), cells)
    options = StripPlotOptions(model_type="interactions", n_starts=30)
    design, _ = strip_plot_design([Factor(name=n, low=-1, high=1) for n in names], layout, options, random_state=1)
    model = " + ".join(names) + " + " + " + ".join(f"{a}:{b}" for a, b in itertools.combinations(names, 2))
    groups = {"row": runs["row"].astype(str), "column": runs["column"].astype(str)}

    def efficiency(settings: pd.DataFrame) -> float:
        frame = settings.assign(**groups)
        found = evaluate_design(frame, model=model, metric="d_efficiency", random_effects={"row": 1.0, "column": 1.0})
        return found["d_efficiency"]

    printed = efficiency(runs[names])
    assert round(printed, 2) == 33.47
    assert efficiency(pd.DataFrame(design, columns=names)) == pytest.approx(printed, rel=1e-9)


# --------------------------------------------------------------------------- what the search returns
def test_row_and_column_factors_are_constant_within_their_rows_and_columns():
    cells = tuple((r, c) for r in range(3) for c in range(4))
    factors = [*CODED, Factor(name="x4", low=-1, high=1)]
    layout = StripPlotLayout(("x1",), ("x2", "x3"), cells)
    design, meta = strip_plot_design(factors, layout, StripPlotOptions(model_type="interactions"), random_state=2)
    frame = pd.DataFrame(design, columns=["x1", "x2", "x3", "x4"]).assign(row=meta["row"], column=meta["column"])
    assert frame.groupby("row")["x1"].nunique().max() == 1
    assert frame.groupby("column")[["x2", "x3"]].nunique().max().max() == 1
    assert len(set(frame["x4"])) > 1  # a run factor is set for each run
    assert meta["backend"] == "strip_plot_exchange"
    assert (meta["row_factors"], meta["column_factors"]) == (["x1"], ["x2", "x3"])


@pytest.mark.parametrize("criterion", ["a_optimal", "i_optimal"])
def test_the_a_and_i_criteria_are_searched_too(criterion):
    cells = tuple((r, c) for r in range(4) for c in range(4))
    layout = StripPlotLayout(("x1",), ("x2",), cells)
    options = StripPlotOptions(criterion=criterion, model_type="interactions", n_starts=3)
    _, meta = strip_plot_design(CODED, layout, options, random_state=0)
    assert meta["optimality_criterion"] == criterion
    assert np.isfinite(meta["trace_criterion"])


def test_categorical_factors_take_their_labels():
    cells = tuple((r, c) for r in range(4) for c in range(3))
    factors = [Factor(name="gas", type="categorical", levels=["A", "B", "C"]), *CODED[:2]]
    design, _ = strip_plot_design(
        factors, StripPlotLayout(("gas",), ("x1",), cells), StripPlotOptions(), random_state=0
    )
    assert set(design[:, 0]) <= {"A", "B", "C"}


def test_the_same_seed_gives_the_same_design():
    cells = tuple((r, c) for r in range(4) for c in range(4))
    layout = StripPlotLayout(("x1", "x2"), ("x3",), cells)
    first, _ = strip_plot_design(CODED, layout, StripPlotOptions(n_starts=3), random_state=7)
    second, _ = strip_plot_design(CODED, layout, StripPlotOptions(n_starts=3), random_state=7)
    np.testing.assert_array_equal(first, second)


def test_the_docstring_example_prints_what_it_shows():
    result = doctest.testmod(split_plot_module, report=False)
    assert result.attempted > 0
    assert result.failed == 0


# --------------------------------------------------------------------------- what is refused
GRID = tuple((r, c) for r in range(4) for c in range(4))


@pytest.mark.parametrize(
    ("layout", "options", "message"),
    [
        (StripPlotLayout(("x1",), ("x2",), GRID), StripPlotOptions(criterion="e_optimal"), "searched for on"),
        (StripPlotLayout(("x1",), ("x2",), GRID), StripPlotOptions(row_variance_ratio=-1.0), "finite number"),
        (StripPlotLayout(("x9",), ("x2",), GRID), StripPlotOptions(), "must be different factors"),
        (StripPlotLayout(("x1",), ("x1",), GRID), StripPlotOptions(), "must be different factors"),
        (StripPlotLayout(("x1",), ("x2",), ((0, -1), (1, 0))), StripPlotOptions(), "whole numbers from 0"),
        (StripPlotLayout(("x1",), ("x2",), ((0, 0), (1, 1))), StripPlotOptions(), "cannot estimate"),
    ],
)
def test_layouts_and_options_the_search_cannot_use_are_refused(layout, options, message):
    with pytest.raises(ValueError, match=message):
        strip_plot_design(CODED, layout, options)
