"""Tests for ``process_improve.experiments.omars_trade_off``.

The capability classes are arithmetic on the foldover estimability frontier, so
the reference values are written down rather than computed: a change to the
formula has to disagree with a table, not with itself. The frontier itself is
proved against the rank law in ``tests/test_omars_estimability.py``; here we
check that the trade-off table reports it faithfully.

Nothing in this module needs a solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments.designs_omars_ilp import (
    _foldover,
    _full_second_order_params,
    _half_pool,
    _min_half_runs,
    _model_rank,
)
from process_improve.experiments.omars_trade_off import (
    CAPABILITIES,
    DEFAULT_FACTORS,
    DEFAULT_RUNS,
    get_omars_trade_off_table_entry,
    omars_minimum_runs,
    omars_trade_off_table,
)

# k -> (satd, quad, full), spelled out from 2k + 1, 2k + 3 and k^2 + k + 1.
THRESHOLDS = {
    3: (7, 9, 13),
    4: (9, 11, 21),
    5: (11, 13, 31),
    6: (13, 15, 43),
    7: (15, 17, 57),
}

# The approved rendering: (n_runs, n_factors) -> cell label.
TABLE_CELLS = {
    (9, 3): "Quad df=2",
    (9, 4): "Satd df=0",
    (9, 5): "",
    (13, 3): "Full df=3",
    (13, 4): "Quad df=4",
    (13, 5): "Quad df=2",
    (13, 6): "Satd df=0",
    (13, 7): "",
    (21, 4): "Full df=6",
    (31, 5): "Full df=10",
    (43, 6): "Full df=15",
    (57, 7): "Full df=21",
    (17, 7): "Quad df=2",
}


class TestMinimumRuns:
    @pytest.mark.parametrize(("k", "expected"), THRESHOLDS.items(), ids=str)
    def test_thresholds_match_the_written_table(self, k, expected):
        satd, quad, full = expected
        assert omars_minimum_runs(k, "satd") == satd
        assert omars_minimum_runs(k, "quad") == quad
        assert omars_minimum_runs(k, "full") == full

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_full_is_the_default(self, k):
        assert omars_minimum_runs(k) == omars_minimum_runs(k, "full")

    @pytest.mark.parametrize("k", list(range(3, 26)))
    def test_every_threshold_is_odd(self, k):
        """A foldover has 2h + 1 runs, so no threshold can be even."""
        assert all(omars_minimum_runs(k, c) % 2 == 1 for c in CAPABILITIES)

    @pytest.mark.parametrize("k", list(range(3, 26)))
    def test_thresholds_increase_with_capability(self, k):
        satd, quad, full = (omars_minimum_runs(k, c) for c in ("satd", "quad", "full"))
        assert satd < quad <= full

    def test_unknown_capability_is_refused(self):
        with pytest.raises(ValueError, match="capability must be one of"):
            omars_minimum_runs(4, "resolution_v")

    @pytest.mark.parametrize("bad", [2, 0, -1, 26, 100])
    def test_factor_count_out_of_range(self, bad):
        with pytest.raises(ValueError, match="OMARS designs need"):
            omars_minimum_runs(bad)

    def test_non_integer_factor_count(self):
        with pytest.raises(ValueError, match='"n_factors" input must be an integer'):
            omars_minimum_runs(4.5)


class TestFullThresholdAgreesWithTheRankLaw:
    """The `Full` threshold is not a convention: it is where the rank arrives.

    Checked directly against the model matrix, so the trade-off table and
    ``designs_omars_ilp`` cannot drift apart silently.
    """

    @pytest.mark.parametrize("k", [3, 4, 5])
    def test_rank_reaches_the_parameter_count_exactly_at_the_threshold(self, k):
        n_half = _min_half_runs(k)
        assert 2 * n_half + 1 == omars_minimum_runs(k, "full")
        pool = _half_pool(k)
        rng = np.random.default_rng(0)
        params = _full_second_order_params(k)
        reached = any(
            _model_rank(_foldover(pool[rng.choice(pool.shape[0], n_half, replace=False)])) == params for _ in range(200)
        )
        assert reached

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_two_runs_short_no_design_can_reach_it(self, k):
        """Pure arithmetic on the bound, so it covers every design at that size."""
        n_half = _min_half_runs(k) - 1
        bound = k + min(n_half + 1, 1 + k * (k + 1) // 2)
        assert bound < _full_second_order_params(k)
        assert 2 * n_half + 1 == omars_minimum_runs(k, "full") - 2


class TestSingleCell:
    @pytest.mark.parametrize(("cell", "label"), TABLE_CELLS.items(), ids=str)
    def test_label_matches_the_approved_rendering(self, cell, label):
        n_runs, n_factors = cell
        assert get_omars_trade_off_table_entry(n_runs, n_factors, display=False).label == label

    def test_full_cell_reports_the_second_order_model(self):
        result = get_omars_trade_off_table_entry(21, 4, display=False)
        assert result.exists
        assert result.capability == "full"
        assert result.tag == "Full"
        assert result.model == "full_second_order"
        assert result.model_params == _full_second_order_params(4) == 15
        assert result.error_df == 6
        assert result.reason == ""

    def test_quad_cell_drops_the_interactions(self):
        result = get_omars_trade_off_table_entry(17, 4, display=False)
        assert result.capability == "quad"
        assert result.model == "main_quadratic"
        assert result.model_params == 1 + 2 * 4
        assert result.error_df == 17 - 9

    def test_saturated_cell_has_no_error_df(self):
        result = get_omars_trade_off_table_entry(9, 4, display=False)
        assert result.capability == "satd"
        assert result.model == "main_quadratic"
        assert result.model_params == 9
        assert result.error_df == 0

    def test_error_df_is_runs_minus_parameters(self):
        for n_runs in (13, 21, 31, 43):
            result = get_omars_trade_off_table_entry(n_runs, 4, display=False)
            if result.capability != "satd":
                assert result.error_df == n_runs - result.model_params

    def test_even_run_counts_are_never_a_design(self):
        result = get_omars_trade_off_table_entry(20, 4, display=False)
        assert not result.exists
        assert result.capability == "none"
        assert result.tag == ""
        assert result.label == ""
        assert result.model is None
        assert "even" in result.reason

    def test_below_the_smallest_design_there_is_nothing(self):
        result = get_omars_trade_off_table_entry(7, 4, display=False)
        assert not result.exists
        assert "below the smallest OMARS design" in result.reason
        assert result.min_runs_satd == 9

    def test_thresholds_travel_with_every_result(self):
        """Even an empty cell says what it would take, which is the point."""
        result = get_omars_trade_off_table_entry(4, 6, display=False)
        assert (result.min_runs_satd, result.min_runs_quad, result.min_runs_full) == THRESHOLDS[6]

    @pytest.mark.parametrize("k", [3, 4, 5, 6, 7])
    def test_capability_never_goes_backwards_as_runs_grow(self, k):
        seen = [
            get_omars_trade_off_table_entry(n, k, display=False).capability
            for n in range(1, omars_minimum_runs(k, "full") + 8, 2)
        ]
        order = {"none": 0, "satd": 1, "quad": 2, "full": 3}
        ranks = [order[c] for c in seen]
        assert ranks == sorted(ranks)

    def test_non_integer_runs_rejected(self):
        with pytest.raises(ValueError, match='"n_runs" input must be an integer'):
            get_omars_trade_off_table_entry(13.5, 4, display=False)

    @pytest.mark.parametrize("bad", [0, -1, -13])
    def test_non_positive_runs_rejected(self, bad):
        with pytest.raises(ValueError, match="number of runs must be positive"):
            get_omars_trade_off_table_entry(bad, 4, display=False)


class TestDisplay:
    def test_report_names_the_class_the_model_and_the_gap(self, capsys):
        get_omars_trade_off_table_entry(17, 4, display=True)
        out = capsys.readouterr().out
        assert "OMARS: 17 runs, 4 factors" in out
        assert "Quad:" in out
        assert "main_quadratic (9 parameters), 8 error df" in out
        assert "Satd 9, Quad 11, Full 21 runs" in out
        assert "4 more runs would reach Full" in out

    def test_a_full_cell_is_not_told_to_buy_more_runs(self, capsys):
        get_omars_trade_off_table_entry(21, 4, display=True)
        assert "more runs would reach Full" not in capsys.readouterr().out

    def test_an_empty_cell_says_what_the_smallest_design_is(self, capsys):
        get_omars_trade_off_table_entry(20, 4, display=True)
        out = capsys.readouterr().out
        assert "No design" in out
        assert "smallest design for 4 factors has 9 runs" in out

    def test_display_is_silent_when_switched_off(self, capsys):
        get_omars_trade_off_table_entry(21, 4, display=False)
        assert capsys.readouterr().out == ""


class TestTable:
    def test_default_shape_and_axis_names(self):
        table = omars_trade_off_table(display=False)
        assert table.shape == (len(DEFAULT_RUNS), len(DEFAULT_FACTORS))
        assert table.index.name == "runs"
        assert table.columns.name == "factors"

    @pytest.mark.parametrize(("cell", "label"), TABLE_CELLS.items(), ids=str)
    def test_cells_match_the_approved_rendering(self, cell, label):
        n_runs, n_factors = cell
        assert omars_trade_off_table(display=False).loc[n_runs, n_factors] == label

    def test_cells_are_self_contained(self):
        """The stored frame repeats ``df=``; only the printed view compresses it."""
        table = omars_trade_off_table(display=False)
        live = [cell for cell in table.to_numpy().ravel() if cell]
        assert all(" df=" in cell for cell in live)

    def test_shape_follows_its_arguments(self):
        table = omars_trade_off_table(runs=(13, 21), factors=(4, 5), display=False)
        assert table.shape == (2, 2)
        assert list(table.index) == [13, 21]
        assert list(table.columns) == [4, 5]

    def test_the_last_row_is_full_everywhere(self):
        """57 runs is the smallest budget reaching Full for all of k = 3..7."""
        row = omars_trade_off_table(display=False).loc[57]
        assert all(cell.startswith("Full") for cell in row)

    def test_capability_worsens_across_a_row(self):
        """More factors on a fixed budget can only buy less."""
        table = omars_trade_off_table(display=False)
        for n_runs in DEFAULT_RUNS:
            tags = [cell.split()[0] for cell in table.loc[n_runs] if cell]
            assert tags == sorted(tags), f"row {n_runs} is not monotone"

    def test_an_out_of_range_factor_count_is_refused(self):
        with pytest.raises(ValueError, match="OMARS designs need"):
            omars_trade_off_table(factors=(3, 30), display=False)


class TestPrintedTable:
    @pytest.fixture
    def printed(self, capsys):
        omars_trade_off_table()
        return capsys.readouterr().out

    def test_header_is_padded_away_from_the_first_column(self, printed):
        header = printed.splitlines()[0]
        assert header.startswith("runs ")
        assert "k=3" in header
        assert "k=7" in header

    def test_df_label_is_written_once_per_column(self, printed):
        assert printed.count("df=") == len(DEFAULT_FACTORS)

    def test_the_label_sits_on_the_first_live_cell_of_the_column(self, printed):
        """For k = 3 that is the 9-run row, its first non-blank entry."""
        row = next(line for line in printed.splitlines() if line.startswith("9 "))
        assert "Quad df=2" in row
        later = next(line for line in printed.splitlines() if line.startswith("13 "))
        assert "Full 3" in later
        assert "Full df=3" not in later

    def test_cells_are_left_aligned(self, printed):
        """The capability staircase should read as an edge, not a ragged column."""
        lines = printed.splitlines()
        starts = [line.index("Full") for line in lines if line.startswith(("13 ", "21 ", "57 "))]
        assert len(set(starts)) == 1

    def test_the_legend_explains_every_tag(self, printed):
        for tag in ("Full:", "Quad:", "Satd:"):
            assert tag in printed
