"""Tests for ``process_improve.experiments.tradeoff``.

The reference values are the trade-off table printed in the course notes
(and shipped as ``inst/trade-off-table.png`` in the companion R package):
the minimum-aberration search here must reproduce every cell of it.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.tradeoff import (
    minimum_aberration_generators,
    trade_off_table,
    tradeoff,
)

# (runs, factors) -> (resolution in roman, generators), read off the course-notes
# trade-off table. Cells where the figure does not print the generators are
# covered by ``TEXTBOOK_RESOLUTIONS`` below instead.
TEXTBOOK_CELLS = {
    (4, 3): ("III", ["C=AB"]),
    (8, 4): ("IV", ["D=ABC"]),
    (8, 5): ("III", ["D=AB", "E=AC"]),
    (8, 6): ("III", ["D=AB", "E=AC", "F=BC"]),
    (8, 7): ("III", ["D=AB", "E=AC", "F=BC", "G=ABC"]),
    (16, 5): ("V", ["E=ABCD"]),
    (16, 6): ("IV", ["E=ABC", "F=ABD"]),
    (16, 7): ("IV", ["E=ABC", "F=ABD", "G=ACD"]),
    (16, 8): ("IV", ["E=ABC", "F=ABD", "G=ACD", "H=BCD"]),
    (32, 6): ("VI", ["F=ABCDE"]),
    (32, 7): ("IV", ["F=ABC", "G=ABDE"]),
    (32, 8): ("IV", ["F=ABC", "G=ABD", "H=ACDE"]),
    (64, 7): ("VII", ["G=ABCDEF"]),
    (64, 8): ("V", ["G=ABCD", "H=ABEF"]),
}

# Cells whose generators the figure crops off, but whose resolution it prints.
TEXTBOOK_RESOLUTIONS = {(16, 9): "III", (32, 9): "IV", (64, 9): "IV"}


class TestAgainstTheTextbookTable:
    @pytest.mark.parametrize(("cell", "expected"), TEXTBOOK_CELLS.items(), ids=str)
    def test_generators_and_resolution_match(self, cell, expected):
        runs, factors = cell
        roman, generators = expected
        result = tradeoff(runs=runs, factors=factors, display=False)
        assert result.roman == roman
        assert result.generators == generators

    @pytest.mark.parametrize(("cell", "roman"), TEXTBOOK_RESOLUTIONS.items(), ids=str)
    def test_resolution_matches_where_generators_are_cropped(self, cell, roman):
        runs, factors = cell
        assert tradeoff(runs=runs, factors=factors, display=False).roman == roman

    def test_table_layout_matches_the_figure(self):
        table = trade_off_table()
        assert table.loc[4, 3] == "2^(3-1) III"
        assert table.loc[8, 3] == "2^3 (full)"
        assert table.loc[16, 3] == "2^3 (twice)"
        assert table.loc[32, 3] == "2^3 (4 times)"
        assert table.loc[64, 3] == "2^3 (8 times)"
        assert table.loc[64, 6] == "2^6 (full)"
        assert table.loc[8, 7] == "2^(7-4) III"

    def test_impossible_cells_are_blank(self):
        """4 runs cannot study 5 factors, and 8 runs cannot study 9."""
        table = trade_off_table()
        assert table.loc[4, 5] == ""
        assert table.loc[8, 9] == ""

    def test_table_shape_follows_its_arguments(self):
        table = trade_off_table(runs=(8, 16), factors=(4, 5))
        assert table.shape == (2, 2)
        assert table.index.name == "runs"
        assert table.columns.name == "factors"


class TestTradeoffResult:
    def test_defining_relation_is_the_full_closure(self):
        """2^(5-2) has 2^2 - 1 = 3 words in its defining relation."""
        result = tradeoff(runs=8, factors=5, display=False)
        assert result.defining_relation == ["I=ABD", "I=ACE", "I=BCDE"]
        assert result.n_generators == 2
        assert result.resolution == 3
        assert result.label == "2^(5-2) III"

    def test_alias_chains_cover_main_effects_and_two_factor_interactions(self):
        """5 main effects + C(5, 2) = 10 two-factor interactions."""
        result = tradeoff(runs=8, factors=5, display=False)
        assert len(result.aliases) == 15
        assert "A = BD + CE + ABCDE" in result.aliases

    def test_resolution_iii_aliases_a_main_effect_with_an_interaction(self):
        result = tradeoff(runs=8, factors=5, display=False)
        main_effect_chain = next(chain for chain in result.aliases if chain.startswith("A ="))
        assert "BD" in main_effect_chain

    def test_resolution_iv_keeps_main_effects_clear_of_interactions(self):
        result = tradeoff(runs=16, factors=6, display=False)
        for name in "ABCDEF":
            chain = next(c for c in result.aliases if c.startswith(f"{name} ="))
            aliases = chain.split(" = ")[1].split(" + ")
            assert all(len(alias) >= 3 for alias in aliases)

    def test_full_factorial_has_no_resolution_and_no_aliasing(self):
        result = tradeoff(runs=16, factors=4, display=False)
        assert result.resolution is None
        assert result.roman is None
        assert result.generators == []
        assert result.defining_relation == []
        assert result.aliases == []
        assert result.replicates == 1
        assert result.label == "2^4 (full)"

    def test_spare_budget_is_reported_as_replication(self):
        result = tradeoff(runs=32, factors=3, display=False)
        assert result.replicates == 4
        assert result.label == "2^3 (4 times)"

    def test_display_prints_a_report(self, capsys):
        tradeoff(runs=8, factors=5, display=True)
        out = capsys.readouterr().out
        assert "With 8 experiments, and 5 factors:" in out
        assert "Resolution: III" in out
        assert "D=AB" in out

    def test_display_is_silent_when_switched_off(self, capsys):
        tradeoff(runs=8, factors=5, display=False)
        assert capsys.readouterr().out == ""

    def test_display_of_a_full_factorial_reports_no_aliasing(self, capsys):
        tradeoff(runs=16, factors=4, display=True)
        assert "No aliasing" in capsys.readouterr().out

    def test_singular_wording_for_a_single_generator(self, capsys):
        tradeoff(runs=16, factors=5, display=True)
        out = capsys.readouterr().out
        assert "Generator:" in out
        assert "Generators:" not in out


class TestInputChecks:
    @pytest.mark.parametrize("bad", [7, 12, 100, 0, -8])
    def test_runs_must_be_a_power_of_two(self, bad):
        with pytest.raises(ValueError, match="power of 2"):
            tradeoff(runs=bad, factors=3, display=False)

    def test_non_integer_runs_rejected(self):
        with pytest.raises(ValueError, match='"runs" input must be an integer'):
            tradeoff(runs=8.5, factors=3, display=False)

    def test_non_integer_factors_rejected(self):
        with pytest.raises(ValueError, match='"factors" input must be an integer'):
            tradeoff(runs=8, factors=3.5, display=False)

    def test_too_few_factors_rejected(self):
        with pytest.raises(ValueError, match="At least 2 factors"):
            tradeoff(runs=8, factors=1, display=False)

    def test_too_many_factors_for_the_budget_rejected(self):
        """8 runs hold at most 7 factors (3 base + 4 interaction columns)."""
        with pytest.raises(ValueError, match="cannot accommodate"):
            tradeoff(runs=8, factors=8, display=False)


class TestMinimumAberrationGenerators:
    def test_full_factorial_is_not_a_fraction(self):
        with pytest.raises(ValueError, match="not a fractional factorial"):
            minimum_aberration_generators(16, 4)

    def test_search_refuses_an_intractable_request(self):
        """A huge search space is refused with an actionable message."""
        with pytest.raises(ValueError, match="above the limit"):
            minimum_aberration_generators(64, 30)

    def test_result_is_cached_and_stable(self):
        assert minimum_aberration_generators(16, 7) is minimum_aberration_generators(16, 7)

    def test_generators_name_each_extra_factor_once(self):
        generators = minimum_aberration_generators(16, 8)
        assert [g.split("=")[0] for g in generators] == ["E", "F", "G", "H"]
