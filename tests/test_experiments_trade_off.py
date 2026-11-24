"""Tests for ``process_improve.experiments.trade_off``.

The reference values are the trade-off table printed in the course notes
(and shipped as ``inst/trade-off-table.png`` in the companion R package):
the minimum-aberration search here must reproduce every cell of it.
"""

from __future__ import annotations

import pytest

from process_improve.experiments.designs_screening import _MIN_ABERRATION
from process_improve.experiments.evaluate import _defining_relation_strings, _signed_defining_relation
from process_improve.experiments.trade_off import (
    SearchLimitError,
    _alias_chains,
    _factor_names,
    get_trade_off_table_entry,
    minimum_aberration_generators,
    trade_off_table,
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
        result = get_trade_off_table_entry(n_runs=runs, n_factors=factors, display=False)
        assert result.roman == roman
        assert result.generators == generators

    @pytest.mark.parametrize(("cell", "roman"), TEXTBOOK_RESOLUTIONS.items(), ids=str)
    def test_resolution_matches_where_generators_are_cropped(self, cell, roman):
        runs, factors = cell
        assert get_trade_off_table_entry(n_runs=runs, n_factors=factors, display=False).roman == roman

    def test_table_layout_matches_the_figure(self):
        table = trade_off_table(display=False)
        assert table.loc[4, 3] == "2^(3-1) III"
        assert table.loc[8, 3] == "2^3 (full)"
        assert table.loc[16, 3] == "2^3 (twice)"
        assert table.loc[32, 3] == "2^3 (4 times)"
        assert table.loc[64, 3] == "2^3 (8 times)"
        assert table.loc[64, 6] == "2^6 (full)"
        assert table.loc[8, 7] == "2^(7-4) III"

    def test_impossible_cells_are_blank(self):
        """4 runs cannot study 5 factors, and 8 runs cannot study 9."""
        table = trade_off_table(display=False)
        assert table.loc[4, 5] == ""
        assert table.loc[8, 9] == ""

    def test_existing_design_whose_search_is_refused_is_not_blank(self):
        """2^(11-5) in 64 runs exists; a refused search falls back to the tabulated catalog."""
        table = trade_off_table(runs=(64,), factors=(11,), display=False)
        assert table.loc[64, 11].startswith("2^(11-5)")

    @pytest.mark.parametrize("bad", [12, 0])
    def test_table_rejects_a_run_count_that_is_not_a_power_of_two(self, bad):
        with pytest.raises(ValueError, match="power of 2"):
            trade_off_table(runs=(bad,), factors=(3,), display=False)

    def test_table_rejects_too_few_factors(self):
        with pytest.raises(ValueError, match="at least 2"):
            trade_off_table(runs=(8,), factors=(1,), display=False)

    def test_table_shape_follows_its_arguments(self):
        table = trade_off_table(runs=(8, 16), factors=(4, 5), display=False)
        assert table.shape == (2, 2)
        assert table.index.name == "runs"
        assert table.columns.name == "factors"

    def test_display_prints_only_when_asked(self, capsys):
        """`display` behaves like it does on the other three trade-off functions."""
        table = trade_off_table(runs=(8,), factors=(4,), display=False)
        assert capsys.readouterr().out == ""

        printed = trade_off_table(runs=(8,), factors=(4,), display=True)
        out = capsys.readouterr().out
        assert "2^(4-1) IV" in out
        assert printed.equals(table)


class TestTradeOffTableEntry:
    def test_defining_relation_is_the_full_closure(self):
        """2^(5-2) has 2^2 - 1 = 3 words in its defining relation."""
        result = get_trade_off_table_entry(n_runs=8, n_factors=5, display=False)
        assert result.defining_relation == ["I=ABD", "I=ACE", "I=BCDE"]
        assert result.n_generators == 2
        assert result.resolution == 3
        assert result.label == "2^(5-2) III"

    def test_alias_chains_cover_main_effects_and_two_factor_interactions(self):
        """5 main effects + C(5, 2) = 10 two-factor interactions."""
        result = get_trade_off_table_entry(n_runs=8, n_factors=5, display=False)
        assert len(result.aliases) == 15
        assert "A = BD + CE + ABCDE" in result.aliases

    def test_resolution_iii_aliases_a_main_effect_with_an_interaction(self):
        result = get_trade_off_table_entry(n_runs=8, n_factors=5, display=False)
        main_effect_chain = next(chain for chain in result.aliases if chain.startswith("A ="))
        assert "BD" in main_effect_chain

    def test_resolution_iv_keeps_main_effects_clear_of_interactions(self):
        result = get_trade_off_table_entry(n_runs=16, n_factors=6, display=False)
        for name in "ABCDEF":
            chain = next(c for c in result.aliases if c.startswith(f"{name} ="))
            aliases = chain.split(" = ")[1].split(" + ")
            assert all(len(alias) >= 3 for alias in aliases)

    def test_full_factorial_has_no_resolution_and_no_aliasing(self):
        result = get_trade_off_table_entry(n_runs=16, n_factors=4, display=False)
        assert result.resolution is None
        assert result.roman is None
        assert result.generators == []
        assert result.defining_relation == []
        assert result.aliases == []
        assert result.n_replicates == 1
        assert result.label == "2^4 (full)"

    def test_a_relation_too_long_to_list_still_gives_the_resolution(self, capsys):
        """25 factors in 32 runs have 2^20 - 1 defining words; listing them and their alias chains took hours."""
        result = get_trade_off_table_entry(n_runs=32, n_factors=25, display=True)
        assert result.label == "2^(25-20) III"
        assert result.defining_relation == []
        assert result.aliases == []
        assert "Not listed: the defining relation has 1,048,575 words." in capsys.readouterr().out
        assert get_trade_off_table_entry(n_runs=32, n_factors=16, display=False).resolution == 4
        assert len(get_trade_off_table_entry(n_runs=16, n_factors=14, display=False).defining_relation) == 1023

    def test_spare_budget_is_reported_as_replication(self):
        result = get_trade_off_table_entry(n_runs=32, n_factors=3, display=False)
        assert result.n_replicates == 4
        assert result.label == "2^3 (4 times)"

    def test_display_prints_a_report(self, capsys):
        get_trade_off_table_entry(n_runs=8, n_factors=5, display=True)
        out = capsys.readouterr().out
        assert "With 8 experiments, and 5 factors:" in out
        assert "Resolution: III" in out
        assert "D=AB" in out

    def test_display_is_silent_when_switched_off(self, capsys):
        get_trade_off_table_entry(n_runs=8, n_factors=5, display=False)
        assert capsys.readouterr().out == ""

    def test_display_of_a_full_factorial_reports_no_aliasing(self, capsys):
        get_trade_off_table_entry(n_runs=16, n_factors=4, display=True)
        assert "No aliasing" in capsys.readouterr().out

    def test_singular_wording_for_a_single_generator(self, capsys):
        get_trade_off_table_entry(n_runs=16, n_factors=5, display=True)
        out = capsys.readouterr().out
        assert "Generator:" in out
        assert "Generators:" not in out


class TestInputChecks:
    @pytest.mark.parametrize("bad", [7, 12, 100, 0, -8])
    def test_runs_must_be_a_power_of_two(self, bad):
        with pytest.raises(ValueError, match="power of 2"):
            get_trade_off_table_entry(n_runs=bad, n_factors=3, display=False)

    def test_non_integer_runs_rejected(self):
        with pytest.raises(ValueError, match='"n_runs" input must be an integer'):
            get_trade_off_table_entry(n_runs=8.5, n_factors=3, display=False)

    def test_non_integer_factors_rejected(self):
        with pytest.raises(ValueError, match='"n_factors" input must be an integer'):
            get_trade_off_table_entry(n_runs=8, n_factors=3.5, display=False)

    def test_too_few_factors_rejected(self):
        with pytest.raises(ValueError, match="At least 2 factors"):
            get_trade_off_table_entry(n_runs=8, n_factors=1, display=False)

    def test_too_many_factors_for_the_budget_rejected(self):
        """8 runs hold at most 7 factors (3 base + 4 interaction columns)."""
        with pytest.raises(ValueError, match="cannot accommodate"):
            get_trade_off_table_entry(n_runs=8, n_factors=8, display=False)

    def test_messages_say_runs_and_factors_in_prose(self):
        with pytest.raises(ValueError, match=r"^8 runs cannot accommodate 8 factors: only 7 factors fit into 8 runs"):
            get_trade_off_table_entry(n_runs=8, n_factors=8, display=False)

    def test_more_factors_than_single_letter_names_rejected(self):
        """The alphabet, minus the identity column ``I``, runs out at 25."""
        with pytest.raises(ValueError, match="single-letter names"):
            get_trade_off_table_entry(n_runs=32, n_factors=26, display=False)


class TestMinimumAberrationGenerators:
    def test_full_factorial_is_not_a_fraction(self):
        with pytest.raises(ValueError, match="not a fractional factorial"):
            minimum_aberration_generators(16, 4)

    def test_search_refuses_an_intractable_request(self):
        """A huge search space is refused with an actionable message."""
        with pytest.raises(SearchLimitError, match="above the limit"):
            minimum_aberration_generators(64, 30)

    @pytest.mark.parametrize(("n_runs", "n_factors", "label"), [(64, 11, "2^(11-5) IV"), (128, 11, "2^(11-4) V")])
    def test_large_cells_fall_back_to_the_catalogue(self, n_runs, n_factors, label):
        """Past the search limit the tabulated minimum-aberration design is used (Chen, Sun and Wu 1993)."""
        assert get_trade_off_table_entry(n_runs=n_runs, n_factors=n_factors, display=False).label == label

    def test_existing_but_unsearched_cell_is_not_blank(self):
        """12 factors fit into 64 runs; a blank cell would say they cannot, though no design is searched or listed."""
        table = trade_off_table(runs=[64], factors=[12], display=False)
        assert table.loc[64, 12] == "?"

    def test_messages_say_runs_and_factors(self):
        """The messages used to read '8 n_runs cannot accommodate 8 n_factors'."""
        with pytest.raises(ValueError, match=r"^8 runs cannot accommodate 8 factors: only 7 factors fit into 8 runs"):
            get_trade_off_table_entry(n_runs=8, n_factors=8, display=False)

    def test_result_is_cached_and_stable(self):
        assert minimum_aberration_generators(16, 7) is minimum_aberration_generators(16, 7)

    def test_generators_name_each_extra_factor_once(self):
        generators = minimum_aberration_generators(16, 8)
        assert [g.split("=")[0] for g in generators] == ["E", "F", "G", "H"]

    def test_no_generators_means_no_alias_chains(self):
        """A design with no defining relation aliases nothing."""
        assert _alias_chains([], ["A", "B", "C"]) == []


class TestDefiningWordsArePositive:
    """The report writes its defining words and alias chains without signs, which is exact only for positive generators.

    Its generators come only from ``minimum_aberration_generators``: from the search, which writes
    ``name=word``, or from the ``_MIN_ABERRATION`` catalog past the search limit. A generator such
    as ``D=-AB`` would be reported as ``I=ABD`` instead of ``I=-ABD``.
    """

    def test_catalogue_has_no_negative_word(self):
        for k, by_runs in _MIN_ABERRATION.items():
            for n_runs, generators in by_runs.items():
                signed = _signed_defining_relation(list(generators), _factor_names(k))
                assert all(sign == 1 for _word, sign in signed), (k, n_runs, generators)

    def test_report_agrees_with_the_signed_defining_relation(self):
        """Each fractional cell of the default table (searched), and each cell taken from the catalog."""
        table = trade_off_table(display=False)
        cells = [(r, k) for r in table.index for k in table.columns if table.loc[r, k].startswith("2^(")]
        cells += [(64, 11), (128, 11), (256, 11)]  # past the search limit
        for n_runs, n_factors in cells:
            entry = get_trade_off_table_entry(n_runs=n_runs, n_factors=n_factors, display=False)
            signed = _defining_relation_strings(entry.generators, _factor_names(n_factors))
            assert entry.defining_relation == signed, (n_runs, n_factors)


# ---------------------------------------------------------------------------
# generate_design's minimum-aberration table agrees with the search here
# ---------------------------------------------------------------------------


def _table_cases() -> list:
    """Every (factors, runs) cell of ``_MIN_ABERRATION`` the exhaustive search can also cover.

    Cells whose search scores more than 50,000 generator sets take several seconds each.
    """
    import math

    from process_improve.experiments.designs_screening import _MIN_ABERRATION
    from process_improve.experiments.trade_off import _MAX_CANDIDATE_SETS

    cases = []
    for k, by_runs in sorted(_MIN_ABERRATION.items()):
        for n_runs, generators in sorted(by_runs.items()):
            n_base = n_runs.bit_length() - 1
            n_sets = math.comb(2**n_base - n_base - 1, k - n_base)
            if n_sets > _MAX_CANDIDATE_SETS:
                continue  # the search refuses it, so the search does not cover this cell
            marks = [pytest.mark.slow] if n_sets > 50_000 else []
            cases.append(pytest.param(k, n_runs, generators, marks=marks, id=f"{k}-factors-{n_runs}-runs"))
    return cases


@pytest.mark.parametrize(("k", "n_runs", "generators"), _table_cases())
def test_generate_design_table_has_minimum_aberration(k: int, n_runs: int, generators: tuple[str, ...]) -> None:
    """The table generate_design uses and the search agree on the word-length pattern, cell by cell.

    A minimum-aberration design is unique only up to relabeling, so the generators may differ;
    the word-length pattern (A3, A4, ...) is what the criterion ranks, and the two must match.
    The patterns are those of Chen, Sun and Wu (1993), e.g. (0, 6, 8, 0, 0, 1) for 2^(9-4) in 32 runs.
    """
    from process_improve.experiments.trade_off import _factor_names, _word_length_pattern

    names = _factor_names(k)
    searched = minimum_aberration_generators(n_runs, k)
    assert _word_length_pattern(generators, names, k) == _word_length_pattern(searched, names, k)


#: The word-length patterns (A3 to A7) of the minimum-aberration fractions in 32 runs that the search is too slow
#: to run on request, as Chen, Sun and Wu (1993) give them; their table is complete for 32 runs.
_PATTERNS_IN_32_RUNS = {
    12: (0, 38, 0, 52, 0),
    13: (0, 55, 0, 96, 0),
    14: (0, 77, 0, 168, 0),
    15: (0, 105, 0, 280, 0),
    16: (0, 140, 0, 448, 0),
    17: (8, 140, 112, 448, 504),
    18: (16, 148, 224, 560, 1008),
    19: (24, 164, 344, 784, 1624),
    20: (32, 188, 480, 1128, 2464),
    21: (40, 220, 641, 1608, 3640),
    22: (48, 263, 832, 2224, 5312),
    23: (56, 315, 1064, 3024, 7616),
    24: (64, 378, 1344, 4032, 10752),
    25: (76, 442, 1656, 5376, 15004),
}


@pytest.mark.parametrize("k", sorted(_PATTERNS_IN_32_RUNS))
def test_the_listed_32_run_fractions_have_the_published_word_length_patterns(k: int) -> None:
    """Past the search limit, 12 to 25 factors in 32 runs come from the table: each has the published pattern."""
    from process_improve.experiments.trade_off import _factor_names, _word_length_pattern

    generators = minimum_aberration_generators(32, k)
    assert _word_length_pattern(generators, _factor_names(k), k)[:5] == _PATTERNS_IN_32_RUNS[k]


@pytest.mark.parametrize(
    "k", [pytest.param(k, marks=[pytest.mark.slow] if 13 <= k <= 23 else []) for k in sorted(_PATTERNS_IN_32_RUNS)]
)
def test_the_listed_32_run_fractions_are_what_the_search_returns(k: int) -> None:
    """The table holds what the exhaustive search finds; it is listed only because the search takes seconds."""
    from process_improve.experiments.designs_screening import _MIN_ABERRATION
    from process_improve.experiments.trade_off import _search_minimum_aberration

    assert _search_minimum_aberration(5, k) == _MIN_ABERRATION[k][32]


def test_a_fractional_factorial_of_20_factors_in_32_runs_is_built_from_the_table() -> None:
    """generate_design builds the minimum-aberration fraction, where the search used to refuse it."""
    from process_improve.experiments import Factor, generate_design

    factors = [Factor(name=f"x{i}", low=-1, high=1) for i in range(20)]
    result = generate_design(factors, design_type="fractional_factorial", budget=32, n_center_points=0)
    assert (result.n_runs, result.resolution) == (32, 3)
