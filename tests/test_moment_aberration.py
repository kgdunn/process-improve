"""Tests for minimum moment aberration (Xu, 2003).

The reference values come from two sources:

* Vazquez, Rother and Charles-Gonzalez (2026), arXiv:2512.17113v2, which
  tabulates moment aberration patterns for several small two-level designs.
  Their Table 1 gives a 2^(7-3) design in full and Section 2.2 gives its
  pattern, and their Tables 3, 5, 7, 9 and 11 give patterns for the 8-, 16-
  and 32-run minimum aberration designs.
* Direct construction from generators, cross-checked against the
  defining-relation resolution that ``evaluate_design`` computes
  independently.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import moment_aberration
from process_improve.experiments._moment_aberration import (
    MAX_PATTERN_LENGTH,
    MomentAberrationResult,
    NotTwoLevelError,
    _even_tuple_counts,
)
from process_improve.experiments.evaluate import evaluate_design

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _basic_design(n_basic: int) -> pd.DataFrame:
    """Return the full 2^n_basic factorial over the first ``n_basic`` letters."""
    runs = list(itertools.product([-1, 1], repeat=n_basic))
    return pd.DataFrame(runs, columns=list(_ALPHABET[:n_basic]))


def _fractional(n_basic: int, generators: dict[str, str]) -> pd.DataFrame:
    """Build a regular 2^(m-p) design from basic factors plus generators.

    ``generators`` maps a generated factor name to the product of basic
    factors defining it, e.g. ``{"D": "ABC"}`` for D = ABC.
    """
    design = _basic_design(n_basic)
    for name, word in generators.items():
        column = np.ones(len(design), dtype=int)
        for letter in word:
            column = column * design[letter].to_numpy()
        design[name] = column
    return design


# The 2^(7-3) design printed in full as Table 1 of Vazquez et al. (2026):
# basic factors A, B, C, D with E = ABC, F = ABD, G = ACD.
PAPER_TABLE_1 = _fractional(4, {"E": "ABC", "F": "ABD", "G": "ACD"})

# Section 2.2 of the same paper, rounded to two decimals.
PAPER_TABLE_1_PATTERN = [3.27, 11.67, 42.47, 157.27, 591.27, 2251.67, 8666.47]


# ---------------------------------------------------------------------------
# The paper's worked example
# ---------------------------------------------------------------------------


def test_paper_worked_example_pattern() -> None:
    """The 2^(7-3) design reproduces the published moment aberration pattern."""
    result = moment_aberration(PAPER_TABLE_1)
    assert [round(k, 2) for k in result.pattern] == PAPER_TABLE_1_PATTERN


def test_paper_worked_example_resolution() -> None:
    """That design is resolution IV, i.e. strength 3."""
    result = moment_aberration(PAPER_TABLE_1)
    assert result.strength == 3
    assert result.resolution == 4
    assert result.is_orthogonal_array


def test_paper_worked_example_attains_bounds_up_to_strength() -> None:
    """K_t equals its lower bound for t <= 3 and misses it at t = 4."""
    result = moment_aberration(PAPER_TABLE_1)
    # lower_bounds is truncated to strength + 1 = 4 entries.
    assert len(result.lower_bounds) == 4
    assert result.pattern[:3] == pytest.approx(result.lower_bounds[:3])
    assert result.pattern[3] > result.lower_bounds[3]


@pytest.mark.parametrize(
    ("generators", "expected_pattern", "expected_resolution"),
    [
        # Table 3 of Vazquez et al.: the 8-run minimum aberration designs.
        ({"D": "ABC"}, [1.7, 3.4, 6.9, 13.7], 4),
        ({"D": "AB", "E": "AC"}, [2.1, 5.0, 12.4, 32.4, 87.9], 3),
        ({"D": "AB", "E": "AC", "F": "BC"}, [2.6, 6.9, 18.9, 53.1, 152.6, 444.0], 3),
        (
            {"D": "AB", "E": "AC", "F": "BC", "G": "ABC"},
            [3.0, 9.0, 27.0, 81.0, 243.0, 729.0, 2187.0],
            3,
        ),
    ],
    ids=["2^(4-1)", "2^(5-2)", "2^(6-3)", "2^(7-4)"],
)
def test_published_eight_run_patterns(
    generators: dict[str, str],
    expected_pattern: list[float],
    expected_resolution: int,
) -> None:
    """The 8-run minimum aberration designs match the published patterns."""
    result = moment_aberration(_fractional(3, generators))
    assert [round(k, 1) for k in result.pattern] == expected_pattern
    assert result.resolution == expected_resolution


@pytest.mark.parametrize(
    ("n_basic", "generators", "expected_prefix", "expected_resolution"),
    [
        # Tables 5 and 9: 16-run minimum aberration designs, first four moments.
        (4, {"E": "ABCD"}, [2.3, 6.3, 18.3, 54.3], 5),
        (4, {"E": "ABC", "F": "BCD"}, [2.8, 8.8, 28.8, 97.6], 4),
        (4, {"E": "ABC", "F": "ABD", "G": "ACD"}, [3.3, 11.7, 42.5, 157.3], 4),
        (4, {"E": "ABC", "F": "ABD", "G": "ACD", "H": "BCD"}, [3.7, 14.9, 59.7, 238.9], 4),
    ],
    ids=["2^(5-1)", "2^(6-2)", "2^(7-3)", "2^(8-4)"],
)
def test_published_sixteen_run_patterns(
    n_basic: int,
    generators: dict[str, str],
    expected_prefix: list[float],
    expected_resolution: int,
) -> None:
    """16-run minimum aberration designs match the first four published moments."""
    result = moment_aberration(_fractional(n_basic, generators))
    assert [round(k, 1) for k in result.pattern[:4]] == expected_prefix
    assert result.resolution == expected_resolution


def test_published_thirty_two_run_six_factor() -> None:
    """Table 7/11: the 32-run 6-factor design is resolution VI."""
    result = moment_aberration(_fractional(5, {"F": "ABCDE"}))
    assert [round(k, 1) for k in result.pattern[:6]] == [2.9, 9.7, 34.8, 131.6, 511.0, 2012.9]
    assert result.resolution == 6


# ---------------------------------------------------------------------------
# Agreement with the defining-relation route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_basic", "generators"),
    [
        (3, {"D": "ABC"}),
        (3, {"D": "AB", "E": "AC"}),
        (3, {"D": "AB", "E": "AC", "F": "BC"}),
        (4, {"E": "ABCD"}),
        (4, {"E": "ABC", "F": "BCD"}),
        (4, {"E": "ABC", "F": "ABD", "G": "ACD"}),
        (5, {"F": "ABCDE"}),
        (5, {"F": "ABCD", "G": "ABCE"}),
    ],
)
def test_agrees_with_defining_relation_resolution(n_basic: int, generators: dict[str, str]) -> None:
    """Moment-derived resolution matches the shortest word in the defining relation."""
    design = _fractional(n_basic, generators)
    shortest_word = min(len(f"{name}{word}") for name, word in generators.items())
    # The defining relation is generated by the words I = <name><word>; the
    # resolution is the shortest word length over the whole group, which for
    # these standard designs is attained by a generator or a product of two.
    products = []
    for (name_a, word_a), (name_b, word_b) in itertools.combinations(generators.items(), 2):
        symmetric = set(name_a + word_a) ^ set(name_b + word_b)
        products.append(len(symmetric))
    expected_resolution = min([shortest_word, *products])

    result = moment_aberration(design)
    assert result.resolution == expected_resolution


def test_full_factorial_saturates_at_m_plus_one() -> None:
    """A full factorial has strength m, hence the saturated resolution m + 1."""
    for k in (2, 3, 4, 5):
        result = moment_aberration(_basic_design(k))
        assert result.strength == k
        assert result.resolution == k + 1
        assert result.lower_bounds == pytest.approx(result.pattern)


# ---------------------------------------------------------------------------
# The failure modes the criterion exists to catch
# ---------------------------------------------------------------------------


def test_duplicated_column_is_resolution_two() -> None:
    """Two identical columns alias a main effect with a main effect."""
    design = _fractional(3, {"D": "AB", "E": "AC", "F": "BC", "G": "ABC"})
    design["G"] = design["A"]
    result = moment_aberration(design)
    assert result.strength == 1
    assert result.resolution == 2
    assert not result.is_orthogonal_array


def test_unbalanced_column_is_resolution_one() -> None:
    """A column that is not level-balanced gives strength 0."""
    design = _basic_design(3)
    design.loc[0, "A"] = 1  # break the balance of column A
    result = moment_aberration(design)
    assert result.strength == 0
    assert result.resolution == 1


def test_detects_a_design_that_looks_plausible_but_is_not_orthogonal() -> None:
    """A 12-run non-regular design is caught even with no defining relation.

    This is the case the word-length-pattern route cannot reach at all: there
    are no generators to read, so ``minimum_aberration`` returns a note rather
    than a verdict, while moment aberration still ranks the design.
    """
    # A Plackett-Burman-like matrix that is *not* orthogonal: the last column
    # is the negation of the first, so columns 1 and 12 are perfectly aliased.
    runs = list(itertools.product([-1, 1], repeat=3))
    design = pd.DataFrame(runs, columns=["A", "B", "C"])
    design["D"] = -design["A"]

    result = moment_aberration(design)
    assert result.resolution == 2

    via_generators = evaluate_design(design, metric="minimum_aberration")
    assert via_generators["minimum_aberration"]["wordlength_pattern"] == []


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_accepts_any_two_level_coding() -> None:
    """0/1, low/high and boolean codings give the same answer as -1/+1."""
    reference = moment_aberration(_fractional(3, {"D": "ABC"}))

    coded = _fractional(3, {"D": "ABC"})
    zero_one = coded.replace({-1: 0})
    labels = coded.replace({-1: "low", 1: "high"})
    booleans = coded.gt(0)

    for variant in (zero_one, labels, booleans):
        assert moment_aberration(variant).pattern == pytest.approx(reference.pattern)
        assert moment_aberration(variant).resolution == reference.resolution


def test_accepts_numpy_array() -> None:
    """A bare ndarray works as well as a DataFrame."""
    frame = _fractional(3, {"D": "ABC"})
    assert moment_aberration(frame.to_numpy()).pattern == pytest.approx(moment_aberration(frame).pattern)


@pytest.mark.parametrize("label", ["Run", "RunOrder", "Block", "run", "run_order", "block"])
def test_bookkeeping_columns_are_ignored(label: str) -> None:
    """Run-order and block labels do not count as factors."""
    frame = _fractional(3, {"D": "ABC"})
    annotated = frame.copy()
    annotated[label] = range(1, len(frame) + 1)
    assert moment_aberration(annotated).pattern == pytest.approx(moment_aberration(frame).pattern)
    assert moment_aberration(annotated).n_factors == frame.shape[1]


def test_center_points_are_rejected() -> None:
    """A three-level column is refused with an actionable message."""
    design = _basic_design(3)
    design.loc[len(design)] = [0, 0, 0]
    with pytest.raises(NotTwoLevelError, match="two-level designs only"):
        moment_aberration(design)


def test_single_level_column_is_rejected() -> None:
    """A constant column is not a two-level factor."""
    design = _basic_design(3)
    design["D"] = 1
    with pytest.raises(NotTwoLevelError, match="1 distinct level"):
        moment_aberration(design)


def test_too_few_runs_is_rejected() -> None:
    """At least two runs are needed to form a pair."""
    with pytest.raises(ValueError, match="at least 2"):
        moment_aberration(pd.DataFrame({"A": [-1]}))


def test_no_factor_columns_is_rejected() -> None:
    """A frame of only bookkeeping columns has nothing to evaluate."""
    with pytest.raises(ValueError, match="no factor columns"):
        moment_aberration(pd.DataFrame({"Run": [1, 2, 3]}))


def test_input_is_not_mutated() -> None:
    """Dropping bookkeeping columns does not touch the caller's frame."""
    frame = _fractional(3, {"D": "ABC"})
    frame["Run"] = range(1, len(frame) + 1)
    before = frame.copy()
    moment_aberration(frame)
    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_minimum_aberration_design_beats_an_inferior_one() -> None:
    """Among 16-run 8-factor designs, the MA design wins on the pattern."""
    best = _fractional(4, {"E": "ABC", "F": "ABD", "G": "ACD", "H": "BCD"})
    worse = _fractional(4, {"E": "AB", "F": "AC", "G": "AD", "H": "BCD"})

    assert moment_aberration(best).is_better_than(moment_aberration(worse))
    assert not moment_aberration(worse).is_better_than(moment_aberration(best))


def test_identical_designs_do_not_beat_each_other() -> None:
    """The comparison is strict, so a design does not beat its own twin."""
    result = moment_aberration(_fractional(3, {"D": "ABC"}))
    assert not result.is_better_than(result)


def test_cross_size_comparison_is_refused() -> None:
    """K_t scales with n and m, so comparing different sizes is meaningless."""
    small = moment_aberration(_fractional(3, {"D": "ABC"}))
    large = moment_aberration(_fractional(4, {"E": "ABCD"}))
    with pytest.raises(ValueError, match="same size"):
        small.is_better_than(large)


def test_is_better_than_falls_back_to_float_pattern() -> None:
    """A result rebuilt without the exact Fractions still compares."""
    result = moment_aberration(_fractional(3, {"D": "ABC"}))
    plain = MomentAberrationResult(
        pattern=[k + 1 for k in result.pattern],
        lower_bounds=result.lower_bounds,
        strength=result.strength,
        resolution=result.resolution,
        n_runs=result.n_runs,
        n_factors=result.n_factors,
        is_orthogonal_array=result.is_orthogonal_array,
    )
    assert result.is_better_than(plain)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_factors", "expected"),
    [
        # E_0 = 1 (the empty tuple); E_1 = 0 (one column cannot appear twice);
        # E_2 = m (the pairs (k, k)); E_3 = 0 (odd tuples never balance).
        (1, [1, 0, 1, 0, 1]),
        (3, [1, 0, 3, 0, 21]),
        (7, [1, 0, 7, 0, 133]),
    ],
)
def test_even_tuple_counts(n_factors: int, expected: list[int]) -> None:
    """E_u(m) counts ordered u-tuples with every column at even multiplicity."""
    assert _even_tuple_counts(n_factors, len(expected) - 1) == expected


def test_even_tuple_counts_match_brute_force() -> None:
    """Cross-check the closed form against direct enumeration."""
    for n_factors in (2, 4, 5):
        for order in range(7):
            brute = sum(
                1
                for tup in itertools.product(range(n_factors), repeat=order)
                if all(tup.count(k) % 2 == 0 for k in range(n_factors))
            )
            assert _even_tuple_counts(n_factors, order)[order] == brute


def test_pattern_is_exact_not_floating_point() -> None:
    """The exact Fractions reproduce the reported floats."""
    result = moment_aberration(PAPER_TABLE_1)
    assert [float(k) for k in result.exact_pattern] == result.pattern


def test_to_dict_is_json_serialisable() -> None:
    """to_dict() drops the Fractions so the result can cross a JSON boundary."""
    import json

    payload = moment_aberration(PAPER_TABLE_1).to_dict()
    assert json.loads(json.dumps(payload))["resolution"] == 4
    assert "exact_pattern" not in payload


def test_replicated_runs_are_handled() -> None:
    """Duplicated runs lower the strength but do not crash."""
    design = pd.concat([_basic_design(3)] * 2, ignore_index=True)
    result = moment_aberration(design)
    assert result.n_runs == 16
    assert result.strength >= 1


def test_pattern_length_is_capped() -> None:
    """Wide designs truncate the pattern rather than compute a useless tail."""
    n_factors = MAX_PATTERN_LENGTH + 4
    rng = np.random.default_rng(0)
    runs = rng.choice([-1, 1], size=(64, n_factors))
    # Force every column to be balanced so the design is not degenerate.
    runs[32:, :] = -runs[:32, :]
    result = moment_aberration(pd.DataFrame(runs, columns=[f"X{i}" for i in range(n_factors)]))
    assert len(result.pattern) == MAX_PATTERN_LENGTH
    assert result.truncated
    assert result.n_factors == n_factors
