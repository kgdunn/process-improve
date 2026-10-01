"""Fractional factorials from the minimum-aberration table (#620).

pyDOE3's ``fracfact_by_res`` could not build the 2^(3-1), 2^(4-1) or 2^(5-1) half
fractions ("design not possible"), and for 7 to 11 factors at resolution V it
returned resolution IV designs labelled V. Designs up to 11 factors now come from a
table of minimum-aberration generators, and the reported resolution is always the
one the design has: these tests measure it from the runs themselves.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from process_improve.experiments import Factor, generate_design
from process_improve.experiments.designs_screening import _MIN_ABERRATION, _TABLE_LETTERS

NAMES = "ABCDEFGHJKLMN"


def _factors(k: int) -> list[Factor]:
    return [Factor(name=name, low=-1, high=1) for name in NAMES[:k]]


def _fraction(k: int, **kwargs: object) -> object:
    return generate_design(_factors(k), design_type="fractional_factorial", n_center_points=0, **kwargs)


def _measured_resolution(coded: np.ndarray) -> int:
    """Return the fewest columns whose elementwise product is constant, found from the runs alone."""
    k = coded.shape[1]
    for size in range(1, k + 1):
        for columns in itertools.combinations(range(k), size):
            product = np.prod(coded[:, list(columns)], axis=1)
            if np.all(product == product[0]):
                return size
    raise AssertionError("The design is not a fraction: no product of columns is constant.")


def _coded(result: object) -> np.ndarray:
    return result.design[list(result.factor_names)].to_numpy(dtype=float)  # type: ignore[attr-defined]


def _word_length_pattern(generator_words: list[int]) -> tuple[int, ...]:
    """Return (A3, A4, ...): how many words of the defining relation have each length.

    Words are bitmasks over the factors; a product of words is their XOR.
    """
    counts: Counter[int] = Counter()
    for size in range(1, len(generator_words) + 1):
        for subset in itertools.combinations(generator_words, size):
            product = 0
            for word in subset:
                product ^= word
            counts[product.bit_count()] += 1
    assert counts[0] == counts[1] == counts[2] == 0, "a word of length 2 or less aliases two main effects"
    return tuple(counts[length] for length in range(3, max(counts) + 1))


def _table_words(k: int, n_runs: int) -> list[int]:
    """Return the generator words of a table entry, e.g. ``E=ABC`` as the bitmask of ABCE."""
    return [sum(1 << _TABLE_LETTERS.index(letter) for letter in g.replace("=", "")) for g in _MIN_ABERRATION[k][n_runs]]


class TestIssueReproducer:
    """The three calls from the issue."""

    def test_five_factors_at_resolution_five(self) -> None:
        result = _fraction(5, resolution=5)
        assert result.n_runs == 16
        assert result.generators == ["E=ABCD"]
        assert result.resolution == 5

    def test_five_factors_by_default(self) -> None:
        result = _fraction(5)
        assert (result.n_runs, result.generators, result.resolution) == (16, ["E=ABCD"], 5)

    def test_four_factors_by_default(self) -> None:
        result = _fraction(4)
        assert (result.n_runs, result.generators, result.resolution) == (8, ["D=ABC"], 4)


@pytest.mark.parametrize(
    ("k", "n_runs", "generator"),
    [
        (3, 4, "C=AB"),
        (4, 8, "D=ABC"),
        (5, 16, "E=ABCD"),
        (6, 32, "F=ABCDE"),
        (7, 64, "G=ABCDEF"),
        (8, 128, "H=ABCDEFG"),
    ],
)
def test_default_is_the_half_fraction(k: int, n_runs: int, generator: str) -> None:
    """Without resolution or generators: 2^(k-1) runs, the last factor the product of the rest."""
    result = _fraction(k)
    assert result.n_runs == n_runs
    assert result.generators == [generator]
    assert result.resolution == k
    assert result.defining_relation == [f"I={NAMES[:k]}"]
    assert _measured_resolution(_coded(result)) == k


#: (k, requested resolution) -> (runs, resolution reached, generators).
_FEWEST_RUNS = {
    (3, 3): (4, 3, ["C=AB"]),
    (4, 3): (8, 4, ["D=ABC"]),
    (4, 4): (8, 4, ["D=ABC"]),
    (5, 3): (8, 3, ["D=AB", "E=AC"]),
    (5, 4): (16, 5, ["E=ABCD"]),
    (5, 5): (16, 5, ["E=ABCD"]),
    (6, 3): (8, 3, ["D=AB", "E=AC", "F=BC"]),
    (6, 4): (16, 4, ["E=ABC", "F=BCD"]),
    (6, 5): (32, 6, ["F=ABCDE"]),
    (7, 3): (8, 3, ["D=AB", "E=AC", "F=BC", "G=ABC"]),
    (7, 4): (16, 4, ["E=ABC", "F=BCD", "G=ACD"]),
    (7, 5): (64, 7, ["G=ABCDEF"]),
    (8, 3): (16, 4, ["E=BCD", "F=ACD", "G=ABC", "H=ABD"]),
    (8, 4): (16, 4, ["E=BCD", "F=ACD", "G=ABC", "H=ABD"]),
    (8, 5): (64, 5, ["G=ABCD", "H=ABEF"]),
}


@pytest.mark.parametrize(("k", "resolution"), sorted(_FEWEST_RUNS))
def test_fewest_runs_that_reach_the_resolution(k: int, resolution: int) -> None:
    """A requested resolution is a minimum: the smallest fraction reaching it is chosen.

    That fraction can do better than asked (five factors at resolution IV get the
    16-run resolution V half fraction, since no 8-run design reaches IV), and the
    result reports what it reaches.
    """
    n_runs, reached, generators = _FEWEST_RUNS[k, resolution]
    result = _fraction(k, resolution=resolution)
    assert result.n_runs == n_runs
    assert result.generators == generators
    assert result.resolution == reached
    assert _measured_resolution(_coded(result)) == reached


#: (A3, A4, ...) of each entry. Each is the minimum over every possible set of
#: generators for that k and run count, found by exhaustive search, so each entry
#: has minimum aberration. The search for 8 and 16 runs is repeated below.
_MIN_WLP = {
    (5, 8): (2, 1),
    (6, 16): (0, 3),
    (6, 8): (4, 3),
    (7, 32): (0, 1, 2),
    (7, 16): (0, 7),
    (7, 8): (7, 7, 0, 0, 1),
    (8, 64): (0, 0, 2, 1),
    (8, 32): (0, 3, 4),
    (8, 16): (0, 14, 0, 0, 0, 1),
    (9, 128): (0, 0, 0, 3),
    (9, 64): (0, 1, 4, 2),
    (9, 32): (0, 6, 8, 0, 0, 1),
    (9, 16): (4, 14, 8, 0, 4, 1),
    (10, 256): (0, 0, 0, 1, 2),
    (10, 128): (0, 0, 3, 3, 1),
    (10, 64): (0, 2, 8, 4, 0, 1),
    (10, 32): (0, 10, 16, 0, 0, 5),
    (10, 16): (8, 18, 16, 8, 8, 5),
    (11, 512): (0, 0, 0, 0, 2, 1),
    (11, 256): (0, 0, 0, 6, 0, 1),
    (11, 128): (0, 0, 6, 6, 2, 1),
    (11, 64): (0, 4, 14, 8, 0, 3, 2),
    (11, 32): (0, 25, 0, 27, 0, 10, 0, 1),
    (11, 16): (12, 26, 28, 24, 20, 13, 4),
}


class TestTable:
    """Every entry of the table is a minimum-aberration fraction."""

    def test_every_run_count_is_listed(self) -> None:
        """Each k lists every fraction from the smallest (resolution III) to the quarter fraction.

        The selection returns the half fraction when no entry reaches a resolution; that
        is only right if no smaller fraction is missing.
        """
        assert set(_MIN_WLP) == {(k, n) for k, cells in _MIN_ABERRATION.items() for n in cells}
        for k, cells in _MIN_ABERRATION.items():
            smallest = math.ceil(math.log2(k + 1))  # 2^m runs hold at most 2^m - 1 factors
            assert sorted(cells) == [2**m for m in range(smallest, k - 1)], k

    @pytest.mark.parametrize(("k", "n_runs"), sorted(_MIN_WLP))
    def test_entry_is_a_fraction_with_the_minimum_pattern(self, k: int, n_runs: int) -> None:
        entry = _MIN_ABERRATION[k][n_runs]
        p = len(entry)
        assert n_runs == 2 ** (k - p)
        # The derived factors are the last p, and each is a product of base factors only.
        assert [g.split("=")[0] for g in entry] == list(_TABLE_LETTERS[k - p : k])
        assert all(_TABLE_LETTERS.index(letter) < k - p for g in entry for letter in g.split("=")[1])
        assert _word_length_pattern(_table_words(k, n_runs)) == _MIN_WLP[k, n_runs]

    @pytest.mark.parametrize(("k", "n_runs"), [cell for cell in sorted(_MIN_WLP) if cell[1] <= 16])
    def test_small_entries_minimum_by_exhaustive_search(self, k: int, n_runs: int) -> None:
        """Score every set of p interaction columns of the m base factors; none beats the table."""
        m = n_runs.bit_length() - 1
        interactions = [mask for mask in range(1 << m) if mask.bit_count() >= 2]
        best = min(
            _word_length_pattern([base | 1 << (m + j) for j, base in enumerate(chosen)])
            for chosen in itertools.combinations(interactions, k - m)
        )
        assert best == _MIN_WLP[k, n_runs]


class TestReportedMetadata:
    """The result carries the generators, defining relation and resolution of the design."""

    def test_explicit_generators_report_their_resolution(self) -> None:
        result = _fraction(5, generators=["D=AB", "E=AC"])
        assert result.resolution == 3
        assert result.defining_relation == ["I=ABD", "I=ACE", "I=BCDE"]

    def test_fractional_ccd_cube_really_is_resolution_v(self) -> None:
        """A resolution-V cube for 7 factors needs 64 runs.

        pyDOE3 gave a 32-run cube labelled V that was resolution IV, so the CCD's check
        for resolution V passed while two-factor interactions were aliased.
        """
        result = generate_design(
            _factors(7), design_type="ccd", cube="fractional", resolution=5, alpha="face_centered", n_center_points=0
        )
        coded = _coded(result)
        cube = coded[np.all(np.abs(coded) == 1, axis=1)]
        assert cube.shape == (64, 7)
        assert result.n_runs == 64 + 2 * 7
        assert _measured_resolution(cube) == result.resolution == 7


class TestBeyondTheTable:
    """More than 11 factors: pyDOE3's search, with its resolution checked."""

    def test_resolution_iv_from_pydoe3(self) -> None:
        result = _fraction(12, resolution=4)
        assert result.n_runs == 32
        assert result.resolution == 4
        assert _measured_resolution(_coded(result)) == 4

    def test_short_of_the_resolution_raises(self) -> None:
        """pyDOE3's 64-run "resolution V" design for 12 factors is resolution IV."""
        with pytest.raises(ValueError, match="only reaches resolution 4"):
            _fraction(12, resolution=5)

    def test_pydoe3_finding_nothing_raises(self) -> None:
        """Sixty factors at resolution 40 would need more base factors than pyDOE3 can name."""
        factors = [Factor(name=f"X{i}", low=-1, high=1) for i in range(60)]
        with pytest.raises(ValueError, match="pyDOE3 found none"):
            generate_design(factors, design_type="fractional_factorial", resolution=40, n_center_points=0)

    def test_more_base_factors_than_letters_raises(self) -> None:
        """The half fraction of 28 factors has 27 base factors; pyDOE3 names at most 26."""
        factors = [Factor(name=f"X{i}", low=-1, high=1) for i in range(28)]
        with pytest.raises(ValueError, match="At most 26 base factors"):
            generate_design(factors, design_type="fractional_factorial", n_center_points=0)

    def test_only_the_half_fraction_reaches_a_high_resolution(self) -> None:
        """No quarter fraction of 12 factors passes resolution floor(2 * 12 / 3) = 8."""
        result = _fraction(12, resolution=9)
        assert result.n_runs == 2**11
        assert result.resolution == 12


class TestInvalidRequests:
    @pytest.mark.parametrize("generator", ["D=", "D=-"])
    def test_generator_naming_no_factors(self, generator: str) -> None:
        """An empty right-hand side used to reach pyDOE3 and fail there with an IndexError."""
        with pytest.raises(ValueError, match="names no factors"):
            _fraction(4, generators=[generator])

    def test_fewer_than_three_factors(self) -> None:
        with pytest.raises(ValueError, match="at least 3 factors"):
            _fraction(2)

    @pytest.mark.parametrize("resolution", [2, 6])
    def test_resolution_out_of_range(self, resolution: int) -> None:
        with pytest.raises(ValueError, match="resolution from 3 to 5"):
            _fraction(5, resolution=resolution)
