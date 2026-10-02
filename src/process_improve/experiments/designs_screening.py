# (c) Kevin Dunn, 2010-2026. MIT License.

"""Screening designs: fractional factorial, Plackett-Burman, Taguchi.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing (center points, replication, randomization,
Column/Expt conversion) is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments._finite_fields import hadamard_matrix
from process_improve.experiments.designs_utils import categorical_codes

try:
    from pyDOE3 import fracfact, pbdesign
except ImportError:  # pragma: no cover - exercised via env-without-pyDOE3
    from process_improve._extras import _MissingExtra

    fracfact = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]
    pbdesign = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

# Minimum-aberration 2^(k-p) fractional factorials for 5 to 11 factors, keyed by the number of
# factors and then by the run count 2^(k-p). Letters name factors by position, in the textbook
# convention A, B, ..., H, J, K, L (I stands for the identity), and the derived factors are the
# last p. The half fraction, with the last factor equal to the product of all the others, is
# built for any k, so it is not listed. The designs are the standard ones (Montgomery, "Design
# and Analysis of Experiments", Chapter 8); an exhaustive search over every generator set
# confirmed that each has minimum aberration, and the tests pin its word-length pattern.
_MIN_ABERRATION: dict[int, dict[int, tuple[str, ...]]] = {
    5: {8: ("D=AB", "E=AC")},
    6: {16: ("E=ABC", "F=BCD"), 8: ("D=AB", "E=AC", "F=BC")},
    7: {
        32: ("F=ABCD", "G=ABDE"),
        16: ("E=ABC", "F=BCD", "G=ACD"),
        8: ("D=AB", "E=AC", "F=BC", "G=ABC"),
    },
    8: {
        64: ("G=ABCD", "H=ABEF"),
        32: ("F=ABC", "G=ABD", "H=BCDE"),
        16: ("E=BCD", "F=ACD", "G=ABC", "H=ABD"),
    },
    9: {
        128: ("H=ACDFG", "J=BCEFG"),
        64: ("G=ABCD", "H=ACEF", "J=CDEF"),
        32: ("F=BCDE", "G=ACDE", "H=ABDE", "J=ABCE"),
        16: ("E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD"),
    },
    10: {
        256: ("J=ABCDEF", "K=ABCDGH"),
        128: ("H=ABCG", "J=BCDE", "K=ACDF"),
        64: ("G=BCDF", "H=ACDF", "J=ABDE", "K=ABCE"),
        32: ("F=ABCD", "G=ABCE", "H=ABDE", "J=ACDE", "K=BCDE"),
        16: ("E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD", "K=AB"),
    },
    11: {
        512: ("K=ABCDEF", "L=ABCGHJ"),
        256: ("J=ABCDE", "K=ABCFG", "L=ABDFH"),
        128: ("H=ABCG", "J=BCDE", "K=ACDF", "L=ABCDEFG"),
        64: ("G=CDE", "H=ABCD", "J=ABF", "K=BDEF", "L=ADEF"),
        32: ("F=ABC", "G=BCD", "H=CDE", "J=ACD", "K=ADE", "L=BDE"),
        16: ("E=ABC", "F=BCD", "G=ACD", "H=ABD", "J=ABCD", "K=AB", "L=AC"),
    },
}
_TABLE_LETTERS = "ABCDEFGHJKL"
_MAX_TABULATED_FACTORS = 11

# (derived factor index, base factor indices): one generator, e.g. D=ABC is (3, [0, 1, 2]).
_Generator = tuple[int, list[int]]


def dispatch_fractional_factorial(
    factors: list[Factor],
    resolution: int | None = None,
    generators: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a 2-level fractional factorial design.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (all treated as 2-level).
    resolution : int or None
        Desired minimum resolution (3 or more). The design is the minimum-aberration
        fraction with the fewest runs that reaches it, so its resolution can be higher.
        When neither *resolution* nor *generators* is given, the design is the half
        fraction 2^(k-1), of resolution k. With *generators*, it is checked instead:
        generators that reach a lower resolution raise ``ValueError``.
    generators : list[str] or None
        Explicit generator strings, e.g. ``["D=ABC", "E=AC"]``.  When given,
        these are translated into the pyDOE3 generator notation.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1) and metadata dict with keys
        ``"generators_used"``, ``"defining_relation"`` and ``"resolution"``, the
        resolution the design achieves. A word of the defining relation that is -1 on
        every run (from a negated generator such as ``"D=-AB"``) is written ``"I=-ABD"``.
        A defining relation of more than 1023 words (more than ten generators) is not
        listed.

    Raises
    ------
    ValueError
        If fewer than 3 factors are given without generators, if *resolution* is below 3
        or above the number of factors, or if *generators* fall short of *resolution*.
    """
    factor_names = [f.name for f in factors]
    k = len(factors)

    if generators:
        derived_idx, rhs_indices = _parse_generators(factor_names, generators)
        coded_matrix = _fracfact_from_indices(k, derived_idx, rhs_indices)
        parsed = [(lhs, rhs) for lhs, (rhs, _negated) in zip(derived_idx, rhs_indices, strict=True)]
        signs = [-1 if negated else 1 for _rhs, negated in rhs_indices]
        generators_used = list(generators)
    else:
        if k < 3:
            raise ValueError(f"A fractional factorial needs at least 3 factors, got {k}; use a full factorial.")
        if resolution is not None and not 3 <= resolution <= k:
            raise ValueError(
                f"A fractional factorial in {k} factors has a resolution from 3 to {k} (the half fraction); "
                f"got resolution={resolution}. Use a full factorial for more."
            )
        if resolution is None:
            parsed = [(k - 1, list(range(k - 1)))]  # the half fraction
        else:
            parsed = _minimum_aberration_generators(k, resolution)
        signs = [1] * len(parsed)
        coded_matrix = _fracfact_from_indices(k, [lhs for lhs, _ in parsed], [(rhs, False) for _, rhs in parsed])
        generators_used = [f"{factor_names[lhs]}={''.join(factor_names[i] for i in rhs)}" for lhs, rhs in parsed]

    words = _defining_word_masks(parsed, signs)
    achieved = min(mask.bit_count() for mask, _ in words)
    if generators and resolution is not None and achieved < resolution:
        raise ValueError(
            f"The generators {list(generators)} give a resolution-{achieved} design, below the resolution={resolution} "
            "asked for. Pass either generators or resolution, or generators that reach the resolution."
        )
    meta: dict = {"generators_used": generators_used, "resolution": achieved}
    if len(words) <= _MAX_LISTED_WORDS:
        words.sort(key=lambda item: (item[0].bit_count(), [i for i in range(k) if item[0] >> i & 1]))
        meta["defining_relation"] = [_format_word(mask, sign, factor_names) for mask, sign in words]
    return coded_matrix, meta


#: Longest defining relation listed in the metadata (ten generators); a longer one is
#: left out, since the generators define it.
_MAX_LISTED_WORDS = 1023


def _format_word(mask: int, sign: int, factor_names: list[str]) -> str:
    """Format a defining word as ``"I=ABD"``, or ``"I=-ABD"`` in a fraction where the word is -1."""
    return ("I=-" if sign < 0 else "I=") + "".join(name for i, name in enumerate(factor_names) if mask >> i & 1)


def _minimum_aberration_generators(k: int, resolution: int) -> list[_Generator]:
    """Choose the fewest-run fraction of resolution at least *resolution*.

    Up to 11 factors the fraction comes from the minimum-aberration table, and the half
    fraction is the answer whenever no smaller tabulated fraction reaches *resolution*.
    Beyond 11 factors :func:`_search_fraction` looks for the smallest fraction that
    reaches it; that fraction is not necessarily of minimum aberration.
    """
    half_fraction = [(k - 1, list(range(k - 1)))]
    if k > _MAX_TABULATED_FACTORS:
        return _search_fraction(k, resolution)
    for _n_runs, entry in sorted(_MIN_ABERRATION.get(k, {}).items()):
        candidate = [(_TABLE_LETTERS.index(g[0]), [_TABLE_LETTERS.index(c) for c in g[2:]]) for g in entry]
        if min(len(word) for word in _defining_words(candidate)) >= resolution:
            return candidate
    return half_fraction


#: Branches the generator search for more than 11 factors may try at one run size
#: before it moves to the next larger size; it keeps the search to about a second.
_SEARCH_NODE_LIMIT = 200_000


def _search_fraction(k: int, resolution: int) -> list[_Generator]:
    """Find a 2^(k-p) fraction of resolution at least *resolution*, trying the fewest runs first.

    At each run size ``2^(k-p)`` a depth-first search picks the ``p`` generator words
    one at a time, keeping only choices that leave every word of the defining relation
    at least *resolution* letters long. The search at a size gives up after
    :data:`_SEARCH_NODE_LIMIT` branches, so a smaller fraction may exist when the
    returned one is larger; the half fraction, of resolution ``k``, always qualifies.
    """
    for n_base in range(max(2, k.bit_length()), k - 1):
        n_extra = k - n_base
        # A generator word is the derived factor times at least resolution - 1 base factors.
        sizes = range(max(2, resolution - 1), n_base + 1)
        if sum(math.comb(n_base, size) for size in sizes) > _SEARCH_NODE_LIMIT:
            continue  # too many candidate words to search at this size
        candidates = [
            sum(1 << i for i in combo)
            for size in sizes
            for combo in itertools.combinations(range(n_base), size)
        ]
        found = _search_generator_words(candidates, n_base, n_extra, resolution)
        if found is not None:
            return [(n_base + j, [i for i in range(n_base) if mask >> i & 1]) for j, mask in enumerate(found)]
    return [(k - 1, list(range(k - 1)))]  # the half fraction, of resolution k


def _search_generator_words(candidates: list[int], n_base: int, n_extra: int, resolution: int) -> list[int] | None:
    """Depth-first search for *n_extra* generator words (bit masks of base factors), or None."""
    nodes = 0

    def extend(start: int, words: list[int], chosen: list[int]) -> list[int] | None:
        nonlocal nodes
        if len(chosen) == n_extra:
            return chosen
        for index in range(start, len(candidates)):
            nodes += 1
            if nodes > _SEARCH_NODE_LIMIT:
                return None
            generator = candidates[index] | (1 << (n_base + len(chosen)))
            new_words = [generator] + [word ^ generator for word in words]
            if all(word.bit_count() >= resolution for word in new_words):
                found = extend(index + 1, words + new_words, [*chosen, candidates[index]])
                if found is not None or nodes > _SEARCH_NODE_LIMIT:
                    return found
        return None

    return extend(0, [], [])


def _defining_word_masks(generators: list[_Generator], signs: list[int] | None = None) -> list[tuple[int, int]]:
    """Return every word of the defining relation as ``(bit mask of factors, sign)``.

    A generator's word is its derived factor times its base factors; the defining
    relation holds every product of those words (symmetric differences, since each
    factor squares to the identity), and a product's sign is the product of the signs
    of the generators in it. No product is the identity: each generator word holds its
    own derived factor.
    """
    words: list[tuple[int, int]] = []
    for (lhs, rhs), sign in zip(generators, signs or [1] * len(generators), strict=True):
        mask = 1 << lhs
        for index in rhs:
            mask ^= 1 << index  # a repeated base factor cancels
        words += [(mask, sign)] + [(word ^ mask, word_sign * sign) for word, word_sign in words]
    return words


def _defining_words(generators: list[_Generator]) -> list[frozenset[int]]:
    """Return every word of the defining relation as a set of factor indices, shortest first."""
    words = [frozenset(i for i in range(mask.bit_length()) if mask >> i & 1) for mask, _ in _defining_word_masks(generators)]
    return sorted(words, key=lambda word: (len(word), sorted(word)))


def _parse_generator_word(word: str, factor_names: list[str]) -> list[int]:
    """Parse a generator word (e.g. ``"ABC"``) into factor indices by NAME.

    Uses the same convention as ``evaluate._parse_word``: character-by-character
    when every factor name is a single character, greedy longest-name-first
    matching otherwise. Unlike that helper, unparseable content raises instead
    of being silently skipped: a generator that does not resolve to real
    factors would otherwise produce a design for the wrong fraction.
    """
    name_to_idx = {name: i for i, name in enumerate(factor_names)}
    indices: list[int] = []
    if all(len(n) == 1 for n in factor_names):
        for ch in word:
            if ch not in name_to_idx:
                raise ValueError(f"Generator word {word!r} refers to {ch!r}, which is not a factor name.")
            indices.append(name_to_idx[ch])
        return indices
    remaining = word
    sorted_names = sorted(name_to_idx, key=len, reverse=True)
    while remaining:
        for name in sorted_names:
            if remaining.startswith(name):
                indices.append(name_to_idx[name])
                remaining = remaining[len(name) :]
                break
        else:
            raise ValueError(f"Generator word {word!r} does not resolve to factor names {factor_names}.")
    return indices


def _parse_generators(factor_names: list[str], generators: list[str]) -> tuple[list[int], list[tuple[list[int], bool]]]:
    """Parse and validate generator strings into (derived indices, rhs terms)."""
    derived_idx: list[int] = []
    rhs_indices: list[tuple[list[int], bool]] = []
    for g in generators:
        if "=" not in g:
            raise ValueError(f"Generator {g!r} must have the form 'D=ABC' (or 'D=-ABC').")
        lhs_word, rhs_word = (part.strip() for part in g.split("=", 1))
        negated = rhs_word.startswith("-")
        rhs_word = rhs_word.lstrip("+-").strip()
        lhs = _parse_generator_word(lhs_word, factor_names)
        if len(lhs) != 1:
            raise ValueError(f"Generator {g!r}: the left-hand side must be exactly one factor.")
        rhs = _parse_generator_word(rhs_word, factor_names)
        if not rhs:
            raise ValueError(f"Generator {g!r}: the right-hand side names no factors.")
        if lhs[0] in rhs:
            raise ValueError(f"Generator {g!r}: the left-hand factor may not appear on the right-hand side.")
        if lhs[0] in derived_idx:
            raise ValueError(f"Generator {g!r}: factor {factor_names[lhs[0]]!r} is derived more than once.")
        derived_idx.append(lhs[0])
        rhs_indices.append((rhs, negated))

    base_idx = [i for i in range(len(factor_names)) if i not in derived_idx]
    for (rhs, _neg), g in zip(rhs_indices, generators, strict=True):
        not_base = [i for i in rhs if i not in base_idx]
        if not_base:
            names = [factor_names[i] for i in not_base]
            raise ValueError(f"Generator {g!r}: right-hand factors {names} are themselves derived factors.")
    return derived_idx, rhs_indices


def _fracfact_from_indices(k: int, derived_idx: list[int], rhs_indices: list[tuple[list[int], bool]]) -> np.ndarray:
    """Build the coded matrix for parsed generators, with column ``i`` belonging to factor ``i``.

    pyDOE3 is handed canonical single letters (so multi-character factor names are
    never misread as products) and returns the base factors followed by the derived
    ones. The columns are then put back in factor order: returning pyDOE3's order
    once swapped columns silently whenever a derived factor was not the last one
    (e.g. ``"B=AC"`` with factors A, B, C).
    """
    base_idx = [i for i in range(k) if i not in derived_idx]

    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(base_idx) > len(letters):
        raise ValueError(f"At most {len(letters)} base factors are supported; got {len(base_idx)}.")
    base_letter = {factor_index: letters[pos] for pos, factor_index in enumerate(base_idx)}
    tokens = [base_letter[i] for i in base_idx]
    for rhs, negated in rhs_indices:
        word = "".join(base_letter[i] for i in rhs)
        tokens.append(f"-{word}" if negated else word)

    # One non-empty token per factor, so pyDOE3 returns exactly k columns, in the order
    # (bases..., derived...); map them back to factor order.
    coded = fracfact(" ".join(tokens))
    reordered = np.empty_like(coded)
    for position, factor_index in enumerate(base_idx + derived_idx):
        reordered[:, factor_index] = coded[:, position]
    return reordered


def plackett_burman_runs(n_factors: int) -> int:
    """Return the number of runs in the Plackett-Burman design for ``n_factors`` factors.

    The smallest multiple of 4 above ``n_factors`` for which a Hadamard matrix is
    built here (every order up to 200 except 92, 116, 156, 172, 184 and 188, which
    step up to the next multiple of 4).
    """
    n = 4 * (n_factors // 4 + 1)
    while hadamard_matrix(n) is None:
        n += 4
    return n


def dispatch_plackett_burman(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Plackett-Burman screening design.

    The ``N``-run design is ``k`` columns of a normalised Hadamard matrix of order
    ``N``, the smallest multiple of 4 above ``k`` that can be built (see
    :func:`plackett_burman_runs`). For the orders pyDOE3 covers (powers of 2, and 12 or
    20 times a power of 2) its matrices are used, which for 12, 20 and 24 runs are the
    cyclic designs Plackett and Burman (1946) published. The other orders (28, 36, 44,
    52, ...) come from the finite-field constructions in
    :mod:`process_improve.experiments._finite_fields`, checked against ``H'H = N I``.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1) and metadata, including the ``construction``.
    """
    k = len(factors)
    n = plackett_burman_runs(k)
    coded_matrix = None
    with contextlib.suppress(AssertionError, IndexError, ValueError):  # pyDOE3 asserts on orders it lacks
        coded_matrix = pbdesign(k)
    if coded_matrix is not None and coded_matrix.shape[0] == n:
        construction = "pyDOE3"
    else:
        hadamard, construction = hadamard_matrix(n)  # type: ignore[misc]  # not None: n was chosen so
        coded_matrix = hadamard[:, 1 : k + 1].astype(float)
    return coded_matrix, {
        "construction": construction,
        "note": f"Plackett-Burman design for {k} factors in {n} runs",
    }


def _is_strength_two(array: np.ndarray) -> bool:
    """Whether every pair of columns shows every level combination equally often."""
    for i, j in itertools.combinations(range(array.shape[1]), 2):
        _, counts = np.unique(array[:, [i, j]], axis=0, return_counts=True)
        if len(counts) != (array[:, i].max() + 1) * (array[:, j].max() + 1) or counts.min() != counts.max():
            return False
    return True


def _taguchi_codes(factor: Factor, n_levels: int) -> np.ndarray:
    """Coded settings for a factor's ``n_levels`` levels: equally spaced, or its own ``levels`` coded to [-1, 1]."""
    if factor.type.value != "categorical" and factor.levels:
        low, high = float(factor.low), float(factor.high)  # type: ignore[arg-type]
        values = np.unique(np.asarray(factor.levels, dtype=float))
        return (values - (low + high) / 2.0) / ((high - low) / 2.0)
    return categorical_codes(n_levels)


def dispatch_taguchi(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Taguchi orthogonal-array design.

    Uses the smallest of pyDOE3's standard arrays that has a column for every factor:
    two levels for a continuous factor, the number of its ``levels`` for a categorical
    factor or a continuous factor given explicit levels. Each factor takes its own
    column (one with its level count, or else one whose levels divide evenly into its
    own, as a 6-level column carries a 3-level factor); the other columns are left
    unused. The chosen columns are checked to be balanced in every pair (strength 2)
    before use. No centre points are added.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (two levels, or the values in ``levels``) and categorical factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1 for two levels, equally spaced codes otherwise) and
        metadata with the ``orthogonal_array`` and the array ``columns`` used.
    """
    from pyDOE3 import get_orthogonal_array, list_orthogonal_arrays  # noqa: PLC0415

    needed = [len(f.levels) if f.levels else 2 for f in factors]
    order = sorted(range(len(factors)), key=lambda i: -needed[i])
    for name in list_orthogonal_arrays():  # in order of run count
        array = get_orthogonal_array(name)
        # Column level counts come from the array itself: some pyDOE3 names do not match
        # their arrays (its "L27(2^1 3^12)" is all three-level, its "L36(3^23)" has eleven
        # two-level columns).
        free = [int(array[:, j].max()) + 1 for j in range(array.shape[1])]
        columns: dict[int, int] = {}
        for i in order:
            open_columns = [c for c in range(len(free)) if c not in columns.values()]
            # A column of exactly the factor's level count, or else one whose levels split
            # evenly into the factor's (a 6-level column carries a 2- or 3-level factor).
            column = next((c for c in open_columns if free[c] == needed[i]), None)
            if column is None:
                column = next((c for c in open_columns if free[c] % needed[i] == 0), None)
            if column is None:
                break
            columns[i] = column
        if len(columns) < len(factors):
            continue
        chosen = [columns[i] for i in range(len(factors))]
        levels = np.column_stack([array[:, chosen[i]] % needed[i] for i in range(len(factors))])
        # pyDOE3's "L64(2^31)" is not orthogonal, so the chosen columns are checked, not trusted.
        if _is_strength_two(levels):
            coded = np.column_stack([_taguchi_codes(f, needed[i])[levels[:, i]] for i, f in enumerate(factors)]).astype(
                float
            )
            return coded, {"orthogonal_array": name, "columns": chosen}
    raise ValueError(
        f"No standard orthogonal array has columns for {len(factors)} factors with {sorted(needed)} levels; "
        "use 'full_factorial' or an optimal design instead."
    )
