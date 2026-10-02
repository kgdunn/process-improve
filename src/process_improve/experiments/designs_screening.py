# (c) Kevin Dunn, 2010-2026. MIT License.

"""Screening designs: fractional factorial, Plackett-Burman, Taguchi.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing (center points, replication, randomization,
Column/Expt conversion) is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments._finite_fields import hadamard_matrix

try:
    from pyDOE3 import fracfact, fracfact_by_res, pbdesign, taguchi_design
except ImportError:  # pragma: no cover - exercised via env-without-pyDOE3
    from process_improve._extras import _MissingExtra

    fracfact = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]
    fracfact_by_res = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]
    pbdesign = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]
    taguchi_design = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]

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
        fraction 2^(k-1), of resolution k. Ignored when *generators* is provided.
    generators : list[str] or None
        Explicit generator strings, e.g. ``["D=ABC", "E=AC"]``.  When given,
        these are translated into the pyDOE3 generator notation.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1) and metadata dict with keys
        ``"generators_used"``, ``"defining_relation"`` and ``"resolution"``, the
        resolution the design achieves. A design with more than 11 factors that needs
        pyDOE3's search reports ``"resolution"`` only.

    Raises
    ------
    ValueError
        If fewer than 3 factors are given without generators, if *resolution* is below 3
        or above the number of factors, or if no checked design reaches it.
    """
    factor_names = [f.name for f in factors]
    k = len(factors)

    if generators:
        derived_idx, rhs_indices = _parse_generators(factor_names, generators)
        coded_matrix = _fracfact_from_indices(k, derived_idx, rhs_indices)
        parsed = [(lhs, rhs) for lhs, (rhs, _negated) in zip(derived_idx, rhs_indices, strict=True)]
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
            chosen = _minimum_aberration_generators(k, resolution)
            if chosen is None:
                return _fracfact_by_res_checked(k, resolution)
            parsed = chosen
        coded_matrix = _fracfact_from_indices(k, [lhs for lhs, _ in parsed], [(rhs, False) for _, rhs in parsed])
        generators_used = [f"{factor_names[lhs]}={''.join(factor_names[i] for i in rhs)}" for lhs, rhs in parsed]

    words = _defining_words(parsed)
    meta = {
        "generators_used": generators_used,
        "defining_relation": ["I=" + "".join(factor_names[i] for i in sorted(word)) for word in words],
        "resolution": min(len(word) for word in words),
    }
    return coded_matrix, meta


def _minimum_aberration_generators(k: int, resolution: int) -> list[_Generator] | None:
    """Choose the fewest-run minimum-aberration fraction of resolution at least *resolution*.

    The half fraction is the answer whenever no smaller fraction reaches *resolution*.
    Returns None for more than 11 factors when a smaller fraction might exist; pyDOE3
    then searches for one.
    """
    half_fraction = [(k - 1, list(range(k - 1)))]
    for _n_runs, entry in sorted(_MIN_ABERRATION.get(k, {}).items()):
        candidate = [(_TABLE_LETTERS.index(g[0]), [_TABLE_LETTERS.index(c) for c in g[2:]]) for g in entry]
        if min(len(word) for word in _defining_words(candidate)) >= resolution:
            return candidate
    # The table lists every fraction up to 11 factors. Beyond that, a quarter fraction
    # reaches at most resolution floor(2k/3), and smaller fractions no more than that.
    if k <= _MAX_TABULATED_FACTORS or resolution > 2 * k // 3:
        return half_fraction
    return None


def _defining_words(generators: list[_Generator]) -> list[frozenset[int]]:
    """Return every word of the defining relation, shortest first.

    A generator's word is its derived factor times its base factors; the defining
    relation holds every product of those words (symmetric differences, since each
    factor squares to the identity).
    """
    generator_words = []
    for lhs, rhs in generators:
        letters = {lhs}
        for index in rhs:
            letters ^= {index}  # a repeated base factor cancels
        generator_words.append(frozenset(letters))
    # No product is the identity: each generator word holds its own derived factor.
    words: set[frozenset[int]] = set()
    for size in range(1, len(generator_words) + 1):
        for subset in itertools.combinations(generator_words, size):
            product: frozenset[int] = frozenset()
            for generator_word in subset:
                product ^= generator_word
            words.add(product)
    return sorted(words, key=lambda word: (len(word), sorted(word)))


def _shortest_word_length(coded: np.ndarray) -> int:
    """Return the resolution of a coded fraction: the fewest columns whose product is constant.

    Shortest candidates are tried first, so the first constant product found is the
    answer. A fraction always has one, since its columns outnumber its base factors.
    """
    k = coded.shape[1]
    return next(
        size
        for size in range(1, k + 1)
        for columns in itertools.combinations(range(k), size)
        if np.all(np.prod(coded[:, columns], axis=1) == np.prod(coded[0, columns]))
    )


def _fracfact_by_res_checked(k: int, resolution: int) -> tuple[np.ndarray, dict]:
    """Search with pyDOE3 beyond the table, then measure the resolution it reached.

    ``fracfact_by_res`` does not always reach the resolution it is asked for: for 7 to
    11 factors at resolution V it returned resolution IV designs. So its design is
    measured from the columns instead of being taken on trust.
    """
    try:
        coded_matrix = fracfact_by_res(k, resolution)[:, :k]
    except ValueError as exc:
        raise ValueError(
            f"No resolution-{resolution} fraction for {k} factors is tabulated, and pyDOE3 found none ({exc}). "
            "Pass explicit generators instead."
        ) from exc
    achieved = _shortest_word_length(coded_matrix)
    if achieved < resolution:
        raise ValueError(
            f"No resolution-{resolution} fraction for {k} factors is tabulated, and pyDOE3's design only reaches "
            f"resolution {achieved}. Pass explicit generators instead."
        )
    return coded_matrix, {"resolution": achieved}


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


def dispatch_taguchi(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Taguchi orthogonal-array design.

    Selects the smallest standard orthogonal array that accommodates all
    factors and their levels.

    Parameters
    ----------
    factors : list[Factor]
        Factors with ``levels`` or 2-level continuous factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1 for 2-level factors) and metadata.
    """
    from pyDOE3 import list_orthogonal_arrays  # noqa: PLC0415

    k = len(factors)
    levels_per_factor: list[list[float]] = []
    for f in factors:
        if f.levels is not None and f.type.value == "categorical":
            levels_per_factor.append(list(range(len(f.levels))))
        else:
            levels_per_factor.append([-1, +1])

    n_levels = [len(lv) for lv in levels_per_factor]
    available = list_orthogonal_arrays()

    # Pick the smallest OA that fits
    selected_oa = None
    for oa_name in available:
        # Parse e.g. "L8(2^7)" to get max_factors and max levels
        parts = oa_name.split("(")
        # Check if this OA can accommodate our factors
        # Simple heuristic: need at least k columns and matching level counts
        inner = parts[1].rstrip(")")
        segments = inner.split(" ")
        total_columns = 0
        max_level = 0
        for seg in segments:
            base, exp = seg.split("^")
            total_columns += int(exp)
            max_level = max(max_level, int(base))

        if total_columns >= k and all(nl <= max_level for nl in n_levels):
            selected_oa = oa_name
            break

    if selected_oa is None:
        raise ValueError(
            f"No standard Taguchi orthogonal array found for {k} factors "
            f"with levels {n_levels}. Consider using a different design type."
        )

    coded_matrix = taguchi_design(selected_oa, levels_per_factor)

    # Trim to the number of factors we actually need
    coded_matrix = coded_matrix[:, :k]

    return coded_matrix, {"orthogonal_array": selected_oa}
