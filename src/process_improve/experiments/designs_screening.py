# (c) Kevin Dunn, 2010-2026. MIT License.

"""Screening designs: fractional factorial, Plackett-Burman, Taguchi.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing (center points, replication, randomization,
Column/Expt conversion) is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
        Desired minimum resolution (3, 4, or 5).  Ignored when *generators*
        is provided.
    generators : list[str] or None
        Explicit generator strings, e.g. ``["D=ABC", "E=AC"]``.  When given,
        these are translated into the pyDOE3 generator notation.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1) and metadata dict with keys
        ``"generators_used"`` and ``"resolution"``.
    """
    k = len(factors)
    meta: dict = {}

    if generators:
        coded_matrix = _fracfact_from_generators(factors, generators)
        meta["generators_used"] = generators
    elif resolution is not None:
        coded_matrix = fracfact_by_res(k, resolution)
        meta["resolution"] = resolution
    else:
        # Default: highest resolution that halves the runs
        res = min(k, 5)
        coded_matrix = fracfact_by_res(k, res)
        meta["resolution"] = res

    # Ensure the matrix has the right number of columns
    if coded_matrix.shape[1] != k:
        # fracfact_by_res may return more/fewer columns; trim or error
        coded_matrix = coded_matrix[:, :k]

    return coded_matrix, meta


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


def _fracfact_from_generators(factors: list[Factor], generators: list[str]) -> np.ndarray:
    """Build a coded fractional-factorial matrix from explicit generators.

    The previous implementation handed pyDOE3 the base factors followed by the
    derived ones and returned the columns in that order, while the caller
    assigns column ``i`` to ``factors[i]``: whenever a generator's left-hand
    factor was not the LAST factor (e.g. ``"B=AC"`` with factors A, B, C), the
    factor columns were silently swapped. It also lower-cased raw factor names
    into the pyDOE3 string, so any multi-character name was misread as a
    product of single-letter factors. Generators are now parsed against the
    real factor names, translated to canonical single letters for pyDOE3, and
    the resulting columns are re-ordered back to the caller's factor order.
    """
    factor_names = [f.name for f in factors]
    k = len(factors)
    derived_idx, rhs_indices = _parse_generators(factor_names, generators)
    base_idx = [i for i in range(k) if i not in derived_idx]

    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(base_idx) > len(letters):
        raise ValueError(f"At most {len(letters)} base factors are supported; got {len(base_idx)}.")
    base_letter = {factor_index: letters[pos] for pos, factor_index in enumerate(base_idx)}
    tokens = [base_letter[i] for i in base_idx]
    for rhs, negated in rhs_indices:
        word = "".join(base_letter[i] for i in rhs)
        tokens.append(f"-{word}" if negated else word)

    coded = fracfact(" ".join(tokens))
    if coded.shape[1] != k:
        raise ValueError(f"pyDOE3 returned {coded.shape[1]} columns for {k} factors; the generators are inconsistent.")
    # pyDOE3 column order is (bases..., derived...); map back to factor order.
    reordered = np.empty_like(coded)
    for position, factor_index in enumerate(base_idx + derived_idx):
        reordered[:, factor_index] = coded[:, position]
    return reordered


def dispatch_plackett_burman(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Plackett-Burman screening design.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / +1) and metadata.
    """
    k = len(factors)
    coded_matrix = pbdesign(k)
    return coded_matrix, {"note": f"Plackett-Burman design for {k} factors in {coded_matrix.shape[0]} runs"}


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
