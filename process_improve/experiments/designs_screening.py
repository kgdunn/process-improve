# (c) Kevin Dunn, 2010-2026. MIT License.

"""Screening designs: fractional factorial, Plackett-Burman, Taguchi.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing (center points, replication, randomization,
Column/Expt conversion) is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyDOE3 import fracfact, fracfact_by_res, pbdesign, taguchi_design

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
        # Convert ["D=ABC", "E=AC"] to pyDOE3 gen string "a b c abc ac"
        # The base factors are the first ones not appearing on the LHS
        lhs_names = [g.split("=")[0].strip() for g in generators]
        factor_names = [f.name for f in factors]
        base_names = [n for n in factor_names if n not in lhs_names]
        gen_parts = [n.lower() for n in base_names]
        for g in generators:
            rhs = g.split("=")[1].strip().lower()
            gen_parts.append(rhs)
        gen_string = " ".join(gen_parts)
        coded_matrix = fracfact(gen_string)
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
