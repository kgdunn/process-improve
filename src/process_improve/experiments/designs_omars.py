# (c) Kevin Dunn, 2010-2026. MIT License.

"""OMARS (Orthogonal Minimally Aliased Response Surface) designs.

OMARS designs are three-level designs (coded ``-1 / 0 / +1``) in which every
main effect is orthogonal to every other main effect **and** to all
second-order terms (the pure quadratics and the two-factor interactions),
confining all aliasing to the second-order block.  They occupy the middle
ground between screening designs and full response-surface designs.

This module provides two things:

* :func:`dispatch_omars` - a constructive generator wired into
  :func:`process_improve.experiments.generate_design` as
  ``design_type="omars"``.  It builds the conference-matrix foldover family of
  OMARS designs; the definitive screening design (DSD) is the minimal member
  of that family, so the construction is shared with
  :func:`process_improve.experiments.designs_response_surface.dispatch_dsd`.
* :func:`omars_properties` and :func:`is_omars` - dependency-free verifiers
  that check the defining OMARS properties on *any* coded design matrix.  Use
  them to validate designs we generate, designs produced by a future
  enumerator, or designs supplied from an external source.

A public catalogue of enumerated OMARS designs exists, but it is unlicensed
and is therefore **not** redistributed with this package.  The verifiers here
let a user cross-check a generated design against such a catalogue offline.

References
----------
.. [1] Núñez Ares, J. and Goos, P. (2020).  "Enumeration and multicriteria
   selection of orthogonal minimally aliased response surface designs."
   *Technometrics*, 62(1):21-36.
.. [2] Jones, B. and Nachtsheim, C. J. (2011).  "A class of three-level
   designs for definitive screening in the presence of second-order
   effects."  *Journal of Quality Technology*, 43(1):1-15.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

# Default numerical tolerance for treating an inner product as zero.  The
# constructive designs are integer-valued so exact-zero comparisons would also
# work, but a small tolerance keeps the verifiers usable on floating-point or
# slightly perturbed matrices.
_DEFAULT_TOL = 1e-9


def _second_order_terms(matrix: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build the second-order model terms (quadratics + two-factor interactions).

    Parameters
    ----------
    matrix : np.ndarray
        Coded design matrix of shape ``(n_runs, n_factors)``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        An ``(n_runs, n_terms)`` array whose columns are the ``k`` pure
        quadratics ``x_i^2`` followed by the ``k (k - 1) / 2`` two-factor
        interactions ``x_i x_j``, and a matching list of human-readable term
        names.
    """
    n_factors = matrix.shape[1]
    columns: list[np.ndarray] = []
    names: list[str] = []
    for i in range(n_factors):
        columns.append(matrix[:, i] * matrix[:, i])
        names.append(f"x{i + 1}^2")
    for i, j in itertools.combinations(range(n_factors), 2):
        columns.append(matrix[:, i] * matrix[:, j])
        names.append(f"x{i + 1}*x{j + 1}")
    if not columns:  # defensive: a zero-factor matrix has no second-order terms
        return np.empty((matrix.shape[0], 0)), names
    return np.column_stack(columns), names


def _max_abs_correlation(terms: np.ndarray) -> float:
    """Return the largest absolute pairwise correlation among the columns of *terms*.

    Constant columns (zero variance) are skipped.  Returns ``0.0`` when fewer
    than two non-constant columns are present.
    """
    if terms.shape[1] < 2:
        return 0.0
    centered = terms - terms.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    keep = norms > 0
    if keep.sum() < 2:
        return 0.0
    unit = centered[:, keep] / norms[keep]
    corr = unit.T @ unit
    off_diagonal = corr - np.diag(np.diag(corr))
    return float(np.abs(off_diagonal).max())


def omars_properties(matrix: np.ndarray, *, tol: float = _DEFAULT_TOL) -> dict:
    """Compute the OMARS-defining properties of a coded design matrix.

    The three properties that define an OMARS design are:

    1. **Three levels** - every entry is one of ``-1``, ``0`` or ``+1``.
    2. **Orthogonal main effects** - the main-effect columns are mutually
       orthogonal and balanced (each column sums to zero).
    3. **Minimally aliased** - every main effect is orthogonal to every
       second-order term (all quadratics and all two-factor interactions);
       aliasing is confined to the second-order block.

    Parameters
    ----------
    matrix : np.ndarray
        Coded design matrix of shape ``(n_runs, n_factors)``.
    tol : float
        Absolute tolerance below which an inner product is treated as zero.

    Returns
    -------
    dict
        A report with the keys ``n_runs``, ``n_factors``, ``levels``,
        ``is_three_level``, ``quadratics_estimable``, ``is_balanced``,
        ``main_effects_orthogonal``, ``main_effects_clear_of_second_order``,
        ``max_main_effect_inner_product``,
        ``max_main_vs_second_order_inner_product``,
        ``max_second_order_correlation`` and ``is_omars``.

    Notes
    -----
    ``quadratics_estimable`` is ``True`` only when every factor takes the
    middle (``0``) level at least once.  A two-level design (e.g. a full
    factorial) would otherwise pass the orthogonality checks but has constant,
    inestimable quadratic terms, so it is not an OMARS design.

    Examples
    --------
    >>> import numpy as np
    >>> # The minimal 3-factor OMARS / DSD (conference-matrix foldover).
    >>> from process_improve.experiments.factor import Factor
    >>> from process_improve.experiments.designs_omars import dispatch_omars
    >>> coded, _ = dispatch_omars([Factor(name=n, low=-1, high=1) for n in "ABC"])
    >>> omars_properties(coded)["is_omars"]
    True
    """
    matrix = np.asarray(matrix, dtype=float)
    n_runs, n_factors = matrix.shape

    levels = sorted({round(float(v), 9) for v in np.unique(matrix)})

    # Three-level validity: every entry is within tol of one of {-1, 0, +1}.
    nearest = np.round(matrix)
    is_three_level = bool(
        np.all(np.abs(matrix - nearest) <= tol) and np.all(np.isin(nearest, (-1.0, 0.0, 1.0)))
    )

    # OMARS factors are genuinely three-level: each must take the middle (0)
    # level at least once, otherwise its pure quadratic is a constant column
    # and cannot be estimated (as in a two-level factorial).
    uses_middle_level = np.any(np.abs(matrix) <= tol, axis=0)
    quadratics_estimable = bool(np.all(uses_middle_level))

    column_sums = np.abs(matrix.sum(axis=0))
    is_balanced = bool(np.all(column_sums <= tol))

    # Main-effect orthogonality: off-diagonal of X'X.
    gram = matrix.T @ matrix
    me_off_diagonal = gram - np.diag(np.diag(gram))
    max_me_inner = float(np.abs(me_off_diagonal).max()) if n_factors > 1 else 0.0
    main_effects_orthogonal = max_me_inner <= tol

    # Main effects clear of all second-order terms.
    second_order, _ = _second_order_terms(matrix)
    if second_order.shape[1] > 0:
        cross = matrix.T @ second_order
        max_me_vs_so = float(np.abs(cross).max())
    else:  # defensive: a zero-factor matrix has no second-order terms
        max_me_vs_so = 0.0
    main_effects_clear = max_me_vs_so <= tol

    max_so_corr = _max_abs_correlation(second_order)

    is_omars = bool(
        is_three_level
        and quadratics_estimable
        and is_balanced
        and main_effects_orthogonal
        and main_effects_clear
    )

    return {
        "n_runs": int(n_runs),
        "n_factors": int(n_factors),
        "levels": levels,
        "is_three_level": is_three_level,
        "quadratics_estimable": quadratics_estimable,
        "is_balanced": is_balanced,
        "main_effects_orthogonal": main_effects_orthogonal,
        "main_effects_clear_of_second_order": main_effects_clear,
        "max_main_effect_inner_product": max_me_inner,
        "max_main_vs_second_order_inner_product": max_me_vs_so,
        "max_second_order_correlation": max_so_corr,
        "is_omars": is_omars,
    }


def is_omars(matrix: np.ndarray, *, tol: float = _DEFAULT_TOL) -> bool:
    """Return ``True`` iff *matrix* satisfies the OMARS-defining properties.

    Convenience wrapper around :func:`omars_properties`.

    Parameters
    ----------
    matrix : np.ndarray
        Coded design matrix of shape ``(n_runs, n_factors)``.
    tol : float
        Absolute tolerance below which an inner product is treated as zero.

    Returns
    -------
    bool
        ``True`` if the design is three-level with orthogonal main effects
        that are clear of all second-order terms.
    """
    return omars_properties(matrix, tol=tol)["is_omars"]


def dispatch_omars(factors: list[Factor], *, verify: bool = True) -> tuple[np.ndarray, dict]:
    """Generate an OMARS design via the conference-matrix foldover construction.

    The construction is shared with :func:`dispatch_dsd`: for ``k`` factors it
    folds a conference matrix ``C`` of order ``m`` over its negative and adds a
    center run, giving ``[C; -C; 0]``.  ``m = k`` for even ``k`` (``2k + 1``
    runs) and ``m = k + 1`` with the last column dropped for odd ``k``
    (``2k + 3`` runs).  This yields the minimal OMARS design for ``k`` factors;
    a future enumerator will expand coverage to the larger, non-foldover
    members of the OMARS family.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (at least three).
    verify : bool
        When ``True`` (default) the generated matrix is checked with
        :func:`is_omars` and the result is recorded in the metadata under
        ``"omars_verified"``.  The check is cheap and guards against the
        degraded orthogonality of the cyclic conference-matrix fallback.

    Returns
    -------
    tuple[np.ndarray, dict]
        The coded design matrix and metadata.  Metadata includes
        ``"construction"`` (the conference-matrix construction used),
        ``"family"`` and, when *verify* is ``True``, ``"omars_verified"``.

    Raises
    ------
    ValueError
        If fewer than three factors are supplied.
    """
    from process_improve.experiments.designs_response_surface import dispatch_dsd  # noqa: PLC0415

    if len(factors) < 3:
        raise ValueError("OMARS designs require at least 3 factors.")

    coded_matrix, dsd_meta = dispatch_dsd(factors)
    meta = {**dsd_meta, "family": "conference_foldover"}
    if verify:
        meta["omars_verified"] = is_omars(coded_matrix)
    return coded_matrix, meta
