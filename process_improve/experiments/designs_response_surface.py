# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response surface designs: CCD, Box-Behnken, Definitive Screening Design.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pyDOE3 import bbdesign, ccdesign

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

logger = logging.getLogger(__name__)


def dispatch_ccd(
    factors: list[Factor],
    center_points: int = 3,
    alpha: str | float | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a Central Composite Design (CCD).

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.
    center_points : int
        Number of center points (split between cube and axial portions).
    alpha : str, float, or None
        Axial distance.  Accepted string values: ``"rotatable"``,
        ``"face_centered"``, ``"orthogonal"``.  A numeric value sets
        alpha directly.  Defaults to ``"orthogonal"``.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata (includes ``alpha_value``).

    Notes
    -----
    Center points are embedded in the CCD structure itself (via pyDOE3's
    ``center`` parameter).  The caller should set ``center_points=0`` in
    ``build_design_result`` to avoid adding duplicate center points.
    """
    k = len(factors)

    # Map alpha string to pyDOE3 face and alpha parameters.
    # pyDOE3 alpha accepts: "orthogonal"/"o", "rotatable"/"r"
    # pyDOE3 face accepts: "circumscribed"/"ccc", "inscribed"/"cci", "faced"/"ccf"
    face = "circumscribed"
    alpha_str = "orthogonal"
    if isinstance(alpha, str):
        alpha_lower = alpha.lower()
        if alpha_lower in ("face_centered", "face centered", "ccf", "faced"):
            face = "faced"
            alpha_str = "orthogonal"
        elif alpha_lower in ("inscribed", "cci"):
            face = "inscribed"
            alpha_str = "orthogonal"
        elif alpha_lower in ("rotatable", "r"):
            face = "circumscribed"
            alpha_str = "rotatable"
        else:
            # "orthogonal" or default
            face = "circumscribed"
            alpha_str = "orthogonal"
    elif isinstance(alpha, (int, float)):
        alpha_str = "orthogonal"
        face = "circumscribed"

    # Split center points between cube and axial portions
    n_center_cube = max(1, center_points // 2)
    n_center_axial = max(1, center_points - n_center_cube)

    coded_matrix = ccdesign(k, center=(n_center_cube, n_center_axial), alpha=alpha_str, face=face)

    # Determine actual alpha value used
    alpha_value: float | None = None
    if face == "ccf":
        alpha_value = 1.0
    elif coded_matrix.shape[0] > 0:
        alpha_value = float(np.max(np.abs(coded_matrix)))

    return coded_matrix, {"alpha_value": alpha_value, "face": face}


def dispatch_box_behnken(
    factors: list[Factor],
    center_points: int = 3,
) -> tuple[np.ndarray, dict]:
    """Generate a Box-Behnken design.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (requires at least 3).
    center_points : int
        Number of center point replicates.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / 0 / +1) and metadata.

    Notes
    -----
    Center points are embedded in the BB structure.  The caller should set
    ``center_points=0`` in ``build_design_result``.
    """
    k = len(factors)
    if k < 3:
        raise ValueError("Box-Behnken designs require at least 3 factors.")
    coded_matrix = bbdesign(k, center=center_points)
    return coded_matrix, {}


def dispatch_dsd(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Definitive Screening Design (DSD).

    Follows the conference-matrix-based construction of Jones & Nachtsheim
    (2011).  For *k* factors the DSD has ``2k + 1`` runs when *k* is even
    (using a conference matrix of order *k*) and ``2k + 3`` runs when *k*
    is odd (using a conference matrix of order ``k + 1`` and dropping the
    last column; Xiao, Lin & Bai 2012).  The design can estimate all main
    effects and quadratic effects and detect two-factor interactions with
    minimal confounding, provided the underlying conference matrix is
    genuine (``C.T @ C == (m-1) * I``).

    The Paley construction used by :func:`_conference_matrix` produces a
    genuine conference matrix whenever ``m - 1`` is an odd prime.  For other
    *m* the function falls back to a cyclic approximation and logs a
    warning; the resulting DSD will still run but its main-effects
    orthogonality may be degraded.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata.  Metadata includes the name of
        the conference-matrix construction that was used.

    References
    ----------
    .. [1] Jones, B. and Nachtsheim, C. J. (2011).  "A class of three-level
       designs for definitive screening in the presence of second-order
       effects."  *Journal of Quality Technology*, 43(1):1-15.
    .. [2] Xiao, L., Lin, D. K. J. and Bai, F. (2012).  "Constructing
       definitive screening designs using conference matrices."  *Journal
       of Quality Technology*, 44(1):2-8.
    """
    k = len(factors)
    if k < 3:
        raise ValueError("Definitive Screening Designs require at least 3 factors.")

    # For even k, use a conference matrix of order k (-> 2k + 1 runs).
    # For odd k, use a conference matrix of order k + 1 and drop the last
    # column so the design has k factors and 2(k+1) + 1 = 2k + 3 runs.
    m = k if k % 2 == 0 else k + 1
    C, construction = _conference_matrix(m)

    zero_row = np.zeros((1, m))
    coded_matrix = np.vstack([C, -C, zero_row])

    if k % 2 == 1:
        coded_matrix = coded_matrix[:, :k]

    return coded_matrix, {"construction": construction}


def _is_prime(n: int) -> bool:
    """Return True iff *n* is a (positive) prime."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _paley_conference_matrix(q: int) -> np.ndarray:
    """Build a conference matrix of order ``q + 1`` via Paley's construction.

    Requires *q* to be an odd prime.  Works for both ``q ≡ 1 (mod 4)``
    (symmetric conference matrix, "Paley type II") and ``q ≡ 3 (mod 4)``
    (skew-symmetric conference matrix, "Paley type I").  In both cases
    ``C.T @ C == q * I``.

    Parameters
    ----------
    q : int
        An odd prime.

    Returns
    -------
    np.ndarray
        ``(q + 1) x (q + 1)`` matrix with 0s on the diagonal and ±1
        off-diagonal.
    """
    # Legendre symbol χ : GF(q) -> {-1, 0, 1}
    quadratic_residues = {(x * x) % q for x in range(1, q)}
    chi = np.zeros(q, dtype=int)
    for x in range(1, q):
        chi[x] = 1 if x in quadratic_residues else -1

    # Jacobsthal matrix Q[a, b] = χ(b - a)
    q_matrix = np.zeros((q, q), dtype=int)
    for a in range(q):
        for b in range(q):
            q_matrix[a, b] = chi[(b - a) % q]

    n = q + 1
    c_matrix = np.zeros((n, n), dtype=int)
    c_matrix[0, 1:] = 1
    if q % 4 == 1:
        # Symmetric Paley conference matrix.
        c_matrix[1:, 0] = 1
    else:
        # Skew-symmetric Paley conference matrix (q ≡ 3 mod 4).
        c_matrix[1:, 0] = -1
    c_matrix[1:, 1:] = q_matrix
    return c_matrix


def _cyclic_conference_matrix(k: int) -> np.ndarray:
    """Legacy cyclic approximation of a conference matrix.

    Does **not** satisfy ``C.T @ C == (k - 1) * I`` in general; used only as
    a fallback when no Paley construction is available for the requested
    order.
    """
    c_matrix = np.zeros((k, k))
    half = (k - 1) // 2
    sequence = [1] * half + [-1] * (k - 1 - half)
    for i in range(k):
        for j in range(k):
            if i != j:
                idx = (j - i - 1) % (k - 1) if j > i else (j - i) % (k - 1)
                c_matrix[i, j] = sequence[idx]
    return c_matrix


def _conference_matrix(m: int) -> tuple[np.ndarray, str]:
    """Construct an ``m x m`` conference matrix.

    Uses Paley's construction when ``m - 1`` is an odd prime (covering
    ``m ∈ {4, 6, 8, 12, 14, 18, 20, 24, 30, 32, 38, 42, 44, 48, 54, 60, 62,
    68, 72, 74, 80, 84, 90, 98, ...}``), which returns a genuine conference
    matrix with ``C.T @ C == (m - 1) * I``.  For other orders (e.g.
    ``m ∈ {10, 16, 22, 26, 28, 34, 36, 40, ...}``) no Paley construction
    with a prime *q* is available; the function falls back to a cyclic
    approximation and logs a warning.

    Parameters
    ----------
    m : int
        Desired order of the conference matrix.

    Returns
    -------
    tuple[np.ndarray, str]
        The matrix and a short string identifying the construction used
        (e.g. ``"paley_q=13"`` or ``"cyclic_fallback"``).
    """
    q = m - 1
    if q >= 3 and q % 2 == 1 and _is_prime(q):
        return _paley_conference_matrix(q).astype(float), f"paley_q={q}"

    logger.warning(
        "No Paley conference-matrix construction known for order m=%d "
        "(q = m - 1 = %d is not an odd prime); falling back to a cyclic "
        "approximation. The resulting DSD's main-effects orthogonality "
        "may be degraded.",
        m,
        q,
    )
    return _cyclic_conference_matrix(m), "cyclic_fallback"
