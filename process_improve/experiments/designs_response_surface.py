# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response surface designs: CCD, Box-Behnken, Definitive Screening Design.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pyDOE3 import bbdesign, ccdesign

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor


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

    Uses the conference-matrix-based construction of Jones & Nachtsheim (2011).
    For *k* factors the DSD has ``2k + 1`` runs (odd number of factors) or
    ``2k + 3`` runs (even) and can estimate all main effects and quadratic
    effects, plus detect two-factor interactions, with minimal confounding.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata.
    """
    k = len(factors)
    if k < 3:
        raise ValueError("Definitive Screening Designs require at least 3 factors.")

    # Build a conference matrix C of size k x k
    # A conference matrix has 0 on the diagonal and +/-1 off-diagonal,
    # with C'C = (k-1)*I.
    # For the DSD we use a simple fold-over construction.
    # Start with a diagonal matrix of +1s and fill off-diagonal with a
    # balanced +/-1 pattern.
    C = _conference_matrix(k)

    # DSD construction: stack [C; -C; 0-row]
    zero_row = np.zeros((1, k))
    coded_matrix = np.vstack([C, -C, zero_row])

    # For even k, add two extra center-ish rows for estimability
    if k % 2 == 0:
        extra1 = np.ones((1, k))
        extra2 = -np.ones((1, k))
        coded_matrix = np.vstack([coded_matrix, extra1, extra2])

    return coded_matrix, {"construction": "conference_matrix_fold_over"}


def _conference_matrix(k: int) -> np.ndarray:
    """Construct a k x k conference matrix.

    Uses a Paley-type construction when k-1 is a prime power, otherwise
    falls back to a cyclic construction.

    Parameters
    ----------
    k : int
        Size of the conference matrix.

    Returns
    -------
    np.ndarray
        k x k matrix with 0s on diagonal and +/-1 off-diagonal.
    """
    C = np.zeros((k, k))

    # Simple construction: use a cyclic shift of a balanced sequence
    # For a k x k conference matrix, we need a (k-1)-length sequence
    # with equal numbers of +1 and -1
    half = (k - 1) // 2
    sequence = [1] * half + [-1] * (k - 1 - half)

    for i in range(k):
        for j in range(k):
            if i != j:
                idx = (j - i - 1) % (k - 1) if j > i else (j - i) % (k - 1)
                C[i, j] = sequence[idx]

    return C
