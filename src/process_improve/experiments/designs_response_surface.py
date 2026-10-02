# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response surface designs: CCD, Box-Behnken, Definitive Screening Design.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments._finite_fields import conference_matrix, conference_order_at_least

try:
    from pyDOE3 import bbdesign, ccdesign
except ImportError:  # pragma: no cover - exercised via env-without-pyDOE3
    from process_improve._extras import _MissingExtra

    bbdesign = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]
    ccdesign = _MissingExtra("pyDOE3", "expt")  # type: ignore[assignment]

if TYPE_CHECKING:
    from process_improve.experiments.factor import Factor

logger = logging.getLogger(__name__)


def dispatch_ccd(  # noqa: PLR0913
    factors: list[Factor],
    n_center_points: int = 3,
    alpha: str | float | None = None,
    cube: str = "full",
    generators: list[str] | None = None,
    resolution: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate a Central Composite Design (CCD).

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.
    n_center_points : int
        Number of center points (split between cube and axial portions).
    alpha : str, float, or None
        Axial distance.  Accepted string values: ``"rotatable"``,
        ``"face_centered"``, ``"orthogonal"``.  A numeric value sets
        alpha directly.  Defaults to ``"orthogonal"``.

        .. note::
           A numeric ``alpha`` is only honored when ``cube="fractional"``.
           For ``cube="full"`` (the default) the underlying pyDOE3
           ``ccdesign`` call does not accept an arbitrary axial distance,
           so a numeric value is silently treated as ``"orthogonal"``.
    cube : str
        How to build the cube (factorial) portion: ``"full"`` (default) uses
        the complete 2^k factorial; ``"fractional"`` uses a resolution-V (or
        higher) fractional factorial, keeping the run count practical for
        k >= 5.
    generators : list[str] or None
        Explicit cube generators (e.g. ``["E=ABCD"]``), used only when
        ``cube="fractional"``.  When omitted, a minimum-aberration
        half-fraction is chosen automatically.
    resolution : int or None
        Desired minimum cube resolution, used only when ``cube="fractional"``
        and *generators* is not given.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata (includes ``alpha_value``).

    Notes
    -----
    Center points are embedded in the CCD structure itself (via pyDOE3's
    ``center`` parameter).  The caller should set ``n_center_points=0`` in
    ``build_design_result`` to avoid adding duplicate center points.
    """
    if cube == "fractional":
        return _dispatch_ccd_fractional(factors, n_center_points, alpha, generators, resolution)
    if cube != "full":
        raise ValueError(f"cube must be 'full' or 'fractional', got {cube!r}.")

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
    n_center_cube = max(1, n_center_points // 2)
    n_center_axial = max(1, n_center_points - n_center_cube)

    coded_matrix = ccdesign(k, center=(n_center_cube, n_center_axial), alpha=alpha_str, face=face)

    # Determine actual alpha value used
    alpha_value: float | None = None
    if face == "ccf":
        alpha_value = 1.0
    elif coded_matrix.shape[0] > 0:
        alpha_value = float(np.max(np.abs(coded_matrix)))

    return coded_matrix, {"alpha_value": alpha_value, "face": face}


def _resolve_fractional_axial_distance(
    alpha: str | float | None,
    n_cube_runs: int,
    k: int,
    n_center_points: int,
) -> tuple[float, str]:
    """Axial (star-point) distance for a fractional-cube CCD.

    Mirrors pyDOE3's :func:`star` formulas, but uses the actual number of
    fractional cube runs *n_cube_runs* in place of the full ``2**k``.

    Parameters
    ----------
    alpha : str, float, or None
        ``"face_centered"`` (alpha = 1), ``"rotatable"``
        (alpha = n_cube_runs ** 0.25), ``"orthogonal"`` / None (the orthogonal
        formula), or a numeric value used directly.
    n_cube_runs : int
        Number of runs in the (fractional) cube portion.
    k : int
        Number of factors.
    n_center_points : int
        Total number of center points; split between the cube and axial blocks
        for the orthogonal-alpha formula.

    Returns
    -------
    tuple[float, str]
        The axial distance and a short label for the design metadata.
    """
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        return float(alpha), "user"

    alpha_lower = alpha.lower() if isinstance(alpha, str) else None
    if alpha_lower in ("face_centered", "face centered", "ccf", "faced"):
        return 1.0, "faced"
    if alpha_lower in ("rotatable", "r"):
        return float(n_cube_runs**0.25), "rotatable"
    if alpha_lower in ("inscribed", "cci"):
        raise ValueError(
            "alpha='inscribed' is not supported with cube='fractional'; "
            "use 'face_centered', 'rotatable', 'orthogonal', or a numeric alpha."
        )

    # "orthogonal", None, or any other string: orthogonal axial distance.
    n_center_cube = n_center_points // 2
    n_center_axial = n_center_points - n_center_cube
    n_axial = 2 * k
    a = (k * (1 + n_center_axial / n_axial) / (1 + n_center_cube / n_cube_runs)) ** 0.5
    return float(a), "orthogonal"


def _dispatch_ccd_fractional(
    factors: list[Factor],
    n_center_points: int,
    alpha: str | float | None,
    generators: list[str] | None,
    resolution: int | None,
) -> tuple[np.ndarray, dict]:
    """Build a CCD whose cube portion is a resolution-V fractional factorial.

    The cube is generated by reusing
    :func:`process_improve.experiments.designs_screening.dispatch_fractional_factorial`,
    then the axial (star) runs and the center runs are stacked on top.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (at least 3).
    n_center_points : int
        Total number of center runs (added once, not split).
    alpha : str, float, or None
        Axial distance specification; see
        :func:`_resolve_fractional_axial_distance`.
    generators : list[str] or None
        Explicit cube generators.  When omitted, a minimum-aberration
        half-fraction (last factor = product of all the others) is used.
    resolution : int or None
        Desired minimum cube resolution; used only when *generators* is None.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata, including ``alpha_value``,
        ``generators_used``, ``defining_relation``, and ``resolution``.
    """
    from process_improve.experiments.designs_screening import dispatch_fractional_factorial  # noqa: PLC0415
    from process_improve.experiments.evaluate import (  # noqa: PLC0415
        _defining_relation_from_generators,
        _word_to_str,
    )

    k = len(factors)
    if k < 3:
        raise ValueError("A fractional-cube CCD requires at least 3 factors; use cube='full' for fewer.")

    factor_names = [f.name for f in factors]

    if generators is None and resolution is None:
        # Minimum-aberration half-fraction: last factor = product of all the others.
        generators = [f"{factor_names[-1]}={''.join(factor_names[:-1])}"]

    cube, frac_meta = dispatch_fractional_factorial(factors, resolution=resolution, generators=generators)
    n_cube_runs = cube.shape[0]

    # Record the cube's generators, defining relation, and (true) resolution.
    used_generators = frac_meta.get("generators_used") or generators
    res = frac_meta.get("resolution")
    defining_relation: list[str] | None = None
    if used_generators:
        words = _defining_relation_from_generators(used_generators, factor_names)
        defining_relation = [f"I={_word_to_str(w, factor_names)}" for w in words]
        if words:
            res = min(len(w) for w in words)

    if res is not None and res < 5:
        raise ValueError(
            f"The fractional cube has resolution {res}, but a CCD needs resolution V or higher so the "
            "full quadratic model is estimable. Supply resolution-V generators or use cube='full'."
        )

    axial_distance, face = _resolve_fractional_axial_distance(alpha, n_cube_runs, k, n_center_points)

    # Axial (star) runs: 2k rows at +/- axial_distance, zeros elsewhere.
    star = np.zeros((2 * k, k))
    for i in range(k):
        star[2 * i : 2 * i + 2, i] = (-axial_distance, axial_distance)

    center = np.zeros((max(0, n_center_points), k))

    coded_matrix = np.vstack([cube, star, center])
    meta = {
        "alpha_value": axial_distance,
        "face": face,
        "cube": "fractional",
        "generators_used": used_generators,
        "defining_relation": defining_relation,
        "resolution": res,
    }
    return coded_matrix, meta


def dispatch_box_behnken(
    factors: list[Factor],
    n_center_points: int = 3,
) -> tuple[np.ndarray, dict]:
    """Generate a Box-Behnken design.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (requires at least 3).
    n_center_points : int
        Number of center point replicates.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / 0 / +1) and metadata.

    Notes
    -----
    Center points are embedded in the BB structure.  The caller should set
    ``n_center_points=0`` in ``build_design_result``.
    """
    k = len(factors)
    if k < 3:
        raise ValueError("Box-Behnken designs require at least 3 factors.")
    coded_matrix = bbdesign(k, center=n_center_points)
    return coded_matrix, {}


def dsd_conference_order(n_factors: int) -> int:
    """Return the order of the conference matrix a definitive screening design for ``n_factors`` factors is built from.

    ``n_factors`` itself when it is even and ``n_factors + 1`` when it is odd, stepped up
    to the next order where a conference matrix can be built: none exists at 22 or 34,
    so 21 and 22 factors use order 24, and 33 and 34 factors use order 38 (36 needs a
    construction not implemented here). The extra columns are fake factors, dropped
    from the design.

    Raises
    ------
    ValueError
        For fewer than 3 factors.
    """
    if n_factors < 3:
        raise ValueError("Definitive Screening Designs require at least 3 factors.")
    return conference_order_at_least(n_factors + n_factors % 2)


def dsd_run_count(n_factors: int) -> int:
    """Return the number of runs in the definitive screening design for ``n_factors`` continuous factors.

    ``2m + 1`` for the conference order ``m`` of :func:`dsd_conference_order`: the familiar
    ``2k + 1`` for even ``k`` and ``2k + 3`` for odd ``k``, except where the order steps up.

    Examples
    --------
    >>> dsd_run_count(6), dsd_run_count(7), dsd_run_count(22)
    (13, 17, 49)
    """
    return 2 * dsd_conference_order(n_factors) + 1


def dispatch_dsd(factors: list[Factor]) -> tuple[np.ndarray, dict]:
    """Generate a Definitive Screening Design (DSD).

    The conference-matrix construction of Jones and Nachtsheim (2011), as given by
    Xiao, Lin and Bai (2012): with ``C`` a conference matrix of order ``m``
    (``C'C = (m - 1) I``), the design is ``[C; -C; 0]`` with ``2m + 1`` runs, and its
    first ``k`` columns are the factors. ``m`` is ``k`` for even ``k`` and ``k + 1`` for
    odd ``k``, stepped up where no conference matrix of that order can be built (see
    :func:`dsd_conference_order`). Main effects are then orthogonal to each other and
    to every quadratic and two-factor interaction column.

    Conference matrices come from :mod:`process_improve.experiments._finite_fields`:
    Paley's construction over GF(q) for every prime power ``q = m - 1`` (including 9,
    25, 27, 49) and the doubling of an antisymmetric matrix (orders 16, 40, 56, ...).
    Every matrix is checked against ``C'C = (m - 1) I`` before use; no approximate
    matrix is ever used.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors, at least three.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata: ``construction`` (the conference-matrix
        construction) and ``conference_order``.

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
    m = dsd_conference_order(k)
    conference, construction = conference_matrix(m)
    coded_matrix = np.vstack([conference, -conference, np.zeros((1, m))]).astype(float)[:, :k]
    return coded_matrix, {"construction": construction, "conference_order": m}
