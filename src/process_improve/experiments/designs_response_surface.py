# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response surface designs: CCD, Box-Behnken, Definitive Screening Design.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

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
    center_points: int = 3,
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
    center_points : int
        Number of center points (split between cube and axial portions).
    alpha : str, float, or None
        Axial distance.  Accepted string values: ``"rotatable"``,
        ``"face_centered"``, ``"orthogonal"``.  A numeric value sets
        alpha directly.  Defaults to ``"orthogonal"``.
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
    ``center`` parameter).  The caller should set ``center_points=0`` in
    ``build_design_result`` to avoid adding duplicate center points.
    """
    if cube == "fractional":
        return _dispatch_ccd_fractional(factors, center_points, alpha, generators, resolution)
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


def _resolve_fractional_axial_distance(
    alpha: str | float | None,
    n_cube_runs: int,
    k: int,
    center_points: int,
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
    center_points : int
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
    n_center_cube = center_points // 2
    n_center_axial = center_points - n_center_cube
    n_axial = 2 * k
    a = (k * (1 + n_center_axial / n_axial) / (1 + n_center_cube / n_cube_runs)) ** 0.5
    return float(a), "orthogonal"


def _dispatch_ccd_fractional(
    factors: list[Factor],
    center_points: int,
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
    center_points : int
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

    axial_distance, face = _resolve_fractional_axial_distance(alpha, n_cube_runs, k, center_points)

    # Axial (star) runs: 2k rows at +/- axial_distance, zeros elsewhere.
    star = np.zeros((2 * k, k))
    for i in range(k):
        star[2 * i : 2 * i + 2, i] = (-axial_distance, axial_distance)

    center = np.zeros((max(0, center_points), k))

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

    Follows the conference-matrix foldover of Jones & Nachtsheim (2011): for a
    conference matrix ``C`` of order *m* the design is ``[C; -C; 0]``, giving
    ``2m + 1`` runs.  For *k* factors the smallest usable order is ``m = k``
    when *k* is even and ``m = k + 1`` when *k* is odd (the surplus column is
    dropped; Xiao, Lin & Bai 2012), so a DSD normally has ``2k + 1`` runs for
    even *k* and ``2k + 3`` runs for odd *k*.

    A conference matrix does not exist for every even order.  Order 22 is the
    first exception: a conference matrix of order ``m ≡ 2 (mod 4)`` exists only
    if ``m - 1`` is a sum of two squares (Belevitch 1950; van Lint & Seidel
    1966), and ``21 = 3 x 7`` is not.  When the minimal order is unavailable
    this function steps up to the next order it can construct and drops the
    surplus columns, which costs runs but keeps the design a genuine DSD.  The
    order actually used is reported in the metadata as ``"conference_order"``.

    Every constructed matrix is checked against the defining property
    ``C.T @ C == (m - 1) * I`` before it is used, so a degraded design can
    never reach the caller.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.  At least three are required.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata.  Metadata keys are
        ``"construction"`` (which conference-matrix construction was used),
        ``"conference_order"`` (the order *m* of that matrix) and, when the
        minimal order was not constructible, ``"minimal_conference_order"``
        (the order that would have been used had it existed).

    Raises
    ------
    ValueError
        If fewer than three factors are supplied, or if no conference matrix
        of usable order can be constructed.

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

    minimal_order = k if k % 2 == 0 else k + 1
    order = _smallest_constructible_conference_order(minimal_order)
    conference, construction = _conference_matrix(order)

    zero_row = np.zeros((1, order))
    coded_matrix = np.vstack([conference, -conference, zero_row])
    if order > k:
        coded_matrix = coded_matrix[:, :k]

    meta = {"construction": construction, "conference_order": order}
    if order > minimal_order:
        meta["minimal_conference_order"] = minimal_order
        logger.info(
            "No conference matrix of order %d exists, so the definitive screening design for %d factors "
            "uses order %d instead: %d runs rather than the %d a minimal design would need.",
            minimal_order,
            k,
            order,
            2 * order + 1,
            2 * minimal_order + 1,
        )

    return coded_matrix, meta


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


def _prime_power_factorization(q: int) -> tuple[int, int] | None:
    """Factor *q* as ``p ** n`` for a prime *p*, or return None.

    Parameters
    ----------
    q : int
        The integer to factor.

    Returns
    -------
    tuple[int, int] or None
        ``(p, n)`` with ``p ** n == q``, or None when *q* is not a prime power
        (including ``q < 2``).
    """
    if q < 2:
        return None
    smallest_divisor = next(d for d in range(2, q + 1) if q % d == 0)
    exponent, remaining = 0, q
    while remaining % smallest_divisor == 0:
        remaining //= smallest_divisor
        exponent += 1
    return (smallest_divisor, exponent) if remaining == 1 else None


def _monic_polynomials(degree: int, p: int) -> list[list[int]]:
    """Enumerate every monic polynomial of the given *degree* over GF(*p*).

    Polynomials are coefficient lists in ascending degree order, so
    ``[2, 0, 1]`` is ``x**2 + 2``.
    """
    polynomials = []
    for index in range(p**degree):
        coefficients, remaining = [], index
        for _ in range(degree):
            coefficients.append(remaining % p)
            remaining //= p
        polynomials.append([*coefficients, 1])
    return polynomials


def _polynomial_remainder(dividend: list[int], divisor: list[int], p: int) -> list[int]:
    """Remainder of *dividend* divided by the monic *divisor*, over GF(*p*).

    Both arguments are coefficient lists in ascending degree order.  The result
    is padded to exactly ``len(divisor) - 1`` coefficients.
    """
    remainder = [c % p for c in dividend]
    degree = len(divisor) - 1
    for i in range(len(remainder) - 1, degree - 1, -1):
        coefficient = remainder[i]
        if coefficient:
            for j in range(degree + 1):
                remainder[i - degree + j] = (remainder[i - degree + j] - coefficient * divisor[j]) % p
    remainder = remainder[:degree]
    return remainder + [0] * (degree - len(remainder))


def _is_irreducible(polynomial: list[int], p: int) -> bool:
    """Return True iff the monic *polynomial* is irreducible over GF(*p*).

    Uses trial division by every monic polynomial of degree up to half the
    degree of *polynomial*, which is inexpensive at the field sizes needed
    here (``p ** n`` of a few hundred at most).
    """
    degree = len(polynomial) - 1
    for divisor_degree in range(1, degree // 2 + 1):
        for divisor in _monic_polynomials(divisor_degree, p):
            if not any(_polynomial_remainder(polynomial, divisor, p)):
                return False
    return True


def _irreducible_polynomial(p: int, n: int) -> list[int]:
    """Return the first monic irreducible polynomial of degree *n* over GF(*p*)."""
    for candidate in _monic_polynomials(n, p):
        if _is_irreducible(candidate, p):
            return candidate
    # Not reachable through the search above: an irreducible polynomial of every
    # degree exists over every finite field.  Kept so that a bug in the
    # irreducibility test surfaces here instead of silently returning None.
    raise ValueError(f"No irreducible polynomial of degree {n} found over GF({p}).")


def _decode_field_element(element: int, p: int, n: int) -> list[int]:
    """Expand the integer label of a GF(``p ** n``) element into its coefficients."""
    coefficients, remaining = [], element
    for _ in range(n):
        coefficients.append(remaining % p)
        remaining //= p
    return coefficients


def _encode_field_element(coefficients: list[int], p: int) -> int:
    """Collapse GF(``p ** n``) coefficients back into an integer label."""
    return sum(c * p**i for i, c in enumerate(coefficients))


def _gf_multiplication_table(p: int, n: int) -> np.ndarray:
    """Multiplication table of GF(``p ** n``), indexed by integer element labels.

    Element *e* stands for the polynomial whose base-*p* digits are the
    coefficients of *e*, taken modulo a fixed irreducible polynomial of degree
    *n*.  For ``n == 1`` this is ordinary multiplication modulo *p*.
    """
    q = p**n
    if n == 1:
        return np.outer(np.arange(p), np.arange(p)) % p

    modulus = _irreducible_polynomial(p, n)
    table = np.zeros((q, q), dtype=int)
    for a in range(q):
        left = _decode_field_element(a, p, n)
        for b in range(a, q):
            right = _decode_field_element(b, p, n)
            product = [0] * (2 * n - 1)
            for i, x in enumerate(left):
                if x:
                    for j, y in enumerate(right):
                        product[i + j] = (product[i + j] + x * y) % p
            table[a, b] = table[b, a] = _encode_field_element(_polynomial_remainder(product, modulus, p), p)
    return table


def _gf_subtraction_table(p: int, n: int) -> np.ndarray:
    """Subtraction table of GF(``p ** n``): entry ``[a, b]`` is ``a - b``.

    Addition in GF(``p ** n``) is coefficient-wise modulo *p*, independent of
    the choice of irreducible polynomial.
    """
    q = p**n
    table = np.zeros((q, q), dtype=int)
    for a in range(q):
        left = _decode_field_element(a, p, n)
        for b in range(q):
            right = _decode_field_element(b, p, n)
            table[a, b] = _encode_field_element([(x - y) % p for x, y in zip(left, right, strict=True)], p)
    return table


def _quadratic_character(p: int, n: int) -> np.ndarray:
    """Legendre symbol on GF(``p ** n``): 0 at zero, +1 on squares, -1 otherwise."""
    q = p**n
    multiplication = _gf_multiplication_table(p, n)
    squares = {int(multiplication[x, x]) for x in range(1, q)}
    character = np.full(q, -1, dtype=int)
    character[0] = 0
    for square in squares:
        character[square] = 1
    return character


def _paley_conference_matrix(q: int) -> np.ndarray:
    """Build a conference matrix of order ``q + 1`` via Paley's construction.

    Requires *q* to be an odd prime power, which covers both ``q ≡ 1 (mod 4)``
    (symmetric conference matrix, "Paley type II") and ``q ≡ 3 (mod 4)``
    (skew-symmetric conference matrix, "Paley type I").  In both cases the
    result satisfies ``C.T @ C == q * I``.

    Prime powers with exponent above one (9, 25, 27, 49, ...) need arithmetic
    in GF(``p ** n``) rather than integers modulo *q*; the field is built from
    the first monic irreducible polynomial of the required degree.  These are
    the orders that make the difference for a DSD: without GF(9), GF(25) and
    GF(27) there is no minimal construction for 9, 10, 25, 26, 27 or 28
    factors.

    Parameters
    ----------
    q : int
        An odd prime power.

    Returns
    -------
    np.ndarray
        ``(q + 1) x (q + 1)`` matrix with 0s on the diagonal and ±1
        off-diagonal.
    """
    factorization = _prime_power_factorization(q)
    if factorization is None or q < 3 or q % 2 == 0:
        raise ValueError(f"Paley's construction needs an odd prime power; got q={q}.")
    p, n = factorization

    character = _quadratic_character(p, n)
    subtraction = _gf_subtraction_table(p, n)

    # Jacobsthal matrix Q[a, b] = chi(b - a), where the subtraction is in the field.
    jacobsthal = character[subtraction.T]

    size = q + 1
    matrix = np.zeros((size, size), dtype=int)
    matrix[0, 1:] = 1
    if q % 4 == 1:
        # Symmetric Paley conference matrix.
        matrix[1:, 0] = 1
    else:
        # Skew-symmetric Paley conference matrix (q ≡ 3 mod 4).
        matrix[1:, 0] = -1
    matrix[1:, 1:] = jacobsthal
    return matrix


_CONFERENCE_MATRIX_16 = np.array(
    [
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [-1, 0, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1],
        [-1, -1, 0, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1],
        [-1, -1, -1, 0, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1],
        [-1, 1, -1, -1, 0, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
        [-1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
        [-1, 1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1, 1],
        [-1, 1, 1, -1, 1, -1, -1, 0, 1, 1, 1, -1, 1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1],
        [-1, 1, 1, 1, -1, 1, -1, -1, -1, 0, -1, -1, 1, -1, 1, 1],
        [-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 0, -1, -1, 1, -1, 1],
        [-1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 0, -1, -1, 1, -1],
        [-1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 0, -1, -1, 1],
        [-1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 0, -1, -1],
        [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 0, -1],
        [-1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 0],
    ]
)
"""Skew-symmetric conference matrix of order 16, ``C.T @ C == 15 * I``.

Needed because 15 is not a prime power, so :func:`_paley_conference_matrix`
cannot reach order 16.  It is the only order below 22 that requires a table:
below it every order is covered by Paley's construction, and at 22 no
conference matrix exists at all.

The table is verified against its defining property on every use, in
:func:`_conference_matrix`, rather than trusted.  That is not ceremony.  The
order-10 table in the same upstream file carries a single mistyped entry (row
17, column 0, ``+1`` where the foldover requires ``-1``), which leaves that
package's 9- and 10-factor designs correlated at r = 0.11 with nothing in place
to notice.

Attribution
-----------
The matrix originates in the conference-matrix catalogue of Xiao, Lin & Bai
(2012), and reached this module through two ports:

- Jacob Albrecht, then at Bristol-Myers Squibb, ported the JMP add-in to MATLAB
  in March 2015 and released it under the BSD 3-Clause licence,
  Copyright (c) 2015 Jacob Albrecht.  The third clause of that licence names
  Bristol-Myers Squibb as the organisation whose name may not be used to
  endorse derived products.  Bristol-Myers Squibb is Albrecht's affiliation
  there, not the copyright holder.
- Daniele Ongari translated the MATLAB code to Python in the
  ``definitive_screening_design`` package (BSD 3-Clause,
  Copyright (c) 2022 Daniele Ongari), retaining Albrecht's header.

Only the numeric table is reused here; none of the surrounding code was copied.

References
----------
.. [1] Xiao, L., Lin, D. K. J. and Bai, F. (2012).  "Constructing definitive
   screening designs using conference matrices."  *Journal of Quality
   Technology*, 44(1):2-8.
"""

_TABULATED_CONFERENCE_MATRICES: dict[int, np.ndarray] = {16: _CONFERENCE_MATRIX_16}

# Upper bound on the conference-matrix order the search will consider.  A DSD
# of order 200 already needs 401 runs, well past any practical screening study.
_MAX_CONFERENCE_ORDER = 200


def _validate_conference_matrix(matrix: np.ndarray, order: int) -> None:
    """Check the defining property ``C.T @ C == (m - 1) * I`` of a conference matrix.

    Raises
    ------
    ValueError
        If the matrix is the wrong shape, has a non-zero diagonal, holds values
        other than 0 and ±1, or fails the orthogonality identity.  A tabulated
        matrix with a single mistyped entry fails here rather than silently
        producing a design with correlated main effects.
    """
    if matrix.shape != (order, order):
        raise ValueError(f"Conference matrix of order {order} has shape {matrix.shape}.")
    if np.any(np.diag(matrix) != 0):
        raise ValueError(f"Conference matrix of order {order} has a non-zero diagonal.")
    if not np.all(np.isin(matrix, (-1, 0, 1))):
        raise ValueError(f"Conference matrix of order {order} holds values outside {{-1, 0, +1}}.")
    expected = (order - 1) * np.eye(order)
    if not np.allclose(matrix.T @ matrix, expected, atol=1e-9):
        raise ValueError(
            f"Conference matrix of order {order} fails C.T @ C == {order - 1} * I; "
            "the design built from it would not be a definitive screening design."
        )


def _conference_matrix(m: int) -> tuple[np.ndarray, str]:
    """Construct an ``m x m`` conference matrix.

    Two constructions are available.  Paley's covers every even order *m* whose
    predecessor ``m - 1`` is an odd prime power, which is most of them.  A small
    table covers order 16, the one gap below order 22; see
    :data:`_CONFERENCE_MATRIX_16` for where that table comes from and who holds
    the copyright on the ports it travelled through.

    Parameters
    ----------
    m : int
        Desired (even) order of the conference matrix.

    Returns
    -------
    tuple[np.ndarray, str]
        The matrix and a short string identifying the construction used
        (``"paley_q=13"``, ``"tabulated_order_16"``).

    Raises
    ------
    ValueError
        If *m* is odd, or if no construction is available at that order.  No
        approximate matrix is ever returned: a design built from one would not
        have orthogonal main effects, which is the defining property of a DSD.
    """
    if m < 2 or m % 2 != 0:
        raise ValueError(f"A conference matrix has even order of at least 2; got m={m}.")

    q = m - 1
    factorization = _prime_power_factorization(q)
    if q >= 3 and factorization is not None:
        matrix = _paley_conference_matrix(q).astype(float)
        construction = f"paley_q={q}"
    elif m in _TABULATED_CONFERENCE_MATRICES:
        matrix = _TABULATED_CONFERENCE_MATRICES[m].astype(float)
        construction = f"tabulated_order_{m}"
    else:
        raise ValueError(f"No conference-matrix construction is available for order m={m}.")

    _validate_conference_matrix(matrix, m)
    return matrix, construction


def _is_constructible_conference_order(m: int) -> bool:
    """Return True iff :func:`_conference_matrix` can build order *m*."""
    if m < 2 or m % 2 != 0:
        return False
    q = m - 1
    return (q >= 3 and _prime_power_factorization(q) is not None) or m in _TABULATED_CONFERENCE_MATRICES


def _smallest_constructible_conference_order(minimum_order: int) -> int:
    """Smallest order at least *minimum_order* for which a conference matrix can be built.

    Parameters
    ----------
    minimum_order : int
        Smallest acceptable (even) order.

    Returns
    -------
    int
        The order that will be used.

    Raises
    ------
    ValueError
        If no order up to ``_MAX_CONFERENCE_ORDER`` can be constructed.
    """
    for order in range(minimum_order, _MAX_CONFERENCE_ORDER + 1, 2):
        if _is_constructible_conference_order(order):
            return order

    raise ValueError(
        f"No conference matrix could be constructed with order between {minimum_order} and "
        f"{_MAX_CONFERENCE_ORDER}; a definitive screening design is not available at this size."
    )


def dsd_conference_order(n_factors: int) -> int:
    """Return the order of the conference matrix a DSD for *n_factors* factors is built from.

    Normally ``n_factors`` for an even count and ``n_factors + 1`` for an odd
    count, but larger when no conference matrix exists at that order (21 and 22
    factors, for instance, need order 24 rather than 22).

    Parameters
    ----------
    n_factors : int
        Number of factors, at least three.

    Returns
    -------
    int
        The conference-matrix order.
    """
    if n_factors < 3:
        raise ValueError("Definitive Screening Designs require at least 3 factors.")
    return _smallest_constructible_conference_order(n_factors if n_factors % 2 == 0 else n_factors + 1)


def dsd_run_count(n_factors: int) -> int:
    """Return the number of runs in the definitive screening design for *n_factors* factors.

    This is ``2m + 1`` for the conference-matrix order *m* returned by
    :func:`dsd_conference_order`, which equals the familiar ``2k + 1`` (even
    *k*) or ``2k + 3`` (odd *k*) except at the factor counts where the minimal
    conference matrix does not exist.

    Parameters
    ----------
    n_factors : int
        Number of factors, at least three.

    Returns
    -------
    int
        Run count, including the design's single centre run.

    Examples
    --------
    >>> dsd_run_count(6), dsd_run_count(7), dsd_run_count(22)
    (13, 17, 49)
    """
    return 2 * dsd_conference_order(n_factors) + 1
