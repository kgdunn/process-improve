# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response surface designs: CCD, Box-Behnken, Definitive Screening Design.

All functions accept a list of ``Factor`` objects and return a raw coded
numpy array.  Post-processing is handled by ``designs_utils.build_design_result``.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np

from process_improve.experiments._classical import bbdesign, ccdesign
from process_improve.experiments._finite_fields import MAX_ORDER, conference_matrix, conference_order_at_least

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
    n_replicates: int = 1,
) -> tuple[np.ndarray, dict]:
    """Generate a Central Composite Design (CCD).

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors.
    n_center_points : int
        Number of center points (split between cube and axial portions).
    alpha : str, float, or None
        Axial distance: ``"rotatable"`` (``F ** 0.25`` for ``F`` cube runs),
        ``"face_centered"`` (1), ``"inscribed"`` (axial runs at +/-1, the cube
        shrunk inside a rotatable design), ``"orthogonal"`` (the default: the
        distance that makes the quadratic columns mutually orthogonal, see
        :func:`orthogonal_alpha`), or a positive number used as the distance
        itself. Any other value raises ``ValueError``.
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
    n_replicates : int
        How many times the cube and axial runs will be replicated (the centre runs
        are not), which the orthogonal axial distance depends on.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix and metadata: ``alpha_value``, the ratio of the axial
        distance to the cube's half-width (for the inscribed design, whose cube is
        shrunk, the axial runs sit at +/-1 and the cube at +/-1/alpha); ``face``, the
        geometry (``"circumscribed"``, ``"faced"`` or ``"inscribed"``); and
        ``alpha_rule``, how alpha was chosen (``"orthogonal"``, ``"rotatable"``,
        ``"face_centered"``, ``"inscribed"`` or ``"user"``).

    Raises
    ------
    ValueError
        If fewer than two factors are given, or *alpha* or *cube* is not recognised.

    Notes
    -----
    Center points are embedded in the CCD structure itself (via ``ccdesign``'s
    ``center`` parameter).  The caller should set ``n_center_points=0`` in
    ``build_design_result`` to avoid adding duplicate center points.
    """
    if len(factors) < 2:
        raise ValueError(f"A central composite design needs at least 2 factors, got {len(factors)}.")
    if cube == "fractional":
        cube_runs, cube_meta = _fractional_cube(factors, generators, resolution)
        axial_distance, rule = _resolve_fractional_axial_distance(
            alpha, len(cube_runs), len(factors), n_center_points, n_replicates
        )
        coded_matrix = _stack_ccd(cube_runs, axial_distance, n_center_points)
        return coded_matrix, {
            "alpha_value": axial_distance,
            "face": _geometry(axial_distance),
            "alpha_rule": rule,
            **cube_meta,
        }
    if cube != "full":
        raise ValueError(f"cube must be 'full' or 'fractional', got {cube!r}.")

    k = len(factors)
    kind = _axial_kind(alpha)
    rule = _ALPHA_NAMES[kind][0] if isinstance(kind, str) else "user"
    if kind == "orthogonal":
        r = n_replicates
        kind = orthogonal_alpha(r * 2**k, r * (2**k + 2 * k) + n_center_points, n_axial_replicates=r)
    # Centre runs are split between the cube and axial blocks, as ccdesign does.
    n_center_cube = n_center_points // 2
    n_center_axial = n_center_points - n_center_cube

    if isinstance(kind, float):
        # An axial distance ccdesign cannot take: build the 2^k cube, the 2k axial runs and the centre runs here.
        cube_runs = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))[:, ::-1]
        coded_matrix = _stack_ccd(cube_runs, kind, n_center_points)
        return coded_matrix, {"alpha_value": kind, "face": _geometry(kind), "alpha_rule": rule}

    face = {"faced": "faced", "inscribed": "inscribed"}.get(kind, "circumscribed")
    coded_matrix = ccdesign(k, center=(n_center_cube, n_center_axial), alpha="rotatable", face=face)
    if face == "inscribed":
        # The axial runs sit at +/-1 and the cube is shrunk to +/-1/alpha.
        cube_rows = coded_matrix[np.all(coded_matrix != 0, axis=1)]
        alpha_value = float(1.0 / np.min(np.abs(cube_rows)))
    else:
        alpha_value = float(np.max(np.abs(coded_matrix)))
    return coded_matrix, {"alpha_value": alpha_value, "face": face, "alpha_rule": rule}


def _geometry(alpha: float) -> str:
    """Name the geometry of a CCD whose cube is at +/-1 and axial runs at +/-alpha."""
    return "faced" if np.isclose(alpha, 1.0) else "circumscribed"


def orthogonal_alpha(n_cube_runs: int, n_runs: int, n_axial_replicates: int = 1) -> float:
    """Axial distance that makes the quadratic columns of a central composite design mutually orthogonal.

    ``alpha = (F * (sqrt(N) - sqrt(F)) ** 2 / (4 * s ** 2)) ** (1 / 4)`` for ``F`` cube runs,
    each axial point run ``s`` times, and ``N`` runs in all, centre runs included (Box and
    Hunter 1957; Myers, Montgomery and Anderson-Cook, *Response Surface Methodology*,
    section 7.4, for ``s = 1``). With it the centred squared columns are orthogonal, so the
    quadratic coefficients are estimated independently of one another.

    The squares of two factors are both non-zero only on the cube runs, so their
    product sums to ``F``; each has the sum ``F + 2 s alpha**2``; the centred columns
    are orthogonal when ``F N = (F + 2 s alpha**2) ** 2``. A design replicated whole
    (``F``, ``s`` and ``N`` all times ``r``) keeps its alpha; one whose centre runs are
    not replicated has relatively fewer of them, and a smaller alpha.

    Parameters
    ----------
    n_cube_runs : int
        Runs in the cube (factorial) portion, full or fractional, replicates included.
    n_runs : int
        Every run of the design: cube, axial and centre runs.
    n_axial_replicates : int
        How many times each of the ``2k`` axial points is run.

    Returns
    -------
    float
        The axial distance in coded units.
    """
    gap = np.sqrt(n_runs) - np.sqrt(n_cube_runs)
    return float((n_cube_runs * gap**2 / (4 * n_axial_replicates**2)) ** 0.25)


#: Accepted spellings of each named axial distance.
_ALPHA_NAMES = {
    "faced": ("face_centered", "face centered", "face_centred", "ccf", "faced"),
    "inscribed": ("inscribed", "cci"),
    "rotatable": ("rotatable", "r"),
    "orthogonal": ("orthogonal", "o"),
}


def _axial_kind(alpha: str | float | None) -> str | float:
    """Resolve ``alpha`` to ``"faced"``, ``"inscribed"``, ``"rotatable"``, ``"orthogonal"`` or a positive number.

    Raises
    ------
    ValueError
        For an unknown name or a non-positive number, instead of silently using the
        orthogonal distance.
    """
    if alpha is None:
        return "orthogonal"
    if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
        if not alpha > 0:
            raise ValueError(f"A numeric alpha (the axial distance) must be positive; got {alpha}.")
        return float(alpha)
    if isinstance(alpha, str):
        for kind, names in _ALPHA_NAMES.items():
            if alpha.lower() in names:
                return kind
    options = ", ".join(repr(names[0]) for names in _ALPHA_NAMES.values())
    raise ValueError(f"Unknown alpha {alpha!r} for a central composite design; use {options} or a positive number.")


def _resolve_fractional_axial_distance(
    alpha: str | float | None,
    n_cube_runs: int,
    k: int,
    n_center_points: int,
    n_replicates: int = 1,
) -> tuple[float, str]:
    """Axial (star-point) distance for a fractional-cube CCD.

    Mirrors the classical axial-distance formulas, but uses the actual number of
    fractional cube runs *n_cube_runs* in place of the full ``2**k``.

    Parameters
    ----------
    alpha : str, float, or None
        ``"face_centered"`` (alpha = 1), ``"rotatable"``
        (alpha = n_cube_runs ** 0.25), ``"orthogonal"`` / None
        (:func:`orthogonal_alpha`), or a numeric value used directly.
    n_cube_runs : int
        Number of runs in the (fractional) cube portion.
    k : int
        Number of factors.
    n_center_points : int
        Total number of center points, which count towards the run total in
        :func:`orthogonal_alpha`.
    n_replicates : int
        How many times the cube and axial runs will be replicated (the centre runs are not).

    Returns
    -------
    tuple[float, str]
        The axial distance and the rule that chose it, recorded as ``alpha_rule``.
    """
    kind = _axial_kind(alpha)
    if isinstance(kind, float):
        return kind, "user"
    if kind == "faced":
        return 1.0, "face_centered"
    if kind == "rotatable":
        return float(n_cube_runs**0.25), "rotatable"
    if kind == "inscribed":
        raise ValueError(
            "alpha='inscribed' is not supported with cube='fractional'; "
            "use 'face_centered', 'rotatable', 'orthogonal', or a numeric alpha."
        )

    r = n_replicates
    total = r * (n_cube_runs + 2 * k) + n_center_points
    return orthogonal_alpha(r * n_cube_runs, total, n_axial_replicates=r), "orthogonal"


def _fractional_cube(
    factors: list[Factor], generators: list[str] | None, resolution: int | None
) -> tuple[np.ndarray, dict]:
    """Build the resolution-V (or higher) fractional factorial cube of a CCD.

    The cube is generated by reusing
    :func:`process_improve.experiments.designs_screening.dispatch_fractional_factorial`.
    Without *generators* or *resolution*, it is the minimum-aberration half-fraction (the
    last factor the product of all the others).

    Returns
    -------
    tuple[np.ndarray, dict]
        The cube runs, and ``cube``, ``generators_used``, ``defining_relation`` and
        ``resolution`` for the metadata.

    Raises
    ------
    ValueError
        For fewer than 3 factors, or a cube of resolution below V.
    """
    from process_improve.experiments.designs_screening import dispatch_fractional_factorial  # noqa: PLC0415

    if len(factors) < 3:
        raise ValueError("A fractional-cube CCD requires at least 3 factors; use cube='full' for fewer.")
    factor_names = [f.name for f in factors]
    if generators is None and resolution is None:
        generators = [f"{factor_names[-1]}={''.join(factor_names[:-1])}"]
    cube, frac_meta = dispatch_fractional_factorial(factors, resolution=resolution, generators=generators)
    res = frac_meta.get("resolution")
    if res is not None and res < 5:
        raise ValueError(
            f"The fractional cube has resolution {res}, but a CCD needs resolution V or higher so the "
            "full quadratic model is estimable. Supply resolution-V generators or use cube='full'."
        )
    return cube, {
        "cube": "fractional",
        "generators_used": frac_meta.get("generators_used") or generators,
        "defining_relation": frac_meta.get("defining_relation"),
        "resolution": res,
    }


def _stack_ccd(cube: np.ndarray, axial_distance: float, n_center_points: int) -> np.ndarray:
    """Stack a CCD: the cube, half the centre runs, the ``2k`` axial runs, the other centre runs.

    The centre runs are split between the cube and axial portions as ``ccdesign`` does,
    so each portion can be a block of its own.
    """
    k = cube.shape[1]
    star = np.zeros((2 * k, k))
    for i in range(k):
        star[2 * i : 2 * i + 2, i] = (-axial_distance, axial_distance)
    n_center_cube = n_center_points // 2
    return np.vstack([cube, np.zeros((n_center_cube, k)), star, np.zeros((n_center_points - n_center_cube, k))])


#: Box and Behnken's (1960) blocks for six and seven factors: each block carries a two-level
#: factorial in the factors it names, with every other factor at its centre. Six factors use
#: their partially balanced design (a pair of factors meets in one or two blocks); seven use
#: the balanced design on the Fano plane, written cyclically (the published table up to a
#: relabelling of the factors). Pairing every two factors, as ``bbdesign`` does for any k, gives the
#: published design for three to five factors but 60 and 84 runs at six and seven.
_BOX_BEHNKEN_BLOCKS: dict[int, tuple[tuple[int, ...], ...]] = {
    6: ((0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)),
    7: ((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (0, 4, 5), (1, 5, 6), (0, 2, 6)),
}


def dispatch_box_behnken(
    factors: list[Factor],
    n_center_points: int = 3,
) -> tuple[np.ndarray, dict]:
    """Generate a Box-Behnken design.

    Three to five factors: a two-level factorial in every pair of factors
    (``bbdesign``), which is the published design. Six and seven factors: the published
    blocks of three factors in ``_BOX_BEHNKEN_BLOCKS`` (48 and 56 runs plus centre
    points). Eight or more factors: all pairs again, since Box and Behnken's larger
    designs are not tabulated here; ``metadata["construction"]`` says ``"all_pairs"``.

    Parameters
    ----------
    factors : list[Factor]
        Continuous factors (requires at least 3).
    n_center_points : int
        Number of center point replicates.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (-1 / 0 / +1) and metadata with the ``construction``.

    Notes
    -----
    Center points are embedded in the BB structure.  The caller should set
    ``n_center_points=0`` in ``build_design_result``.

    References
    ----------
    Box, G. E. P. and Behnken, D. W. (1960). Some new three level designs for the study
    of quantitative variables. *Technometrics*, 2(4), 455-475.
    """
    k = len(factors)
    if k < 3:
        raise ValueError("Box-Behnken designs require at least 3 factors.")
    blocks = _BOX_BEHNKEN_BLOCKS.get(k)
    if blocks is None:
        construction = "published_pairs" if k <= 5 else "all_pairs"
        return bbdesign(k, center=n_center_points), {"construction": construction}
    rows = []
    for block in blocks:
        for corner in itertools.product((-1.0, 1.0), repeat=len(block)):
            row = np.zeros(k)
            row[list(block)] = corner
            rows.append(row)
    rows.extend(np.zeros(k) for _ in range(n_center_points))
    return np.asarray(rows), {"construction": "published_blocks_of_three"}


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


def dsd_run_count(n_factors: int, n_categorical: int = 0) -> int:
    """Return the number of runs in the definitive screening design for ``n_factors`` factors.

    ``2m + 1`` for the conference order ``m`` of :func:`dsd_conference_order`: the familiar
    ``2k + 1`` for even ``k`` and ``2k + 3`` for odd ``k``, except where the order steps up.
    Two-level categorical factors (counted in ``n_factors``) need two centre runs instead
    of one (Jones and Nachtsheim 2013).

    Examples
    --------
    >>> dsd_run_count(6), dsd_run_count(7), dsd_run_count(22), dsd_run_count(6, n_categorical=2)
    (13, 17, 49, 14)
    """
    return 2 * dsd_conference_order(n_factors) + (2 if n_categorical else 1)


def _dsd_order_for_budget(k: int, n_categorical: int, budget: int | None) -> int:
    """Conference order for ``k`` factors: the minimal one, or the largest that fits ``budget`` runs.

    Columns beyond ``k`` are fake factors (Jones and Nachtsheim 2017): they are not run as
    factors, but the foldover keeps them orthogonal to everything, so they give the
    analysis error degrees of freedom free of any second-order effect.
    """
    minimal = dsd_conference_order(k)
    if budget is None:
        return minimal
    extra = 2 if n_categorical else 1
    if budget < 2 * minimal + extra:
        raise ValueError(
            f"A definitive screening design for {k} factors needs at least {2 * minimal + extra} runs; "
            f"budget={budget} is too small."
        )
    order = minimal
    while order + 2 <= MAX_ORDER and 2 * (order + 2) + extra <= budget:
        order += 2
    while conference_order_at_least(order) != order:  # step down to an order that can be built
        order -= 2
    return order


def dispatch_dsd(factors: list[Factor], budget: int | None = None) -> tuple[np.ndarray, dict]:
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

    Two-level categorical factors use the DSD-augment method of Jones and Nachtsheim
    (2013): a categorical factor takes a conference-matrix column like any other, its
    two zeros (one in each half of the foldover) become ``+z`` and ``-z``, and the
    single centre run becomes two, with the categorical factors at ``+b`` and ``-b``.
    The signs are chosen to maximise the determinant of the main-effects information
    matrix. Every main effect stays orthogonal to every second-order effect, and the
    continuous main effects stay orthogonal to each other. Resolving the zeros costs
    some orthogonality between a categorical main effect and the other main effects:
    it is slightly correlated with each continuous main effect (``|r|`` about 0.17 for
    four continuous factors and one categorical factor in 14 runs) and with the other
    categorical main effects.

    A ``budget`` larger than the minimal design adds fake factors: the conference
    matrix of the largest buildable order whose design fits the budget is used, and its
    extra columns are left out. Jones and Nachtsheim (2017) recommend this, because the
    unused columns give error degrees of freedom free of any second-order effect, which
    the analysis of a DSD (:func:`process_improve.experiments.analyze_omars`) needs.

    Parameters
    ----------
    factors : list[Factor]
        At least three factors: continuous, or categorical with exactly two levels.
    budget : int or None
        Largest number of runs. ``None`` gives the minimal design.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coded design matrix (categorical factors at -1 / +1) and metadata:
        ``construction``, ``conference_order``, ``fake_factors`` and, with categorical
        factors, ``n_categorical``.

    References
    ----------
    * Jones, B. and Nachtsheim, C. J. (2011).  "A class of three-level
      designs for definitive screening in the presence of second-order
      effects."  *Journal of Quality Technology*, 43(1):1-15.
    * Xiao, L., Lin, D. K. J. and Bai, F. (2012).  "Constructing
      definitive screening designs using conference matrices."  *Journal
      of Quality Technology*, 44(1):2-8.
    * Jones, B. and Nachtsheim, C. J. (2013).  "Definitive screening
      designs with added two-level categorical factors."  *Journal of Quality
      Technology*, 45(2):121-129.
    * Jones, B. and Nachtsheim, C. J. (2017).  "Effective design-based model
      selection for definitive screening designs."  *Technometrics*, 59(3):319-329.
    """
    k = len(factors)
    categorical = [j for j, f in enumerate(factors) if f.type.value == "categorical"]
    for j in categorical:
        if len(factors[j].levels or []) != 2:
            raise ValueError(
                f"Factor {factors[j].name!r} has {len(factors[j].levels or [])} levels; definitive screening "
                "designs take two-level categorical factors only (Jones and Nachtsheim 2013). Use "
                'design_type="d_optimal" or "i_optimal" for a factor with more levels.'
            )
    m = _dsd_order_for_budget(k, len(categorical), budget)
    conference, construction = conference_matrix(m)
    folded = np.vstack([conference, -conference]).astype(float)[:, :k]
    meta: dict = {"construction": construction, "conference_order": m, "fake_factors": m - k}
    if not categorical:
        return np.vstack([folded, np.zeros((1, k))]), meta
    meta["n_categorical"] = len(categorical)
    return _dsd_augment(folded, categorical), meta


#: Up to this many categorical factors, every sign choice of the DSD-augment method is tried.
_MAX_EXHAUSTIVE_CATEGORICAL = 7


def _dsd_augment(folded: np.ndarray, categorical: list[int]) -> np.ndarray:
    """Jones and Nachtsheim's (2013) DSD-augment: resolve the categorical columns' zeros and add two centre runs."""
    n_half, k = folded.shape[0] // 2, folded.shape[1]
    zero_rows = [int(np.flatnonzero(folded[:n_half, j] == 0)[0]) for j in categorical]
    c = len(categorical)
    if c <= _MAX_EXHAUSTIVE_CATEGORICAL:
        candidates = (np.array(signs) for signs in itertools.product((-1.0, 1.0), repeat=2 * c))
    else:  # one-at-a-time flips of an alternating pattern, linear in c (the paper uses an exchange here)
        seed = np.where(np.arange(2 * c) % 2 == 0, 1.0, -1.0)
        candidates = (seed * np.where(np.arange(2 * c) == i, -1.0, 1.0) for i in range(-1, 2 * c))
    best, best_value = folded, -np.inf
    for signs in candidates:
        design = np.vstack([folded, np.zeros((2, k))])
        for position, (j, row) in enumerate(zip(categorical, zero_rows, strict=True)):
            design[row, j], design[row + n_half, j] = signs[position], -signs[position]
            design[-2, j], design[-1, j] = signs[c + position], -signs[c + position]
        model = np.column_stack([np.ones(len(design)), design])
        value = float(np.linalg.slogdet(model.T @ model)[1])
        if value > best_value + 1e-12:
            best, best_value = design, value
    return best
