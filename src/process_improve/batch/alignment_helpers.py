"""Functions in this files will ONLY use NumPy, and are therefore candidates for speed up with Numba."""

from collections.abc import Callable

import numpy as np

# ENG-13 (#295): numba lives in the optional ``[fast]`` extra. Without it
# the module still imports and the functions still run; ``@jit`` falls back
# to a no-op decorator so the code executes as plain Python (slower, but
# functionally identical).
try:
    from numba import jit
except ImportError:
    from typing import Any

    def jit(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]  # noqa: ANN401
        """No-op fallback when numba (the 'fast' extra) is not installed."""

        def _decorator(func: Callable) -> Callable:
            return func

        # Support both ``@jit`` (no args) and ``@jit(nopython=True)``.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


def full_band(n_test: int, n_ref: int) -> np.ndarray:
    """
    Build the unconstrained band: every reference row is reachable from every test sample.

    This is the default, and reproduces the behaviour the module had before bands were
    configurable (#197).

    Parameters
    ----------
    n_test : int
        Number of samples in the test batch (columns of the DTW cost matrix).
    n_ref : int
        Number of samples in the reference batch (rows of the DTW cost matrix).

    Returns
    -------
    np.ndarray
        A ``(n_test, 2)`` integer array. Row ``n`` gives the half-open range of
        reference rows ``[band[n, 0], band[n, 1])`` that test sample ``n`` may map to.
    """
    band = np.zeros((n_test, 2), dtype=np.int64)
    band[:, 1] = n_ref
    return band


def sakoe_chiba_band(n_test: int, n_ref: int, window: float) -> np.ndarray:
    """
    Build a Sakoe-Chiba band: a corridor of fixed width around the diagonal.

    The corridor is centred on the line joining ``(0, 0)`` to ``(n_ref - 1, n_test - 1)``,
    so it is correct when the two batches have different lengths, and it always contains
    both corner cells.

    Parameters
    ----------
    n_test, n_ref : int
        Lengths of the test and reference batches.
    window : int or float
        The half-width of the corridor. An ``int`` is a number of reference rows; a
        ``float`` is a fraction of the reference length, so ``window=0.1`` allows a
        warp of 10% of the batch duration. ``bool`` is rejected, since ``True`` as a
        one-row radius is almost certainly a mistake.

    Returns
    -------
    np.ndarray
        A ``(n_test, 2)`` integer array of half-open row ranges, as :func:`full_band`.

    Notes
    -----
    The radius is raised to at least ``ceil((n_ref - 1) / (n_test - 1))`` when the
    batches differ in length. A corridor narrower than the diagonal's own step would
    leave gaps that no monotone warping path can cross, making the alignment infeasible
    rather than merely constrained.

    References
    ----------
    Sakoe, H. and Chiba, S. (1978). "Dynamic programming algorithm optimization for
    spoken word recognition", *IEEE Transactions on Acoustics, Speech, and Signal
    Processing*, 26(1), 43-49.
    """
    _check_lengths(n_test, n_ref)
    if isinstance(window, bool):
        raise TypeError(f"`window` must be an int (rows) or a float (fraction); got {window!r}.")

    if isinstance(window, (int, np.integer)):
        radius = int(window)
        if radius < 1:
            raise ValueError(f"An integer `window` is a number of rows and must be at least 1; got {radius}.")
    else:
        fraction = float(window)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"A float `window` is a fraction and must lie in (0, 1]; got {fraction}.")
        radius = max(1, int(np.ceil(fraction * (n_ref - 1))))

    if n_test == 1 or n_ref == 1:
        # A single column (or row) leaves no freedom to constrain.
        return full_band(n_test, n_ref)

    slope = (n_ref - 1) / (n_test - 1)
    radius = max(radius, int(np.ceil(slope)))
    centre = np.arange(n_test) * slope
    band = np.empty((n_test, 2), dtype=np.int64)
    band[:, 0] = np.clip(np.ceil(centre - radius), 0, n_ref - 1)
    band[:, 1] = np.clip(np.floor(centre + radius) + 1, 1, n_ref)
    return band


def itakura_band(n_test: int, n_ref: int, max_slope: float) -> np.ndarray:
    """
    Build an Itakura parallelogram: a corridor defined by bounds on the local slope.

    Unlike a Sakoe-Chiba band, this is narrow at the two corners and widest in the
    middle, which forbids the long flat runs (many test samples mapped to one
    reference row) that a fixed-width corridor still permits.

    The slope bounds are taken relative to the diagonal ``(n_ref - 1) / (n_test - 1)``,
    not relative to 1, so ``max_slope`` means the same thing whatever the two lengths.

    Parameters
    ----------
    n_test, n_ref : int
        Lengths of the test and reference batches.
    max_slope : float
        How much faster than the diagonal the path may advance, and (as its
        reciprocal) how much slower. Must be at least 1.0; ``1.0`` admits only the
        diagonal itself.

    Returns
    -------
    np.ndarray
        A ``(n_test, 2)`` integer array of half-open row ranges, as :func:`full_band`.

    Raises
    ------
    ValueError
        If ``max_slope`` is below 1.0, or is too small for these two lengths to leave a
        corridor a warping path can follow. The message names the smallest value that
        does work, or says that none does.

    Notes
    -----
    The parallelogram's edges are rounded outwards, so a reference row is admitted when
    the continuous corridor overlaps it at all. Rounding inwards can empty a column
    whose corridor is thinner than one row, which happens whenever the two batches
    differ much in length.

    Batches of very different duration cannot be constrained this way at any slope:
    the corners of the parallelogram are one cell wide, so a path that must climb many
    reference rows per test sample has nowhere to start. :func:`sakoe_chiba_band` has
    no such limitation and is the better choice there.

    References
    ----------
    Itakura, F. (1975). "Minimum prediction residual principle applied to speech
    recognition", *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
    23(1), 67-72.
    """
    _check_lengths(n_test, n_ref)
    slope_limit = float(max_slope)
    if slope_limit < 1.0:
        raise ValueError(f"`max_slope` must be at least 1.0; got {slope_limit}.")

    if n_test == 1 or n_ref == 1:
        return full_band(n_test, n_ref)

    band = _itakura_corridor(n_test, n_ref, slope_limit)
    try:
        validate_band(band, n_test, n_ref)
    except ValueError as exc:
        raise ValueError(_itakura_advice(n_test, n_ref, slope_limit, exc)) from None
    return band


def _itakura_corridor(n_test: int, n_ref: int, slope_limit: float) -> np.ndarray:
    """Build the parallelogram for :func:`itakura_band`, without validating it."""
    last_test, last_ref = n_test - 1, n_ref - 1
    diagonal = last_ref / last_test
    fastest, slowest = diagonal * slope_limit, diagonal / slope_limit
    samples = np.arange(n_test)
    # Rounded outwards: a row is in the corridor if the corridor touches it at all.
    lower = np.floor(np.maximum(slowest * samples, last_ref - fastest * (last_test - samples)))
    upper = np.ceil(np.minimum(fastest * samples, last_ref - slowest * (last_test - samples))) + 1
    band = np.empty((n_test, 2), dtype=np.int64)
    band[:, 0] = np.clip(lower, 0, last_ref)
    band[:, 1] = np.clip(upper, 1, n_ref)
    return band


def _itakura_advice(n_test: int, n_ref: int, slope_limit: float, failure: ValueError) -> str:
    """
    Explain an infeasible Itakura parallelogram, naming the smallest slope that works.

    The smallest workable slope is found by bisection on the same construction rather
    than from a closed form, so the number quoted is the one the code will actually
    accept. This runs only on the error path.
    """

    def feasible(candidate: float) -> bool:
        try:
            validate_band(_itakura_corridor(n_test, n_ref, candidate), n_test, n_ref)
        except ValueError:
            return False
        return True

    ceiling = 1.0
    while ceiling <= 1024.0 and not feasible(ceiling):
        ceiling *= 2.0

    preamble = f"`max_slope`={slope_limit} leaves no usable corridor for n_test={n_test}, n_ref={n_ref}: {failure}"
    if ceiling > 1024.0:
        return (
            f"{preamble} These two lengths differ too much for an Itakura parallelogram at any "
            f"slope; use a Sakoe-Chiba band instead."
        )

    floor_ = ceiling / 2.0
    for _ in range(40):
        midpoint = (floor_ + ceiling) / 2.0
        if feasible(midpoint):
            ceiling = midpoint
        else:
            floor_ = midpoint

    # Quote a value that is itself accepted. The bisection lands just above the
    # threshold, and rounding that for display can drop below it again, so round
    # upwards and confirm before quoting.
    threshold = ceiling
    for _ in range(20):
        candidate = _round_up_4_significant_figures(threshold)
        if feasible(candidate):
            return f"{preamble} Use max_slope >= {candidate:g}."
        threshold *= 1.001
    return f"{preamble} Widen `max_slope`; the threshold is near {ceiling:.4g}."


def _round_up_4_significant_figures(value: float) -> float:
    """Round up to four significant figures, so a quoted threshold never falls below it."""
    if value <= 0.0:
        return value
    scale = 10.0 ** (3 - int(np.floor(np.log10(value))))
    return float(np.ceil(value * scale) / scale)


def sakoe_chiba(window: float) -> Callable[[int, int], np.ndarray]:
    """
    Return a band constraint that builds a Sakoe-Chiba corridor of the given window.

    The two batch lengths are not known until each pair is aligned, so a constraint is
    passed around as a callable of ``(n_test, n_ref)``:

    >>> from process_improve.batch.alignment_helpers import sakoe_chiba
    >>> settings = {"band": sakoe_chiba(window=0.1)}     # 10% of the batch duration

    See :func:`sakoe_chiba_band` for the meaning of ``window``.
    """

    def band(n_test: int, n_ref: int) -> np.ndarray:
        return sakoe_chiba_band(n_test, n_ref, window)

    return band


def itakura(max_slope: float) -> Callable[[int, int], np.ndarray]:
    """
    Return a band constraint that builds an Itakura parallelogram of the given slope.

    >>> from process_improve.batch.alignment_helpers import itakura
    >>> settings = {"band": itakura(max_slope=2.0)}

    See :func:`itakura_band` for the meaning of ``max_slope``.
    """

    def band(n_test: int, n_ref: int) -> np.ndarray:
        return itakura_band(n_test, n_ref, max_slope)

    return band


def resolve_band(band: object, n_test: int, n_ref: int) -> np.ndarray:
    """
    Turn any accepted band specification into a validated ``(n_test, 2)`` integer array.

    Parameters
    ----------
    band : None, np.ndarray, or callable
        ``None`` gives :func:`full_band`. An array is used as given. A callable is
        invoked as ``band(n_test, n_ref)``; :func:`sakoe_chiba` and :func:`itakura`
        return callables of that shape.
    n_test, n_ref : int
        Lengths of the test and reference batches.

    Returns
    -------
    np.ndarray
        The band, contiguous ``int64``, already passed through :func:`validate_band`.

    Notes
    -----
    A callable is deliberately resolved here, outside the numba-compiled kernel: a
    Python callable cannot be invoked from ``nopython`` code, so the geometry is built
    once per alignment and only the finished array crosses into the kernel.
    """
    _check_lengths(n_test, n_ref)
    if band is None:
        return full_band(n_test, n_ref)

    resolved = band(n_test, n_ref) if callable(band) else band
    as_array = np.ascontiguousarray(resolved)
    if not np.issubdtype(as_array.dtype, np.integer):
        raise TypeError(f"A band must hold integer row bounds; got dtype {as_array.dtype}.")

    as_array = as_array.astype(np.int64, copy=False)
    validate_band(as_array, n_test, n_ref)
    return as_array


def validate_band(band: np.ndarray, n_test: int, n_ref: int) -> None:
    """
    Check that a band describes a corridor at least one monotone warping path can follow.

    Raises ``ValueError`` naming the first condition that fails. An invalid band is
    worth rejecting up front: out-of-band cells of the cost matrix stay NaN, and a
    path that has to step into one cannot be backtracked.

    Parameters
    ----------
    band : np.ndarray
        Candidate ``(n_test, 2)`` array of half-open reference-row ranges.
    n_test, n_ref : int
        Lengths of the test and reference batches.
    """
    if band.shape != (n_test, 2):
        raise ValueError(f"A band must have shape ({n_test}, 2) for these two batches; got {band.shape}.")

    lower, upper = band[:, 0], band[:, 1]
    if np.any(lower < 0) or np.any(upper > n_ref):
        raise ValueError(f"Band row bounds must lie within [0, {n_ref}]; got [{lower.min()}, {upper.max()}].")

    if np.any(lower >= upper):
        first = int(np.argmax(lower >= upper))
        raise ValueError(
            f"Every test sample needs at least one reachable reference row, but the band is "
            f"empty at test sample {first}: [{lower[first]}, {upper[first]})."
        )

    if np.any(np.diff(lower) < 0) or np.any(np.diff(upper) < 0):
        raise ValueError("Band row bounds must be non-decreasing, so that the corridor moves forward only.")

    if lower[0] != 0:
        raise ValueError(f"The band must admit the starting cell (0, 0), but test sample 0 starts at row {lower[0]}.")

    if upper[-1] != n_ref:
        raise ValueError(
            f"The band must admit the final cell ({n_ref - 1}, {n_test - 1}), but the last test "
            f"sample reaches only row {upper[-1] - 1}."
        )

    # A cell (m, n) is entered from (m, n-1), (m-1, n-1) or (m-1, n), so the lowest
    # row of column n must be no higher than one above the top of column n-1.
    gaps = lower[1:] > upper[:-1]
    if np.any(gaps):
        first = int(np.argmax(gaps)) + 1
        raise ValueError(
            f"The band is disconnected between test samples {first - 1} and {first}: no warping "
            f"path can step from rows [{lower[first - 1]}, {upper[first - 1]}) to "
            f"[{lower[first]}, {upper[first]}). Widen the constraint."
        )


def _check_lengths(n_test: int, n_ref: int) -> None:
    """Reject batch lengths that cannot describe a cost matrix."""
    if n_test < 1 or n_ref < 1:
        raise ValueError(f"Both batches need at least one sample; got n_test={n_test}, n_ref={n_ref}.")


def distance_matrix(
    test: np.ndarray,
    ref: np.ndarray,
    weight_matrix: np.ndarray,
    band: object = None,
) -> np.ndarray:
    """
    Compute the accumulated DTW cost matrix between test and reference batch trajectories.

    Parameters
    ----------
    test : np.ndarray
        The test batch, ``(n_test, n_tags)``; it will be aligned to ``ref``.
    ref : np.ndarray
        The reference batch, ``(n_ref, n_tags)``.
    weight_matrix : np.ndarray
        The ``(n_tags, n_tags)`` weighting of the per-tag deviations.
    band : None, np.ndarray, or callable, optional
        A band constraint, resolved by :func:`resolve_band`. The default ``None``
        places no constraint, which is what this function did before #197.

    Returns
    -------
    np.ndarray
        The ``(n_ref, n_test)`` accumulated cost matrix ``D``. Cells outside the band
        are ``NaN``: they are not reachable, and :func:`backtrack_optimal_path` will
        not step into one.
    """
    resolved = resolve_band(band, test.shape[0], ref.shape[0])
    return _banded_distance_matrix(test, ref, weight_matrix, resolved)


@jit(nopython=True)
def _banded_distance_matrix(
    test: np.ndarray,
    ref: np.ndarray,
    weight_matrix: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Fill the accumulated cost matrix inside ``band``; see :func:`distance_matrix`."""
    nt = test.shape[0]  # 'test' data; will be align to the 'reference' data
    nr = ref.shape[0]
    dist = np.zeros((nr, nt)) * np.nan

    # Mahalanobis distance, computed only where the band admits it. Filling every cell
    # first and then constraining only the accumulation would leave the whole function
    # quadratic in the two batch lengths however narrow the band: on random 700-sample
    # series a 10% band was then 7x faster, against 359x once the cost is restricted too.
    #
    # The two cases are separate loops rather than one loop over runtime bounds. The
    # bounds are monotone, so these two entries decide whether the band spans everything;
    # when it does, the original whole-array expression is used. A slice of the same
    # extent taken with runtime bounds is not free: numba cannot prove it contiguous and
    # compiles a slower matmul, which cost the unconstrained default 4x at 700 samples.
    if band[nt - 1, 0] == 0 and band[0, 1] == nr:
        for idx in np.arange(nt):
            deviation = test[idx] - ref
            dist[:, idx] = np.diag(deviation @ weight_matrix @ deviation.T)
    else:
        for idx in np.arange(nt):
            lower, upper = band[idx, 0], band[idx, 1]
            deviation = test[idx] - ref[lower:upper]
            dist[lower:upper, idx] = np.diag(deviation @ weight_matrix @ deviation.T)

    D = np.zeros((nr, nt)) * np.nan
    D[0, 0] = dist[0, 0]

    # The first row and the first column are the two edges of the matrix that have a
    # single predecessor each, so they accumulate directly. They are filled only as
    # far as the band admits: filling them regardless (as the unbanded version did)
    # would let a path run along an edge outside the corridor and re-enter it later.
    for jdx in np.arange(1, nt):
        if band[jdx, 0] > 0:
            break
        D[0, jdx] = dist[0, jdx] + D[0, jdx - 1]

    for kdx in np.arange(1, band[0, 1]):
        D[kdx, 0] = dist[kdx, 0] + D[kdx - 1, 0]

    for n in np.arange(1, nt):
        for m in np.arange(max(1, band[n, 0]), band[n, 1]):
            # index here must be integer!
            D[m, n] = dist[m, n] + np.nanmin([D[m, n - 1], D[m - 1, n - 1], D[m - 1, n]])

    return D


@jit(nopython=True)
def _best_predecessor(diagonal: float, horizontal: float, vertical: float) -> int:
    """
    Pick the cheapest reachable predecessor of a cost-matrix cell.

    Returns 0 for the diagonal step, 1 for the horizontal, 2 for the vertical, and -1
    when all three are NaN, which means the cell is unreachable. NaN entries are
    skipped rather than compared: they are cells outside a band constraint (#197).
    Among finite costs the order of the tests makes the diagonal win a tie, then the
    horizontal, which is what the previous comparison chain did.
    """
    best = np.inf
    choice = -1
    if not np.isnan(diagonal) and diagonal < best:
        best, choice = diagonal, 0
    if not np.isnan(horizontal) and horizontal < best:
        best, choice = horizontal, 1
    if not np.isnan(vertical) and vertical < best:
        choice = 2
    return choice


@jit(nopython=True)
def backtrack_optimal_path(D: np.ndarray) -> tuple[np.ndarray, float]:
    """Backtrack through the distance matrix to find the optimal warping path.

    Returns the path and the DTW distance, which is the ACCUMULATED cost at
    the end of the alignment, ``D[-1, -1]``. An earlier version summed the
    cumulative ``D`` entries along the path - a sum of prefix sums that grows
    super-linearly with path length and is not a distance; the per-batch
    alignment-quality numbers built from it were meaningless.

    Only finite predecessors are considered. Cells outside a band constraint (#197)
    stay NaN, and NaN fails every ``<=`` comparison, so the previous three-way
    comparison chain fell through to a bare ``AssertionError`` the moment a NaN
    neighbour was reached. On an unconstrained matrix the choice, ties included, is
    the same as before: the diagonal step wins a tie, then the horizontal.
    """
    nr, nt = D.shape
    nr -= 1
    nt -= 1
    distance = float(D[nr, nt])
    if np.isnan(distance):
        raise ValueError(
            "The final cell of the DTW cost matrix is not reachable, so no warping path "
            "exists. This means the band constraint is too narrow for these two batches."
        )
    path = [
        [nr, nt],
    ]
    while (nt + nr) != 0:
        if nt == 0:
            nr -= 1
        elif nr == 0:
            nt -= 1
        else:
            # Commented-code here is to read, but for Numba JIT, the other code is able to be
            # compiled. They give the same results in regular Python.
            # number = np.argmin([D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt]])
            choice = _best_predecessor(D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt])
            if choice == 0:
                nt -= 1
                nr -= 1
            elif choice == 1:
                nt -= 1
            elif choice == 2:
                nr -= 1
            else:
                raise ValueError(
                    "The DTW warping path reached a cell with no reachable predecessor. "
                    "This means the band constraint leaves the corridor disconnected."
                )

        path.append([nr, nt])

    # All done:
    path.reverse()
    return np.array(path), distance
