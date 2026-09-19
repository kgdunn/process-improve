# (c) Kevin Dunn, 2010-2026. MIT License.

"""Gap filling and smoothing for batch trajectory data.

Two jobs that look like one-liners and are not, because a batch data set is a
*collection* of trajectories, not one long series.

**Filling.** The obvious ``df.bfill().ffill()`` is wrong in three specific ways
on trajectory data, and :func:`fill_gaps` fixes each one:

1. It holds the last value flat across the gap. For a ramp, a conversion curve,
   or anything monotone, that manufactures a step followed by a plateau, and the
   derivative-based features (:func:`~process_improve.batch.features.f_slope`,
   :func:`~process_improve.batch.features.f_rupture`) then report the step as a
   real process event. Interpolating between the two observed ends does not.
2. It bridges a gap of any length without saying so. Two missing samples and two
   hundred missing samples are not the same claim about the process, but they
   produce the same silent, confident output.
3. ``bfill`` at the start of a batch fills from the future. Offline that is
   merely surprising; carried into
   :class:`~process_improve.batch._batch_monitor.BatchMonitor`, where a partial
   batch is scored as it runs, it is a leak.

The third point is why the default leaves leading and trailing gaps alone rather
than extrapolating, and the second is why a filled gap is capped at *limit*
samples. What is left over stays ``NaN`` on purpose: the estimators here
(:class:`~process_improve.multivariate.methods.PCA`, ``PLS``,
:class:`~process_improve.batch._batch_pca.BatchPCA`) all have NIPALS
missing-data paths, so an honest hole is worth more to them than an invented
number. Filling exists to remove the *short* gaps that only add noise.

**Smoothing.** :func:`smooth_trajectories` runs within each batch, never across
the join between two of them. Concatenating batches and calling ``rolling()`` or
``savgol_filter`` once is the standard way to get this wrong: the filter window
straddles the boundary and the end of one batch leaks into the start of the next.

Savitzky-Golay is the default because it fits a local polynomial rather than
averaging, so it preserves the height and area of a peak up to its polynomial
order. Those are exactly the quantities
:func:`~process_improve.batch.features.f_max` and
:func:`~process_improve.batch.features.f_area` go on to measure, and a moving
average would shrink both before they were measured. ``"lowess"`` is the
alternative when the trajectory carries spikes: its robustness iterations
downweight them, where Savitzky-Golay smears each spike across its window.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

#: Savitzky-Golay needs a window strictly longer than its polynomial order, and
#: an odd one so the fitted point sits at the window's centre.
_MIN_WINDOW = 3
#: Relative size below which a fit counts as having left the data where it was.
#: LOWESS reproduces its input to rounding, not bit-for-bit, when the robustness
#: weights collapse, so the comparison needs a tolerance scaled to the data.
_UNCHANGED_TOL = 1e-9

FillMethod = Literal["linear", "time", "pchip", "nearest"]
EdgePolicy = Literal["leave", "nearest"]
SmoothMethod = Literal["savgol", "lowess"]


def _columns_to_treat(batch: pd.DataFrame, columns: list[str] | None) -> list[str]:
    """Return the numeric columns to work on, defaulting to all of them."""
    if columns is not None:
        missing = [name for name in columns if name not in batch.columns]
        if missing:
            msg = f"Columns {missing} are not in the batch data; it has {list(batch.columns)}."
            raise ValueError(msg)
        return list(columns)
    return [name for name in batch.columns if pd.api.types.is_numeric_dtype(batch[name])]


def _run_lengths(missing: np.ndarray) -> np.ndarray:
    """Return, for each position, the length of the run of ``True`` it belongs to.

    Observed positions get 0. Used to decide which gaps are short enough to fill,
    before any filling happens, so the decision is made on the real gap and not
    on what an earlier pass left behind.
    """
    lengths = np.zeros(missing.shape, dtype=int)
    if not missing.any():
        return lengths
    # Boundaries of each run of missing values.
    padded = np.concatenate(([False], missing, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        lengths[start:stop] = stop - start
    return lengths


def fill_gaps(
    batches: dict[str, pd.DataFrame],
    *,
    method: FillMethod = "linear",
    limit: int | None = 5,
    edge: EdgePolicy = "leave",
    columns: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fill short interior gaps in each batch, and report what was left alone.

    Parameters
    ----------
    batches : dict[str, pd.DataFrame]
        Batch identifier to that batch's trajectory frame, one row per sample.
    method : {"linear", "time", "pchip", "nearest"}
        How to interpolate across an interior gap. ``"linear"`` joins the two
        observed ends. ``"time"`` does the same but weights by the index, which
        is the one to use when samples are unevenly spaced in time and the index
        carries that time. ``"pchip"`` is shape-preserving cubic: smoother than
        linear, and unlike a plain cubic it will not overshoot into an impossible
        value such as a negative concentration. ``"nearest"`` holds the closer of
        the two ends, which is the old behaviour, restricted to interior gaps.
    limit : int or None
        The longest run of consecutive missing samples that may be filled.
        Anything longer stays ``NaN``, because the estimators in this package
        read ``NaN`` as "not measured" and an invented trajectory as fact.
        ``None`` fills every interior gap, however long.
    edge : {"leave", "nearest"}
        What to do with missing samples before the first or after the last
        observation in a column. ``"leave"`` keeps them missing, which is the
        default because filling the start of a batch means reading backwards from
        the future. ``"nearest"`` extends the first and last observed values
        outwards, matching what ``bfill().ffill()`` did.
    columns : list[str] or None
        Columns to fill; every numeric column by default. Non-numeric columns,
        such as a batch label carried alongside the trajectory, are never touched.

    Returns
    -------
    tuple[dict[str, pd.DataFrame], pd.DataFrame]
        The filled batches, and a report indexed by batch identifier with one
        column per treated variable holding ``"filled/remaining"`` counts, plus
        ``n_samples``. The report is the point: it says how much of the data
        being modelled was measured and how much was interpolated.

    Raises
    ------
    ValueError
        If *batches* is empty, *limit* is not positive, or *columns* names a
        column the data does not have.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> ramp = pd.DataFrame({"T": [20.0, 25.0, np.nan, 35.0, 40.0]})
    >>> filled, report = fill_gaps({"B1": ramp})
    >>> float(filled["B1"]["T"].iloc[2])
    30.0
    >>> report.loc["B1", "T"]
    '1/0'
    """
    if not batches:
        msg = "At least one batch is required."
        raise ValueError(msg)
    if limit is not None and limit < 1:
        msg = f"limit must be a positive number of samples, or None; got {limit}."
        raise ValueError(msg)

    filled: dict[str, pd.DataFrame] = {}
    rows: dict[str, dict[str, Any]] = {}

    for batch_id, batch in batches.items():
        out = batch.copy()
        row: dict[str, Any] = {"n_samples": len(batch)}

        for name in _columns_to_treat(batch, columns):
            series = out[name]
            missing = series.isna().to_numpy()
            if not missing.any():
                row[name] = f"0/{0}"
                continue

            # Decide on the original gaps, before anything is filled.
            too_long = _run_lengths(missing) > (limit if limit is not None else len(series))
            interior = series.interpolate(method=method, limit_area="inside")
            if edge == "nearest":
                interior = interior.ffill().bfill()
            interior[too_long & missing] = np.nan

            out[name] = interior
            still_missing = int(interior.isna().sum())
            row[name] = f"{int(missing.sum()) - still_missing}/{still_missing}"

        filled[batch_id] = out
        rows[batch_id] = row

    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "batch_id"
    logger.debug("fill_gaps: %d batches, method=%r, limit=%r, edge=%r", len(batches), method, limit, edge)
    return filled, report


def _savgol_column(values: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay filter one column, leaving any missing samples missing.

    ``savgol_filter`` spreads a single ``NaN`` across its whole window, so the
    filter runs on the observed samples only and the gaps are put back afterwards.
    """
    observed = np.isfinite(values)
    n_observed = int(observed.sum())
    if n_observed <= polyorder:
        return values.copy()

    usable = min(window, n_observed)
    if usable % 2 == 0:
        usable -= 1
    if usable <= polyorder or usable < _MIN_WINDOW:
        return values.copy()

    out = values.copy()
    out[observed] = savgol_filter(values[observed], usable, polyorder)
    return out


def _lowess_column(
    values: np.ndarray,
    index: np.ndarray,
    frac: float,
    iterations: int,
) -> tuple[np.ndarray, bool]:
    """LOWESS one column against its index, leaving any missing samples missing.

    Returns the smoothed column and whether the robustness weights collapsed.

    The collapse is worth naming. LOWESS scales its robustness weights by
    ``6 * median(|residual|)``. On a trajectory that the local fits reproduce
    exactly away from a few spikes, a perfect ramp or a flat-lined sensor, that
    median is zero, every weight degenerates, and ``lowess`` hands back the input
    untouched. Nothing raises, the output is finite and the right length, and the
    only symptom is that the "smoothed" trajectory still carries its spikes. It
    is the same implosion a median-of-differences scale estimator suffers under a
    tied majority, in a place nobody looks for it, so it is detected rather than
    left to be noticed downstream.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: PLC0415

    observed = np.isfinite(values)
    if int(observed.sum()) < _MIN_WINDOW:
        return values.copy(), False

    y, x = values[observed], index[observed]
    fitted = lowess(y, x, frac=frac, it=iterations, return_sorted=False)

    def unchanged(candidate: np.ndarray) -> bool:
        return bool(np.max(np.abs(candidate - y)) <= _UNCHANGED_TOL * max(1.0, float(np.ptp(y))))

    collapsed = False
    if iterations > 0 and unchanged(fitted):
        without_robustness = lowess(y, x, frac=frac, it=0, return_sorted=False)
        if not unchanged(without_robustness):
            collapsed = True
            fitted = without_robustness

    out = values.copy()
    out[observed] = fitted
    return out, collapsed


def _validate_smoothing(
    batches: dict[str, pd.DataFrame],
    method: str,
    window: int,
    polyorder: int,
    frac: float,
) -> None:
    """Reject settings that cannot produce a smoothed trajectory."""
    if not batches:
        msg = "At least one batch is required."
        raise ValueError(msg)
    if method not in ("savgol", "lowess"):
        msg = f"method must be 'savgol' or 'lowess'; got {method!r}."
        raise ValueError(msg)
    if polyorder >= window:
        msg = f"polyorder={polyorder} must be below window={window}."
        raise ValueError(msg)
    if not 0.0 < frac <= 1.0:
        msg = f"frac must be in (0, 1]; got {frac}."
        raise ValueError(msg)


def _warn_about_smoothing(shortened: list[str], collapsed: list[str], window: int) -> None:
    """Report the batches and columns that were not smoothed as asked."""
    if collapsed:
        warnings.warn(
            f"LOWESS robustness weights collapsed on {len(collapsed)} column(s): "
            f"{sorted(collapsed)[:5]}{' ...' if len(collapsed) > 5 else ''}. Their residuals away from the "
            f"outliers are essentially zero, so the 6*median(|residual|) scale is zero and every weight "
            f"degenerates; lowess then returns the column untouched. They were smoothed with iterations=0 "
            f"instead, which smooths but does not reject outliers. Pass method='savgol', or check whether "
            f"those columns are flat-lined or heavily quantised.",
            UserWarning,
            stacklevel=3,
        )
    if shortened:
        warnings.warn(
            f"{len(shortened)} batch(es) are shorter than window={window}, so their window was shrunk to fit: "
            f"{sorted(shortened)[:5]}{' ...' if len(shortened) > 5 else ''}. They are smoothed less than the "
            f"rest, which matters if the smoothed trajectories are compared across batches.",
            UserWarning,
            stacklevel=3,
        )


def smooth_trajectories(  # noqa: PLR0913 - the two smoothers' settings are not interchangeable, and
    # naming them beats a settings dict whose valid keys depend on `method`
    batches: dict[str, pd.DataFrame],
    *,
    method: SmoothMethod = "savgol",
    window: int = 11,
    polyorder: int = 2,
    frac: float = 0.15,
    iterations: int = 3,
    columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Smooth each batch's trajectories, one batch at a time.

    The per-batch loop is the substance, not an implementation detail.
    Concatenating batches and filtering once lets the window straddle the join,
    so the tail of one batch bleeds into the head of the next and both ends are
    quietly wrong. Here a batch never sees another batch's samples.

    Parameters
    ----------
    batches : dict[str, pd.DataFrame]
        Batch identifier to that batch's trajectory frame.
    method : {"savgol", "lowess"}
        ``"savgol"`` fits a local polynomial, so it preserves peak height and
        area up to *polyorder*; use it when
        :func:`~process_improve.batch.features.f_max`,
        :func:`~process_improve.batch.features.f_area` or a slope is measured
        downstream, since a moving average would shrink all three before they
        were measured. ``"lowess"`` is locally weighted regression with
        robustness iterations: slower, and the one to reach for when the
        trajectory carries spikes, which Savitzky-Golay smears across its window
        instead of rejecting.
    window : int
        Savitzky-Golay window length in samples. Forced odd, and shrunk to fit a
        batch shorter than the window rather than failing on it, since batches
        legitimately differ in length before alignment.
    polyorder : int
        Degree of the local polynomial. Must be below *window*.
    frac : float
        LOWESS bandwidth: the fraction of each batch used for each local fit.
    iterations : int
        LOWESS robustness iterations. 0 turns robustness off; the default of 3
        is what makes it reject spikes.
    columns : list[str] or None
        Columns to smooth; every numeric column by default.

    Returns
    -------
    dict[str, pd.DataFrame]
        The smoothed batches. Missing samples stay missing: neither filter can
        see through a gap, so run :func:`fill_gaps` first if the short ones
        should be closed.

    Raises
    ------
    ValueError
        If *batches* is empty, *method* is unknown, *polyorder* is not below
        *window*, or *frac* is outside (0, 1].

    Warns
    -----
    UserWarning
        When a batch is too short for the requested window, so its window was
        shrunk. Silently returning a differently-smoothed batch would make the
        collection inconsistent without saying so.
    UserWarning
        When LOWESS's robustness weights collapse on a column, which makes
        ``lowess`` hand the column back untouched. See :func:`_lowess_column`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> clean = np.linspace(0, 10, 60) ** 0.5
    >>> noisy = pd.DataFrame({"y": clean + rng.normal(scale=0.05, size=60)})
    >>> out = smooth_trajectories({"B1": noisy}, window=11, polyorder=2)
    >>> bool(np.std(out["B1"]["y"] - clean) < np.std(noisy["y"] - clean))
    True
    """
    _validate_smoothing(batches, method, window, polyorder, frac)

    smoothed: dict[str, pd.DataFrame] = {}
    shortened: list[str] = []
    collapsed: list[str] = []

    for batch_id, batch in batches.items():
        out = batch.copy()
        if method == "savgol" and len(batch) < window:
            shortened.append(str(batch_id))

        position = np.arange(len(batch), dtype=float)
        for name in _columns_to_treat(batch, columns):
            values = batch[name].to_numpy(dtype=float)
            if method == "savgol":
                out[name] = _savgol_column(values, window, polyorder)
            else:
                out[name], degenerate = _lowess_column(values, position, frac, iterations)
                if degenerate:
                    collapsed.append(f"{batch_id}.{name}")

        smoothed[batch_id] = out

    _warn_about_smoothing(shortened, collapsed, window)
    return smoothed
