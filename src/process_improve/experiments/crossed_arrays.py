# (c) Kevin Dunn, 2010-2026. MIT License.
"""Crossed arrays for robust design: an inner array of controllable factors run at every point of an outer array.

In a robust-design study (Taguchi; Montgomery, *Design and Analysis of Experiments*, chapter 12) the
*inner array* sets the factors that will be controlled in production, and the *outer array* the noise
factors, which can be controlled only during the experiment. Crossing the two runs every inner run at
every outer run, so each inner run gives as many readings as the outer array has runs, and their mean
and variance show how the controllable settings fare across the noise. :func:`signal_to_noise` sums up
each inner run's readings as one of Taguchi's signal-to-noise ratios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from process_improve.experiments._analyses._shared import BLOCK_COL, WHOLE_PLOT_COL
from process_improve.experiments.factor import DesignResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

#: Taguchi's signal-to-noise ratios, in decibels, of each row of readings; a blank reading is left out.
_RATIOS: dict[str, Callable[[pd.DataFrame], ArrayLike]] = {
    "larger": lambda y: -10 * np.log10((1 / y**2).mean(axis=1)),
    "smaller": lambda y: -10 * np.log10((y**2).mean(axis=1)),
    "nominal": lambda y: 10 * np.log10(y.mean(axis=1) ** 2 / y.var(axis=1, ddof=1)),
}


def cross_arrays(
    inner: DesignResult | pd.DataFrame,
    outer: DesignResult | pd.DataFrame,
    *,
    units: Literal["coded", "actual"] = "coded",
) -> pd.DataFrame:
    """Run every run of an inner array at every run of an outer array.

    Parameters
    ----------
    inner : DesignResult or pandas.DataFrame
        The inner array: the controllable factors, one run per row.
    outer : DesignResult or pandas.DataFrame
        The outer array: the noise factors, one run per row. To read each pair of runs more than
        once, repeat the outer runs (``generate_design(..., n_replicates=3)``).
    units : {"coded", "actual"}, default "coded"
        For a :class:`DesignResult`, which of its tables to cross: ``design`` or ``design_actual``.
        A DataFrame is crossed as it is given.

    Returns
    -------
    pandas.DataFrame
        One row for each pair of runs, ``len(inner) * len(outer)`` rows, ordered by the inner run
        and then by the outer run. The columns are ``inner_run`` and ``outer_run`` (the position of
        the run in its array, from 1), then the inner array's factors and the outer array's. The
        ``RunOrder`` column of a design is left out. To lay the readings out as the sources print
        them, one row per inner run and one column per outer run, use
        ``crossed.pivot(index="inner_run", columns="outer_run", values="y")``.

    Raises
    ------
    ValueError
        If an array has no runs or no factors, has a ``Block`` or ``WholePlot`` column, or shares a
        factor name with the other array, or if ``units`` is neither ``"coded"`` nor ``"actual"``.

    Notes
    -----
    A :class:`DesignResult` is crossed in the order of its table, which is its randomised run order.
    For the standard order, pass ``result.design.iloc[numpy.argsort(result.run_order)]``.

    The mean and the variance of each inner run's readings, the summaries of a robust-design study,
    are ``crossed.groupby("inner_run")["y"].agg(["mean", "var"])``.

    Examples
    --------
    Montgomery's leaf springs (Table 12.1): a 2^(4-1) inner array in the controllable factors A to D,
    run three times at each level of the noise factor E, so each inner run gives six readings.

    >>> from process_improve.experiments import Factor, cross_arrays, generate_design
    >>> inner = generate_design([Factor(name=n, low=-1, high=1) for n in "ABCD"], "fractional_factorial",
    ...                         n_center_points=0, random_state=1)
    >>> outer = generate_design([Factor(name="E", low=-1, high=1)], "full_factorial", n_center_points=0,
    ...                         n_replicates=3, random_state=1)
    >>> crossed = cross_arrays(inner, outer)
    >>> crossed.shape
    (48, 7)
    >>> list(crossed.columns)
    ['inner_run', 'outer_run', 'A', 'B', 'C', 'D', 'E']
    """
    if units not in ("coded", "actual"):
        msg = f"units must be 'coded' or 'actual', not {units!r}."
        raise ValueError(msg)
    inside, outside = _factors(inner, "inner", units), _factors(outer, "outer", units)
    shared = sorted(set(inside.columns) & set(outside.columns))
    if shared:
        msg = f"The inner and outer arrays both have the factors {shared}; a factor belongs to one array."
        raise ValueError(msg)
    rows = np.repeat(np.arange(len(inside)), len(outside))
    columns = np.tile(np.arange(len(outside)), len(inside))
    crossed = pd.concat(
        [inside.iloc[rows].reset_index(drop=True), outside.iloc[columns].reset_index(drop=True)], axis=1
    )
    crossed.insert(0, "outer_run", columns + 1)
    crossed.insert(0, "inner_run", rows + 1)
    return crossed


def _factors(design: DesignResult | pd.DataFrame, which: str, units: str) -> pd.DataFrame:
    """Return the factor columns of an array; refuse an array with blocks or whole plots."""
    if isinstance(design, DesignResult):
        frame = pd.DataFrame(design.design_actual if units == "actual" else design.design)
    else:
        frame = pd.DataFrame(design)
    structure = [c for c in (BLOCK_COL, WHOLE_PLOT_COL) if c in frame.columns]
    if structure:
        msg = (
            f"The {which} array has a {structure[0]} column, and cross_arrays crosses arrays without blocks "
            "or whole plots. Drop the column to cross the runs alone."
        )
        raise ValueError(msg)
    frame = frame.drop(columns="RunOrder", errors="ignore")
    if frame.empty:
        msg = f"The {which} array has no runs or no factors."
        raise ValueError(msg)
    return frame


def signal_to_noise(readings: ArrayLike, kind: Literal["larger", "smaller", "nominal"]) -> float | np.ndarray:
    """Return Taguchi's signal-to-noise ratio of repeated readings, in decibels: the higher, the better.

    Parameters
    ----------
    readings : array_like
        The readings at one setting, or a table of them with one row for each setting (each
        inner run of a crossed array) and one column for each reading. A blank (NaN) reading
        is left out.
    kind : {"larger", "smaller", "nominal"}
        What makes a reading good. ``"larger"`` (larger is better) is -10 log10 of the mean of
        1/y^2, ``"smaller"`` (smaller is better) is -10 log10 of the mean of y^2, and
        ``"nominal"`` (nominal is best) is 10 log10 of the squared mean over the variance.

    Returns
    -------
    float or numpy.ndarray
        The ratio of the readings, or one ratio for each row of a table.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the three, or ``readings`` is not one or two dimensional.

    Notes
    -----
    Larger is better for a reading of 0 is -inf, and nominal is best for readings that do not vary
    is +inf. Nominal is best needs two readings for the variance: with fewer it is NaN.

    For nominal is best, Taguchi's two-step optimisation then picks the settings with the highest
    ratio and moves the mean to the target with a factor that changes the mean but not the ratio.

    Examples
    --------
    Schmidt and Launsby's signal-to-noise example: the first two settings of a 2^(3-1) design, each
    read twice.

    >>> from process_improve.experiments import signal_to_noise
    >>> signal_to_noise([[20, 30], [9, 11]], "nominal").round(2)
    array([10.97, 16.99])
    >>> round(signal_to_noise([20, 30], "larger"), 2)
    27.43
    """
    if kind not in _RATIOS:
        msg = f"kind must be 'larger', 'smaller' or 'nominal', not {kind!r}."
        raise ValueError(msg)
    values = np.asarray(readings, dtype=float)
    if values.ndim not in (1, 2):
        msg = f"readings must be one setting's readings or a table of them, not an array of {values.ndim} dimensions."
        raise ValueError(msg)
    with np.errstate(divide="ignore"):
        ratios = np.asarray(_RATIOS[kind](pd.DataFrame(np.atleast_2d(values))), dtype=float)
    return float(ratios[0]) if values.ndim == 1 else ratios
