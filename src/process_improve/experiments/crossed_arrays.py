# (c) Kevin Dunn, 2010-2026. MIT License.
"""Crossed arrays for robust design: an inner array of controllable factors run at every point of an outer array.

In a robust-design study (Taguchi; Montgomery, *Design and Analysis of Experiments*, chapter 12) the
*inner array* sets the factors that will be controlled in production, and the *outer array* the noise
factors, which can be controlled only during the experiment. Crossing the two runs every inner run at
every outer run, so each inner run gives as many readings as the outer array has runs, and their mean
and variance show how the controllable settings fare across the noise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from process_improve.experiments._analyses._shared import BLOCK_COL, WHOLE_PLOT_COL
from process_improve.experiments.factor import DesignResult


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
