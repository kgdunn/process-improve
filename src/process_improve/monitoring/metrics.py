import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ..univariate.metrics import Sn


def calculate_cpk(  # noqa: C901
    df: pd.DataFrame,
    which_column: str,
    specifications: tuple[float, float] = (np.nan, np.nan),
    trim_percentile: float = 2.5,
) -> Bunch:
    """
    Calculate the process capability, Cpk, near either the lower or the upper limit [will be
    automatically determined which].

    Process capability, nearer the lower limit = (avg - lower_spec)/(3 x std deviation)
    Process capability, nearer the upper limit = (upper_spec - avg)/(3 x std deviation)

    Parameters
    ----------
    df : pd.DataFrame
        Raw data, at least one column is numeric.
    which_column : str
        Indicates which is the column of data that should be used for the Cpk calculation.
    specifications : tuple of (lower, upper), optional
        A 2-tuple ``(lower_spec, upper_spec)`` of the lower and upper specification limits.
        Each element may be:

        * a numeric value, when the specification is constant over time;
        * a string, interpreted as a column name in ``df`` whose values give the
          per-row specification (use this when the specification changes over time);
        * ``None``, in which case the corresponding spec is estimated from the data
          using ``trim_percentile`` (a percentile-based robust limit).

        Default is ``(np.nan, np.nan)``, which treats both specs as numeric NaN
        and yields NaN for the corresponding side of the Cpk calculation.
    trim_percentile : float, optional
        Controls two things. (1) When a specification limit is missing, ``trim_percentile`` is
        used as a percentile on the data (in percent) to estimate that limit: the lower spec is
        set to ``np.nanpercentile(data, trim_percentile)`` and the upper spec to
        ``np.nanpercentile(data, 100 - trim_percentile)``. Default ``2.5`` therefore yields the
        2.5th and 97.5th percentiles. (2) When ``trim_percentile > 0`` the centre/spread used in
        the Cpk formula switch from mean/std to robust alternatives (median and ``Sn``); when 0
        the classical mean/std are used.

    Returns
    -------
    sklearn.utils.Bunch
        A bunch with the following fields:

        * ``cpk``: the Cpk value (the limiting, i.e. smaller, of the two sides).
        * ``center``: the center (mean or median) of the limiting side, i.e. of
          the distance-to-specification metric.
        * ``spread``: the spread (standard deviation or Sn) of the limiting side.
        * ``rsd``: the relative standard deviation OF THE DATA COLUMN, as a
          percentage: ``spread / center-of-the-data * 100``. (Earlier versions
          divided by the distance-to-spec centre, which is not an RSD of
          anything and changed value when the spec limit moved.)

    Notes
    -----
    The spread is estimated from ALL the values in the column (the overall
    long-term variation), not from a within-subgroup range or moving range.
    Strictly this makes the statistic a *performance* index (Ppk-style) rather
    than a short-term capability index (Cpk with sigma-hat = Rbar/d2); on a
    stable process the two coincide, and on an unstable one this value is the
    more honest of the two.
    """
    if trim_percentile < 0:
        raise ValueError(f"trim_percentile must be non-negative; got {trim_percentile}.")
    if trim_percentile >= 40:
        raise ValueError(f"trim_percentile must be < 40 (typically <= 10-20); got {trim_percentile}.")
    lower_spec, upper_spec = specifications

    if lower_spec is None:
        Cpk_lower_spec = float(np.nanpercentile(df[which_column].values, trim_percentile))
    elif isinstance(lower_spec, str):
        Cpk_lower_spec = df[lower_spec]
    else:
        Cpk_lower_spec = float(lower_spec)

    if upper_spec is None:
        Cpk_upper_spec = float(np.nanpercentile(df[which_column].values, 100 - trim_percentile))
    elif isinstance(upper_spec, str):
        Cpk_upper_spec = df[upper_spec]
    else:
        Cpk_upper_spec = float(upper_spec)

    metric_lower = df[which_column] - Cpk_lower_spec
    metric_upper = Cpk_upper_spec - df[which_column]

    if trim_percentile > 0:
        center_lower, center_upper = float(metric_lower.median()), float(metric_upper.median())
        spread_lower, spread_upper = float(Sn(metric_lower)), float(Sn(metric_upper))
        data_center, data_spread = float(df[which_column].median()), float(Sn(df[which_column]))
    else:
        center_lower, center_upper = float(metric_lower.mean()), float(metric_upper.mean())
        spread_lower, spread_upper = float(metric_lower.std()), float(metric_upper.std())
        data_center, data_spread = float(df[which_column].mean()), float(df[which_column].std())

    # A column with no spread (constant data, or only one non-NaN value)
    # makes Cpk undefined: a bare division would silently yield inf / NaN.
    # Emit a clear warning and return NaN per side -- callers can then
    # distinguish "Cpk could not be computed" from a numeric result. SEC-24
    # (#273).
    import warnings  # noqa: PLC0415

    def _safe_ratio(numer: float, denom: float, side: str) -> float:
        if not (denom > 0):
            warnings.warn(
                f"Cpk_{side}: spread is zero or non-finite; returning NaN. "
                "Likely cause: constant column or only one non-NaN value.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        return numer / (3 * denom)

    cpk_lower = _safe_ratio(center_lower, spread_lower, "lower")
    cpk_upper = _safe_ratio(center_upper, spread_upper, "upper")

    # The Cpk is the smaller (limiting) of the two sides; report the centre,
    # spread and RSD of whichever side that is. A NaN side never wins over a
    # finite one.
    if np.isnan(cpk_upper) or (not np.isnan(cpk_lower) and cpk_lower <= cpk_upper):
        cpk, center, spread = cpk_lower, center_lower, spread_lower
    else:
        cpk, center, spread = cpk_upper, center_upper, spread_upper

    # RSD of the DATA: the spread of (x - constant) equals the spread of x,
    # but the centre must be the centre of the data itself, not the centre of
    # the distance-to-spec metric (which made the "RSD" depend on where the
    # specification sat). data_center / data_spread are computed from the data
    # column directly above, alongside the per-side statistics.
    rsd = (data_spread / data_center) * 100 if abs(data_center) > 0 else float("nan")
    return Bunch(cpk=cpk, center=center, spread=spread, rsd=rsd)


_RENAMED = {"calculate_Cpk": "calculate_cpk"}


def __getattr__(name: str) -> None:
    """Raise a helpful error when a renamed module attribute is accessed."""
    if name in _RENAMED:
        new = _RENAMED[name]
        raise AttributeError(
            f"{name!r} has been renamed to {new!r}. Use: from process_improve.monitoring.metrics import {new}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
