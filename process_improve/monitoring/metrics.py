import numpy as np
import pandas as pd

from ..univariate.metrics import Sn


def calculate_cpk(
    df: pd.DataFrame,
    which_column: str,
    specifications: tuple[float, float] = (np.nan, np.nan),
    trim_percentile: float = 2.5,
) -> float:
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
        If non-zero, then robust alternatives are used. The value specified is the percentile of
        data that is trimmed away; by default 2.5 percent on the left, and 2.5% on the right.

    Returns
    -------
    float
        The Cpk value.
    """
    assert trim_percentile >= 0
    assert trim_percentile < 40  # typically a max of 10 to 20 is advised.
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
        center_lower, center_upper = metric_lower.median(), metric_upper.median()
        spread_lower, spread_upper = Sn(metric_lower), Sn(metric_upper)
    else:
        center_lower, center_upper = metric_lower.mean(), metric_upper.mean()
        spread_lower, spread_upper = metric_lower.std(), metric_upper.std()

    # TODO: return the RSD also: rsd = (spread / center) * 100
    return np.nanmin([center_lower / (3 * spread_lower), center_upper / (3 * spread_upper)])


_RENAMED = {"calculate_Cpk": "calculate_cpk"}

def __getattr__(name: str) -> None:
    """Raise a helpful error when a renamed module attribute is accessed."""
    if name in _RENAMED:
        new = _RENAMED[name]
        raise AttributeError(
            f"{name!r} has been renamed to {new!r}. "
            f"Use: from process_improve.monitoring.metrics import {new}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
