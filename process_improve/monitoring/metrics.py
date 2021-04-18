import numpy as np
import pandas as pd
from ..univariate.metrics import Sn


def calculate_Cpk(
    df: pd.DataFrame,
    which_column: str,
    specifications=(np.NaN, np.NaN),
    trim_percentile: float = 5.0,
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
    specifications : tuple, optional
        Either a value, if the specification is constant over time; if the specification changes
        over time, then use two column names here, one of which is the lower specification and
        the second is the upper specification.
    trim_percentile : float, optional
        If non-zero, then robust alternatives are used. The value specified here is the percentile
        of the data that is trimmed away; by default 5 percent on the left, and 5% on the right.

    Returns
    -------
    float
        The Cpk value.
    """
    assert trim_percentile >= 0
    assert trim_percentile < 40  # typically a max of 10 to 20 is advised.
    lower_spec, upper_spec = specifications

    if lower_spec is None:
        Cpk_lower_spec = float(
            np.nanpercentile(df[which_column].values, [trim_percentile])
        )
    elif isinstance(lower_spec, str):
        Cpk_lower_spec = df[lower_spec]
    else:
        Cpk_lower_spec = float(lower_spec)

    if upper_spec is None:
        Cpk_upper_spec = float(
            np.nanpercentile(df[which_column].values, [100 - trim_percentile])
        )
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
    Cpk = np.nanmin(
        [center_lower / (3 * spread_lower), center_upper / (3 * spread_upper)]
    )
    return Cpk
