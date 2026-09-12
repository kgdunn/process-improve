from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import iqr, median_abs_deviation

if TYPE_CHECKING:
    from pandas.core.groupby import DataFrameGroupBy

try:
    import ruptures as rpt
except ImportError:  # pragma: no cover - exercised via env-without-ruptures
    from .._extras import _MissingExtra

    rpt = _MissingExtra("ruptures", "batch")  # type: ignore[assignment]

from ..bivariate.methods import find_elbow_point
from ..regression.methods import repeated_median_slope

# General
# ------------------------------------------


def _prepare_data(  # noqa: C901, PLR0912
    df: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
    age_col: str | None = None,
) -> tuple[DataFrameGroupBy, list[str], pd.DataFrame, pd.DataFrame]:
    """
    General function, used for all feature extractions.

    1. Groups the ``df`` by batch firstly, and by phase, secondly.
    2. Creates the output dataframe to write the results to.
    """

    # Special case: a single series. Convert it to a dataframe
    if isinstance(df, pd.Series) and (tags is None) and isinstance(df.index, pd.DatetimeIndex):
        name = "tag" if df.name is None else df.name

        tags = [name]
        df = pd.DataFrame(df, columns=tags)
        ##  NOT SURE WHAT THIS IS FOR df.insert(0, SETTINGS["batch_number_tag"], 1)

    # If no phase_col: add a fake column with all the same values
    if phase_col is None:
        phase_col = "__phase_grouper__"

    if phase_col not in df:
        # Cannot use "None" or 'nan'; else it won't group in the
        # groupby([batch_col, phase_col]) statement below.
        df.insert(0, phase_col, "NoPhases")

    # If no batch_col: add a fake column with all the same values
    if batch_col not in df or batch_col is None:
        batch_col = "__batch_grouper__"
        if batch_col not in df:
            # Cannot use "None" or 'nan'; else it won't group in the
            # groupby([batch_col, phase_col]) statement below.
            df.insert(0, batch_col, 1)

    if tags is None:
        tags = list(df.columns)
        if "__phase_grouper__" in tags:
            tags.remove("__phase_grouper__")
        if "__batch_grouper__" in tags:
            tags.remove("__batch_grouper__")

    # The user has provided a single tag name as a string, instead of a list of strings.
    if isinstance(tags, str):
        tags = [tags]

    # Check that all these columns actually exist in the df
    missing_tags = [column_name for column_name in tags if column_name not in df.columns]
    if missing_tags:
        raise KeyError(f"Tag(s) not found in the dataframe columns: {missing_tags}.")

    # First make a copy! else, it will repeatedly add these. Ensure it is
    # a unique list too.
    tags = tags.copy()
    tags.extend([batch_col, phase_col])
    if age_col and age_col in df:
        tags.append(age_col)
    seen_tags = []
    for tag in tags:
        if tag not in seen_tags:
            seen_tags.append(tag)

    tags = seen_tags

    if age_col and age_col in df:
        # Feature operations that require to know the time axis can
        # use a duplicate of the dataframe, and then the index from the
        # time axis to do their calculations.

        df_out = df[tags].copy().set_index(age_col).sort_index()
        tags.remove(age_col)
        grouped_data = df_out.groupby([batch_col, phase_col], as_index=True)

    else:
        df_out = df[tags].copy()
        grouped_data = df_out.groupby([batch_col, phase_col], as_index=True)

    # Remove the extra tags
    tags = [x for x in tags if x not in [batch_col, phase_col]]

    # Create the storage DF that will used to fill the outputs
    multiindex = df.index.unique()
    output = pd.DataFrame(index=multiindex)
    output.columns.name = "__features__"
    return (grouped_data, tags, output, df_out)


# Location-based features
# ------------------------------------------
def f_mean(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    mean.

    The arithmetic mean for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.
    """
    base_name = "mean"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.mean()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))[f_names]


def f_median(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    median.

    The median for the given tags in ``tags``, for each unique batch in the ``batch_col``
    indicator column, and within each unique phase, per batch, of the ``phase_col`` column.
    """
    base_name = "median"
    prepared, tags, output, _df_out = _prepare_data(data, tags, batch_col, phase_col)

    # If the feature is to be calculated within a specific range of time, then provide the
    # `slicer` input and the name of the `age_col` (column containing time evolution) which is
    # to be sliced.

    # `slicer` is a 2-element tuple: (start, stop): indicating the time points to be used during
    # which the feature is calculated. The bounds are inclusive.

    # if age_col and slicer:
    #     # The `age_col` is the index.
    #     rows = {}
    #     for idx, subgroup in prepared:
    #         rows[idx] = subgroup[
    #             (subgroup.index >= slicer[0]) & (subgroup.index <= slicer[1])
    #         ].median()

    #     # Transpose it!
    #     output = pd.DataFrame(data=rows).T
    #     f_names = [f"{tag}_{base_name}_{slicer[0]}_to_{slicer[1]}" for tag in tags]

    # The following is preferred. Not the above. Leaving it here, because it could be useful
    # to see how the slicer was constructed.
    # NOTE: an alternative for the above, is not to provide the extra function inputs of
    # `age_col` and `slicer`, but that the user pre-slices their own data.
    #
    # >>>  df = snip_head_of_all_batches(df, time_tag, snip_off_time=100)
    # >>>  df = snip_tail_of_all_batches(df, time_tag: str, snip_off_time=150)
    # >>>  f_mean(df, tags, batch_col, phase_col)

    output = prepared.median()
    f_names = [(tag + "_" + base_name) for tag in tags]
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))[f_names]


# Scale-based features
# ------------------------------------------
def f_std(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    std.

    The standard deviation for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    See also: f_iqr
    """
    base_name = "std"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.std()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_iqr(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    iqr.

    The InterQuartile Range (IQR) for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    The IQR is a robust variant of the standard deviation.
    The difference between the 75th percentile and the 25th percentile of a
    sample this is the 25 % trimmed range, an example of an L - estimator.

    See also: f_std
    """
    base_name = "iqr"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.agg(iqr)
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_robust_mad(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    robust_mad.

    The Median Absolute Deviation (MAD) for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    The MAD is a robust alternative to the standard deviation. It is scaled by
    the normal-consistency factor (~1.4826), so that for normally distributed
    data it estimates the same quantity as ``f_std``.

    See also: f_std, f_iqr
    """
    base_name = "robust_mad"

    def _mad(col: pd.Series) -> float:
        return float(median_abs_deviation(col, scale="normal", nan_policy="omit"))

    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.agg(_mad)
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


# Cumulative features
# ------------------------------------------
def f_sum(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    sum.

    The SUM within each tag for for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    If the x-axis (time) data are evenly-spaced, then this is directly
    proportional to the area under the trace (curve/trajectory).

    See also: f_cumsum
    """
    base_name = "sum"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.sum()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_area(
    data: pd.DataFrame,
    time_tag: str,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    area.

    The AREA of each tag for for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column against
    the time-based curve.

    The spacing of the x-axis is taken into account, so, this will produce
    accurate areas if the data are not evenly-spaced in time along the x-axis.

    The area is calculated using the trapezoidal rule.

    See also: f_sum, f_cumsum
    """
    base_name = "area"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col, age_col=time_tag)
    f_names = [(tag + "_" + base_name) for tag in tags]

    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum() * 0.0
    for batch_id, this_batch in prepared:
        # Average width of the base * height. For time series, the average width of the base is
        # the same as the sampling intervals. Therefore the diff of the index.
        half_base_factor = np.diff(this_batch.index)
        for tag in tags:
            # Now the sum makes sense: gets the area under the curve by adding
            # up the smaller trapezoids: consider the trapezoids as rotated by 90 degrees.
            # The parallel edges are lying vertically. Area of each one is the average of the
            # heights on the left and the right, multiplied by delta distance on the horizontal
            # index axis. Area = average(parallel lengths) * height
            # where height = delta distance on the horizontal axis.
            left_vals = np.asarray(this_batch[tag].iloc[0:-1].to_numpy())
            right_vals = np.asarray(this_batch[tag].iloc[1:].to_numpy())
            area = ((left_vals + right_vals) / 2 * half_base_factor).sum()
            output.loc[batch_id, tag] = area

    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


# Breakpoint detection:  rupture / breakpoint within a particular tag.
# ------------------------------------------
def f_rupture(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
    settings: dict | None = None,
) -> pd.DataFrame:
    """
    Feature:    rupture.

    The change points (breakpoints) of each tag in ``tags``, for each unique batch in the
    ``batch_col`` indicator column, and within each unique phase, per batch, of the
    ``phase_col`` column.

    A change point is a position where the statistical behaviour of the trajectory
    shifts: the mean steps, the variance opens up, or the correlation structure changes.
    In a batch context these usually mark the transitions between operating stages
    (heat-up finishing, a feed starting, an agitator changing speed), which is why they
    are worth extracting even when no phase column is recorded.

    Detection is done by the ``ruptures`` package, using PELT (Pruned Exact Linear Time),
    an exact search whose cost is linear in the length of the signal. PELT does not need
    to be told how many change points to look for; the ``penalty`` decides that instead.

    Parameters
    ----------
    data : pd.DataFrame
        Batch data in the long (melted) format the other feature functions take.
    tags : list[str], optional
        Which tags to search. Defaults to every non-grouping column.
    batch_col, phase_col : str, optional
        The batch and phase indicator columns, as elsewhere in this module.
    settings : dict, optional
        Detector options::

            {
                "model": "rbf",     # cost function; see below
                "penalty": None,    # cost of one more change point; None means log(n)
                "min_size": 2,      # smallest number of samples in a segment
                "jump": 5,          # only consider change points at multiples of this
            }

        ``model`` is the cost function ``ruptures`` minimises: ``"rbf"`` (the default)
        detects any change in distribution through a kernel, ``"l2"`` detects a change in
        the mean, ``"l1"`` is its outlier-resistant counterpart, and ``"normal"`` detects
        a change in mean or variance.

        The default is ``"rbf"`` because its kernel cost is bounded, which makes it
        insensitive to the tag's units: multiplying a signal by 1000 leaves the detected
        change points unchanged. ``"l2"`` is not, and on the same rescaled signal the same
        penalty turned one true change point into 37 spurious ones. Scale the tag, or
        scale the penalty with its variance, before choosing ``"l2"``.

        ``penalty`` decides how many change points come back: larger values return fewer.
        ``None`` (the default) uses ``log(n)`` for a signal of ``n`` samples, the
        BIC-style choice, which adapts to the length of the batch. Measured on a single
        step of five standard deviations, it recovers the change point exactly at 60, 200
        and 1000 samples under both ``"rbf"`` and ``"l2"``, and on pure noise of those
        lengths it reports at most one spurious change point.

        ``jump`` trades resolution for speed; pass 1 to consider every sample.

    Returns
    -------
    pd.DataFrame
        Indexed like the other feature functions, with one ``<tag>_rupture`` column per
        tag. **Unlike them, the cells are not numeric**: each holds a tuple of integer
        positions into that batch's rows, which is the only faithful shape for a result
        whose length varies from batch to batch. Position ``i`` means the change occurs
        between row ``i - 1`` and row ``i``.

        The trailing sentinel ``ruptures`` appends (the length of the signal, which is
        not a change point) is removed, so an empty tuple means no change was detected.

        For a numeric feature to put in a model matrix, take the count::

            breaks = f_rupture(data, tags=["Temperature"], batch_col="batch_id")
            counts = breaks.map(len)

    Raises
    ------
    ImportError
        If ``ruptures`` is not installed. It is part of the optional ``batch`` extra.
    ValueError
        If ``settings`` carries a key this function does not recognize.

    Notes
    -----
    This function returns the change points rather than drawing them. ``ruptures`` has a
    ``display`` helper that uses matplotlib; plotting belongs to the caller, and the rest
    of this project plots with plotly.

    See also: f_elbow, f_cross
    """
    default_settings: dict = {"model": "rbf", "penalty": None, "min_size": 2, "jump": 5}
    if settings:
        unknown = set(settings) - set(default_settings)
        if unknown:
            raise ValueError(
                f"f_rupture got unrecognized settings {sorted(unknown)}; expected any of {sorted(default_settings)}."
            )
        default_settings.update(settings)

    settings = default_settings
    base_name = "rupture"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]

    # Build the output frame the way f_area does, so it carries the (batch, phase)
    # MultiIndex that `prepared` iterates over. Object dtype, because every cell holds a
    # tuple and the tuples differ in length.
    output = (prepared.sum() * 0.0).astype(object)
    for batch_id, this_batch in prepared:
        for tag in tags:
            output.loc[batch_id, tag] = _change_points(np.asarray(this_batch[tag].to_numpy(), dtype=float), settings)

    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def _change_points(signal: np.ndarray, settings: dict) -> tuple[int, ...]:
    """
    Run PELT on one signal and return its change points, without the trailing sentinel.

    ``ruptures`` always ends its result with the length of the signal, which marks the
    end of the last segment rather than a change, and it raises rather than returning
    nothing when the signal is too short to hold two segments. Both are normalised here
    to an empty tuple, so a caller can compare batches without special cases.

    A ``penalty`` of ``None`` becomes ``log(n)``, which is why it is resolved per signal
    rather than once for the whole frame: batches differ in length.
    """
    min_size = int(settings["min_size"])
    if signal.size < 2 * min_size or not np.all(np.isfinite(signal)):
        # Too short to split, or carrying missing data that the cost function cannot use.
        return ()

    penalty = float(np.log(signal.size)) if settings["penalty"] is None else float(settings["penalty"])
    algorithm = rpt.Pelt(model=settings["model"], min_size=min_size, jump=int(settings["jump"])).fit(signal)
    breakpoints = algorithm.predict(pen=penalty)
    return tuple(int(position) for position in breakpoints if position < signal.size)


# Extreme features
# ------------------------------------------
def f_min(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    min.

    The minimum value attained by each tag, for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    To get the time-point when the minimum occured: `f_agemin`.

    See also: f_agemin, f_max
    """
    base_name = "min"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.min()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_max(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    max.

    The maximum value attained by each tag, for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    To get the time-point when the maximum occured: `f_agemax`.

    See also: f_min
    """
    base_name = "max"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.max()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_agemin(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    agemin.

    The age - the index label, i.e. the time stamp or sample number - at which
    each tag attained its minimum value, for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    See also: f_min, f_agemax
    """
    base_name = "agemin"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.idxmin()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_agemax(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    agemax.

    The age - the index label, i.e. the time stamp or sample number - at which
    each tag attained its maximum value, for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    See also: f_max, f_agemin
    """
    base_name = "agemax"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.idxmax()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_last(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    endpoint.

    The final value attained by each tag, for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    If you want to know *how many* rows [i.e. the last row], then consider
    using the `f_count` feature.

    See also: f_sum, f_count
    """
    base_name = "endpoint"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.last()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def f_count(
    data: pd.DataFrame,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    count.

    The number of non-missing observations for each tag, for the given tags
    in ``tags``, for each unique batch in the ``batch_col`` indicator column,
    and within each unique phase, per batch, of the ``phase_col`` column.

    For data without internal gaps this count equals the 1-based index of the
    final row, so it can also be used as that index for other calculations.

    See also: f_sum, f_last
    """
    base_name = "count"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.count()
    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


# Shape-based features
# ------------------------------------------
def f_slope(  # noqa: PLR0913
    data: pd.DataFrame,
    x_axis_tag: str,
    tags: list[str] | None = None,
    batch_col: str | None = None,
    phase_col: str | None = None,
    age_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    slope.

    The slope of the given `tags` for each unique batch in the `batch_col` indicator column,
    of the `phase_col` column.

    The slope is calculated against whichever variable is given by `x_axis_tag`. If this is the
    `age_col` of the batch (i.e. time duration), ensure that `age_col` is also specified.

    """
    base_name = "slope"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col, age_col)
    f_names = [(tag + "_" + base_name) for tag in tags]

    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum()
    for batch_id, batch_data in prepared:
        if x_axis_tag not in batch_data:
            batch_data = batch_data.reset_index()  # noqa: PLW2901
        for tag in tags:
            x_vals = np.asarray(batch_data[x_axis_tag].to_numpy())
            output.loc[batch_id, tag] = repeated_median_slope(x_vals, np.asarray(batch_data[tag].to_numpy()))

    return output.rename(columns=dict(zip(tags, f_names, strict=False)))


def cross(
    series: pd.Series,
    threshold: int | None = 0,
    direction: str | None = "cross",
    only_index: bool | None = False,
    first_point_only: bool | None = False,
) -> np.ndarray | list:
    """
    Given a Series returns all the index values where the data values equal
    the 'threshold' value. Will first drop all missing values from the series.

    `direction`` can be 'rising' (for rising edge), 'falling' (for only falling
    edge), or 'cross' for both edges.

    If `only_index` is True (default False), then it will return the 0-based
    index where crossing occur *just after*. E.g. if the returned index is 135,
    then the crossing takes place at, or after, index 135, but before index 136.

    If the setting `first_point_only` is set to True, only the first point where
    the crossing occurs is reported. The rest are ignored. Default = all
    crossings are report (i.e. `first_point_only=False`).

    https://stackoverflow.com/questions/10475488/calculating-crossing-intercept-
    points-of-a-series-or-dataframe
    """
    # Find if values are above or bellow y-value crossing:
    series_no_na = series.dropna()
    above = np.asarray(series_no_na.to_numpy()) > threshold
    below = np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings: np.ndarray | list = []
    # Find indexes on left side of crossing point
    if direction == "rising":
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == "falling":
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    if len(idxs) and first_point_only:
        idxs = idxs[0]

    # Calculate x crossings with interpolation using formula for a straight line
    index_values = np.asarray(series_no_na.index.to_numpy())
    data_values = np.asarray(series_no_na.to_numpy())
    x1 = index_values[idxs]
    x2 = index_values[idxs + 1]
    y1 = data_values[idxs]
    y2 = data_values[idxs + 1]

    if only_index:
        return idxs

    try:
        x_crossings = (threshold - y1) * (x2 - x1) / (y2 - y1) + x1
    except TypeError:
        #  If it is a type that cannot be subtracted or multiplied:
        x_crossings = idxs

    return x_crossings


def f_crossing(  # noqa: PLR0913
    data: pd.DataFrame,
    tag: str,
    time_tag: str,
    threshold: int = 0,
    direction: str = "cross",
    only_index: bool = False,
    batch_col: str | None = None,
    phase_col: str | None = None,
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    cross.

    The time (`time_tag`) value at which `tag` crosses a certain numeric
    `threshold``, either `direction='rising'`` (for rising edge), or
    `direction='falling'`' (for falling edge), or 'cross' for both edges.

    The time when the crossing occurs is found by linear interpolation
    between the indices. If you prefer the index itself, use `only_index=True`,
    but the default for that setting is `False`.

    Does this for each unique batch in the `batch_col` indicator column. The
    `phase_col` argument is accepted for signature-compatibility with the
    other ``f_*`` features but is not consumed here: the crossing is looked
    up over the whole batch, not per phase.

    `suffix`: what to add to the data tag, to name to this feature.

    Note: NaN is returned for a given batch, if the crossing is not found.

    """
    base_name = f"cross-{int(threshold)}" if suffix is None else str(suffix)

    if not isinstance(tag, str):
        raise TypeError(f"tag must be a string; got {type(tag).__name__}.")
    if tag not in data:
        raise KeyError(f"Desired tag ['{tag}'] not found in the dataframe.")

    prepared, _tags, output, _ = _prepare_data(
        data,
        [tag],
        batch_col,
        phase_col=None,
        age_col=time_tag,
    )

    f_name = tag + "_" + base_name

    def _cross_first(x: pd.DataFrame) -> np.ndarray | list:
        return cross(x[tag], threshold, direction, only_index=only_index, first_point_only=True)

    # pandas-stubs' DFCallable1 protocol does not include ``np.ndarray`` among the allowed
    # return types, although groupby.apply accepts array-returning callables at runtime.
    output = prepared.apply(_cross_first)  # type: ignore[call-overload]

    return pd.DataFrame(data={f_name: output})


def f_elbow(  # noqa: PLR0913
    data: pd.DataFrame,
    x_axis_tag: str,
    tags: list[str] | None = None,
    only_index: bool | None = False,
    batch_col: str | None = None,
    phase_col: str | None = None,
) -> pd.DataFrame:
    """
    Feature:    elbow.

    The "elbow" of the given ``tags`` for each unique batch in the ``batch_col`` indicator column,
    of the ``phase_col`` column.

    The elbow is calculated against whichever variable is given by `x_axis_tag` (usually a time-
    based tag).

    The function returns the *value* on the x-axis where the elbox occurs. Sometimes you might
    want the *index* of the value, so you can also find the corresponding y-axis value. Use
    `only_index=True` for such cases.
    """
    base_name = "elbow"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col, age_col=x_axis_tag)
    f_names = [(tag + "_" + base_name) for tag in tags]

    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum()
    for batch_id, batch_data in prepared:
        if x_axis_tag not in batch_data:
            batch_data = batch_data.reset_index()  # noqa: PLW2901
        for tag in tags:
            subset = batch_data[[x_axis_tag, tag]]
            subset = subset.dropna()
            x_vals = np.asarray(subset[x_axis_tag].to_numpy())

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                elbow_index = find_elbow_point(x_vals, np.asarray(subset[tag].to_numpy()))
                if elbow_index < 0:
                    elbow_index = np.nan

                if only_index:
                    output.loc[batch_id, tag] = elbow_index
                elif np.isnan(elbow_index):
                    output.loc[batch_id, tag] = np.nan
                else:
                    output.loc[batch_id, tag] = x_vals[int(elbow_index)]

    return output.rename(columns=dict(zip(tags, f_names, strict=False)))
