from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import norm, iqr

from ..regression.methods import repeated_median_slope
from ..bivariate.methods import find_elbow_point

# General
# ------------------------------------------


def _prepare_data(
    df: pd.DataFrame, tags=None, batch_col=None, phase_col=None, age_col=None
):
    """
    General function, used for all feature extractions.

    1. Groups the ``df`` by batch firstly, and by phase, secondly.
    2. Creates the output dataframe to write the results to.
    """

    # Special case: a single series. Convert it to a dataframe
    if (
        isinstance(df, pd.Series)
        and (tags is None)
        and isinstance(df.index, pd.DatetimeIndex)
    ):
        if df.name is None:
            name = "tag"
        else:
            name = df.name

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
    assert all(column_name in list(df) for column_name in tags)

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

        df_out = df[tags].copy()
        df_out.set_index(age_col, inplace=True)
        df_out.sort_index(inplace=True)
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
def f_mean(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    mean

    The arithmetic mean for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.
    """
    base_name = "mean"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.mean()
    return output.rename(columns=dict(zip(tags, f_names)))[f_names]


def f_median(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    median

    The median for the given tags in ``tags``, for each unique batch in the ``batch_col``
    indicator column, and within each unique phase, per batch, of the ``phase_col`` column.
    """
    base_name = "median"
    prepared, tags, output, df_out = _prepare_data(data, tags, batch_col, phase_col)

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
    return output.rename(columns=dict(zip(tags, f_names)))[f_names]


# Scale-based features
# ------------------------------------------
def f_std(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    std

    The standard deviation for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    See also: f_mad, f_iqr
    """
    base_name = "std"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.std()
    return output.rename(columns=dict(zip(tags, f_names)))


def f_iqr(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    iqr

    The InterQuartile Range (IQR) for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    The IQR is a robust variant of the standard deviation.
    The difference between the 75th percentile and the 25th percentile of a
    sample this is the 25 % trimmed range, an example of an L - estimator.

    See also: f_std, f_mad
    """
    base_name = "iqr"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.agg(iqr)
    return output.rename(columns=dict(zip(tags, f_names)))


def f_mad(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    mad

    The MEAN (not MEDIAN) Absolute Deviation for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    The mean absolute deviation (MAD) is a measure of the variability of a
    univariate sample of quantitative data. For values in a sequence
    X1, X2, ..., Xn, the ``mad`` is the mean of the absolute deviations from
    the data's mean.

    Since the mean can be biased by outliers, the MAD can also be biased. If
    an unbiased estimate is required, see `f_robust_mad`.

    See also: f_std, f_iqr, f_robust_mad
    """
    base_name = "mad"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.mad()
    return output.rename(columns=dict(zip(tags, f_names)))


def f_robust_mad(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    mad

    The MEDIAN (not MEAN) Absolute Deviation for the given tags in ``tags``,
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.

    In statistics, the median absolute deviation (MAD) is a robust measure of
    the variability of a univariate sample of quantitative data.

    For a univariate data set X1, X2, ..., Xn, the MAD is defined as the
    median of the absolute deviations from the data's median.

    from scipy.stats import norm as Gaussian
    c_MAD_constant = Gaussian.ppf(3/4.0)
    median = np.nanmedian(x)
    mad = np.nanmedian((np.fabs(x - median)) / c_MAD_constant)

    The constant correction factor is so that MAD agrees with standard
    deviation for normally distributed data.

    See also: f_mad, f_std, f_iqr,
    """
    c_MAD_const = norm.ppf(3 / 4.0)

    base_name = "mad_robust"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]

    assert False, "This next line of code fails. Fix it."
    output = (np.fabs(prepared - prepared.median())).median() / c_MAD_const

    return output.rename(columns=dict(zip(tags, f_names)))


# Cumulative features
# ------------------------------------------
def f_sum(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    sum

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
    return output.rename(columns=dict(zip(tags, f_names)))


def f_area(data: pd.DataFrame, time_tag, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    area

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
    prepared, tags, output, _ = _prepare_data(
        data, tags, batch_col, phase_col, age_col=time_tag
    )
    f_names = [(tag + "_" + base_name) for tag in tags]

    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum()
    for batch_id, this_batch in prepared:
        # Average width of the base * height. For time series, the average width of the base is
        # the same as the sampling intervals. Therefore the diff of the index.
        half_base_factor = np.diff(this_batch.index)
        for tag in tags:
            # Now the sum makes sense: gets the area under the curve by adding
            # up the smaller trapezoids: consider the trapezoids as rotated by 90 degrees.
            # The parallel edges are lying vertically. Area of each one is the average of the
            # heights on the left and the right, multiplied by delta distance on the horizontal
            # index axis. Area = average(parallel lenghts) * height
            # where height = delta distance on the horizontal axis.
            area = (
                (this_batch[tag].iloc[0:-1].values + this_batch[tag].iloc[1:])
                / 2
                * half_base_factor
            ).sum()
            output.loc[batch_id][tag] = area

            # TODO: check this out still
            # Trapezoidal rule for integrated area, still add the "delta X" constant:
            # # https://en.wikipedia.org/wiki/Trapezoidal_rule
            # area = (
            #     this_batch[tag].values[0]
            #     + 2 * sum(this_batch[tag].values[1:-1])
            #     + this_batch[tag].values[-1]
            # ) / 2

    # output.add_suffix('_' + base_name)
    return output.rename(columns=dict(zip(tags, f_names)))


# Breakpoint detection:  rupture / breakpoint within a particular tag.
# ------------------------------------------
def f_rupture(data: pd.DataFrame, columns=None, batch_col=None, phase_col=None):
    """
    Feature:    rupture

    The breakpoint in a given tag in ``columns`` (usually it is 1 tag),
    for each unique batch in the ``batch_col`` indicator column, and
    within each unique phase, per batch, of the ``phase_col`` column.
    """
    # Handle phase detection based on 1 column for now.
    assert len(columns) == 1

    # TODO: see https://github.com/deepcharles/ruptures

    # base_name = "rupture"
    # prepared = _prepare_data(data, columns, batch_col, phase_col)
    # feature_columns = prepared['columns']
    # grouper = prepared['data']

    # import ruptures as rpt
    # import matplotlib.pyplot as plt
    # output = pd.DataFrame()
    # for batch_id, subset in grouper:
    # signal = subset[columns[0]].values
    # algo = rpt.Pelt(model="rbf").fit(signal)
    # result = algo.predict(pen=100)
    # print(result)
    # plt.show()
    # fig = rpt.display(signal, result, computed_chg_pts=result)
    # fig.save(batchid)


# Extreme features
# ------------------------------------------
def f_min(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    min

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
    return output.rename(columns=dict(zip(tags, f_names)))


def f_max(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    max

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
    return output.rename(columns=dict(zip(tags, f_names)))


def f_agemin(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    # TODO
    pass


def f_agemax(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    # TODO
    pass


def f_last(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    endpoint

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
    return output.rename(columns=dict(zip(tags, f_names)))


def f_count(data: pd.DataFrame, tags=None, batch_col=None, phase_col=None):
    """
    Feature:    count

    The index number of the final value for each tag, for the given tags
    in ``tags``, for each unique batch in the ``batch_col`` indicator column,
    and within each unique phase, per batch, of the ``phase_col`` column.

    Can be useful to get the 1-based index (it is a count!), and to then
    use that index for other calculation purposes.

    See also: f_sum, f_last
    """
    base_name = "count"
    prepared, tags, output, _ = _prepare_data(data, tags, batch_col, phase_col)
    f_names = [(tag + "_" + base_name) for tag in tags]
    output = prepared.count()
    return output.rename(columns=dict(zip(tags, f_names)))


# Shape-based features
# ------------------------------------------
def f_slope(
    data: pd.DataFrame,
    x_axis_tag: str,
    tags=None,
    batch_col: Optional[str] = None,
    phase_col: Optional[str] = None,
    age_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Feature:    slope

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
    for batch_id, this_batch in prepared:
        if x_axis_tag not in this_batch:
            this_batch.reset_index(inplace=True)
        for tag in tags:
            X = this_batch[x_axis_tag]
            output.loc[batch_id][tag] = repeated_median_slope(X, this_batch[tag])

    return output.rename(columns=dict(zip(tags, f_names)))


def cross(
    series: pd.Series,
    threshold: Optional[int] = 0,
    direction: Optional[str] = "cross",
    only_index: Optional[bool] = False,
    first_point_only: Optional[bool] = False,
) -> list:
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
    above = series_no_na.values > threshold
    below = np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
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
    x1 = series_no_na.index.values[idxs]
    x2 = series_no_na.index.values[idxs + 1]
    y1 = series_no_na.values[idxs]
    y2 = series_no_na.values[idxs + 1]

    if only_index:
        return idxs

    try:
        x_crossings = (threshold - y1) * (x2 - x1) / (y2 - y1) + x1
    except TypeError:
        #  If it is a type that cannot be subtracted or multiplied:
        x_crossings = idxs

    return x_crossings


def f_crossing(
    data: pd.DataFrame,
    tag: str,
    time_tag: str,
    threshold: int = 0,
    direction: str = "cross",
    only_index: bool = False,
    batch_col: Optional[str] = None,
    phase_col: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Feature:    cross

    The time (`time_tag`) value at which `tag` crosses a certain numeric
    `threshold``, either `direction='rising'`` (for rising edge), or
    `direction='falling'`' (for falling edge), or 'cross' for both edges.

    The time when the crossing occurs is found by linear interpolation
    between the indices. If you prefer the index itself, use `only_index=True`,
    but the default for that setting is `False`.

    Does this for each unique batch in the `batch_col` indicator column, and
    within each unique phase, per batch, of the `phase_col` column.

    `suffix`: what to add to the data tag, to name to this feature.

    Note: NaN is returned for a given batch and phase, if the crossing is not
    found.

    """
    if suffix is None:
        base_name = f"cross-{int(threshold)}"
    else:
        base_name = str(suffix)

    assert isinstance(tag, str)
    assert tag in data, f"Desired tag ['{tag}'] not found in the dataframe."

    prepared, tags, output, _ = _prepare_data(
        data,
        [tag],
        batch_col,
        phase_col=None,
        age_col=time_tag,
    )

    f_name = tag + "_" + base_name

    output = prepared.apply(
        lambda x: cross(
            x[tag], threshold, direction, only_index=only_index, first_point_only=True
        )
    )

    return pd.DataFrame(data={f_name: output})


def f_elbow(
    data: pd.DataFrame,
    x_axis_tag: str,
    tags=None,
    only_index: Optional[bool] = False,
    batch_col: Optional[str] = None,
    phase_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Feature:    elbow

    The "elbow" of the given ``tags`` for each unique batch in the ``batch_col`` indicator column,
    of the ``phase_col`` column.

    The elbow is calculated against whichever variable is given by `x_axis_tag` (usually a time-
    based tag).

    The function returns the *value* on the x-axis where the elbox occurs. Sometimes you might
    want the *index* of the value, so you can also find the corresponding y-axis value. Use
    `only_index=True` for such cases.
    """
    import warnings

    base_name = "elbow"
    prepared, tags, output, _ = _prepare_data(
        data, tags, batch_col, phase_col, age_col=x_axis_tag
    )
    f_names = [(tag + "_" + base_name) for tag in tags]

    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum()
    for batch_id, this_batch in prepared:

        if x_axis_tag not in this_batch:
            this_batch.reset_index(inplace=True)
        for tag in tags:
            subset = this_batch[[x_axis_tag, tag]]
            subset = subset.dropna()
            X = subset[x_axis_tag]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                elbow_index = find_elbow_point(X, subset[tag])
                if elbow_index < 0:
                    elbow_index = np.nan

                if only_index:
                    output.loc[batch_id][tag] = elbow_index
                else:
                    if np.isnan(elbow_index):
                        output.loc[batch_id][tag] = np.isnan
                    else:
                        output.loc[batch_id][tag] = X[elbow_index]

    return output.rename(columns=dict(zip(tags, f_names)))
