import logging
from typing import Optional
import pandas as pd
import numpy as np

from ..regression.methods import repeated_median_slope

_LOG = logging.getLogger(__name__)
_LOG.setLevel("INFO")


# General
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
    _LOG.debug(f"Calculated f_mean for {len(tags)} tags.")
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

    _LOG.debug(f"Calculated f_median for {len(tags)} tags.")
    return output.rename(columns=dict(zip(tags, f_names)))[f_names]


# Cumulative features
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
    _LOG.debug(f"Calculated f_sum for {len(tags)} tags.")
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

    _LOG.debug(f"Calculated f_area for {len(tags)} tags.")
    # output.add_suffix('_' + base_name)
    return output.rename(columns=dict(zip(tags, f_names)))


# Shape-based features
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

    _LOG.debug(
        f"Can take time to get slopes for {len(tags)} tags on {len(prepared)} batches."
    )
    # We will overwrite all entries in this dataframe, one-by-one
    output = prepared.sum()
    for batch_id, this_batch in prepared:
        if x_axis_tag not in this_batch:
            this_batch.reset_index(inplace=True)
        for tag in tags:
            X = this_batch[x_axis_tag]
            output.loc[batch_id][tag] = repeated_median_slope(X, this_batch[tag])

    _LOG.info(f"Calculated slopes for {len(tags)} tags.")
    return output.rename(columns=dict(zip(tags, f_names)))
