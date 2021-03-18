from typing import Dict, List
import numpy as np
import pandas as pd

# dtwalign: https://github.com/statefb/dtwalign
from dtwalign import dtw


def determine_scaling(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: List = None,
    robust: bool = True,
) -> pd.DataFrame:
    """
    Scales the batch data according to the variable ranges.

    Parameters
    ----------
    dict_df : Dict[str, pd.DataFrame]
        Batch data, in the standard format.

    Returns
    -------
    range_scalers: DataFrame

        J rows
        2 columns:
            column 1 = Range of each tag. Approximated as the typical delta from q98 - q02
            column 2 = Typical minimum of each tag. Robustly calculated.

    TODO: put this in a scikit-learn style: .fit() and .apply() style
    """
    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns

    collector_rnge = []
    collector_mins = []
    for _, batch in batches.items():
        if robust:
            rnge = batch[columns_to_align].quantile(0.98) - batch[
                columns_to_align
            ].quantile(0.02)
        else:
            rnge = batch[columns_to_align].max() - batch[columns_to_align].min()

        rnge[rnge.values == 0] = 1.0
        collector_rnge.append(rnge)
        collector_mins.append(batch.min())

    if robust:
        scalings = pd.concat(
            [
                pd.DataFrame(collector_rnge).median(),
                pd.DataFrame(collector_mins).median(),
            ],
            axis=1,
        )
    else:
        scalings = pd.concat(
            [pd.DataFrame(collector_rnge).mean(), pd.DataFrame(collector_mins).mean()],
            axis=1,
        )
    scalings.columns = ["Range", "Minimum"]
    scalings["Minimum"] = 0.0
    return scalings


def apply_scaling(
    batches: Dict[str, pd.DataFrame],
    scale_df: pd.DataFrame,
    columns_to_align: List = None,
) -> dict:
    """Scales the batches according to the information in the scaling dataframe.

    Parameters
    ----------
    batches : Dict[str, pd.DataFrame]
        The batches, in standard format.
    scale_df : pd.DataFrame
        The scaling dataframe, from `determine_scaling`

    Returns
    -------
    dict
        The scaled batch data.
    """
    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns
    out = {}
    for batch_id, batch in batches.items():
        out[batch_id] = batch[columns_to_align].copy()
        for tag, column in out[batch_id].iteritems():
            out[batch_id][tag] = (column - scale_df.loc[tag, "Minimum"]) / scale_df.loc[
                tag, "Range"
            ]
    return out


def reverse_scaling(
    batches: Dict[str, pd.DataFrame],
    scale_df: pd.DataFrame,
    columns_to_align: List = None,
):
    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns
    out = {}
    for batch_id, batch in batches.items():
        out[batch_id] = batch[columns_to_align].copy()
        for tag, column in out[batch_id].iteritems():
            out[batch_id][tag] = (
                column * scale_df.loc[tag, "Range"] + scale_df.loc[tag, "Minimum"]
            )
    return out


def batch_dtw(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: list,
    reference_batch: str,
    maximum_iterations: int = 25,
) -> dict:
    """
    Synchronize, via iterative DTW, with weighting.

    Algorithm: Kassidas et al. (2004):  https://doi.org/10.1002/aic.690440412

    Parameters
    ----------
    batches : Dict[str, pd.DataFrame]
        Batch data, in the standard format.
    columns_to_align : list
        Which columns to use during the alignment process. The others are aligned, but
        get no weight, and therefore do not influence the objective function.
    reference_batch : str
        Which key in the `batches` is the reference batch.
    maximum_iterations : int
        The maximum number of iterations allowed.

    Returns
    -------
    dict
        Various outputs relevant to the alignment.
        TODO: Document completely later.

    Notation
    --------
    I = number of batches: index = i
    i = index for the batches
    J = number of tags (columns in each batch)
    j = index for the tags
    k = index into the rows of each batch, the samples: 0 ... k ... K_i
    """
    assert maximum_iterations >= 3, "At least 3 iterations are required"

    scale_df = determine_scaling(batches=batches, columns_to_align=columns_to_align)
    batches_scaled = apply_scaling(batches, scale_df, columns_to_align)
    refbatch_sc = batches_scaled[reference_batch]
    prewarp = {}
    distances = []
    for batch_id, batch in batches_scaled.items():

        # x: will be DTW'ed to get it to align better with the reference.
        res = dtw(
            batch.values,
            y=refbatch_sc.values,
            window_type="none",
            window_size=None,
            step_pattern="symmetric2",
            dist_only=False,
            open_begin=False,
            open_end=False,
        )  # see Kassidas, page 180
        x_warping_path = res.get_warping_path(target="query")

        # Store the 'prewarped' data
        prewarp[batch_id] = batches_scaled[batch_id].iloc[x_warping_path]
        # batch.insert(2, "WarpPath", res.get_warping_path(target="reference"))
        # b#atch.insert(3, "sequence", list(range(batch.shape[0])))
        # prewarp[batch_id].insert(2, "sequence", list(range(prewarp[batch_id].shape[0])), True)
        distances.append({"batch_id": batch_id, "Distance": res.distance})

        # TODO : Come back to page 181 of thesis: where more than 1 point in the target trajectory
        #        is aligned with the reference: compute the average

    #
    dist_df = pd.DataFrame(distances).set_index("batch_id")
    # dist_df.hist("Distance", bins=50)

    # Now find the average trajectory, but ignore problematic batches: top 5% of the distances
    problematic_threshold = dist_df["Distance"].quantile(0.95)

    # Find the average trajectory
    avg_trajectory = {}
    for column in columns_to_align:
        # Changes at each iter: just a placeholder variable
        rawdata = pd.DataFrame()
        for batch_id, batch in prewarp.items():
            # Greater than (not greater than or equal): because if you chose the alignment tag
            # poorly, it might be that many batches have zero distance.
            if dist_df.loc[batch_id]["Distance"] > problematic_threshold:
                # TODO: log this:: print(f'Skipping problematic batch: {batch_id}')
                continue
            rawdata[batch_id] = batch[column].values

        avg_trajectory[column] = rawdata.median(axis=1)

    del rawdata

    # Step 3: deviation of "synced trajectories" from "average trajectory":
    # Will use the median centered trajectories

    # Rows = batch_id; Columns= tags to be aligned
    weights = pd.DataFrame(index=prewarp.keys(), columns=columns_to_align)
    for column in columns_to_align:
        avg_traj = avg_trajectory[column]
        for batch_id, batch in prewarp.items():
            if dist_df.loc[batch_id]["Distance"] >= problematic_threshold:
                continue

            # TODO: try quadratic weights, but for now I will use the sum of the absolute values.
            weights.loc[batch_id, column] = np.abs(
                batch[column].values - avg_traj.values
            ).sum()

    # invert_W = 1 / weights.sum(axis=0)
    # diagonal_W = len(columns_to_align)/invert_W.sum() * invert_W

    # Assume we have done some iterations. Now check if we repeat alignment based on the average
    # trajectory of the aligned data.
    # I am not going to do this yet. I suspect the gain is fairly minimal.

    # Melt the aligned data into a long matrix
    columns_to_export = columns_to_align.copy()
    # columns_to_export.insert(0, "sequence")
    # columns_to_export.insert(0, "batch_id")
    aligned_df = pd.DataFrame()
    for batch_id, batch in prewarp.items():
        copy_batch = batch.copy()
        copy_batch["batch_id"] = batch_id
        aligned_df = aligned_df.append(copy_batch[columns_to_export])

    try:
        columns_to_export.remove("batch_id")
    except ValueError:
        pass

    aligned_wide_df = aligned_df.pivot(
        index="batch_id", columns="sequence", values=columns_to_export
    )
    new_labels = [
        "-".join(item)
        for item in zip(
            aligned_wide_df.columns.get_level_values(0),
            [f"{val*60:.0f}" for val in aligned_wide_df.columns.get_level_values(1)],
        )
    ]
    aligned_wide_df.columns = new_labels

    return dict(
        scalings=scale_df,
        all_batches_sc=batches_scaled,
        prewarp=prewarp,
        problematic_threshold=problematic_threshold,
        avg_trajectory=avg_trajectory,
        weights=weights,
        aligned_df=aligned_df,
        aligned_wide_df=aligned_wide_df,
    )
