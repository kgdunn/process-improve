from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba import jit

# # dtwalign: https://github.com/statefb/dtwalign
# from dtwalign import dtw_from_distance_matrix
epsqrt = np.sqrt(np.finfo(float).eps)


def determine_scaling(
    batches: Dict[str, pd.DataFrame], columns_to_align: List = None, settings: dict = dict,
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
    # This will be clumsy, until we have Python 3.9
    default_settings = {"robust": True}
    default_settings.update(settings)
    settings = default_settings
    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns

    collector_rnge = []
    collector_mins = []
    for _, batch in batches.items():
        if settings["robust"]:
            rnge = batch[columns_to_align].quantile(0.98) - batch[columns_to_align].quantile(0.02)
        else:
            rnge = batch[columns_to_align].max() - batch[columns_to_align].min()

        rnge[rnge.values == 0] = 1.0
        collector_rnge.append(rnge)
        collector_mins.append(batch.min())

    if settings["robust"]:
        scalings = pd.concat(
            [pd.DataFrame(collector_rnge).median(), pd.DataFrame(collector_mins).median(),],
            axis=1,
        )
    else:
        scalings = pd.concat(
            [pd.DataFrame(collector_rnge).mean(), pd.DataFrame(collector_mins).mean()], axis=1,
        )
    scalings.columns = ["Range", "Minimum"]
    scalings["Minimum"] = 0.0
    return scalings


def apply_scaling(
    batches: Dict[str, pd.DataFrame], scale_df: pd.DataFrame, columns_to_align: List = None,
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
    batches: Dict[str, pd.DataFrame], scale_df: pd.DataFrame, columns_to_align: List = None,
):
    # TODO: for now, `batches` must be a dict of batches. Allow it to be a dataframe for 1 batch

    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns
    out = {}
    for batch_id, batch in batches.items():
        out[batch_id] = batch[columns_to_align].copy()
        for tag, column in out[batch_id].iteritems():
            out[batch_id][tag] = column * scale_df.loc[tag, "Range"] + scale_df.loc[tag, "Minimum"]
    return out


def distance_matrix(test, ref, weight):
    # TODO: allow user to specify `band`. The code below assumes that `band` is fixed as shown
    # here, so therefore, if user provide `band`, the code needs to be adjusted.
    nt = test.shape[0]  # 'test' data; will be align to the 'reference' data
    nr = ref.shape[0]
    band = np.ones((nt, 2))
    band[:, 1] *= int(nr)
    dist = np.zeros((nr, nt))

    # Mahalanobis distance:
    for idx, row in test.reset_index(drop=True).iterrows():
        dist[:, idx] = np.diag((row - ref) @ weight @ ((row - ref).values.T))

    # TODO: Sakoe-Chiba constraints could still be added
    D = np.zeros((nr, nt)) * np.NaN
    D[0, 0] = dist[0, 0]
    for idx in np.arange(1, nt):
        D[0, idx] = dist[0, idx] + D[0, idx - 1]

    for idx in np.arange(1, nr):
        D[idx, 0] = dist[idx, 0] + D[idx - 1, 0]

    for n in np.arange(1, nt):
        for m in np.arange(max((1, band[n, 0])), int(band[n, 1])):
            # index here must be integer!
            D[m, n] = dist[m, n] + np.nanmin([D[m, n - 1], D[m - 1, n - 1], D[m - 1, n]])

    return D


def backtrack_optimal_path(D: np.ndarray):
    nr, nt = D.shape
    nr -= 1
    nt -= 1
    path = np.array(((nr, nt),), dtype=np.int64)
    path_sum = 0.0

    while (nt + nr) != 0:
        if nt == 0:
            nr -= 1
            path_sum += D[nr, nt]
        elif nr == 0:
            nt -= 1
            path_sum += D[nr, nt]
        else:
            number = np.argmin([D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt]])
            a, b, c = D[nr - 1, nt - 1], D[nr, nt - 1], D[nr - 1, nt]
            if (a <= b) & (a <= c):
                assert number == 0
                path_sum += D[nr - 1, nt - 1]
                nt -= 1
                nr -= 1
            elif (b <= a) & (b <= c):
                assert number == 1
                path_sum += D[nr, nt - 1]
                nt -= 1
            elif (c <= a) & (c <= b):
                assert number == 2
                path_sum += D[nr - 1, nt]
                nr -= 1
            else:
                assert False

        path = np.vstack(([nr, nt], path))

    return path, path_sum


class DTWresult:
    """Result class."""

    def __init__(
        self,
        synched: np.ndarray,
        penalty_matrix: np.ndarray,
        warping_path: np.ndarray,
        distance: float,
        normalized_distance: float,
    ):
        self.synched = synched
        self.penalty_matrix = penalty_matrix
        self.warping_path = warping_path
        self.distance = distance
        self.normalized_distance = normalized_distance


def dtw(test, ref, weight, penalty_matrix=None):
    # TODO: optimize: store the `D` matrix per batch, relative to a given reference, to
    #       avoid recomputing it everytime.
    show_plot = False
    nt = test.shape[0]  # 'test' data; will be align to the 'reference' data
    nr = ref.shape[0]
    assert test.shape[1] == ref.shape[1]

    if penalty_matrix:
        D = penalty_matrix
    else:
        D = distance_matrix(test, ref, weight)

    path, distance = backtrack_optimal_path(D)
    warping_path = np.zeros(nr)
    for idx in range(nr):
        warping_path[idx] = path[np.where(path[:, 0] == idx)[0][-1], 1]

    # Now align the `test` batch:
    row = 0
    synched = np.zeros((nr, test.shape[1]))
    synched[row, :] = test.iloc[path[0, 1], :]
    temp = ref.iloc[path[0, 0], :]
    for idx in np.arange(1, path.shape[0]):
        if path[idx, 0] != path[idx - 1, 0]:
            row += 1
            synched[row, :] = temp = test.iloc[path[idx, 1], :]

        if path[idx, 0] == path[idx - 1, 1]:
            # TODO : Come back to page 181 of thesis: where more than 1 point in the target
            #        trajectory is aligned with the reference: compute the average,
            temp = np.vstack((temp, test.iloc[path[idx, 1], :]))
            synched[row, :] = np.nanmean(temp)

    if show_plot:
        X = np.arange(0, nt, 1)
        Y = np.arange(0, nr, 1)
        X, Y = np.meshgrid(X, Y)
        fig = go.Figure(
            data=[
                go.Mesh3d(x=X.ravel(), y=Y.ravel(), z=D.ravel(), color="lightpink", opacity=0.90)
            ]
        )
        fig.show()

    return DTWresult(synched, D, warping_path, distance, normalized_distance=distance / (nr + nt))


def batch_dtw(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: list,
    reference_batch: str,
    settings: dict = dict,
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
    default_settings = {"maximum_iterations": 25, "robust": True}
    default_settings.update(settings)
    settings = default_settings
    assert settings["maximum_iterations"] >= 3, "At least 3 iterations are required"

    scale_df = determine_scaling(
        batches=batches, columns_to_align=columns_to_align, settings=settings
    )
    batches_scaled = apply_scaling(batches, scale_df, columns_to_align)
    refbatch_sc = batches_scaled[reference_batch]
    weight = np.diag(np.ones(refbatch_sc.shape[1]))
    aligned_batches = {}
    distances = []
    average_batch = refbatch_sc.copy().reset_index(drop=True) * 0.0
    successful_alignments = 0
    for batch_id, batch in batches_scaled.items():
        print(batch_id)
        try:
            # see Kassidas, page 180
            result = dtw(batch, refbatch_sc, weight=weight)
            average_batch += result.synched
            aligned_batches[batch_id] = result
            successful_alignments += 1
        except ValueError:
            print(f"Failed on batch {batch_id}")

        distances.append(
            {
                "batch_id": batch_id,
                "Distance": result.distance,
                "Normalized distance": result.normalized_distance,
            }
        )

    average_batch /= successful_alignments

    # Deviations from the average batch:
    weight_vector = np.zeros((1, average_batch.shape[1]))
    for batch_id, result in aligned_batches.items():
        weight_vector += np.nansum(np.power(result.synched - average_batch, 2), axis=0)
        # TODO: use quadratic weights for now, but try sum of the absolute values instead
        #  np.abs(result.synched - average_batch).sum(axis=0)

    weight_vector = np.where(weight_vector > epsqrt, weight_vector, 10000)

    weight_update = np.diag(
        (weight_vector / np.sum(weight_vector) * len(columns_to_align)).ravel()
    )
    delta_weight = np.abs(weight_update - weight)

    # TODO : if change in delt_weight is small, terminate early

    dist_df = pd.DataFrame(distances).set_index("batch_id")
    # dist_df.hist("Distance", bins=50)

    # Now find the average trajectory, but ignore problematic batches: top 5% of the distances
    problematic_threshold = dist_df["Distance"].quantile(0.95)

    # Assume we have done some iterations. Now check if we repeat alignment based on the average
    # trajectory of the aligned data.
    # I am not going to do this yet. I suspect the gain is fairly minimal.

    # # Melt the aligned data into a long matrix
    # columns_to_export = columns_to_align.copy()
    # # columns_to_export.insert(0, "sequence")
    # # columns_to_export.insert(0, "batch_id")
    # aligned_df = pd.DataFrame()
    # for batch_id, batch in prewarp.items():
    #     copy_batch = batch.copy()
    #     copy_batch["batch_id"] = batch_id
    #     aligned_df = aligned_df.append(copy_batch[columns_to_export])

    # try:
    #     columns_to_export.remove("batch_id")
    # except ValueError:
    #     pass

    # aligned_wide_df = aligned_df.pivot(
    #     index="batch_id", columns="sequence", values=columns_to_export
    # )
    # new_labels = [
    #     "-".join(item)
    #     for item in zip(
    #         aligned_wide_df.columns.get_level_values(0),
    #         [f"{val*60:.0f}" for val in aligned_wide_df.columns.get_level_values(1)],
    #     )
    # ]
    # aligned_wide_df.columns = new_labels

    return dict(
        scale_df=scale_df,
        all_batches_sc=batches_scaled,
        problematic_threshold=problematic_threshold,
        average_batch=average_batch,
        weight_vector=weight_vector,
        aligned_df=aligned_df,
        aligned_wide_df=aligned_wide_df,
    )
