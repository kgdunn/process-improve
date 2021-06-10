from typing import Dict, List, Any, Union

import numpy as np
import pandas as pd
import scipy as sp
import plotly.graph_objects as go

from .alignment_helpers import distance_matrix, backtrack_optimal_path
from .data_input import dict_to_wide, check_valid_batch_dict, melted_to_dict
from ..multivariate.methods import MCUVScaler, PCA

epsqrt = np.sqrt(np.finfo(float).eps)


def determine_scaling(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: List = None,
    settings: dict = None,
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
    if settings:
        default_settings.update(settings)

    settings = default_settings
    if columns_to_align is None:
        columns_to_align = batches[list(batches.keys())[0]].columns

    collector_rnge = []
    collector_mins = []
    for _, batch in batches.items():
        if settings["robust"]:
            # TODO: consider f_iqr feature here. Would that work?
            rnge = batch[columns_to_align].quantile(0.98) - batch[columns_to_align].quantile(0.02)
        else:
            rnge = batch[columns_to_align].max() - batch[columns_to_align].min()

        rnge[rnge.values == 0] = 1.0
        collector_rnge.append(rnge)
        collector_mins.append(batch.min(axis=0))

    if settings["robust"]:
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
    # TODO: handle the case of DataFrames still
    if columns_to_align is None:
        if isinstance(batches, dict):
            batch1 = batches[list(batches.keys())[0]]
            columns_to_align = batch1.columns
        elif isinstance(batches, pd.DataFrame):
            columns_to_align = batches.columns
        else:
            assert False, "Undefined input type"
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
    # TODO: handle the case of DataFrames still
    if columns_to_align is None:
        if isinstance(batches, dict):
            columns_to_align = batches[list(batches.keys())[0]].columns
        elif isinstance(batches, pd.DataFrame):
            columns_to_align = batches.columns
        else:
            assert False, "Undefined input type"
    out = {}
    for batch_id, batch in batches.items():
        out[batch_id] = batch[columns_to_align].copy()
        for tag, column in out[batch_id].iteritems():
            out[batch_id][tag] = column * scale_df.loc[tag, "Range"] + scale_df.loc[tag, "Minimum"]
    return out


class DTWresult:
    """Result class."""

    def __init__(
        self,
        synced: np.ndarray,
        penalty_matrix: np.ndarray,
        md_path: np.ndarray,  # multi-dimensional path through distance mesh D = penalty_matrix
        warping_path: np.ndarray,
        distance: float,
        normalized_distance: float,
    ):
        self.synced = synced
        self.penalty_matrix = penalty_matrix
        self.md_path = md_path
        self.warping_path = warping_path
        self.distance = distance
        self.normalized_distance = normalized_distance


def align_with_path(md_path, batch, initial_row):
    row = 0
    nr = md_path[:, 0].max() + 1  # to account for the zero-based indexing
    synced = pd.DataFrame(np.zeros((nr, batch.shape[1])), columns=batch.columns)
    synced.iloc[row, :] = batch.iloc[md_path[0, 1], :]
    temp = initial_row
    for idx in np.arange(1, md_path.shape[0]):
        if md_path[idx, 0] != md_path[idx - 1, 0]:
            row += 1
            synced.iloc[row, :] = temp = batch.iloc[md_path[idx, 1], :]

        else:
            # TODO : Come back to page 181 of thesis: where more than 1 point in the target
            #        trajectory is aligned with the reference: compute the average,
            temp = np.vstack((temp, batch.iloc[md_path[idx, 1], :]))
            synced.iloc[row, :] = np.nanmean(temp, axis=0)

    return pd.DataFrame(
        synced,
    )


def dtw_core(test, ref, weight_matrix: np.ndarray):

    show_plot = False
    nt = test.shape[0]  # 'test' data; will be align to the 'reference' data
    nr = ref.shape[0]
    assert test.shape[1] == ref.shape[1]

    D = distance_matrix(test.values, ref.values, weight_matrix)
    md_path, distance = backtrack_optimal_path(D)
    warping_path = np.zeros(nr)
    for idx in range(nr):
        warping_path[idx] = md_path[np.where(md_path[:, 0] == idx)[0][-1], 1]

    # Now align the `test` batch:
    initial_row = ref.iloc[md_path[0, 0], :].copy()
    synced = align_with_path(md_path=md_path, batch=test, initial_row=initial_row)

    if show_plot:  # for debugging
        X = np.arange(0, nt, 1)
        Y = np.arange(0, nr, 1)
        X, Y = np.meshgrid(X, Y)
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=X.ravel(),
                    y=Y.ravel(),
                    z=D.ravel(),
                    color="lightpink",
                    opacity=0.90,
                )
            ]
        )
        fig.show()

    return DTWresult(
        synced,
        D,
        md_path,
        warping_path,
        distance,
        normalized_distance=distance / (nr + nt),
    )


def one_iteration_dtw(
    batches_scaled: dict,
    refbatch_sc: pd.DataFrame,
    weight_matrix: np.ndarray,
    settings: dict = None,
):
    default_settings = {"show_progress": True, "subsample": 1}
    if settings:
        default_settings.update(settings)
    settings = default_settings

    aligned_batches = {}
    distances = []
    average_batch = refbatch_sc.copy().reset_index(drop=True) * 0.0
    successful_alignments = 0
    for batch_id, batch in batches_scaled.items():
        try:
            # see Kassidas, page 180
            batch_subset = batch.iloc[:: int(settings["subsample"]), :]
            result = dtw_core(batch_subset, refbatch_sc, weight_matrix=weight_matrix)
            average_batch += result.synced
            aligned_batches[batch_id] = result
            successful_alignments += 1
            distances.append(
                {
                    "batch_id": batch_id,
                    "Distance": result.distance,
                    "Normalized distance": result.normalized_distance,
                }
            )
            if settings["show_progress"]:
                message = f"  * {batch_id}: distance = {result.distance}"
                print(message)
        except ValueError:
            assert False, f"Failed on batch {batch_id}"

    average_batch /= successful_alignments

    return aligned_batches, average_batch


def batch_dtw(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: list,
    reference_batch: str,
    settings: dict = None,
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
        Which key in the `batches` is the reference batch to use.
    settings : dict
        Default settings are = {
            "maximum_iterations": 25,  # maximum iterations (stops here, even if not converged)
            "tolerance": 0.1,  # convergence tolerance
            "robust": True,  # use robust scaling
            "show_progress": True,  # show progress
            "subsample": 1,  # use every sample
            "interpolate_time_axis_maximum": 100,  # interpolates everything to be on this scale
            "interpolate_time_axis_delta": 1,  # with this resolution
            "interpolate_method": "cubic" # any method from scipy.interpolate.interp1d allowed
        }

        The default settings will therefore resampled the time axis to have 100 data points,
        starting at 0 and ending at 99. You might want more points (change the delta), or use
        a different x-axis maximum.

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
    default_settings: Dict[str, Any] = dict(
        maximum_iterations=25,  # maximum iterations (stops here, even if not converged)
        tolerance=0.1,  # convergence tolerance
        robust=True,  # use robust scaling
        show_progress=True,  # show progress
        subsample=1,  # use every sample
        interpolate_time_axis_maximum=100,  # interpolates everything to be on this scale
        interpolate_time_axis_delta=1,
        interpolate_method="cubic",  # any method from scipy.interpolate.interp1d allowed
    )
    if settings:
        default_settings.update(settings)
    settings = default_settings
    assert settings["maximum_iterations"] >= 3, "At least 3 iterations are required"
    assert reference_batch in batches, "`reference_batch` was not found in the dict of batches."

    assert check_valid_batch_dict(batches, no_nan=True)

    scale_df = determine_scaling(
        batches=batches, columns_to_align=columns_to_align, settings=settings
    )
    batches_scaled = apply_scaling(batches, scale_df, columns_to_align)
    refbatch_sc = batches_scaled[reference_batch].iloc[:: int(settings["subsample"]), :]
    weight_vector = np.ones(refbatch_sc.shape[1])
    weight_matrix = np.diag(weight_vector)
    weight_history = np.zeros_like(weight_vector) * np.nan
    average_batch = None
    delta_weight = np.linalg.norm(weight_vector)
    iter = 0
    while (np.linalg.norm(delta_weight) > settings["tolerance"]) and (
        iter <= settings["maximum_iterations"]
    ):
        if settings["show_progress"]:
            message = f"Iter = {iter} and norm = {np.linalg.norm(delta_weight)}"
            print(message)

        iter += 1
        weight_matrix = np.diag(weight_vector)
        weight_history = np.vstack((weight_history, weight_vector.copy()))

        if iter > 3:
            refbatch_sc = average_batch

        aligned_batches, average_batch = one_iteration_dtw(
            batches_scaled=batches_scaled,
            refbatch_sc=refbatch_sc,
            weight_matrix=weight_matrix,
            settings=settings,
        )

        # Deviations from the average batch:
        next_weights = np.zeros((1, refbatch_sc.shape[1]))
        for batch_id, result in aligned_batches.items():
            next_weights += np.nansum(np.power(result.synced - average_batch, 2), axis=0)
            # TODO: use quadratic weights for now, but try sum of the absolute values instead
            #  np.abs(result.synced - average_batch).sum(axis=0)

            # TODO: leave out worst batches when computing the weights
            # dist_df = pd.DataFrame(distances).set_index("batch_id")
            # dist_df.hist("Distance", bins=50)
            # Now find the average trajectory, but ignore problematic batches:
            #      for example, top 5% of the distances.
            # TODO: make this a configurable setting
            # problematic_threshold = dist_df["Distance"].quantile(0.95)

        next_weights = 1.0 / np.where(next_weights > epsqrt, next_weights, 10000)
        weight_vector = (next_weights / np.sum(next_weights) * len(columns_to_align)).ravel()
        # If change in delta_weight is small, we terminate early; no need to fine-tune excessively.
        delta_weight = np.diag(weight_matrix) - weight_vector  # old - new

    # OK, the weights are found: now use the last iteration's result to get back to original
    # scaling for the trajectories
    weight_history = weight_history[1:, :]
    aligned_df = pd.DataFrame()
    new_time_axis = np.arange(
        0,
        settings["interpolate_time_axis_maximum"],
        settings["interpolate_time_axis_delta"],
    )

    for batch_id, result in aligned_batches.items():
        if settings["show_progress"]:
            message = f"Iterpolating values for batch = {batch_id}"
            print(message)

        initial_row = batches[batch_id].iloc[result.md_path[0, 0], :].copy()
        synced = align_with_path(
            result.md_path,
            batches[batch_id].iloc[:: int(settings["subsample"]), :],
            initial_row=initial_row,
        )
        # Resample the trajectories of the aligned data now along this sequence.
        sequence = np.linspace(
            0,
            settings["interpolate_time_axis_maximum"] - settings["interpolate_time_axis_delta"],
            synced.shape[0],
        )
        assert new_time_axis.min() == sequence.min()
        assert new_time_axis.max() == sequence.max()

        synced_interpolated = pd.DataFrame()
        for column, _ in synced.iteritems():
            if column in ["batch_id", "_sequence_"]:
                continue

            interp_column = sp.interpolate.interp1d(
                sequence,
                synced[column],
                kind=settings["interpolate_method"],
                assume_sorted=True,
            )
            synced_interpolated[column] = interp_column(new_time_axis)

        # Pop in an extra column at the start of the df
        synced_interpolated.insert(0, "batch_id", batch_id)
        synced_interpolated.set_index(new_time_axis)

        # Overwrite existing dataframe with this, unscaled, and interpolated dataframe.
        result.synced = synced_interpolated
        aligned_df = aligned_df.append(synced_interpolated)

    # Make the batch_id label consistent
    aligned_df["batch_id"] = aligned_df["batch_id"].astype(type(batch_id))

    last_average_batch = reverse_scaling(dict(avg=average_batch), scale_df)["avg"]
    aligned_batch_dfdict = melted_to_dict(aligned_df, batch_id_col="batch_id")

    return dict(
        scale_df=scale_df,
        aligned_batch_objects=aligned_batches,
        aligned_batch_dfdict=aligned_batch_dfdict,
        last_average_batch=last_average_batch,
        weight_history=pd.DataFrame(weight_history, columns=columns_to_align),
    )


def resample_to_reference(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: list,
    reference_batch: str,
    settings: dict = None,
) -> dict:
    """Resamples all `batches` (only the `columns_to_align`) to the duration of batch with
    identifier `reference`.

    Parameters
    ----------
    batches : Dict[str, pd.DataFrame]
        Batch data, in the standard format.
    columns_to_align : list
        Which columns to use. Others are ignored.
    reference_batch : str
        Which key in the `batches` is the reference batch.
    settings : dict, optional
        [description], by default None

    Returns
    -------
    dict
        Batch data, in the standard format.
    """
    default_settings = {
        "interpolate_kind": "cubic",  # must be a valid "scipy.interpolate.interp1d" `kind`
    }
    if settings:
        default_settings.update(settings)
    settings = default_settings
    out = {}
    target_time = np.arange(0, batches[reference_batch].shape[0])
    target_time = target_time / target_time[-1]

    for batch_id, batch in batches.items():
        to_resample = np.arange(0, batch.shape[0])
        to_resample = to_resample / to_resample[-1]
        out_df = {}
        for column, series in batch.iteritems():
            if column in columns_to_align:
                out_df[column] = sp.interpolate.interp1d(
                    to_resample,
                    series.values,
                    copy=False,
                    kind=settings["interpolate_kind"],
                )(target_time)

        out[batch_id] = pd.DataFrame(out_df)

    return out


def find_average_length(batches: Dict[str, pd.DataFrame], settings: dict = None):
    """
    Find the batch in `batches` with the average length.

    Parameters
    ----------
    batches : Dict[str, pd.DataFrame]
        Batch data, in the standard format.
    settings : dict
        Default settings are = {
            "robust": True,  # use robust (median)
        }

    Returns
    -------
    One of the dictionary keys from `batches`.

    """
    default_settings = {
        "robust": True,  # use robust scaling
    }
    if settings:
        default_settings.update(settings)
    settings = default_settings

    batch_lengths = pd.Series({batch_id: df.shape[0] for batch_id, df in batches.items()})
    if settings["robust"]:
        # If multiple batches of the median length, return the last one.
        return batch_lengths.index[
            np.where((batch_lengths == batch_lengths.median()).values)[0][-1]
        ]
    else:
        return batch_lengths.index[(batch_lengths - batch_lengths.mean()).abs().argmin()]


def find_reference_batch(
    batches: Dict[str, pd.DataFrame],
    columns_to_align: list,
    settings: Dict[str, Any] = None,
):
    """
    Find a reference batch. Assumes NO missing data.

    Starts with the average duration batch; resamples (simple interpolation) of all batches to
    that duration. Unfolds that resampled data. Does PCA on the wide, unfolded data. Fits,
    by default, 4 components. Excludes all batches with Hotelling's T2 > 90% limit. Refits PCA
    with 4 components. Finds the batch which has the multivariate combination of scores which are
    the smallest (i.e. closest to the model center) and ensures this batch has SPE < 50% of the
    model limit.

    Parameters
    ----------
    batches : Dict[str, pd.DataFrame]
        Batch data, in the standard format.
    columns_to_align : list
        Which columns to use. Others are ignored.
    settings : dict
        Default settings are = {
            "robust": True,  # use robust scaling
            "subsample": 1,  # use every sample
            "method": "pca_most_average", # the most average batch from a crudely aligned PCA
            "n_components": 4,
        }

    Returns
    -------
    One of the dictionary keys from `batches`.

    """
    default_settings: Dict[str, Union[int, float, str, bool]] = {
        "robust": True,  # use robust scaling
        "subsample": 1,  # use every sample
        "method": "pca_most_average",
        "n_components": 4,
    }
    if isinstance(settings, dict):
        default_settings.update(settings)
    settings = default_settings

    assert check_valid_batch_dict(batches)

    # Starts with the average duration batch.
    initial_reference_id = find_average_length(batches, settings)

    # Resamples (simple interpolation) of all batches to that duration.
    resampled = resample_to_reference(
        batches,
        columns_to_align,
        reference_batch=initial_reference_id,
        settings=settings,
    )

    # Unfolds that resampled data.
    basewide = dict_to_wide(resampled)

    # Does PCA on the wide, unfolded data. A=4
    scaler = MCUVScaler().fit(basewide)
    mcuv = scaler.fit_transform(basewide)
    n_components = int(np.floor(settings["n_components"]))
    pca_first = PCA(n_components=n_components).fit(mcuv)

    # Excludes all batches with Hotelling's T2 > 90% limit.
    T2_limit_90 = pca_first.T2_limit(0.90)
    to_keep = pca_first.Hotellings_T2.iloc[:, -1] < T2_limit_90

    # Refits PCA with A=4 on a subset of the batches, to avoid biasing the PCA model too much.
    basewide = basewide.loc[to_keep, :]
    scaler = MCUVScaler().fit(basewide)
    mcuv = scaler.fit_transform(basewide)
    pca_second = PCA(n_components=n_components).fit(mcuv)

    # Finds batch with scores; and ensures this batch has SPE < 50% of the model limit.
    metrics = pd.DataFrame(
        {
            "HT2": pca_second.Hotellings_T2.iloc[:, -1],
            "SPE": pca_second.squared_prediction_error.iloc[:, -1],
        }
    )
    metrics = metrics.sort_values(by=["HT2", "SPE"])
    metrics = metrics.query(f"SPE < {pca_second.SPE_limit(0.5)}")
    return metrics.index[0]
