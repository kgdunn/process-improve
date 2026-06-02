# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
from __future__ import annotations

import time
import typing
import warnings
from collections.abc import Callable, KeysView
from functools import partial

import numpy as np
import pandas as pd

# ENG-13 (#295): plotting deps live in the ``[plotting]`` extra. The
# ``_MissingExtra`` stand-in lets module-import succeed for the algorithm
# surface (``PCA``, ``PLS``, ...) while any actual plot call raises a
# clear "install the extra" ImportError.
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra
    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

try:
    import ridgeplot
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra
    ridgeplot = _MissingExtra("ridgeplot", "plotting")  # type: ignore[assignment]
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context, clone
from sklearn.metrics import r2_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_is_fitted
from tqdm import tqdm

from .._linalg import safe_inverse
from .._random import check_random_state
from ..univariate.metrics import detect_outliers_esd
from ..visualization.themes import REFERENCE_LINE_COLOR

# ENG-01: shared primitives now live in ``_common`` (re-exported here for
# backward compatibility; see the module docstring above).
from ._common import (
    SpecificationWarning,
    _nz,
    epsqrt,
)
from ._diagnostics import (
    eigenvalue_summary,
    observation_contributions,
    project_variables,
    rv2_coefficient,
    rv_coefficient,
    squared_cosine,
    vip,
)
from ._limits import (
    ellipse_coordinates,
    hotellings_t2_limit,
    score_limit,
    spe_calculation,
    spe_limit,
)
from ._nipals import (
    internal_pls_nipals_fit_one_pc,
    nan_to_zeros,
    quick_regress,
    regress_a_space_on_b_row,
    ssq,
    terminate_check,
)
from ._pca import PCA
from ._pls import PLS
from ._preprocessing import MCUVScaler, center, scale
from .plots import (
    coefficient_plot,
    correlation_loadings_plot,
    explained_variance_plot,
    loading_plot,
    predictions_vs_observed_plot,
    score_plot,
    spe_plot,
    t2_plot,
)


class DataFrameDict(dict):
    """Container for the partitionable (Z, F) and static (Y) data blocks used by TPLS."""

    def __init__(self, datadict: dict[str, dict[str, pd.DataFrame]]):
        """
        Initialize a DataFrameDict to handle partitionable and static dataframes.

        datadict: Dictionary with 3 keys, one for each block: Z, F and Y.
                  Each block is itself a dictionary of dataframes: dict[str, dict[str, pd.DataFrame]]
        """

        self.partitionable_blocks: list[str] = ["Z", "F", "Y"]
        self.datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            self.datadict[block] = datadict.get(block, {})
        first_group = next(iter(self.datadict["F"].keys()))
        self.n_samples = self.datadict["F"][first_group].shape[0]
        self.shape = (self.n_samples, len(self.datadict))

        # Some basic checks: each dataframe inside each block has the same number of rows
        for block in set(self.partitionable_blocks) & set(self.datadict.keys()):
            for group, df in self.datadict[block].items():
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Expected a DataFrame for block {block}, group '{group}'; got instead{type(df)}.")
                if df.shape[0] != self.n_samples:
                    raise ValueError(
                        f"DataFrames in block {block} must have the same number of rows ({self.n_samples}). "
                        f"Group {group} has {df.shape[0]} rows."
                    )

    def keys(self) -> KeysView[str]:
        """Return the keys of the DataFrameDict."""
        return self.datadict.keys()

    def __setitem__(self, key: str, value: pd.DataFrame | dict) -> None:
        """Set a DataFrame for a specific key in the DataFrameDict."""
        if key not in self.partitionable_blocks:
            raise KeyError(f"Key {key} is not a valid partitionable block. Valid keys are: {self.partitionable_blocks}")

        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame for key {key}, got {type(value)}.")
        if value.shape[0] != self.n_samples:
            raise ValueError(
                f"DataFrames in block {key} must have the same number of rows ({self.n_samples}). "
                f"Provided DataFrame has {value.shape[0]} rows."
            )
        self.datadict[key] = value

    def __getitem__(self, lookup: int | list[int] | list[np.int64] | str) -> DataFrameDict | dict[str, pd.DataFrame]:
        """Return a new DataFrameDict with partitioned data."""

        if isinstance(lookup, str):
            return self.datadict[lookup]  # returns the `dict[str, pd.DataFrame]` version of the function

        datadict: dict[str, dict[str, pd.DataFrame]] = {}
        for block in self.partitionable_blocks:
            datadict[block] = {}
            for group, df in self.datadict[block].items():
                match lookup:
                    case int() | np.integer():
                        datadict[block][group] = df.iloc[[lookup]]
                    case list():
                        datadict[block][group] = df.iloc[lookup]
                    case np.ndarray():
                        datadict[block][group] = df.iloc[lookup.tolist()]
                    case tuple():
                        if lookup[1] == Ellipsis:
                            datadict[block][group] = df.iloc[[int(item) for item in lookup[0]]]
                        else:
                            raise TypeError(f"Invalid tuple structure for lookup: {lookup}")
                    case _:
                        raise TypeError(
                            f"Lookup must be an int, list of ints, or a string. Got {lookup}; {type(lookup)}"
                        )

        return DataFrameDict(datadict)

    def __len__(self):
        """Return the number of samples in the DataFrameDict."""
        return self.n_samples

    def __repr__(self):
        """Return a string representation of the DataFrameDict."""
        groups_in_block_f = list(self.datadict["F"].keys())
        groups_in_block_z = list(self.datadict["Z"].keys())
        groups_in_block_y = list(self.datadict["Y"].keys())
        output = f"DataFrameDict with {len(self)} samples and {len(self.datadict)} blocks: {list(self.datadict.keys())}"
        output += f"\n  F groups: {groups_in_block_f}"
        output += f"\n  Z groups: {groups_in_block_z}"
        output += f"\n  Y groups: {groups_in_block_y}"
        return output


class TPLS(RegressorMixin, BaseEstimator):
    """
    TPLS algorithm for T-shaped data structures (we also include standard pre-processing of the data inside this class).

    Source: Garcia-Munoz, https://doi.org/10.1016/j.chemolab.2014.02.006, Chem.Intell.Lab.Sys. v133, p 49 to 62, 2014.

    We change the notation from the original paper to avoid confusion with a generic "X" matrix, and match symbols
    that are more natural for our use.

    Notation mapping (paper → this code):

    - X^T → D: ``d_matrix`` (external), ``d_mats`` (internal) - Database of properties
    - X → D^T: transposed D (not used directly)
    - R → F: ``f_mats`` - Formula matrices
    - Z → Z: ``z_mats`` - Process conditions
    - Y → Y: ``y_mats`` - Quality indicators

    Notes
    1. Matrices in F, Z and Y must all have the same number of rows.
    2. Columns in F must be the same as the **rows** in D.
    3. Conditions in Z may be missing (turning it into an L-shaped data structure).

    Parameters
    ----------
    n_components : int
        A parameter used to specify the number of components.

    d_matrix : dict[str, dict[str, pd.DataFrame]]
        A dictionary containing the properties of each group of materials.
        Keys are group names; values are DataFrames with properties as columns and materials as rows.
        This "D" matrix is provided once at construction and reused for fitting, prediction and
        cross-validation.

    max_iter : int, optional
        The maximum number of iterations for the TPLS algorithm. Default is 500.

    skip_f_matrix_preprocessing : bool, optional
        If True, the F (formula) matrices are used as-is, skipping the internal
        centering and scaling of the F block. Default is False.

    Notes
    -----
    The input ``X`` passed to :meth:`fit` and :meth:`predict` is a dictionary with 3 keys:

    - ``F``: Formula matrices (rows = blends, columns = materials).
      ``F = {"Group A": df_formulas_a, "Group B": df_formulas_b, ...}``
    - ``Z``: Process conditions - one row per blend, one column per condition.
    - ``Y``: Product quality indicators - one row per blend, one column per indicator.

    The ``D`` matrix (database of material properties) is supplied once at
    construction via the ``d_matrix`` argument; it is not part of ``X``.

    Attributes
    ----------
    n_samples : int
        The number of samples (rows) in the training data

    n_substances : int
        The number of substances (columns) in the training data, i.e. the number of materials in the F matrix.


    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng()
    >>>
    >>> n_props_a, n_props_b = 6, 4            # Two groups of properties: A and B.
    >>> n_materials_a, n_materials_b = 12, 8   # Number of materials in each group.
    >>> n_formulas = 40                        # Number of formulas in matrix F.
    >>> n_outputs = 3
    >>> n_conditions = 2
    >>>
    >>> properties = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_materials_a, n_props_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_materials_b, n_props_b))),
    >>> }
    >>> formulas = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_b))),
    >>> }
    >>> process_conditions = {"Conditions": pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))}
    >>> quality_indicators = {"Quality":    pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))}
    >>> all_data = {"Z": process_conditions, "F": formulas, "Y": quality_indicators}
    >>> estimator = TPLS(n_components=4, d_matrix=properties)
    >>> estimator.fit(all_data)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
        "max_iter": [int],
        "d_matrix": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        d_matrix: dict,
        max_iter: int = 500,
        skip_f_matrix_preprocessing: bool = False,
    ):
        super().__init__()
        if n_components <= 0:
            raise ValueError(f"n_components must be positive; got {n_components}.")
        self.n_components = n_components

        self.d_matrix = d_matrix  # This is required input dict containing the properties for each group.
        if not isinstance(self.d_matrix, dict):
            raise TypeError(
                f"d_matrix must be a dict of DataFrames; got {type(self.d_matrix).__name__}."
            )
        if not all(isinstance(df, pd.DataFrame) for df in self.d_matrix.values()):
            raise TypeError("d_matrix must contain pandas DataFrames as values.")

        self.max_iter = max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {self.max_iter}.")

        self.skip_f_matrix_preprocessing = skip_f_matrix_preprocessing

        self.is_fitted_ = False
        self.n_substances = 0
        self.n_samples = 0
        self.tolerance_ = np.sqrt(np.finfo(float).eps)
        self.required_blocks_ = {"D", "F", "Y", "Z"}  # "Z" block is optional; an empty one is added if not provided
        # "required_inputs" used in the sense of inputs to this class; not in the sense of a "model input"
        self.required_inputs_ = {"F", "Y", "Z"}
        self.plot = Plot(self)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataFrameDict, y: None = None) -> TPLS:  # noqa: ARG002, PLR0915
        """Fit the preprocessing parameters and also the latent variable model from the training data.

        Parameters
        ----------
        X : {dictionary of dataframes}, keys that must be present: "F", "Z", and "Y"
            The training input samples. See documentation in the class definition for more information on each matrix.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(X, DataFrameDict):
            raise TypeError(f"X must be a DataFrameDict; got {type(X).__name__}.")
        self._input_data_checks(X)
        group_keys = [str(key) for key in self.d_matrix]

        # Storage for pre-processing and the raw matrices
        self.fitting_statistics: dict[str, list] = {"iterations": [], "convergance_tolerance": [], "milliseconds": []}
        self.preproc_: dict[str, dict[str, dict[str, pd.Series]]] = {key: {} for key in self.required_blocks_}
        self.sums_of_squares_: list[dict[str, dict[str, np.ndarray]]] = [{key: {} for key in self.required_blocks_}]
        # These are *fractional* R2 values, i.e. always less than or equal to 1.0.
        # As a list: entry 0 is zeros; entry 1 is after fitting the first component, and so on.
        # The keys are the blocks, and the values are dictionaries with group keys as keys.
        # The values are the R2 values for each column in the block.
        self.r2_frac: list[dict[str, dict[str, np.ndarray]]] = [{key: {} for key in self.required_blocks_}]
        self.feature_importance: dict[str, dict[str, pd.Series]] = {key: {} for key in self.required_blocks_}

        self.d_mats: dict[str, np.ndarray] = {key: self.d_matrix[key].values.copy() for key in group_keys}
        self.f_mats: dict[str, np.ndarray] = {key: X["F"][key].values.copy() for key in group_keys}
        self.z_mats: dict[str, np.ndarray] = {key: X["Z"][key].values.copy() for key in X["Z"]}
        self.y_mats: dict[str, np.ndarray] = {key: X["Y"][key].values.copy() for key in X["Y"]}

        # Empty model coefficients
        self.n_substances = sum(self.f_mats[key].shape[1] for key in group_keys)
        self.n_conditions = sum(self.z_mats[key].shape[1] for key in self.z_mats)
        self.n_outputs = sum(self.y_mats[key].shape[1] for key in self.y_mats)
        self.n_samples = self.f_mats[group_keys[0]].shape[0]

        # Learn the centering and scaling parameters
        for key in X["Y"]:
            self.preproc_["Y"][key] = {}
            self.preproc_["Y"][key]["center"], self.preproc_["Y"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["Y"][key])
            )
        for key in X["Z"]:
            self.preproc_["Z"][key] = {}
            self.preproc_["Z"][key]["center"], self.preproc_["Z"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["Z"][key])
            )
        for key, df_d in self.d_matrix.items():
            self.preproc_["D"][key] = {}
            self.preproc_["D"][key]["center"], self.preproc_["D"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(df_d)
            )
            self.preproc_["D"][key]["block"] = pd.Series([np.sqrt(df_d.shape[1])])  # <-- sqrt(number of properties!)
            #
            # Also do the same for the formula matrix
            self.preproc_["F"][key] = {}
            self.preproc_["F"][key]["center"], self.preproc_["F"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["F"][key])
            )

        # Then implement the preprocessing on the raw data
        self._preprocess_data()

        # Sum of square values for each column in each block (dicts) per component (elements in the list)
        # The first entry is after centering and scale (baseline variance) but before fitting any components.
        # The second entry is after fitting one component, and so on.
        # You can sum the sums-of-squares values for all columns to get the total variance for each block.
        self.sums_of_squares_ = [
            {
                "D": {key: np.nansum(self.d_mats[key] ** 2, axis=0) for key in group_keys},
                "F": {key: np.nansum(self.f_mats[key] ** 2, axis=0) for key in group_keys},
                "Z": {key: np.nansum(self.z_mats[key] ** 2, axis=0) for key in X["Z"]},
                "Y": {key: np.nansum(self.y_mats[key] ** 2, axis=0) for key in X["Y"]},
            }
        ]
        self.r2_frac = [
            {
                "D": {key: np.zeros(self.d_mats[key].shape[1]) for key in group_keys},
                "F": {key: np.zeros(self.f_mats[key].shape[1]) for key in group_keys},
                "Z": {key: np.zeros(self.z_mats[key].shape[1]) for key in self.z_mats},
                "Y": {key: np.zeros(self.y_mats[key].shape[1]) for key in self.y_mats},
            }
        ]

        # Then set missing data values to zeros (not because we are ignoring the values), but because we will use
        # the missing value maps to identify where the missing values are and therefore ignore them. But set to zero,
        # so these values have no influence on the calculations.
        self.d_mats = {key: nan_to_zeros(self.d_mats[key]) for key in group_keys}
        self.f_mats = {key: nan_to_zeros(self.f_mats[key]) for key in group_keys}
        self.z_mats = {key: nan_to_zeros(self.z_mats[key]) for key in X["Z"]}
        self.y_mats = {key: nan_to_zeros(self.y_mats[key]) for key in X["Y"]}

        # Storage for the model objects. Make a copy only of the Numpy values to use in the Estimator.
        self.observation_names = X["F"][group_keys[0]].index
        self.property_names = {key: self.d_matrix[key].columns.to_list() for key in group_keys}
        self.material_names = {key: self.d_matrix[key].index.to_list() for key in group_keys}
        self.condition_names = {key: X["Z"][key].columns.to_list() for key in X["Z"]}
        self.quality_names = {key: X["Y"][key].columns.to_list() for key in X["Y"]}

        # Create the missing value maps, except we store the opposite, i.e., not missing, since these are more useful.
        # We refer to these as `pmaps` in the code (present maps, as opposed to `mmap` or missing maps).
        self.not_na_d = {key: ~np.isnan(self.d_matrix[key].values) for key in self.d_mats}
        self.not_na_f = {key: ~np.isnan(X["F"][key].values) for key in self.f_mats}
        self.not_na_z = {key: ~np.isnan(X["Z"][key].values) for key in self.z_mats}
        self.not_na_y = {key: ~np.isnan(X["Y"][key].values) for key in self.y_mats}

        # Model parameters. Naming convention: x_i_j
        # x = block letter (P, W, R, T, etc)
        # i = block type: `scores` [for the observations (rows)] or `loadings` [for the variables (columns)]
        # j = block name [z, f, d, y, super]
        # ----------------
        self.t_scores_super: pd.DataFrame = pd.DataFrame(index=self.observation_names)
        self.r_loadings_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.material_names[key]) for key in group_keys
        }
        self.w_loadings_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.condition_names[key]) for key in self.z_mats
        }
        self.w_loadings_super = pd.DataFrame(index=["Z", "F"] if self.n_conditions > 0 else ["F"])
        # Capture the correlation of the properties in D; for the last component.
        self.s_loadings_d: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.property_names[key]) for key in group_keys
        }
        # Captures the deflation of the properties in D; for the last component.
        self.v_loadings_d: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.property_names[key]) for key in group_keys
        }
        self.p_loadings_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.material_names[key]) for key in group_keys
        }
        self.p_loadings_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.condition_names[key]) for key in self.z_mats
        }
        self.q_loadings_y: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.quality_names[key]) for key in self.y_mats
        }

        # Model performance
        # -----------------
        # 1. Prediction matrices (hat matrices for Y-space) in pre-processed space
        self.hat_: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.observation_names, columns=self.quality_names[key], dtype=float).fillna(0)
            for key in self.y_mats
        }
        # 2. Prediction matrix for the Y-space only, and then scaled back to the original space
        self.hat: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=self.observation_names, columns=self.quality_names[key], dtype=float).fillna(0)
            for key in self.y_mats
        }
        # 3. Squared prediction error (SPE) for each observation, per component, per block
        self.spe: dict[str, dict[str, pd.DataFrame]] = {key: {} for key in self.required_blocks_}
        self.spe_limit: dict[str, dict[str, Callable]] = {key: {} for key in self.required_blocks_}

        # 4. Hotelling's T2 values for each observation, per component
        self.hotellings_t2: pd.DataFrame = pd.DataFrame()
        self.hotellings_t2_limit: Callable = hotellings_t2_limit
        self.scaling_factor_for_scores = pd.Series()
        self.ellipse_coordinates: Callable = ellipse_coordinates

        self._fit_iterative_regressions()
        self.is_fitted_ = True
        return self

    def predict(self, X: DataFrameDict) -> Bunch:  # noqa: C901, PLR0912, PLR0915
        """
        Model inference on new data.

        This will pre-process the new data and apply those subsequently to the latent variable model.

        Example
        -------

        # Training phase:
        estimator = TPLS(n_components=2).fit(training_data)

        # Testing/inference phase:
        new_data = {"Z": ..., "F": ...}  # you need at least the F block for a new prediction. "Z" is optional.
        predictions = estimator.predict(new_data)

        Parameters
        ----------
        X : DataFrameDict
            The input samples.

        Returns
        -------
        y : dict
            Returns an array of prediction objects. More details to come here later. Please ask.
        """
        check_is_fitted(self)  # Check if fit had been called
        if not isinstance(X, DataFrameDict):
            raise TypeError(f"X must be a DataFrameDict; got {type(X).__name__}.")

        # TODO: Check consistency on the data: the columns names in the new data must match the columns names in the
        # training data.
        x_f: dict[str, pd.DataFrame] = {key: X["F"][key].copy() for key in X["F"]}
        x_z: dict[str, pd.DataFrame] = {key: X["Z"][key].copy() for key in X["Z"]}

        for key, df_f in x_f.items():
            if not self.skip_f_matrix_preprocessing:
                x_f[key] = (df_f - self.preproc_["F"][key]["center"]) / self.preproc_["F"][key]["scale"]

        for key, df_z in x_z.items():
            x_z[key] = (df_z - self.preproc_["Z"][key]["center"]) / self.preproc_["Z"][key]["scale"]

        not_na_f = {key: ~np.isnan(X["F"][key].values) for key in X["F"]}
        not_na_z = {key: ~np.isnan(X["Z"][key].values) for key in X["Z"]}
        names_observations = X["F"][next(iter(X["F"]))].index
        num_obs = names_observations.shape[0]
        spe_f: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=x_f[key].index, columns=range(1, self.n_components + 1)) for key in x_f
        }
        spe_z: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=x_z[key].index, columns=range(1, self.n_components + 1)) for key in x_z
        }

        t_scores_super = pd.DataFrame(index=names_observations, columns=range(1, self.n_components + 1), dtype=float)
        # Hotelling's T2 values, after so many components. In other words, in column 3, it is the Hotelling's T2
        # computed with 3 components.
        hotellings_t2 = pd.DataFrame(index=names_observations, columns=range(1, self.n_components + 1), dtype=float)
        # Predictions are returned in un-scaled form, so they are in the same units as the training data.
        hat: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(index=names_observations, columns=self.quality_names[key], dtype=float)
            for key in self.y_mats
        }

        for key, df_f in x_f.items():
            if df_f.shape[0] != num_obs:
                raise ValueError(
                    f"All formula blocks must have the same number of rows; "
                    f"group [{key}] has {df_f.shape[0]} rows, expected {num_obs}."
                )
            if set(df_f.columns) != set(self.material_names[key]):
                raise ValueError(
                    f"Columns in block F, group [{key}] must match training data column names for each material."
                )

        for key, df_z in x_z.items():
            if df_z.shape[0] != num_obs:
                raise ValueError(
                    f"All condition blocks must have the same number of rows; "
                    f"group [{key}] has {df_z.shape[0]} rows, expected {num_obs}."
                )
            if set(df_z.columns) != set(self.condition_names[key]):
                raise ValueError(
                    f"Column names in block Z, group [{key}] must match training data column names."
                )

        for pc_a in range(self.n_components):
            # Regress the row of each new formula block on the r_loadings_f, to get the t-score for that pc_a component.
            # Add up the t-score as you go block by block.
            score_f_a = np.zeros(num_obs)
            denominators = np.zeros(num_obs)
            for key, df_x_f in x_f.items():
                b_row = np.array(self.r_loadings_f[key].iloc[:, pc_a].values)
                # Tile row-by-row to create `n_rows`, and maps missing entries to zero, so they have no effect
                denom = np.tile(b_row, (num_obs, 1)) * not_na_f[key]
                score_f_a += np.array(np.sum(df_x_f.values * denom, axis=1))  # numerator portion
                denominators += np.sum((denom * not_na_f[key]) ** 2, axis=1)

            denominators[denominators == 0] = np.nan  # Guard should not be needed; should never be zeros in here.
            score_f_a /= denominators

            # Repeat for the Z-space: regress the row of each new Z block on the w-loadings, to get the
            # t-score for that pc_a. It seems redundant to divide by w'w, since w is already normalized, but if there
            # are missing values, then that correction is needed, to avoid dividing by a larger value than is fair.
            if self.n_conditions > 0:
                score_z_a = np.zeros(num_obs)
                denominators = np.zeros(num_obs)
                for key, df_x_z in x_z.items():
                    b_row = np.array(self.w_loadings_z[key].iloc[:, pc_a].values)
                    denom = np.tile(b_row, (num_obs, 1)) * not_na_z[key]
                    score_z_a += np.array(np.sum(df_x_z.values * denom, axis=1))
                    denominators += np.sum((denom * not_na_z[key]) ** 2, axis=1)

                # Multiply the individual block scores by the super-weights, to get the super-scores.
                # After transposing below, rows are the observations, and columns are the blocks: [Z, F]
                super_score_a = np.vstack([score_z_a, score_f_a]).T @ np.asarray(
                    self.w_loadings_super.iloc[:, pc_a].values
                ).reshape(-1, 1)
            else:
                # The w_loadings_super are just "1" or "-1" in this case
                super_score_a = score_f_a.reshape(-1, 1) * self.w_loadings_super.iloc[:, pc_a].values

            # Deflate each block (key) in x_f matrices with the super_scores, to get values for the next iteration,
            # and to compute SPE.
            explained_f = {
                key: super_score_a @ np.asarray(self.p_loadings_f[key].iloc[:, pc_a].values).reshape(1, -1)
                for key in x_f
            }
            for key, df_x_f in x_f.items():
                x_f[key] -= explained_f[key]
                spe_f[key].iloc[:, pc_a] = np.sqrt(np.sum(np.square(df_x_f), axis=1))

            explained_z = {
                key: super_score_a @ np.asarray(self.p_loadings_z[key].iloc[:, pc_a].values).reshape(1, -1)
                for key in x_z
            }
            for key, df_x_z in x_z.items():
                x_z[key] -= explained_z[key]
                spe_z[key].iloc[:, pc_a] = np.sqrt(np.sum(np.square(df_x_z), axis=1))

            # Store values for the final output
            t_scores_super.iloc[:, pc_a] = super_score_a.flatten()
            hotellings_t2.iloc[:, pc_a] = np.sum(super_score_a**2, axis=1)

        # After the loop has repeated `self.n_components` times: calculate the predictions using the full set of super
        # scores and the q-loadings for the Y-space.
        for key in self.y_mats:
            hat[key].iloc[:, :] = (t_scores_super.values @ self.q_loadings_y[key].values.T) * self.preproc_["Y"][key][
                "scale"
            ].values[None, :] + self.preproc_["Y"][key]["center"].values[None, :]

        # Calculate the T2 values: for all the spaces
        hotellings_t2.iloc[:, :] = (
            # Last item in the statement here is not super_scores.values !! we want the result back as a DataFrame
            t_scores_super.values @ np.diag(np.power(1 / self.scaling_factor_for_scores.values, 2), 0) * t_scores_super
        ).cumsum(axis="columns")

        return Bunch(
            hat=hat,
            t_scores_super=t_scores_super,
            spe={"Z": spe_z, "F": spe_f},
            hotellings_t2=hotellings_t2,
        )

    def display_results(self, show_cumulative_stats: bool = True) -> str:
        """Display the results of the model fitting."""

        if not self.is_fitted_:
            raise RuntimeError("The model is not fitted yet. Please call `fit` first.")

        output = f"Hotelling's T2 limit [95% limit]: {self.hotellings_t2_limit():.4g}\n"
        output += f"                     [99% limit]: {self.hotellings_t2_limit(0.99):.4g}\n"
        # output += f"SPE limits: {self.spe_limit['Y'](self.spe['Y'])}\n"
        sep = "------ ---------- ---------- ---------- ---------- -------------\n"
        output += sep
        if show_cumulative_stats:
            header = "LV #   sum(R2: D) sum(R2: Z) sum(R2: F) sum(R2: Y)|    ms [iter]"
        else:
            header = "LV #        R2: D      R2: Z      R2: F      R2: Y|    ms [iter]"

        output += header + "\n" + sep
        r2_d_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["D"].values()])
        r2_z_a_prior = (
            np.mean([r2val.mean() for r2val in self.r2_frac[0]["Z"].values()]) if self.n_conditions > 0 else 0
        )
        r2_f_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["F"].values()])
        r2_y_a_prior = np.mean([r2val.mean() for r2val in self.r2_frac[0]["Y"].values()])
        for a in range(1, self.n_components + 1):
            r2_d_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["D"].values()])
            r2_z_a = (
                np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["Z"].values()]) if self.n_conditions > 0 else 0
            )
            r2_f_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["F"].values()])
            r2_y_a = np.mean([np.nanmean(r2val) for r2val in self.r2_frac[a]["Y"].values()])
            if show_cumulative_stats:
                r2_d_a += r2_d_a_prior
                r2_z_a += r2_z_a_prior
                r2_f_a += r2_f_a_prior
                r2_y_a += r2_y_a_prior

            r2_d_a_prior = r2_d_a
            r2_z_a_prior = r2_z_a
            r2_f_a_prior = r2_f_a
            r2_y_a_prior = r2_y_a
            r2_z_a = f"{r2_z_a * 100:>10.1f}" if self.n_conditions > 0 else "        -"

            # Calculate time per iteration for this component
            time_ms = self.fitting_statistics["milliseconds"][a - 1]
            iterations = self.fitting_statistics["iterations"][a - 1]
            time_iter = f"{time_ms:>5.0f} [{iterations:>4d}]"

            line = (
                f"LV {a:<2}  {r2_d_a * 100:>10.1f} {r2_z_a} {r2_f_a * 100:>10.1f} {r2_y_a * 100:>10.1f}|{time_iter:>13}"
            )
            if self.fitting_statistics["iterations"][a - 1] >= self.max_iter:
                line += "** (max iter reached)"
            output += line + "\n"

        output += sep
        ms_per_iter = round(
            sum(self.fitting_statistics["milliseconds"]) / sum(self.fitting_statistics["iterations"]), 2
        )
        output += f"Timing: {ms_per_iter} ms/iter; {sum(self.fitting_statistics['iterations'])} iterations required\n"
        output += f"Total time: {sum(self.fitting_statistics['milliseconds']) / 1000:.2f} seconds\n"
        output += f"Average tolerance: {np.mean(self.fitting_statistics['convergance_tolerance']):.4g}\n"
        output += "Settings\n---------\n"
        output += f"n_components: {self.n_components}\n"
        output += f"max_iter: {self.max_iter}\n"
        output += f"skip_f_matrix_preprocessing: {self.skip_f_matrix_preprocessing}\n"

        return output

    def score(self, X: DataFrameDict, y: None = None, sample_weight: None | np.ndarray = None) -> float:  # noqa: ARG002
        """Return r2_score` on test data.

        See RegressorMixin.score for more details.

        Parameters
        ----------
        X : DataFrameDict
            Test samples.

        y : Not used. In the `X` input, there is a already a "Y" block. This will be the Y-data.

        sample_weight : Not used.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)``.
        """
        predictions = self.predict(X)
        y_pred = predictions.hat
        y_actual = X["Y"]
        if not y_actual:
            msg = "y_actual is empty: X['Y'] must contain at least one block to compute a score."
            raise ValueError(msg)
        r2_key = 0.0
        count = 0
        for key in y_actual:
            r2_key += r2_score(y_true=y_actual[key], y_pred=y_pred[key], sample_weight=sample_weight)
            _ = np.corrcoef(y_actual[key].values.ravel(), y_pred[key].values.ravel())
            count += 1
        return r2_key / count

    def help(self) -> str:
        """Help for the TPLS Estimator.

        Data organization
        -----------------

        Quick tips
        ----------
        Build model:                tpls = TPLS(n_components=2, d_matrix=d_matrix).fit(X)
        Get model's predictions:    tpls.hat            <-- the hat-matrix, i.e., the predictions
        Predict on new data:        tpls.predict(X_new)
        See model summary:          tpls.display_results()
        This help page:             tpls.help()

        Statistical values
        ------------------

        .t_scores_super             Super scores for the entire model                           [pd.DataFrame]
        .hotellings_t2              Hotelling's T2 values for each observation, per component   [pd.DataFrame]
        .spe                        Squared prediction error for each block                     [dict of pd.DataFrames]


        .hotellings_t2_limit()      Returns the Hotelling's T2 limit for the model              [float]
        .spe_limit[block]()         Return the SPE limit for the block; e.g. .spe_limit["Y"]() [float]


        TODO:
        self.hotellings_t2: pd.DataFrame = pd.DataFrame()
        self.hotellings_t2_limit: Callable = hotellings_t2_limit
        self.scaling_factor_for_scores = pd.Series()
        self.ellipse_coordinates: Callable = ellipse_coordinates

        """

        # Return this function's docstring as the help text.
        # Dedent the self.__docs__ string and return that
        return self.help.__doc__.replace("        ", "").replace("\n\n", "\n").strip()

    def _input_data_checks(self, X: DataFrameDict) -> None:
        """Check the incoming data."""
        if not isinstance(X, DataFrameDict):
            raise TypeError(
                f"The input data must be a DataFrameDict; got {type(X).__name__}."
            )
        if set(X.keys()) != self.required_inputs_:
            raise ValueError(
                f"Expected keys: {self.required_inputs_}, got: {set(X.keys())}."
            )
        group_keys = [str(key) for key in self.d_matrix]
        if set(X["F"]) != set(group_keys):
            raise ValueError("The keys in F must match the keys in D.")

        for key in X["Y"]:
            self._validate_df(X["Y"][key])
        for key in X["Z"]:
            self._validate_df(X["Z"][key])
        for key in self.d_matrix:
            self._validate_df(self.d_matrix[key])
            if key not in X["F"]:
                raise ValueError(
                    f"Block/group name '{key}' in D must also be present in F."
                )
            self._validate_df(X["F"][key])  # this also ensures the keys in F are the same as in D

    def _learn_center_and_scaling_parameters(self, y: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Learn the centering and scaling parameters for the output space.

        Parameters
        ----------
        y : pd.DataFrame
            The output space.

        Returns
        -------
        centering : pd.Series
            The centering parameters.

        scaling : pd.Series
            The scaling parameters.
        """
        centering = y.mean(axis="index")
        scaling = y.std(ddof=1, axis="index") if y.shape[0] > 1 else pd.Series(1.0, index=y.columns)
        scaling[scaling < self.tolerance_] = float("nan")  # columns with little/no variance: set as nan
        return centering, scaling

    def _validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a single dataframe using `check_array` from scikit-learn.

        Parameters
        ----------
        df : {pd.DataFrame}

        Returns
        -------
        y : {pd.DataFrame}
            Returns the input dataframe.
        """
        # Ensure all columns are dtype "float64" or "int64"
        if not all(good_cols := [isinstance(col, (np.dtypes.Float64DType, np.dtypes.IntDType)) for col in df.dtypes]):
            bad_columns = df.columns[[not item for item in good_cols]].to_list()
            raise ValueError(
                f"All columns in the DataFrame must be of type float64 or int64. Bad columns: {bad_columns}"
            )

        return check_array(
            df, accept_sparse=False, ensure_all_finite="allow-nan", ensure_2d=True, allow_nd=False, ensure_min_samples=1
        )

    def _has_converged(self, starting_vector: np.ndarray, revised_vector: np.ndarray, iterations: int) -> bool:
        """
        Terminate the iterative algorithm when any one of these conditions is True.

        #. scores converge: the norm between two successive iterations is smaller than a tolerance
        #. maximum number of iterations is reached
        """
        # Floor ``||starting_vector||`` -- when both vectors collapse to
        # zero (degenerate block / fully-deflated component) the bare
        # division produced NaN, the convergence test became permanently
        # False, and the loop ran to max_iter. SEC-21 (#270) sub-item 1.
        delta_gap = float(
            np.linalg.norm(starting_vector - revised_vector, ord=None)
            / _nz(np.linalg.norm(starting_vector, ord=None))
        )
        converged = delta_gap < self.tolerance_
        max_iter = iterations >= self.max_iter
        return bool(np.any([max_iter, converged]))

    def _store_model_coefficients(  # noqa: PLR0913
        self,
        pc_a_column: int,  # one-based index for the component
        t_super_i: np.ndarray,
        r_i: dict[str, np.ndarray],
        w_i_z: dict[str, np.ndarray],
        w_super_i: np.ndarray,
        s_i: dict[str, np.ndarray],
    ) -> None:
        """Store the model coefficients for later use."""

        self.t_scores_super = self.t_scores_super.join(
            pd.DataFrame(t_super_i, index=self.observation_names, columns=[pc_a_column])
        )

        # These are loadings really, not scores, for each group in the F block.
        self.r_loadings_f = {
            key: self.r_loadings_f[key].join(
                pd.DataFrame(r_i[key], index=self.material_names[key], columns=[pc_a_column])
            )
            for key in r_i
        }

        # These are the loadings for the Z space
        self.w_loadings_z = {
            key: self.w_loadings_z[key].join(
                pd.DataFrame(w_i_z[key], index=self.condition_names[key], columns=[pc_a_column])
            )
            for key in w_i_z
        }

        self.w_loadings_super = self.w_loadings_super.join(
            pd.DataFrame(w_super_i, index=["Z", "F"] if self.n_conditions > 0 else ["F"], columns=[pc_a_column])
        )

        self.s_loadings_d = {
            key: self.s_loadings_d[key].join(
                pd.DataFrame(s_i[key], index=self.property_names[key], columns=[pc_a_column])
            )
            for key in s_i
        }

    def _calculate_and_store_deflation_matrices(
        self,
        pc_a: int,
        t_super_i: np.ndarray,
        q_super_i: np.ndarray,
        r_i: dict[str, np.ndarray],
    ) -> None:
        """
        Calculate and store the deflation matrices for the TPLS model.

        Deflate the matrices stored in the instance object.

        Returns the prediction matrices in a dictionary.
        """
        # Step 13: Deflate the Z matrix with a loadings vector, pz_b (_b is for block)
        pz_b = {
            key: regress_a_space_on_b_row(df_z.T, t_super_i.T, pmap_z.T)
            for key, df_z, pmap_z in zip(self.z_mats.keys(), self.z_mats.values(), self.not_na_z.values(), strict=True)
        }
        for key in self.z_mats:
            self.z_mats[key] -= (t_super_i @ pz_b[key].T) * self.not_na_z[key]
        self.p_loadings_z = {
            key: self.p_loadings_z[key].join(pd.DataFrame(pz_b[key], index=self.condition_names[key], columns=[pc_a]))
            for key in pz_b
        }

        # Step 13. p_i = F_i' t_i / t_i't_i. Regress the columns of F_i on t_i; store slope coeff in vectors p_i.
        # Note: the "t" vector is the t_i vector from the inner PLS model, marked as "Tt" in figure 4 of the paper.
        # It is the score column from the super score matrix regression onto Y.
        pf_i = {
            key: regress_a_space_on_b_row(df_f.T, t_super_i.T, pmap_f.T)
            for key, df_f, pmap_f in zip(self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True)
        }
        self.p_loadings_f = {
            key: self.p_loadings_f[key].join(pd.DataFrame(pf_i[key], index=self.material_names[key], columns=[pc_a]))
            for key in pf_i
        }
        # Step 13: v_i = D_i' r_i / r_i'r_i. Regress the rows of D_i (properties) on r_i; store slopes in v_i.
        self.v_loadings_d = {
            key: self.v_loadings_d[key].join(
                pd.DataFrame(
                    regress_a_space_on_b_row(df_d.T, r_i[key].T, pmap_d.T),
                    index=self.property_names[key],
                    columns=[pc_a],
                )
            )
            for key, df_d, pmap_d in zip(self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True)
        }
        # Step 14. Do the actual deflation.
        for key in self.d_mats:
            # Step to deflate F matrix
            self.f_mats[key] -= (t_super_i @ pf_i[key].T) * self.not_na_f[key]

            # Two sets of matrices to deflate: properties D and formulas F.
            self.d_mats[key] -= (r_i[key] @ self.v_loadings_d[key].iloc[:, [-1]].T) * self.not_na_d[key]

        # Deflate the Y-space as well
        self.q_loadings_y = {
            key: self.q_loadings_y[key].join(pd.DataFrame(q_super_i, index=self.quality_names[key], columns=[pc_a]))
            for key in self.y_mats
        }
        for key in self.y_mats:
            self.hat_[key] += t_super_i @ q_super_i.T
            self.y_mats[key] -= (t_super_i @ q_super_i.T) * self.not_na_y[key]

    def _update_performance_statistics(self) -> None:
        """Calculate and store the performance statistics of the model, such as SSQ, R2, etc."""
        # Calculate the sums of squares for each block, per column.
        # Note: the `ddof=0` is used to calculate the population variance, which is proportional to the SSQ.
        calc_ssq = {
            "D": {key: np.nansum(self.d_mats[key] ** 2, axis=0) for key in self.d_mats},
            "F": {key: np.nansum(self.f_mats[key] ** 2, axis=0) for key in self.f_mats},
            "Z": {key: np.nansum(self.z_mats[key] ** 2, axis=0) for key in self.z_mats},
            "Y": {key: np.nansum(self.y_mats[key] ** 2, axis=0) for key in self.y_mats},
        }
        self.sums_of_squares_.append(calc_ssq)

        # Calculate the incremental (not cumulative!) R2 values for each block, per column:
        # Cumulative R2 values can be found by summation. The R2 values are **always** fractional (between 0 and 1).
        ssq_prior_pc = self.sums_of_squares_[-2]
        ssq_start_0 = self.sums_of_squares_[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Ignore warnings about division by zero, since some columns might have no variance.
            calc_r2 = {
                "D": {
                    key: (ssq_prior_pc["D"][key] - calc_ssq["D"][key]) / ssq_start_0["D"][key] for key in self.d_mats
                },
                "F": {
                    key: (ssq_prior_pc["F"][key] - calc_ssq["F"][key]) / ssq_start_0["F"][key] for key in self.f_mats
                },
                "Z": {
                    key: (ssq_prior_pc["Z"][key] - calc_ssq["Z"][key]) / ssq_start_0["Z"][key] for key in self.z_mats
                },
                "Y": {
                    key: (ssq_prior_pc["Y"][key] - calc_ssq["Y"][key]) / ssq_start_0["Y"][key] for key in self.y_mats
                },
            }
            self.r2_frac.append(calc_r2)

        # VIP for each block, for given number of components we currently have. VIP are cumulative.
        # For the D-block and F-block: you want to be able to use the VIPs to compare across the entire block, without
        # regards to the banding in groups that might have to be done. So use the total R2 for that block, and do not
        # work per group.
        r2_d_a: list[float] = [
            float(np.mean([np.nanmean(r2val) for r2val in r2_frac["D"].values()])) for r2_frac in self.r2_frac
        ]
        r2_f_a: list[float] = [
            float(np.mean([np.nanmean(r2val) for r2val in r2_frac["F"].values()])) for r2_frac in self.r2_frac
        ]
        loadings_s = np.concatenate(list(self.s_loadings_d.values()))
        loadings_f = np.concatenate(list(self.p_loadings_f.values()))
        vip_d = self._calculate_vip(loadings_s, np.array(r2_d_a[1:]))
        vip_f = self._calculate_vip(loadings_f, np.array(r2_f_a[1:]))
        # Split the `vip_d` back into the original groups that it was merged from.
        # For example: if there are two groups, one with 17 columns, and the second with 3 columns, then there are
        # a total of 20 values in `vip_d`, and the first 17 values correspond to the first group, and the last 3 values.
        # Create a dictionary with the group names as keys, and the VIP values as values, split correctly:
        vip_split_d = {}
        start = 0
        for key in self.property_names:
            end = start + len(self.property_names[key])
            vip_split_d[key] = pd.Series(vip_d[start:end], index=self.property_names[key])
            start = end

        vip_split_f = {}
        start = 0
        for key in self.material_names:
            end = start + len(self.material_names[key])
            vip_split_f[key] = pd.Series(vip_f[start:end], index=self.material_names[key])
            start = end

        self.feature_importance["D"] = vip_split_d  # TODO: should it not be based on deflated matrices? S(V^TS)^{-1}
        self.feature_importance["F"] = vip_split_f  # TODO: should it not be based on deflated matrices? P(_^TP)^{-1}

    def vip(self, block: str | None = None) -> dict[str, dict[str, pd.Series]] | dict[str, pd.Series]:
        """Return Variable Importance in Projection (VIP) scores for TPLS blocks.

        VIP scores are computed during fitting for the D-block (material properties) and
        F-block (formulation variables) and stored in :attr:`feature_importance`.

        Parameters
        ----------
        block : str or None, default=None
            Which block to return. Must be ``"D"`` or ``"F"``, or ``None`` to
            return all blocks.

        Returns
        -------
        dict
            If *block* is ``None``: ``{"D": {group: pd.Series, ...}, "F": {group: pd.Series, ...}}``.
            If *block* is ``"D"`` or ``"F"``: the inner dict ``{group: pd.Series, ...}`` for that block,
            where each ``pd.Series`` is indexed by feature names.

        Raises
        ------
        ValueError
            If the model is not fitted or *block* is not ``"D"``, ``"F"``, or ``None``.

        Examples
        --------
        >>> tpls = TPLS(...).fit(data)
        >>> tpls.vip()          # all blocks
        >>> tpls.vip("D")       # D-block only → {group_name: pd.Series, ...}
        """
        check_is_fitted(self, "feature_importance")
        if block is None:
            return self.feature_importance
        if block not in ("D", "F"):
            msg = f"block must be 'D', 'F', or None; got {block!r}."
            raise ValueError(msg)
        return self.feature_importance[block]

    def _calculate_vip(self, loadings: np.ndarray, r2_vector: np.ndarray) -> np.ndarray:
        """Calculate the VIP values for the current component.

        The `loadings` has as many rows as there are feature varaibles, and A columns, where A = number of components.
        The `r2_vector` is a vector of fractional R^2 values for the current component, with `A` entries.
        The `r2_vector` values should be between 0 and 1; the fraction of variance explained by the component for that
        given `loadings` matrix.

        The VIP values are calculated as follows:
            VIP = sqrt(n * sum((r2_vector * (loadings ** 2)) / sum(r2_vector)))

        where n is the number of features (rows in the loadings matrix).
        """
        # VIP = sqrt(n * sum((r2_vector * (loadings ** 2)) / sum(r2_vector)))
        n = loadings.shape[0]
        r2_vector = r2_vector.reshape(1, -1)  # Ensure r2_vector is a row vector
        return np.sqrt(n * np.sum(r2_vector * (loadings**2), axis=1) / np.sum(r2_vector))

    def _calculate_model_statistics_and_limits(self) -> None:
        """Calculate and store the model limits.

        Limits calculated:
        1. Hotelling's T2 limits
        2. Squared prediction error limits

        Other calculations:
        1. The model's Y-space predictions are scaled back to the original space.
        """

        # Calculate the Hotelling's T2 values, and limits. Could do a ddof correction (n-1) for the variance matrix.
        variance_matrix = self.t_scores_super.T @ self.t_scores_super / self.t_scores_super.shape[0]
        t2_values = np.sum(
            (self.t_scores_super.values @ safe_inverse(variance_matrix, what="super-score covariance"))
            * self.t_scores_super.values,
            axis=1,
        )
        self.hotellings_t2 = pd.DataFrame(
            t2_values,
            index=self.observation_names,
            columns=["Hotelling's T^2"],
        )
        self.hotellings_t2_limit = partial(
            hotellings_t2_limit, n_components=self.n_components, n_rows=self.hotellings_t2.shape[0]
        )
        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(np.diag(variance_matrix)),
            index=[a + 1 for a in range(self.n_components)],
            name="Standard deviation per score",
        )
        self.ellipse_coordinates = partial(
            ellipse_coordinates,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.t_scores_super.shape[0],
        )

        # Squared prediction error limits. This is a measure of the prediction error = difference between the actual
        # and predicted values. Since the matrices are deflated by the predictive part of the model already, the
        # data in these matrices is already the prediction error. Calculate the **squared** portion, and store it.
        column_name = [f"SPE with A={self.n_components}"]
        self.spe["Y"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.y_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.y_mats
        }
        self.spe_limit["Y"] = {key: partial(spe_calculation, self.spe["Y"][key].values) for key in self.y_mats}
        self.spe["Z"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.z_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.z_mats
        }
        self.spe_limit["Z"] = {key: partial(spe_calculation, self.spe["Z"][key].values) for key in self.z_mats}

        # SPE for the D-space. There are two options: per property feature, or per material feature.
        self.spe["D"] = {key: self.d_mats[key].pow(2).sum(axis="columns").pow(0.5) for key in self.d_mats}
        self.spe_limit["D"] = {key: partial(spe_calculation, self.spe["D"][key].values) for key in self.d_mats}
        self.spe["F"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.f_mats[key]), axis=1, keepdims=True)),
                index=self.observation_names,
                columns=column_name,
            )
            for key in self.f_mats
        }
        self.spe_limit["F"] = {key: partial(spe_calculation, self.spe["F"][key].values) for key in self.f_mats}

        # Y-space predictions
        for key in self.y_mats:
            # The Y-space predictions are already in the pre-processed space, so we need to scale them back to the
            self.hat[key] = pd.DataFrame(self.hat_[key], index=self.observation_names, columns=self.quality_names[key])
            self.hat[key] = self.hat[key].multiply(self.preproc_["Y"][key]["scale"].values[None, :], axis=1)
            self.hat[key] += self.preproc_["Y"][key]["center"].values[None, :]

    def _preprocess_data(self) -> None:
        """Pre-process the training data."""

        for key in self.f_mats:
            if not self.skip_f_matrix_preprocessing:
                self.f_mats[key] = (
                    self.f_mats[key] - self.preproc_["F"][key]["center"].values[None, :]
                ) / self.preproc_["F"][key]["scale"].values[None, :]

            self.d_mats[key] = (
                (self.d_mats[key] - self.preproc_["D"][key]["center"].values[None, :])
                / self.preproc_["D"][key]["scale"].values[None, :]
                / self.preproc_["D"][key]["block"][0]  # scalar!
            )
        for key in self.z_mats:
            self.z_mats[key] = (self.z_mats[key] - self.preproc_["Z"][key]["center"].values[None, :]) / self.preproc_[
                "Z"
            ][key]["scale"].values[None, :]

        for key in self.y_mats:
            self.y_mats[key] = (self.y_mats[key] - self.preproc_["Y"][key]["center"].values[None, :]) / self.preproc_[
                "Y"
            ][key]["scale"].values[None, :]

        # Test that all blocks and groups within a block have a mean of 0 and a standard deviation of 1.
        # Note the extra complexity for checking columns that have perfectly zero variance.
        # Internal invariants on the just-preprocessed matrices, not user input.
        for key in self.z_mats:
            assert np.allclose(np.nanmean(self.z_mats[key], axis=0), 0, atol=1e-6)  # post-centering invariant
            for item in np.nanstd(self.z_mats[key], axis=0, ddof=1):
                if item != 0:
                    assert np.isclose(item, 1)  # post-scaling invariant

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            for key in self.f_mats:
                if not self.skip_f_matrix_preprocessing:
                    vector = np.nanmean(self.f_mats[key], axis=0)
                    vector[np.isnan(vector)] = 0
                    assert np.allclose(vector, 0, atol=1e-6)  # post-centering invariant

                    vector = np.nanstd(self.f_mats[key], axis=0, ddof=1)
                    vector[np.isnan(vector)] = 1
                    assert np.allclose(vector, 1)  # post-scaling invariant

                vector = np.nanmean(self.d_mats[key], axis=0)
                vector[np.isnan(vector)] = 0
                assert np.allclose(vector, 0, atol=1e-6)  # post-centering invariant
                vector = np.nanstd(self.d_mats[key], axis=0, ddof=1) * self.preproc_["D"][key]["block"].values[0]
                vector[np.isnan(vector)] = 1
                assert np.allclose(vector, 1)  # post-scaling invariant

        # Checks on the Y-block: post-centering / post-scaling invariants.
        assert all(  # post-centering invariant on every Y block
            np.allclose(np.nanmean(self.y_mats[key], axis=0), 0, atol=1e-6) for key in self.y_mats
        )
        assert all(  # post-scaling invariant on every Y block
            np.allclose(np.where((in_array := np.nanstd(self.y_mats[key], axis=0, ddof=1)) == 0, 1, in_array), 1)
            for key in self.y_mats
        )

    def _fit_iterative_regressions(self) -> None:
        """Fit the model via iterative regressions and store the model coefficients in the class instance."""

        # Formula matrix: assemble all not-na maps from blocks in F: make a single matrix.
        pmap_f = np.concatenate(list(self.not_na_f.values()), axis=1)

        # Follow the steps in the paper on page 54
        for pc_a in range(self.n_components):
            n_iter = 0
            milliseconds_start = time.time()

            # Step 1: Select any column in Y as initial guess (they have all be scaled anyway, so it doesn't matter)
            u_super_i = next(iter(self.y_mats.values()))[:, [0]]
            u_prior = u_super_i + 1

            while not self._has_converged(starting_vector=u_prior, revised_vector=u_super_i, iterations=n_iter):
                n_iter += 1
                u_prior = u_super_i.copy()
                # Step 2. h_i = F_i' u / u'u. Regress the columns of F on u_i, and store the slope coeff in vectors h_i
                h_i = {
                    key: regress_a_space_on_b_row(df_f.T, u_super_i.T, pmap_f.T)
                    for key, df_f, pmap_f in zip(
                        self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True
                    )
                }

                # Step 3. s_i = D_i' h_i / h_i'h_i. Regress the rows of D_i on h_i, and store slope coeff in vectors s_i
                s_i = {
                    key: regress_a_space_on_b_row(df_d.T, h_i[key].T, pmap_d.T)
                    for key, df_d, pmap_d in zip(
                        self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True
                    )
                }
                # Step 4: combine the entries in s_i to form a joint `s` and normalize it to unit length.
                joint_s_normalized = np.linalg.norm(np.concatenate(list(s_i.values())))
                s_i = {key: s / joint_s_normalized for key, s in s_i.items()}

                # Step 5: r_i = D_i s_i / s_i's_i. Regress columns of D_i on s_i, and store slope coefficients in r_i.
                r_i = {
                    key: regress_a_space_on_b_row(df_d, s_i[key].T, self.not_na_d[key])
                    for key, df_d in zip(self.d_mats.keys(), self.d_mats.values(), strict=True)
                }

                # Step 6: Combine the entries in r_i to form a joint r (which is the name of the method in the paper).
                #         Horizontally concatenate all matrices in F_i to form a joint F matrix.
                #         Regress rows of the joint F matrix onto the joint r vector. Store coeff in block scores, t_f
                joint_r = np.concatenate(list(r_i.values()))
                joint_f = np.concatenate(list(self.f_mats.values()), axis=1)
                t_f = regress_a_space_on_b_row(joint_f, joint_r.T, pmap_f)

                # If there is a Condition matrix (non-empty Z block)
                if self.n_conditions > 0:
                    # Step 7: w_i = Z_i' u / u'u. Regress the columns of Z on u_i, and store the slope coefficients
                    #         in vectors w_i.
                    w_i_z = {
                        key: regress_a_space_on_b_row(df_z.T, u_super_i.T, self.not_na_z[key].T)
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }

                    # Step 8: Normalize joint w to unit length. See MB-PLS by Westerhuis et al. 1998. This is normal.
                    # Floor each per-block norm so a degenerate Z block doesn't yield NaN. SEC-21 (#270) sub-item 1.
                    w_i_z = {key: w / _nz(np.linalg.norm(w)) for key, w in w_i_z.items()}

                    # Step 9: regress rows of Z on w_i, and store slope coefficients in t_z. There is an error in the
                    #        paper here, but in figure 4 it is clear what should be happening.
                    t_zb = {
                        key: regress_a_space_on_b_row(df_z, w_i_z[key].T, self.not_na_z[key])
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }
                    t_z = np.concatenate(list(t_zb.values()), axis=1)

                else:
                    # Step 7: No Z block. Take an empty matrix across to the the superblock.
                    w_i_z = {}
                    t_z = np.zeros((t_f.shape[0], 0))  # empty matrix: in other words, no Z block

                # Step 10: Combine t_z and t_f to form a joint t matrix.
                t_combined = np.concatenate([t_z, t_f], axis=1)

                # Step 11: Build an inner PLS model: using the t_combined as the X matrix, and the Y (quality space)
                #          as the Y matrix.
                inner_pls = internal_pls_nipals_fit_one_pc(
                    x_space=t_combined,
                    y_space=np.array(next(iter(self.y_mats.values()))),
                    x_present_map=np.ones(t_combined.shape).astype(bool),
                    y_present_map=np.array(next(iter(self.not_na_y.values()))),
                )
                u_super_i = inner_pls["u_i"]  # only used for convergence check; not stored or used further
                t_super_i = inner_pls["t_i"]
                q_super_i = inner_pls["q_i"]
                w_super_i = inner_pls["w_i"]

            # After convergance. Step 12: Now store information.
            # =================
            # Floor ``||u_prior||`` -- see SEC-21 (#270) sub-item 1.
            delta_gap = float(
                np.linalg.norm(u_prior - u_super_i, ord=None) / _nz(np.linalg.norm(u_prior, ord=None))
            )
            self.fitting_statistics["iterations"].append(n_iter)
            self.fitting_statistics["convergance_tolerance"].append(delta_gap)
            self.fitting_statistics["milliseconds"].append((time.time() - milliseconds_start) * 1000)

            # Store model coefficients
            self._store_model_coefficients(
                pc_a + 1, t_super_i=t_super_i, r_i=r_i, w_i_z=w_i_z, w_super_i=w_super_i, s_i=s_i
            )

            # Calculate and store the deflation vectors. See equation 7 on page 55.
            self._calculate_and_store_deflation_matrices(pc_a + 1, t_super_i=t_super_i, q_super_i=q_super_i, r_i=r_i)

            # Update performance statistics for this component
            self._update_performance_statistics()

        # Step 15: Calculate the final model limits (after all components have been fitted).
        self._calculate_model_statistics_and_limits()

    # def _calculate_r2_score(
    #     self, y_true: dict[str, pd.DataFrame], y_pred: dict[str, pd.DataFrame], sample_weight: np.ndarray|None = None
    # ) -> float:
    #     """Calculate R^2 score across all Y blocks."""
    #     total_ss_res = 0.0
    #     total_ss_tot = 1e-10

    # for key in y_true.keys():
    #     y_true_values = y_true[key].values
    #     y_pred_values = y_pred[key].values

    #     # Handle sample weights
    #     if sample_weight is not None:
    #         weights = sample_weight.reshape(-1, 1)
    #         # Residual sum of squares (weighted)
    #         ss_res = np.sum(weights * (y_true_values - y_pred_values) ** 2)
    #         # Total sum of squares (weighted)
    #         y_mean_weighted = np.average(y_true_values, weights=sample_weight.flatten(), axis=0)
    #         ss_tot = np.sum(weights * (y_true_values - y_mean_weighted) ** 2)
    #     else:
    #         # Residual sum of squares
    #         ss_res = np.sum((y_true_values - y_pred_values) ** 2)
    #         # Total sum of squares
    #         y_mean = np.mean(y_true_values, axis=0)
    #         ss_tot = np.sum((y_true_values - y_mean) ** 2)

    #     total_ss_res = total_ss_res + ss_res
    #     total_ss_tot = total_ss_tot + ss_tot

    # # Calculate R² = 1 - (SS_res / SS_tot)
    # if total_ss_tot == 0:
    #     return 0.0 if total_ss_res == 0 else float("-inf")

    #    return 1.0 - (total_ss_res / total_ss_tot)


class MBPLS(RegressorMixin, BaseEstimator):
    r"""Multi-block PLS (hierarchical / superblock formulation).

    Generic multi-block PLS as described by Westerhuis, Kourti & MacGregor
    (1998) and Westerhuis & Smilde (2001). Each X-block is preprocessed
    independently (mean-centred and unit-variance scaled), then divided by
    ``sqrt(K_b)`` so that blocks of unequal width contribute fairly to the
    super-score.

    Parameters
    ----------
    n_components : int
        Number of latent variables to extract.
    max_iter : int, default=500
        Maximum NIPALS iterations per latent variable.
    tol : float or None, default=None
        Convergence tolerance on the change in the Y-block score. If
        ``None``, ``np.finfo(float).eps ** (6/7)`` is used (matching the
        legacy multi-block reference implementation).
    algorithm : str, default="auto"
        Algorithm to use for fitting the model.

        - ``"auto"``: dense vectorised hierarchical NIPALS when every
          block (X and Y) is complete; mask-aware NIPALS when any block
          contains missing values.
        - ``"dense"``: dense vectorised hierarchical NIPALS. Raises if
          any block contains missing values.
        - ``"nipals"``: mask-aware hierarchical NIPALS. Always uses the
          NaN-tolerant inner-loop primitives, even when the data is
          complete (slower than ``"dense"`` but produces equivalent
          results).

    missing_data_settings : dict or None, default=None
        Settings for the iterative ``"nipals"`` path. Keys: ``md_tol``
        (convergence tolerance on the score-vector change between
        iterations), ``md_max_iter`` (maximum NIPALS iterations per
        component). Defaults to ``{"md_tol": epsqrt, "md_max_iter": 1000}``.

    Attributes (after fitting)
    --------------------------
    block_names_ : list[str]
        Ordered list of X-block names (the keys of the input dict).
    block_widths_ : dict[str, int]
        Number of variables in each X-block.
    super_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block (consensus) X-scores ``T``.
    super_y_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block Y-scores ``U``.
    super_weights_ : pd.DataFrame, shape (n_blocks, n_components)
        Super-block weights ``w_super``; rows indexed by block name.
    super_y_loadings_ : pd.DataFrame, shape (n_targets, n_components)
        Y-block loadings ``c``.
    block_scores_ : dict[str, pd.DataFrame]
        Per-block X-scores ``t_b``, each shape ``(n_samples, n_components)``.
    block_weights_ : dict[str, pd.DataFrame]
        Per-block X-weights ``w_b``, each shape ``(K_b, n_components)``.
        Each column has unit norm.
    block_loadings_ : dict[str, pd.DataFrame]
        Per-block X-loadings ``p_b`` (used for deflation), each shape
        ``(K_b, n_components)``.
    predictions_ : pd.DataFrame, shape (n_samples, n_targets)
        In-sample Y predictions on the *original* scale.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variance of the super-score per component (ddof=1).
    scaling_factor_for_super_scores_ : pd.Series
        ``sqrt(explained_variance_)`` per component.
    fitting_info_ : dict
        Per-component iteration count and timing.
    has_missing_data_ : bool
        Whether any X-block or Y had NaN values.
    algorithm_ : str
        The resolved algorithm actually used for the fit. With
        ``algorithm="auto"``, this is ``"dense"`` for complete data
        and ``"nipals"`` for NaN-containing data.

    Notes
    -----
    Block weighting uses the convention :math:`X_b / \sqrt{K_b}` so that
    every block contributes the same total sum of squares to the
    super-score, regardless of how many variables it has.

    Missing data
    ------------
    When any X-block or Y contains NaN entries, the ``"auto"``
    algorithm routes to a mask-aware NIPALS variant. The X-block
    weights, block scores, block loadings used for deflation, Y-block
    loadings and Y-block scores are each computed as a regression that
    uses only the observed entries; the masked sum-of-squares is used
    as the denominator so missing values neither bias the latent
    direction nor contribute to the score. The mask is preserved
    across components automatically because deflation propagates NaN
    through subtraction. This is the standard skip-NaN NIPALS update;
    see Walczak & Massart (2001) and Arteaga & Ferrer (2002).

    The fit refuses to run if any X-block or Y has a column with all
    entries missing, or a row with all entries missing for that
    block; either case leaves the masked denominator at zero. Drop or
    impute such rows or columns before fitting. Predict-time score
    estimation for new observations with NaN (Trimmed Score Regression
    / Projection to the Model Plane) is a separate follow-up.

    References
    ----------
    Westerhuis, J. A., Kourti, T. & MacGregor, J. F. *Analysis of
    multiblock and hierarchical PCA and PLS models.* Journal of
    Chemometrics, 12 (1998), 301-321.

    Westerhuis, J. A. & Smilde, A. K. *Deflation in multiblock PLS.*
    Journal of Chemometrics, 15 (2001), 485-493.

    Walczak, B. & Massart, D. L. *Dealing with missing data: Part I.*
    Chemom. Intell. Lab. Syst., 58 (2001), 15-27.

    Arteaga, F. & Ferrer, A. *Dealing with missing data in MSPC: several
    methods, different interpretations, some examples.* J. Chemometrics,
    16 (2002), 408-418.
    """

    _valid_algorithms: typing.ClassVar[list[str]] = ["auto", "dense", "nipals"]

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
        "max_iter": [int],
        "tol": [float, None],
        "algorithm": [str],
        "missing_data_settings": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 500,
        tol: float | None = None,
        algorithm: str = "auto",
        missing_data_settings: dict | None = None,
    ):
        super().__init__()
        if n_components <= 0:
            raise ValueError(f"n_components must be positive; got {n_components}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {max_iter}.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.missing_data_settings = missing_data_settings

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> MBPLS:  # noqa: C901, PLR0912, PLR0915
        """Fit the multi-block PLS model.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            X-blocks. Keys are block names; values are DataFrames sharing the
            same row index (and row count). Each block is preprocessed
            independently.
        y : pd.DataFrame
            Y-block. Same row index / row count as the X-blocks.
        """
        if not isinstance(X, dict) or len(X) == 0:
            raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        for name, block in X.items():
            if not isinstance(block, pd.DataFrame):
                raise TypeError(f"X['{name}'] must be a pandas DataFrame; got {type(block).__name__}.")

        self.block_names_: list[str] = list(X.keys())
        first = X[self.block_names_[0]]
        n_samples = first.shape[0]
        for name in self.block_names_:
            if X[name].shape[0] != n_samples:
                raise ValueError(
                    f"All X-blocks must have the same row count. Block '{name}' has "
                    f"{X[name].shape[0]} rows; expected {n_samples}."
                )
        if y.shape[0] != n_samples:
            raise ValueError(f"y has {y.shape[0]} rows; expected {n_samples} to match X-blocks.")

        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._y_columns = y.columns
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}

        self.n_samples_ = int(n_samples)
        self.n_targets_ = int(y.shape[1])
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        n_components = int(self.n_components)
        n_blocks = len(self.block_names_)

        self.has_missing_data_ = any(np.any(X[name].isna().values) for name in self.block_names_) or bool(
            np.any(y.isna().values)
        )
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognised. Must be one of {self._valid_algorithms}."
            )
        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "dense"
        if algo == "dense" and self.has_missing_data_:
            raise ValueError("Algorithm 'dense' cannot handle missing data. Use 'nipals' or 'auto' instead.")
        self.algorithm_ = algo

        # Resolve iterative-algorithm settings (used by the 'nipals' path).
        settings = {"md_tol": epsqrt, "md_max_iter": 1000}
        if isinstance(self.missing_data_settings, dict):
            settings.update(self.missing_data_settings)
        settings["md_max_iter"] = int(settings["md_max_iter"])
        if algo == "nipals":
            if not settings["md_tol"] < 10:
                raise ValueError("Tolerance should not be too large.")
            if not settings["md_tol"] > epsqrt**1.95:
                raise ValueError("Tolerance must exceed machine precision.")
            # Degeneracy guards: any column or any (block, row) entirely NaN
            # leaves the masked NIPALS denominator at zero, which would
            # silently produce a spurious score or loading. Refuse the fit
            # rather than coerce the user into a misleading result.
            for name in self.block_names_:
                values = X[name].values
                col_all_nan = np.all(np.isnan(values), axis=0)
                if np.any(col_all_nan):
                    bad = X[name].columns[col_all_nan].tolist()
                    raise ValueError(
                        f"Block '{name}' has columns with all values missing: {bad}. "
                        "Drop these columns before fitting."
                    )
                row_all_nan = np.all(np.isnan(values), axis=1)
                if np.any(row_all_nan):
                    bad_rows = np.where(row_all_nan)[0].tolist()
                    raise ValueError(
                        f"Block '{name}' has rows with all values missing at positions {bad_rows}. "
                        "Drop these observations or impute them before fitting."
                    )
            y_values = y.values
            y_col_all_nan = np.all(np.isnan(y_values), axis=0)
            if np.any(y_col_all_nan):
                bad = y.columns[y_col_all_nan].tolist()
                raise ValueError(
                    f"Y has columns with all values missing: {bad}. Drop these targets before fitting."
                )
            y_row_all_nan = np.all(np.isnan(y_values), axis=1)
            if np.any(y_row_all_nan):
                bad_rows = np.where(y_row_all_nan)[0].tolist()
                raise ValueError(
                    f"Y has rows with all values missing at positions {bad_rows}. "
                    "Drop these observations or impute them before fitting."
                )

        # Preprocess each X-block and Y independently
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        self.y_preproc_ = MCUVScaler().fit(y)
        x_blocks_pp: dict[str, np.ndarray] = {
            name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_
        }
        y_pp = self.y_preproc_.transform(y).values.astype(float)

        # Algorithmic block weighting: X_b / sqrt(K_b)
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        # Storage (numpy arrays during fit; wrapped in pandas at the end)
        super_scores_np = np.zeros((n_samples, n_components))
        super_y_scores_np = np.zeros((n_samples, n_components))
        super_weights_np = np.zeros((n_blocks, n_components))
        super_y_loadings_np = np.zeros((self.n_targets_, n_components))
        block_scores_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }
        block_weights_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        block_loadings_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }

        x_def: dict[str, np.ndarray] = {name: x_blocks_pp[name].copy() for name in self.block_names_}
        y_def = y_pp.copy()

        # Initial sums of squares (for R^2 bookkeeping)
        ssq_x_init = {name: float(np.nansum(x_blocks_pp[name] ** 2)) for name in self.block_names_}
        ssq_y_init = float(np.nansum(y_pp ** 2))
        ssq_x_init_per_var = {
            name: np.nansum(x_blocks_pp[name] ** 2, axis=0) for name in self.block_names_
        }
        ssq_y_init_per_var = np.nansum(y_pp ** 2, axis=0)

        # Per-component cumulative R^2 storage (filled inside the loop)
        r2_x_block_cum = np.zeros((n_blocks, n_components))
        r2_x_var_cum: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        r2_y_cum = np.zeros(n_components)
        r2_y_var_cum = np.zeros((self.n_targets_, n_components))
        block_spe_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }

        tol = float(np.finfo(float).eps ** (6 / 7)) if self.tol is None else float(self.tol)
        timing = np.zeros(n_components)
        iterations = np.zeros(n_components, dtype=int)
        rng = np.random.default_rng(0)

        for a in range(n_components):
            start = time.time()
            u_a = rng.standard_normal(n_samples)
            prev = u_a * 2
            local_w: dict[str, np.ndarray] = {}
            local_t: dict[str, np.ndarray] = {}
            t_b_summary = np.zeros((n_samples, n_blocks))
            t_super = np.zeros(n_samples)
            w_s = np.zeros(n_blocks)
            c_a = np.zeros(self.n_targets_)
            itern = 0
            while np.linalg.norm(prev - u_a) > tol and itern < self.max_iter:
                prev = u_a
                if algo == "nipals":
                    # Mask-aware NIPALS: each projection is a per-column (or
                    # per-row) regression that uses only the entries that
                    # are not NaN, and divides by the masked sum of squares.
                    # Reuses the same primitives as single-block PCA NIPALS.
                    u_a_col = u_a.reshape(-1, 1)
                    for b_idx, name in enumerate(self.block_names_):
                        w_b = quick_regress(x_def[name], u_a_col).flatten()
                        w_b = w_b / _nz(np.sqrt(ssq(w_b.reshape(-1, 1))))
                        t_b = quick_regress(x_def[name], w_b.reshape(-1, 1)).flatten() / sqrt_kb[name]
                        local_w[name] = w_b
                        local_t[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                else:
                    for b_idx, name in enumerate(self.block_names_):
                        w_b = x_def[name].T @ u_a / _nz(u_a @ u_a)
                        w_b = w_b / _nz(np.linalg.norm(w_b))
                        t_b = x_def[name] @ w_b / _nz(w_b @ w_b) / sqrt_kb[name]
                        local_w[name] = w_b
                        local_t[name] = t_b
                        t_b_summary[:, b_idx] = t_b

                w_s = t_b_summary.T @ u_a / _nz(u_a @ u_a)
                w_s = w_s / _nz(np.linalg.norm(w_s))
                t_super = t_b_summary @ w_s / _nz(w_s @ w_s)
                if algo == "nipals":
                    t_super_col = t_super.reshape(-1, 1)
                    c_a = quick_regress(y_def, t_super_col).flatten()
                    u_a = quick_regress(y_def, c_a.reshape(-1, 1)).flatten()
                else:
                    c_a = y_def.T @ t_super / _nz(t_super @ t_super)
                    u_a = y_def @ c_a / _nz(c_a @ c_a)
                itern += 1

            # Sign convention: largest |w_super| element positive
            flip_idx = int(np.argmax(np.abs(w_s)))
            if w_s[flip_idx] < 0:
                w_s = -w_s
                t_super = -t_super
                u_a = -u_a
                c_a = -c_a
                for name in self.block_names_:
                    local_w[name] = -local_w[name]
                    local_t[name] = -local_t[name]

            # Deflate using the super-score
            t_super_col = t_super.reshape(-1, 1)
            for name in self.block_names_:
                if algo == "nipals":
                    p_b = quick_regress(x_def[name], t_super_col).flatten()
                else:
                    p_b = x_def[name].T @ t_super / _nz(t_super @ t_super)
                x_def[name] = x_def[name] - np.outer(t_super, p_b)
                block_loadings_np[name][:, a] = p_b
                block_weights_np[name][:, a] = local_w[name]
                block_scores_np[name][:, a] = local_t[name]
            y_def = y_def - np.outer(t_super, c_a)

            super_scores_np[:, a] = t_super
            super_y_scores_np[:, a] = u_a
            super_weights_np[:, a] = w_s
            super_y_loadings_np[:, a] = c_a

            # Track per-block cumulative R^2_X and per-Y-variable cumulative R^2_Y
            for b_idx, name in enumerate(self.block_names_):
                ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
                # R^2 is undefined for a zero-variance block/column; report NaN
                # rather than dividing by zero (inf/nan + warning) or returning a
                # misleading 1.0.
                r2_x_block_cum[b_idx, a] = (
                    1 - np.sum(ssq_remain_per_var) / ssq_x_init[name] if ssq_x_init[name] > 0 else np.nan
                )
                r2_x_var_cum[name][:, a] = np.where(
                    ssq_x_init_per_var[name] > 0,
                    1 - ssq_remain_per_var / np.where(ssq_x_init_per_var[name] > 0, ssq_x_init_per_var[name], 1.0),
                    np.nan,
                )
                block_spe_np[name][:, a] = np.sqrt(np.nansum(x_def[name] ** 2, axis=1))
            ssq_y_remain_per_var = np.nansum(y_def ** 2, axis=0)
            r2_y_cum[a] = 1 - np.sum(ssq_y_remain_per_var) / ssq_y_init if ssq_y_init > 0 else np.nan
            r2_y_var_cum[:, a] = np.where(
                ssq_y_init_per_var > 0,
                1 - ssq_y_remain_per_var / np.where(ssq_y_init_per_var > 0, ssq_y_init_per_var, 1.0),
                np.nan,
            )

            timing[a] = time.time() - start
            iterations[a] = itern

        component_names = list(range(1, n_components + 1))
        self.super_scores_ = pd.DataFrame(super_scores_np, index=self._sample_index, columns=component_names)
        self.super_y_scores_ = pd.DataFrame(super_y_scores_np, index=self._sample_index, columns=component_names)
        self.super_weights_ = pd.DataFrame(super_weights_np, index=self.block_names_, columns=component_names)
        self.super_y_loadings_ = pd.DataFrame(super_y_loadings_np, index=self._y_columns, columns=component_names)

        self.block_scores_ = {
            name: pd.DataFrame(block_scores_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_weights_ = {
            name: pd.DataFrame(
                block_weights_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(
                block_loadings_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }

        # In-sample predictions on the original Y scale
        y_hat_pp = super_scores_np @ super_y_loadings_np.T
        y_hat = self.y_preproc_.inverse_transform(pd.DataFrame(y_hat_pp, columns=self._y_columns))
        y_hat.index = self._sample_index
        self.predictions_ = y_hat

        self.explained_variance_ = np.diag(super_scores_np.T @ super_scores_np) / max(1, n_samples - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )
        converged = iterations < self.max_iter
        self.fitting_info_ = {"timing": timing, "iterations": iterations, "converged": converged}
        if not np.all(converged):
            failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
            warnings.warn(
                f"MBPLS NIPALS did not converge within max_iter={self.max_iter} for "
                f"component(s) {failed}; results for those components may be unreliable.",
                SpecificationWarning,
                stacklevel=2,
            )

        # --- Per-component (incremental) R^2 from cumulative ---
        r2_x_block_per_a = np.zeros_like(r2_x_block_cum)
        r2_x_block_per_a[:, 0] = r2_x_block_cum[:, 0]
        if n_components > 1:
            r2_x_block_per_a[:, 1:] = np.diff(r2_x_block_cum, axis=1)
        r2_y_per_a = np.empty_like(r2_y_cum)
        r2_y_per_a[0] = r2_y_cum[0]
        if n_components > 1:
            r2_y_per_a[1:] = np.diff(r2_y_cum)

        self.r2_x_per_block_cumulative_ = pd.DataFrame(
            r2_x_block_cum, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_block_per_component_ = pd.DataFrame(
            r2_x_block_per_a, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_variable_ = {
            name: pd.DataFrame(r2_x_var_cum[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }
        self.r2_y_cumulative_ = pd.Series(r2_y_cum, index=component_names, name="Cumulative R²Y")
        self.r2_y_per_component_ = pd.Series(r2_y_per_a, index=component_names, name="R²Y per component")
        self.r2_y_per_variable_ = pd.DataFrame(r2_y_var_cum, index=self._y_columns, columns=component_names)

        # --- Per-block VIP and super VIP ---
        # Per-block VIP_jb = sqrt(K_b * sum_a(r2_x_block_a * w_b[j,a]^2) / sum_a r2_x_block_a)
        self.block_vip_: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            r2 = r2_x_block_per_a[b_idx, :]
            if np.sum(r2) > 0:
                w = self.block_weights_[name].values  # (K_b, A)
                vip_b = np.sqrt(self.block_widths_[name] * np.sum(r2 * w**2, axis=1) / np.sum(r2))
            else:
                vip_b = np.zeros(self.block_widths_[name])
            self.block_vip_[name] = pd.Series(vip_b, index=self._block_columns[name], name=f"VIP[{name}]")

        # Super VIP_b = sqrt(B * sum_a(r2_y_a * w_super[b,a]^2) / sum_a r2_y_a)
        if np.sum(r2_y_per_a) > 0:
            ws = self.super_weights_.values  # (B, A)
            super_vip = np.sqrt(n_blocks * np.sum(r2_y_per_a * ws**2, axis=1) / np.sum(r2_y_per_a))
        else:
            super_vip = np.zeros(n_blocks)
        self.super_vip_ = pd.Series(super_vip, index=self.block_names_, name="Super VIP")

        # --- Per-block SPE (already accumulated) and per-block / super Hotelling's T^2 ---
        self.block_spe_ = {
            name: pd.DataFrame(block_spe_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }

        # Cumulative T^2 from block scores and super scores (using per-component score variance)
        block_t2: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            scores_np = self.block_scores_[name].values  # (N, A)
            score_var = np.var(scores_np, axis=0, ddof=1)
            score_var = np.where(score_var > 0, score_var, 1.0)
            t2 = np.cumsum((scores_np**2) / score_var, axis=1)
            block_t2[name] = t2
        self.block_hotellings_t2_ = {
            name: pd.DataFrame(block_t2[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        super_t2 = np.cumsum((super_scores_np**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

        # --- Convenience method bindings ---
        self.hotellings_t2_limit = partial(hotellings_t2_limit, n_components=n_components, n_rows=n_samples)

        return self

    def block_spe_limit(self, block: str, conf_level: float = 0.95) -> float:
        """SPE limit for one X-block using the Nomikos & MacGregor chi-square approximation.

        Operates on the same scale as ``block_spe_[block]`` (sqrt of row sum
        of squares), so the value can be drawn directly on a SPE plot.
        """
        check_is_fitted(self, "block_spe_")
        if block not in self.block_spe_:
            raise KeyError(f"Unknown block '{block}'. Known blocks: {list(self.block_spe_)}.")
        return spe_calculation(self.block_spe_[block].iloc[:, -1].values, conf_level=conf_level)

    def super_spe_limit(self, conf_level: float = 0.95) -> float:
        """SPE limit for the merged super-block (sum of per-block SPE squared)."""
        check_is_fitted(self, "block_spe_")
        merged_spe_squared = np.zeros(self.n_samples_)
        for name in self.block_names_:
            merged_spe_squared += self.block_spe_[name].iloc[:, -1].values ** 2
        return spe_calculation(np.sqrt(merged_spe_squared), conf_level=conf_level)

    def spe_contributions(self, X: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Per-variable squared residuals for each X-block (SPE contributions).

        For each new observation and each X-block, reconstruct the block as
        ``T_super @ P_b^T`` (matching the deflation step used during fit) and
        return the squared per-variable residuals. Useful for fault diagnosis:
        the variable with the largest contribution is the most likely culprit
        for a high SPE.

        Returns
        -------
        dict[str, pd.DataFrame]
            One DataFrame per block, shape ``(n_samples, K_b)``. Values are
            preprocessed-scale squared residuals (centred and scaled inside
            the model). Sum across columns equals ``block_spe_[b].iloc[:, -1] ** 2``.
        """
        check_is_fitted(self, "block_loadings_")
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks: {sorted(missing)}.")

        result = self._project(X)
        super_scores = result.super_scores.values  # (N, A)
        out: dict[str, pd.DataFrame] = {}
        sample_index = next(iter(result.block_scores.values())).index
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            x_pp = self.preproc_[name].transform(block).values.astype(float)
            x_hat = super_scores @ self.block_loadings_[name].values.T
            residuals_sq = (x_pp - x_hat) ** 2
            out[name] = pd.DataFrame(residuals_sq, index=sample_index, columns=self._block_columns[name])
        return out

    def score_contributions(
        self,
        t_super_start: np.ndarray | pd.Series,
        t_super_end: np.ndarray | pd.Series | None = None,
        components: list[int] | None = None,
        *,
        weighted: bool = False,
    ) -> dict[str, pd.Series]:
        r"""Per-block per-variable contributions to a super-score movement.

        The multi-block analogue of :meth:`PCA.score_contributions`. Decomposes
        a super-score-space delta back into preprocessed-scale variable-space
        deltas, one per X-block.

        For MBPLS, the super-score at component *a* is built from the per-block
        scores (which themselves are weighted regressions of the block on
        ``w_b``), so the back-projection through ``w_super[b,a] * w_b[:,a] /
        sqrt(K_b)`` gives the variable contribution.

        Parameters
        ----------
        t_super_start : array-like, shape (n_components,)
            Super-score row of the observation of interest. Typically a row
            from ``self.super_scores_`` or from ``predict(X_new).super_scores``.
        t_super_end : array-like, optional
            Reference point in super-score space. Defaults to the model
            centre (zeros).
        components : list of int, optional
            **1-based** component indices to decompose over. ``None`` (default)
            uses all components - appropriate for Hotelling's T² contributions.
        weighted : bool, default=False
            If ``True``, divide the super-score delta by
            ``sqrt(explained_variance_)`` per component before back-projecting,
            giving contributions to the T² statistic instead of the Euclidean
            super-score distance.

        Returns
        -------
        dict[str, pd.Series]
            One Series per X-block (length ``K_b``), indexed by variable
            (column) name.
        """
        check_is_fitted(self, "block_weights_")
        t_start = np.asarray(t_super_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_super_end is None else np.asarray(t_super_end, dtype=float)
        idx = np.arange(self.n_components) if components is None else np.array(components) - 1
        dt = t_end[idx] - t_start[idx]  # (len(idx),)
        if weighted:
            # ``explained_variance_[a] == 0`` for a degenerate component
            # would silently produce inf/NaN weighted contributions.
            # Clamp the divisor so weighting is a no-op on such components.
            # SEC-21 (#270) sub-item 5.
            ev = np.asarray(self.explained_variance_)[idx]
            dt = dt / np.sqrt(np.where(ev > epsqrt, ev, 1.0))

        out: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            sqrt_kb = float(np.sqrt(self.block_widths_[name]))
            ws = self.super_weights_.values[b_idx, idx]  # (len(idx),)
            wb = self.block_weights_[name].values[:, idx]  # (K_b, len(idx))
            # Effective per-component contribution per variable: w_super[b] * w_b[:,a] / sqrt(K_b)
            effective = wb * (ws / sqrt_kb)  # (K_b, len(idx))
            contrib = effective @ dt  # (K_b,)
            out[name] = pd.Series(contrib, index=self._block_columns[name], name=f"score_contributions[{name}]")
        return out

    def super_score_plot(self, pc_horiz: int = 1, pc_vert: int = 2) -> go.Figure:
        """Scatter plot of super-scores for two components."""
        check_is_fitted(self, "super_scores_")
        a_max = int(self.n_components)
        if not (1 <= pc_horiz <= a_max and 1 <= pc_vert <= a_max):
            raise ValueError(f"pc_horiz and pc_vert must be in 1..{a_max}.")
        x = self.super_scores_[pc_horiz].values
        y = self.super_scores_[pc_vert].values
        labels = [str(i) for i in self.super_scores_.index]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    name="Super-scores",
                )
            ]
        )
        fig.update_layout(
            xaxis_title=f"t_super[{pc_horiz}]",
            yaxis_title=f"t_super[{pc_vert}]",
            title=f"MBPLS super-score plot: PC{pc_horiz} vs PC{pc_vert}",
        )
        return fig

    def super_weights_bar_plot(self, component: int = 1) -> go.Figure:
        """Bar plot of super-weights ``w_super`` for a single component."""
        check_is_fitted(self, "super_weights_")
        a_max = int(self.n_components)
        if not (1 <= component <= a_max):
            raise ValueError(f"component must be in 1..{a_max}.")
        weights = self.super_weights_[component]
        fig = go.Figure(data=[go.Bar(x=list(weights.index), y=weights.values, name=f"w_super[{component}]")])
        fig.update_layout(
            xaxis_title="Block",
            yaxis_title=f"w_super[{component}]",
            title=f"MBPLS super-weights, component {component}",
        )
        return fig

    def predictions_vs_observed_plot(self, y_observed: pd.DataFrame, variable: str | None = None) -> go.Figure:
        """Scatter plot of predicted vs observed Y, with y=x reference and RMSEE annotation.

        Parameters
        ----------
        y_observed : pd.DataFrame
            The observed Y on the original scale, same columns as the training Y.
        variable : str or None, default=None
            If given, plot only that Y-variable. If ``None``, plot the first one.
        """
        check_is_fitted(self, "predictions_")
        if variable is None:
            variable = str(self.predictions_.columns[0])
        if variable not in self.predictions_.columns:
            raise ValueError(f"Unknown Y-variable '{variable}'. Known: {list(self.predictions_.columns)}.")
        observed = pd.Series(y_observed[variable].values, name="observed").reset_index(drop=True)
        predicted = pd.Series(self.predictions_[variable].values, name="predicted").reset_index(drop=True)
        rmsee = float(np.sqrt(np.mean((observed.values - predicted.values) ** 2)))
        lo = float(min(observed.min(), predicted.min()))
        hi = float(max(observed.max(), predicted.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        fig = go.Figure(
            data=[
                go.Scatter(x=observed, y=predicted, mode="markers", name="Predicted vs observed"),
                go.Scatter(
                    x=[lo - pad, hi + pad],
                    y=[lo - pad, hi + pad],
                    mode="lines",
                    line={"color": REFERENCE_LINE_COLOR, "dash": "dash"},
                    name="y = x",
                ),
            ]
        )
        fig.add_annotation(
            x=lo + 0.05 * (hi - lo),
            y=hi - 0.05 * (hi - lo),
            text=f"RMSEE = {rmsee:.4g}",
            showarrow=False,
        )
        fig.update_layout(
            xaxis_title=f"Observed: {variable}",
            yaxis_title=f"Predicted: {variable}",
            title=f"Predicted vs observed for {variable}",
        )
        return fig

    def display_results(self, show_cumulative: bool = True) -> str:
        """Format a short text summary of per-block R²X, overall R²Y, iterations and timing."""
        check_is_fitted(self, "super_scores_")
        rows: list[str] = []
        rows.append(f"MBPLS model: {self.n_components} component(s), {len(self.block_names_)} X-block(s)")
        header = "  PC | " + " | ".join(f"R²X[{name}]" for name in self.block_names_) + " | R²Y"
        rows.append(header)
        rows.append("-" * len(header))
        for a in range(self.n_components):
            cells = [f"{i:>3d}" for i in [a + 1]]
            for name in self.block_names_:
                src = self.r2_x_per_block_cumulative_ if show_cumulative else self.r2_x_per_block_per_component_
                cells.append(f"{src.loc[name].iloc[a]:>9.4f}")
            r2y_src = self.r2_y_cumulative_ if show_cumulative else self.r2_y_per_component_
            cells.append(f"{r2y_src.iloc[a]:>9.4f}")
            rows.append(" | ".join(cells))
        rows.append("")
        rows.append(f"  Iterations per PC: {list(self.fitting_info_['iterations'])}")
        rows.append(f"  Time per PC (ms):  {[round(float(t * 1000), 1) for t in self.fitting_info_['timing']]}")
        return "\n".join(rows)

    def transform(self, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Project new data to super-scores using the fitted model."""
        check_is_fitted(self, "super_weights_")
        return self._project(X).super_scores

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data and predict Y on the original scale.

        Returns a :class:`sklearn.utils.Bunch` with fields ``super_scores``,
        ``block_scores`` (dict[str, DataFrame]) and ``predictions`` (DataFrame
        on original Y scale).
        """
        check_is_fitted(self, "super_weights_")
        return self._project(X)

    def _project(self, X: dict[str, pd.DataFrame]) -> Bunch:
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks for prediction: {sorted(missing)}.")

        # Preprocess each block
        x_pp: dict[str, np.ndarray] = {}
        sample_index = None
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            if block.shape[1] != self.block_widths_[name]:
                raise ValueError(
                    f"Block '{name}' must have {self.block_widths_[name]} columns; got {block.shape[1]}."
                )
            x_pp[name] = self.preproc_[name].transform(block).values.astype(float)
            if sample_index is None:
                sample_index = block.index

        n_components = int(self.n_components)
        n_new = next(iter(x_pp.values())).shape[0]
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        super_scores = np.zeros((n_new, n_components))
        block_scores: dict[str, np.ndarray] = {
            name: np.zeros((n_new, n_components)) for name in self.block_names_
        }

        x_def = {name: x_pp[name].copy() for name in self.block_names_}
        for a in range(n_components):
            t_b_row = np.zeros((n_new, len(self.block_names_)))
            for b_idx, name in enumerate(self.block_names_):
                w_b = self.block_weights_[name].values[:, a]
                t_b = x_def[name] @ w_b / sqrt_kb[name]
                block_scores[name][:, a] = t_b
                t_b_row[:, b_idx] = t_b
            w_s = self.super_weights_.values[:, a]
            t_super = t_b_row @ w_s
            super_scores[:, a] = t_super
            for name in self.block_names_:
                p_b = self.block_loadings_[name].values[:, a]
                x_def[name] = x_def[name] - np.outer(t_super, p_b)

        component_names = list(range(1, n_components + 1))
        super_scores_df = pd.DataFrame(super_scores, index=sample_index, columns=component_names)
        block_scores_df = {
            name: pd.DataFrame(block_scores[name], index=sample_index, columns=component_names)
            for name in self.block_names_
        }
        y_hat_pp = super_scores @ self.super_y_loadings_.values.T
        predictions = self.y_preproc_.inverse_transform(pd.DataFrame(y_hat_pp, columns=self._y_columns))
        predictions.index = sample_index

        # Per-block SPE for new observations (residual after final deflation)
        block_spe = {
            name: pd.Series(
                np.sqrt(np.nansum(x_def[name] ** 2, axis=1)), index=sample_index, name=f"SPE[{name}]"
            )
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        hotellings_t2 = pd.Series(
            np.sum((super_scores**2) / super_score_var, axis=1), index=sample_index, name="Hotelling's T²"
        )

        return Bunch(
            super_scores=super_scores_df,
            block_scores=block_scores_df,
            predictions=predictions,
            block_spe=block_spe,
            hotellings_t2=hotellings_t2,
        )


def randomization_test_mbpls(
    model: MBPLS,
    X: dict[str, pd.DataFrame],
    y: pd.DataFrame,
    n_permutations: int = 200,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    r"""Randomization (permutation) test for component significance in MBPLS.

    For each component ``a``, the null hypothesis is "there is no real
    relationship between X and Y at this component"; the test permutes the
    rows of ``y``, refits a fresh MBPLS with the same number of components,
    and recomputes the test statistic. The risk is the fraction of
    permutations whose statistic equals or exceeds the original model's.

    Statistic: per-component absolute correlation between the super X-score
    and the super Y-score, ``|t_super(:,a)' u_super(:,a)| / (||t|| * ||u||)``,
    matching the legacy ConnectMV randomization-objective for PLS.

    Parameters
    ----------
    model : MBPLS
        A fitted MBPLS model.
    X, y : dict[str, DataFrame], DataFrame
        The same training data used to fit ``model``.
    n_permutations : int, default=200
        Number of Y-row permutations to evaluate.
    seed : int or None, default=None
        Seed for the permutation RNG (``None`` uses non-reproducible
        randomness).

    Returns
    -------
    pd.DataFrame
        Indexed by component ``1..A`` with columns:

        - ``observed`` : the actual model's per-component statistic.
        - ``risk_pct`` : fraction (in %) of permutations with statistic
          ``>= observed``. Low values (e.g. < 5%) suggest the component is
          significant; values near 50% suggest the component is no better
          than chance.

    References
    ----------
    Wiklund, S., Nilsson, D., Eriksson, L., Sjöström, M., Wold, S. &
    Faber, K. *A randomization test for PLS component selection.* J.
    Chemometrics, 21 (2007), 427-439.
    """
    check_is_fitted(model, "super_scores_")
    rng = np.random.default_rng(seed)
    a_components = int(model.n_components)

    def _objective(mod: MBPLS) -> np.ndarray:
        t = mod.super_scores_.values
        u = mod.super_y_scores_.values
        out = np.zeros(t.shape[1])
        for a in range(t.shape[1]):
            num = float(np.abs(t[:, a] @ u[:, a]))
            denom = float(np.linalg.norm(t[:, a]) * np.linalg.norm(u[:, a]))
            # SEC-33 (#282): float ``==`` zero only catches the exact-zero
            # case; a sub-eps near-zero denom produced a meaningless ratio
            # that the permutation test treated as an observed statistic.
            out[a] = 0.0 if denom <= epsqrt else num / denom
        return out

    observed = _objective(model)
    n_exceed = np.zeros(a_components, dtype=int)
    n_samples = y.shape[0]
    for _ in range(int(n_permutations)):
        perm_idx = rng.permutation(n_samples)
        y_perm = y.iloc[perm_idx].reset_index(drop=True)
        # Reset X indices to align row positions (otherwise pandas will
        # join on index and silently misalign).
        x_reset = {name: X[name].reset_index(drop=True) for name in X}
        permuted_model = MBPLS(n_components=a_components).fit(x_reset, y_perm)
        stat = _objective(permuted_model)
        n_exceed += (stat >= observed).astype(int)

    component_names = list(range(1, a_components + 1))
    risk_pct = 100.0 * n_exceed / n_permutations
    return pd.DataFrame(
        {"observed": observed, "risk_pct": risk_pct},
        index=pd.Index(component_names, name="component"),
    )


class MBPCA(TransformerMixin, BaseEstimator):
    r"""Multi-block PCA (hierarchical / consensus PCA).

    Generic multi-block PCA following the consensus-PCA / hierarchical PCA
    formulation of Westerhuis, Kourti & MacGregor (1998). Each X-block is
    preprocessed independently (mean-centred and unit-variance scaled),
    then divided by ``sqrt(K_b)`` so blocks of unequal width contribute
    fairly to the consensus super-score.

    The hierarchical NIPALS loop alternates: (i) regress each block on the
    super-score to get block loadings and block scores, (ii) collect block
    scores into a super-block, (iii) regress the super-block to get a new
    super-score / super-loading, repeat to convergence. After convergence,
    deflate every block by the super-score and the corresponding block
    loading scaled by the super-loading element.

    Parameters
    ----------
    n_components : int
    max_iter : int, default=500
    tol : float or None, default=None
        Convergence tolerance on the super-score change. ``None`` uses
        ``np.finfo(float).eps ** (9/10)`` (matches the legacy reference).
    algorithm : str, default="auto"
        Algorithm to use for fitting the model.

        - ``"auto"``: dense vectorised hierarchical NIPALS when the data is
          complete; mask-aware NIPALS (NaN-tolerant) when any block contains
          missing values.
        - ``"dense"``: dense vectorised hierarchical NIPALS. Raises if any
          block contains missing values.
        - ``"nipals"``: mask-aware hierarchical NIPALS. Always uses the
          NaN-tolerant inner-loop primitives, even when the data is
          complete (slower than ``"dense"`` but produces equivalent
          results).

    missing_data_settings : dict or None, default=None
        Settings for the iterative ``"nipals"`` path. Keys: ``md_tol``
        (convergence tolerance on the score-vector change between
        iterations), ``md_max_iter`` (maximum NIPALS iterations per
        component). Defaults to ``{"md_tol": epsqrt, "md_max_iter": 1000}``.

    Attributes (after fitting)
    --------------------------
    block_names_, block_widths_                       (as MBPLS)
    super_scores_                                     DataFrame (N x A)
    super_loadings_                                   DataFrame (B x A)
    block_scores_, block_loadings_                    dict[str, DataFrame]
    r2_x_per_block_cumulative_, r2_x_per_block_per_component_
    r2_x_per_variable_                                dict[str, DataFrame]
    block_vip_, super_vip_
    block_spe_, block_hotellings_t2_, super_hotellings_t2_
    explained_variance_, scaling_factor_for_super_scores_
    fitting_info_, has_missing_data_, algorithm_

    Notes
    -----
    The deflation step is :math:`X_b \leftarrow X_b - t_{\rm super}\,
    (p_b\,p_s[b]\,\sqrt{K_b})^\top`, derived in Westerhuis et al. 1998.
    The legacy MATLAB ``mbpca.m`` had this step marked as broken by the
    original author; this implementation re-derives it directly from the
    paper and is independently validated against the pure-numpy reference
    oracles in the test suite.

    Missing data
    ------------
    When any block contains NaN entries, the ``"auto"`` algorithm
    routes to a mask-aware NIPALS variant. Each per-block projection
    in the inner loop is computed as a regression that uses only the
    observed entries; the masked sum-of-squares is used as the
    denominator so missing values neither bias the loading direction
    nor contribute to the score. The mask is preserved across
    components automatically because deflation propagates NaN through
    subtraction. This is the standard skip-NaN NIPALS update; see
    Walczak & Massart (2001) and Arteaga & Ferrer (2002).

    The fit refuses to run if any block has a column with all entries
    missing, or any block has a row with all entries missing for that
    block; either case leaves the masked denominator at zero. Drop or
    impute such rows or columns before fitting. Predict-time score
    estimation for new observations with NaN (Trimmed Score Regression
    / Projection to the Model Plane) is a separate follow-up.

    References
    ----------
    Westerhuis, J. A., Kourti, T. & MacGregor, J. F. *Analysis of
    multiblock and hierarchical PCA and PLS models.* J. Chemometrics, 12
    (1998), 301-321.

    Walczak, B. & Massart, D. L. *Dealing with missing data: Part I.*
    Chemom. Intell. Lab. Syst., 58 (2001), 15-27.

    Arteaga, F. & Ferrer, A. *Dealing with missing data in MSPC: several
    methods, different interpretations, some examples.* J. Chemometrics,
    16 (2002), 408-418.
    """

    _valid_algorithms: typing.ClassVar[list[str]] = ["auto", "dense", "nipals"]

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
        "max_iter": [int],
        "tol": [float, None],
        "algorithm": [str],
        "missing_data_settings": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 500,
        tol: float | None = None,
        algorithm: str = "auto",
        missing_data_settings: dict | None = None,
    ):
        super().__init__()
        if n_components <= 0:
            raise ValueError(f"n_components must be positive; got {n_components}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {max_iter}.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.missing_data_settings = missing_data_settings

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, pd.DataFrame], y: None = None) -> MBPCA:  # noqa: ARG002, C901, PLR0912, PLR0915
        """Fit the multi-block PCA model."""
        if not isinstance(X, dict) or len(X) == 0:
            raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
        for name, block in X.items():
            if not isinstance(block, pd.DataFrame):
                raise TypeError(f"X['{name}'] must be a pandas DataFrame; got {type(block).__name__}.")

        self.block_names_: list[str] = list(X.keys())
        first = X[self.block_names_[0]]
        n_samples = first.shape[0]
        for name in self.block_names_:
            if X[name].shape[0] != n_samples:
                raise ValueError(
                    f"All X-blocks must have the same row count. Block '{name}' has "
                    f"{X[name].shape[0]} rows; expected {n_samples}."
                )

        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}

        self.n_samples_ = int(n_samples)
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        n_components = int(self.n_components)
        n_blocks = len(self.block_names_)

        self.has_missing_data_ = any(np.any(X[name].isna().values) for name in self.block_names_)
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognised. "
                f"Must be one of {self._valid_algorithms}."
            )
        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "dense"
        if algo == "dense" and self.has_missing_data_:
            raise ValueError(
                "Algorithm 'dense' cannot handle missing data. "
                "Use 'nipals' or 'auto' instead."
            )
        self.algorithm_ = algo

        # Resolve iterative-algorithm settings (used by the 'nipals' path).
        settings = {"md_tol": epsqrt, "md_max_iter": 1000}
        if isinstance(self.missing_data_settings, dict):
            settings.update(self.missing_data_settings)
        settings["md_max_iter"] = int(settings["md_max_iter"])
        if algo == "nipals":
            if not settings["md_tol"] < 10:
                raise ValueError("Tolerance should not be too large.")
            if not settings["md_tol"] > epsqrt**1.95:
                raise ValueError("Tolerance must exceed machine precision.")
            # Degeneracy guards: any column or any (block, row) entirely NaN
            # leaves the masked NIPALS denominator at zero, which would
            # silently produce a spurious score or loading. Refuse the fit
            # rather than coerce the user into a misleading result.
            for name in self.block_names_:
                values = X[name].values
                col_all_nan = np.all(np.isnan(values), axis=0)
                if np.any(col_all_nan):
                    bad = X[name].columns[col_all_nan].tolist()
                    raise ValueError(
                        f"Block '{name}' has columns with all values missing: {bad}. "
                        "Drop these columns before fitting."
                    )
                row_all_nan = np.all(np.isnan(values), axis=1)
                if np.any(row_all_nan):
                    bad_rows = np.where(row_all_nan)[0].tolist()
                    raise ValueError(
                        f"Block '{name}' has rows with all values missing at positions {bad_rows}. "
                        "Drop these observations or impute them before fitting."
                    )

        # Preprocess each block independently
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        x_blocks_pp: dict[str, np.ndarray] = {
            name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_
        }
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        # Working copies for deflation and stats accumulation
        x_def: dict[str, np.ndarray] = {name: x_blocks_pp[name].copy() for name in self.block_names_}
        ssq_x_init = {name: float(np.nansum(x_blocks_pp[name] ** 2)) for name in self.block_names_}
        ssq_x_init_per_var = {
            name: np.nansum(x_blocks_pp[name] ** 2, axis=0) for name in self.block_names_
        }

        super_scores_np = np.zeros((n_samples, n_components))
        super_loadings_np = np.zeros((n_blocks, n_components))
        block_scores_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }
        block_loadings_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        r2_x_block_cum = np.zeros((n_blocks, n_components))
        r2_x_var_cum: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        block_spe_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }

        tol = float(np.finfo(float).eps ** (9 / 10)) if self.tol is None else float(self.tol)
        timing = np.zeros(n_components)
        iterations = np.zeros(n_components, dtype=int)
        rng = np.random.default_rng(0)

        for a in range(n_components):
            start = time.time()
            t_super = rng.standard_normal(n_samples)
            prev = t_super * 2
            t_b_summary = np.zeros((n_samples, n_blocks))
            local_loadings: dict[str, np.ndarray] = {}
            local_scores: dict[str, np.ndarray] = {}
            p_s = np.zeros(n_blocks)
            itern = 0
            while np.linalg.norm(prev - t_super) > tol and itern < self.max_iter:
                prev = t_super
                if algo == "nipals":
                    # Mask-aware NIPALS: each projection is a per-column (or
                    # per-row) regression that uses only the entries that
                    # are not NaN, and divides by the masked sum of squares.
                    # Reuses the same primitives as single-block PCA NIPALS.
                    t_super_col = t_super.reshape(-1, 1)
                    for b_idx, name in enumerate(self.block_names_):
                        p_b = quick_regress(x_def[name], t_super_col).flatten()
                        p_b = p_b / _nz(np.sqrt(ssq(p_b.reshape(-1, 1))))
                        t_b = quick_regress(x_def[name], p_b.reshape(-1, 1)).flatten() / sqrt_kb[name]
                        local_loadings[name] = p_b
                        local_scores[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                else:
                    for b_idx, name in enumerate(self.block_names_):
                        p_b = x_def[name].T @ t_super / _nz(t_super @ t_super)
                        p_b = p_b / _nz(np.linalg.norm(p_b))
                        t_b = x_def[name] @ p_b / _nz(p_b @ p_b) / sqrt_kb[name]
                        local_loadings[name] = p_b
                        local_scores[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                p_s = t_b_summary.T @ t_super / _nz(t_super @ t_super)
                p_s = p_s / _nz(np.linalg.norm(p_s))
                t_super = t_b_summary @ p_s / _nz(p_s @ p_s)
                itern += 1

            # Sign convention: largest |super_loading| element positive
            flip_idx = int(np.argmax(np.abs(p_s)))
            if p_s[flip_idx] < 0:
                p_s = -p_s
                t_super = -t_super
                for name in self.block_names_:
                    local_loadings[name] = -local_loadings[name]
                    local_scores[name] = -local_scores[name]

            # Deflate each block using the super-score and block loading scaled by super-loading
            for b_idx, name in enumerate(self.block_names_):
                p_deflate = local_loadings[name] * p_s[b_idx] * sqrt_kb[name]
                x_def[name] = x_def[name] - np.outer(t_super, p_deflate)
                block_loadings_np[name][:, a] = local_loadings[name]
                block_scores_np[name][:, a] = local_scores[name]

            super_scores_np[:, a] = t_super
            super_loadings_np[:, a] = p_s

            # Per-block cumulative R²X and SPE
            for b_idx, name in enumerate(self.block_names_):
                ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
                # R^2 is undefined for a zero-variance block/column; report NaN
                # rather than dividing by zero (inf/nan + warning) or 1.0.
                r2_x_block_cum[b_idx, a] = (
                    1 - np.sum(ssq_remain_per_var) / ssq_x_init[name] if ssq_x_init[name] > 0 else np.nan
                )
                r2_x_var_cum[name][:, a] = np.where(
                    ssq_x_init_per_var[name] > 0,
                    1 - ssq_remain_per_var / np.where(ssq_x_init_per_var[name] > 0, ssq_x_init_per_var[name], 1.0),
                    np.nan,
                )
                block_spe_np[name][:, a] = np.sqrt(np.nansum(x_def[name] ** 2, axis=1))

            timing[a] = time.time() - start
            iterations[a] = itern

        # Wrap in pandas containers
        component_names = list(range(1, n_components + 1))
        self.super_scores_ = pd.DataFrame(super_scores_np, index=self._sample_index, columns=component_names)
        self.super_loadings_ = pd.DataFrame(
            super_loadings_np, index=self.block_names_, columns=component_names
        )
        self.block_scores_ = {
            name: pd.DataFrame(block_scores_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(
                block_loadings_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }

        self.explained_variance_ = np.diag(super_scores_np.T @ super_scores_np) / max(1, n_samples - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )
        converged = iterations < self.max_iter
        self.fitting_info_ = {"timing": timing, "iterations": iterations, "converged": converged}
        if not np.all(converged):
            failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
            warnings.warn(
                f"MBPCA NIPALS did not converge within max_iter={self.max_iter} for "
                f"component(s) {failed}; results for those components may be unreliable.",
                SpecificationWarning,
                stacklevel=2,
            )

        # Per-component (incremental) R²X
        r2_x_block_per_a = np.zeros_like(r2_x_block_cum)
        r2_x_block_per_a[:, 0] = r2_x_block_cum[:, 0]
        if n_components > 1:
            r2_x_block_per_a[:, 1:] = np.diff(r2_x_block_cum, axis=1)

        self.r2_x_per_block_cumulative_ = pd.DataFrame(
            r2_x_block_cum, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_block_per_component_ = pd.DataFrame(
            r2_x_block_per_a, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_variable_ = {
            name: pd.DataFrame(r2_x_var_cum[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }

        # VIPs (per-block: variance-of-X explanation; super: same on super-loadings)
        self.block_vip_: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            r2 = r2_x_block_per_a[b_idx, :]
            if np.sum(r2) > 0:
                p = self.block_loadings_[name].values
                vip_b = np.sqrt(self.block_widths_[name] * np.sum(r2 * p**2, axis=1) / np.sum(r2))
            else:
                vip_b = np.zeros(self.block_widths_[name])
            self.block_vip_[name] = pd.Series(vip_b, index=self._block_columns[name], name=f"VIP[{name}]")

        # Per-block SPE / T² and super T²
        self.block_spe_ = {
            name: pd.DataFrame(block_spe_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        block_t2: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            scores_np = self.block_scores_[name].values
            score_var = np.var(scores_np, axis=0, ddof=1)
            score_var = np.where(score_var > 0, score_var, 1.0)
            block_t2[name] = np.cumsum((scores_np**2) / score_var, axis=1)
        self.block_hotellings_t2_ = {
            name: pd.DataFrame(block_t2[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        super_t2 = np.cumsum((super_scores_np**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

        self.hotellings_t2_limit = partial(hotellings_t2_limit, n_components=n_components, n_rows=n_samples)
        return self

    def transform(self, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Project new data to super-scores using the fitted model."""
        check_is_fitted(self, "super_loadings_")
        return self._project(X).super_scores

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data; return super_scores, block_scores, block_spe, hotellings_t2."""
        check_is_fitted(self, "super_loadings_")
        return self._project(X)

    def _project(self, X: dict[str, pd.DataFrame]) -> Bunch:
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks for prediction: {sorted(missing)}.")

        x_pp: dict[str, np.ndarray] = {}
        sample_index = None
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            if block.shape[1] != self.block_widths_[name]:
                raise ValueError(
                    f"Block '{name}' must have {self.block_widths_[name]} columns; got {block.shape[1]}."
                )
            x_pp[name] = self.preproc_[name].transform(block).values.astype(float)
            if sample_index is None:
                sample_index = block.index

        n_components = int(self.n_components)
        n_new = next(iter(x_pp.values())).shape[0]
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        super_scores = np.zeros((n_new, n_components))
        block_scores: dict[str, np.ndarray] = {
            name: np.zeros((n_new, n_components)) for name in self.block_names_
        }
        x_def = {name: x_pp[name].copy() for name in self.block_names_}

        for a in range(n_components):
            t_b_row = np.zeros((n_new, len(self.block_names_)))
            for b_idx, name in enumerate(self.block_names_):
                p_b = self.block_loadings_[name].values[:, a]
                t_b = x_def[name] @ p_b / _nz(p_b @ p_b) / sqrt_kb[name]
                block_scores[name][:, a] = t_b
                t_b_row[:, b_idx] = t_b
            p_s = self.super_loadings_.values[:, a]
            t_super = t_b_row @ p_s / _nz(p_s @ p_s)
            super_scores[:, a] = t_super
            for b_idx, name in enumerate(self.block_names_):
                p_b = self.block_loadings_[name].values[:, a]
                p_deflate = p_b * p_s[b_idx] * sqrt_kb[name]
                x_def[name] = x_def[name] - np.outer(t_super, p_deflate)

        component_names = list(range(1, n_components + 1))
        super_scores_df = pd.DataFrame(super_scores, index=sample_index, columns=component_names)
        block_scores_df = {
            name: pd.DataFrame(block_scores[name], index=sample_index, columns=component_names)
            for name in self.block_names_
        }
        block_spe = {
            name: pd.Series(
                np.sqrt(np.nansum(x_def[name] ** 2, axis=1)), index=sample_index, name=f"SPE[{name}]"
            )
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        hotellings_t2 = pd.Series(
            np.sum((super_scores**2) / super_score_var, axis=1), index=sample_index, name="Hotelling's T²"
        )
        return Bunch(
            super_scores=super_scores_df,
            block_scores=block_scores_df,
            block_spe=block_spe,
            hotellings_t2=hotellings_t2,
        )

    def block_spe_limit(self, block: str, conf_level: float = 0.95) -> float:
        """SPE limit for one X-block (Nomikos & MacGregor chi-square approximation)."""
        check_is_fitted(self, "block_spe_")
        if block not in self.block_spe_:
            raise KeyError(f"Unknown block '{block}'. Known blocks: {list(self.block_spe_)}.")
        return spe_calculation(self.block_spe_[block].iloc[:, -1].values, conf_level=conf_level)

    def super_spe_limit(self, conf_level: float = 0.95) -> float:
        """SPE limit for the merged super-block."""
        check_is_fitted(self, "block_spe_")
        merged_spe_squared = np.zeros(self.n_samples_)
        for name in self.block_names_:
            merged_spe_squared += self.block_spe_[name].iloc[:, -1].values ** 2
        return spe_calculation(np.sqrt(merged_spe_squared), conf_level=conf_level)

    def spe_contributions(self, X: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Per-variable squared residuals for each X-block (SPE contributions).

        Reconstruction matches the MBPCA deflation step:
        ``X_b = T_super @ (P_b * p_super[b] * sqrt(K_b))^T`` summed over
        components. Returns squared residuals on the preprocessed scale; sum
        across columns equals ``block_spe_[b].iloc[:, -1] ** 2``.
        """
        check_is_fitted(self, "block_loadings_")
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks: {sorted(missing)}.")

        result = self._project(X)
        super_scores = result.super_scores.values  # (N, A)
        sample_index = next(iter(result.block_scores.values())).index
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        out: dict[str, pd.DataFrame] = {}
        for b_idx, name in enumerate(self.block_names_):
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            x_pp = self.preproc_[name].transform(block).values.astype(float)
            # X_b reconstruction summed over components
            p_b = self.block_loadings_[name].values  # (K_b, A)
            p_s = self.super_loadings_.values[b_idx, :]  # (A,)
            p_eff = p_b * p_s * sqrt_kb[name]  # (K_b, A) effective loading
            x_hat = super_scores @ p_eff.T
            residuals_sq = (x_pp - x_hat) ** 2
            out[name] = pd.DataFrame(residuals_sq, index=sample_index, columns=self._block_columns[name])
        return out

    def score_contributions(
        self,
        t_super_start: np.ndarray | pd.Series,
        t_super_end: np.ndarray | pd.Series | None = None,
        components: list[int] | None = None,
        *,
        weighted: bool = False,
    ) -> dict[str, pd.Series]:
        r"""Per-block per-variable contributions to a super-score movement (MBPCA).

        Decomposes a super-score-space delta into preprocessed-scale variable
        contributions per X-block. The MBPCA back-projection mirrors the
        deflation step used during fit:

        .. math::

            \text{contrib}_{b,j} = \sum_a (\Delta t_\mathrm{super}[a]) \cdot
            P_b[j, a] \cdot p_\mathrm{super}[b, a] \cdot \sqrt{K_b}

        See :meth:`MBPLS.score_contributions` for the parameter and return
        descriptions; the API is identical.
        """
        check_is_fitted(self, "block_loadings_")
        t_start = np.asarray(t_super_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_super_end is None else np.asarray(t_super_end, dtype=float)
        idx = np.arange(self.n_components) if components is None else np.array(components) - 1
        dt = t_end[idx] - t_start[idx]
        if weighted:
            # ``explained_variance_[a] == 0`` for a degenerate component
            # would silently produce inf/NaN weighted contributions.
            # Clamp the divisor so weighting is a no-op on such components.
            # SEC-21 (#270) sub-item 5.
            ev = np.asarray(self.explained_variance_)[idx]
            dt = dt / np.sqrt(np.where(ev > epsqrt, ev, 1.0))

        out: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            sqrt_kb = float(np.sqrt(self.block_widths_[name]))
            ps = self.super_loadings_.values[b_idx, idx]  # (len(idx),)
            pb = self.block_loadings_[name].values[:, idx]  # (K_b, len(idx))
            effective = pb * (ps * sqrt_kb)  # (K_b, len(idx))
            contrib = effective @ dt  # (K_b,)
            out[name] = pd.Series(contrib, index=self._block_columns[name], name=f"score_contributions[{name}]")
        return out

    def super_score_plot(self, pc_horiz: int = 1, pc_vert: int = 2) -> go.Figure:
        """Scatter plot of MBPCA super-scores for two components."""
        check_is_fitted(self, "super_scores_")
        a_max = int(self.n_components)
        if not (1 <= pc_horiz <= a_max and 1 <= pc_vert <= a_max):
            raise ValueError(f"pc_horiz and pc_vert must be in 1..{a_max}.")
        x = self.super_scores_[pc_horiz].values
        y = self.super_scores_[pc_vert].values
        labels = [str(i) for i in self.super_scores_.index]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    name="Super-scores",
                )
            ]
        )
        fig.update_layout(
            xaxis_title=f"t_super[{pc_horiz}]",
            yaxis_title=f"t_super[{pc_vert}]",
            title=f"MBPCA super-score plot: PC{pc_horiz} vs PC{pc_vert}",
        )
        return fig

    def super_loadings_bar_plot(self, component: int = 1) -> go.Figure:
        """Bar plot of MBPCA super-loadings for a single component."""
        check_is_fitted(self, "super_loadings_")
        a_max = int(self.n_components)
        if not (1 <= component <= a_max):
            raise ValueError(f"component must be in 1..{a_max}.")
        loadings = self.super_loadings_[component]
        fig = go.Figure(data=[go.Bar(x=list(loadings.index), y=loadings.values, name=f"p_super[{component}]")])
        fig.update_layout(
            xaxis_title="Block",
            yaxis_title=f"p_super[{component}]",
            title=f"MBPCA super-loadings, component {component}",
        )
        return fig

    def display_results(self, show_cumulative: bool = True) -> str:
        """Format a short text summary of per-block R²X, iterations and timing."""
        check_is_fitted(self, "super_scores_")
        rows: list[str] = [f"MBPCA model: {self.n_components} component(s), {len(self.block_names_)} X-block(s)"]
        header = "  PC | " + " | ".join(f"R²X[{name}]" for name in self.block_names_)
        rows.append(header)
        rows.append("-" * len(header))
        src = self.r2_x_per_block_cumulative_ if show_cumulative else self.r2_x_per_block_per_component_
        for a in range(self.n_components):
            cells = [f"{a + 1:>3d}"]
            cells.extend(f"{src.loc[name].iloc[a]:>9.4f}" for name in self.block_names_)
            rows.append(" | ".join(cells))
        rows.append("")
        rows.append(f"  Iterations per PC: {list(self.fitting_info_['iterations'])}")
        rows.append(f"  Time per PC (ms):  {[round(float(t * 1000), 1) for t in self.fitting_info_['timing']]}")
        return "\n".join(rows)


class Plot:
    """Create plots of estimators."""

    def __init__(self, parent: BaseEstimator) -> None:
        self._parent = parent

    def scores(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a score plot."""
        return score_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)

    def loadings(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:
        """Generate a loading plot."""
        return loading_plot(self, pc_horiz=pc_horiz, pc_vert=pc_vert, **kwargs)


class Resampler:
    """Base class for resampling methods."""

    def __init__(  # noqa: PLR0913
        self,
        estimator: BaseEstimator,
        x: DataFrameDict,
        accessor: Callable,
        use_jackknife: bool = True,
        bootstrap_rounds: int = 0,
        fraction_excluded: float = 0.0,
        random_state: int | np.random.Generator | None = None,
    ):
        """Initialize the resampling method.

        The `accessor` is a callable that takes an estimator and returns the parameters of interest.

        Mutually exclusive parameters:
            * `use_jackknife` flag indicates whether to use jackknife resampling (leave out one sample; rebuild)
            * `bootstrap_rounds` specifies the number of bootstrap rounds if applicable (resample data with replacement)
            * `fraction_excluded` specifies the fraction of data to exclude in each resample (for fractional resampling)

        Only one of these parameters should be set at a time.

        Parameters
        ----------
        random_state : int, np.random.Generator, or None, optional
            Seeds the RNG used by ``bootstrap()`` and ``fractional()``;
            see ``docs/development/reproducibility.rst`` (ENG-08). Pass
            the same int twice to get bit-identical resamples; pass
            ``None`` for fresh entropy on each call. ``jackknife()``
            is deterministic and ignores this parameter.
        """
        if not isinstance(estimator, BaseEstimator):
            raise TypeError("estimator must be a BaseEstimator instance.")
        self.estimator = estimator

        if not isinstance(x, DataFrameDict):
            raise TypeError("x must be a DataFrameDict instance.")
        self.x = x

        if not callable(accessor):
            raise TypeError("accessor must be a callable function.")
        self.accessor = accessor

        self.use_jackknife = use_jackknife
        self.bootstrap_rounds = int(bootstrap_rounds)
        self.fraction_excluded = float(fraction_excluded)
        if self.use_jackknife and self.bootstrap_rounds > 0 and self.fraction_excluded > 0.0:
            raise ValueError(
                (
                    "`use_jackknife`, `bootstrap_rounds`, and `fraction_excluded` are mutually exclusive. ",
                    "Set only one of them.",
                )
            )

        # Resolve random_state up front so the same instance can be
        # called twice and produce bit-identical resamples (ENG-08).
        # Keep the original value for repr / debugging.
        self.random_state = random_state
        self._rng = check_random_state(random_state)

        self.parameters: list = []
        self.n_resamples = 0

    def resample(self, show_progress: bool = True) -> Resampler:
        """Perform the resampling."""
        if self.use_jackknife:
            return self.jackknife(show_progress=show_progress)
        elif self.bootstrap_rounds > 0:
            return self.bootstrap(show_progress=show_progress)
        elif self.fraction_excluded > 0.0:
            return self.fractional(show_progress=show_progress)
        else:
            raise ValueError("Either use_jackknife or bootstrap_rounds must be set.")

    def jackknife(self, show_progress: bool) -> Resampler:
        """Perform jackknife resampling on the given estimator."""
        self.parameters = []
        indices = np.arange(len(self.x))
        for i in tqdm(range(len(self.x)), desc="Jackknife Resampling", disable=not show_progress):
            leave_one_out_indices = indices[indices != i]
            x_train = self.x[leave_one_out_indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")
        return self

    def bootstrap(self, show_progress: bool) -> Resampler:
        """Perform bootstrap resampling on the given estimator."""
        self.parameters = []

        # Generate bootstrap samples, resample with replacement, in a loop of self.bootstrap_rounds iterations.
        # The shared ``self._rng`` is seeded via the constructor's ``random_state`` (ENG-08).
        for _ in tqdm(range(self.bootstrap_rounds), desc="Bootstrap Resampling", disable=not show_progress):
            # Resample indices with replacement

            indices = self._rng.choice(len(self.x), size=len(self.x), replace=True)
            x_train = self.x[indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")

        return self

    def fractional(self, show_progress: bool) -> Resampler:
        """Perform fractional resampling on the given estimator.

        Will repeat N times (N = number of rows in x), each time leaving out a fraction of the data as specified by
        self.fraction_excluded.
        """
        self.parameters = []

        # The shared ``self._rng`` is seeded via the constructor's ``random_state`` (ENG-08).
        # Re-validate here: the __init__ guard can be bypassed by mutating
        # ``fraction_excluded`` to 0 (or out of range) before calling fractional().
        if not 0.0 < self.fraction_excluded < 1.0:
            raise ValueError(
                f"`fraction_excluded` must be in the open interval (0, 1) to perform fractional "
                f"resampling, got {self.fraction_excluded}."
            )
        n_groups = int(1 / self.fraction_excluded)
        for _ in tqdm(range(len(self.x)), desc="Fractional Resampling", disable=not show_progress):
            # Find the indices to leave out
            all_indices = np.arange(len(self.x))
            self._rng.shuffle(all_indices)
            groups = np.array_split(all_indices, n_groups)
            rows_to_drop = groups[0]
            train_indices = np.setdiff1d(all_indices, rows_to_drop)
            x_train = self.x[train_indices]
            parameter = self.accessor(clone(self.estimator).fit(x_train))
            self.parameters.append(parameter)

        self.n_resamples = len(self.parameters)
        if self.n_resamples == 0:
            raise ValueError("No resamples were generated. Check your data and parameters.")

        return self

    def plot_results(self, cutoff: float | None = None) -> go.Figure:
        """
        Plot the results of the resampling.

        A vertical line can be added at the specified cutoff value. If `cutoff` is None, no vertical line is added.
        """
        parameters = pd.DataFrame(self.parameters)
        size_per_sample = len(self.parameters[0])

        # Resort the columns of the parameters DataFrame by the .median() value of each column
        parameters = parameters.reindex(parameters.median().sort_values(ascending=False).index, axis=1)

        fig = ridgeplot.ridgeplot(
            samples=parameters.to_numpy().T.reshape((size_per_sample, 1, self.n_resamples)),
            # bandwidth=4,
            kde_points=np.linspace(0, 2, 500),
            colorscale="viridis",
            colormode="row-index",
            opacity=0.6,
            labels=parameters.columns.tolist(),
            spacing=0.1,
            norm="probability",
        )
        if cutoff is not None:
            fig.add_vline(
                x=cutoff, line_color="red", line_dash="dash", annotation_text="Cutoff", annotation_position="top left"
            )
        fig.update_layout(
            font_size=16,
            plot_bgcolor="white",
            xaxis=dict(
                title="Parameter Value",
                showgrid=True,
                zeroline=False,
            ),
            yaxis=dict(
                title="Parameter Index",
                showgrid=True,
                zeroline=False,
                showticklabels=True,
            ),
            title="Resampling Results",
        )
        return fig



# ENG-23 (#305): explicit ``__all__`` so the thin re-exporter ``methods.py``
# can do ``from ._pca_pls import *`` without triggering CodeQL's
# py/polluting-import warning. List enumerated to mirror the public surface
# the prior ``methods.py`` exposed -- every name visible at module level,
# minus stdlib / 3rd-party imports and underscore-prefixed helpers.
__all__ = [
    "MBPCA",
    "MBPLS",
    "PCA",
    "PLS",
    "REFERENCE_LINE_COLOR",
    "TPLS",
    "DataFrameDict",
    "MCUVScaler",
    "Plot",
    "Resampler",
    "SpecificationWarning",
    "center",
    "check_random_state",
    "coefficient_plot",
    "correlation_loadings_plot",
    "detect_outliers_esd",
    "eigenvalue_summary",
    "ellipse_coordinates",
    "epsqrt",
    "explained_variance_plot",
    "hotellings_t2_limit",
    "internal_pls_nipals_fit_one_pc",
    "loading_plot",
    "nan_to_zeros",
    "observation_contributions",
    "predictions_vs_observed_plot",
    "project_variables",
    "quick_regress",
    "randomization_test_mbpls",
    "regress_a_space_on_b_row",
    "rv2_coefficient",
    "rv_coefficient",
    "safe_inverse",
    "scale",
    "score_limit",
    "score_plot",
    "spe_calculation",
    "spe_limit",
    "spe_plot",
    "squared_cosine",
    "ssq",
    "t2_plot",
    "terminate_check",
    "vip",
]
