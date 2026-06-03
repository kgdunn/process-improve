# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""T-PLS (PLS for T-shaped data structures) estimator and its data container (ENG-01).

Holds :class:`DataFrameDict`, the container for the partitionable (Z, F) and
static (Y) data blocks, and :class:`TPLS`, the sklearn-compatible T-PLS
regressor. ("TPLS" here means PLS for T-shaped data structures, not "Total PLS"
or "Three-way PLS".)
"""

from __future__ import annotations

import logging
import time
import typing
import warnings
from collections.abc import Callable, KeysView
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.metrics import r2_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_array, check_is_fitted

from .._linalg import safe_inverse
from ._common import _nz
from ._limits import ellipse_coordinates as _ellipse_coordinates
from ._limits import hotellings_t2_limit as _hotellings_t2_limit
from ._limits import spe_calculation
from ._nipals import internal_pls_nipals_fit_one_pc, nan_to_zeros, regress_a_space_on_b_row
from .plots import Plot


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


logger = logging.getLogger(__name__)


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

    # ENG-05: convenience methods forwarding to the standalone functions. These
    # used to be ``functools.partial`` instances bound in ``fit``; defining them
    # as real methods keeps ``help`` / ``inspect.signature`` accurate and the
    # fitted model picklable.
    def hotellings_t2_limit(self, conf_level: float = 0.95) -> float:
        """Hotelling's T2 limit at the given confidence level (see :func:`hotellings_t2_limit`)."""
        return _hotellings_t2_limit(
            conf_level=conf_level,
            n_components=self.n_components,
            n_rows=self.hotellings_t2.shape[0],
        )

    def ellipse_coordinates(
        self,
        score_horiz: int,
        score_vert: int,
        conf_level: float = 0.95,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the T2 confidence ellipse (see :func:`ellipse_coordinates`)."""
        return _ellipse_coordinates(
            score_horiz=score_horiz,
            score_vert=score_vert,
            conf_level=conf_level,
            n_points=n_points,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores,
            n_rows=self.t_scores_super.shape[0],
        )

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
        self.scaling_factor_for_scores = pd.Series()

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
        self.scaling_factor_for_scores = pd.Series(
            np.sqrt(np.diag(variance_matrix)),
            index=[a + 1 for a in range(self.n_components)],
            name="Standard deviation per score",
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

            logger.debug("TPLS: super-component %d converged in %d iterations", pc_a + 1, n_iter)

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

