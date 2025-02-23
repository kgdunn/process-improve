import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_array, check_is_fitted

epsqrt = np.sqrt(np.finfo(float).eps)


def nan_to_zeros(in_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NaN to zero and return a NaN map."""

    nan_map = np.isnan(in_array)
    in_array[nan_map] = 0
    return in_array, nan_map


def regress_y_space_on_x(y_space: np.ndarray, x_space: np.ndarray, y_space_nan_map: np.ndarray) -> np.ndarray:
    """
    Project the rows of `y_space` onto the vector `x_space`. Neither of these two inputs may have missing values.

    The `y_space_nan_map` has `True` entries where `y_space` originally had NaN values. The `x_space` may never have
    missing values.

    y_space = [n_rows x j_cols]
    x_space = [j_cols x 1]
    Returns   [n_rows x 1]
    """

    b_mat = np.tile(x_space.T, (y_space.shape[0], 1))  # tiles, row-by-row the `x_space` row vector, to create `n_rows`
    denominator = np.sum((b_mat * ~y_space_nan_map) ** 2, axis=1).astype("float")
    denominator[denominator == 0] = np.nan
    return np.array((np.sum(y_space * b_mat, axis=1)) / denominator).reshape(-1, 1)


# ------- Tests -------


def test_nan_to_zeros() -> None:
    """Test the `nan_to_zeros` function."""
    in_array = np.array([[1, 2, np.nan], [4, 5, 6], [float("nan"), 8, 9]])
    out_array, nan_map = nan_to_zeros(in_array)
    assert np.allclose(out_array, np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]))
    assert np.allclose(nan_map, np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]))


def test_regress_y_space_on_x() -> None:
    """Test the `regress_y_space_on_x` function."""
    x_space = np.array([1, 2, 3, 4])
    y_space = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, float("NaN"), 4],
            [float("NaN"), float("NaN"), 3, float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), 4],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [6, 4, 2, 0],
        ]
    )

    y_space_filled, y_space_nan_map = nan_to_zeros(y_space)
    regression_vector = regress_y_space_on_x(y_space_filled, x_space, y_space_nan_map)
    assert np.allclose(regression_vector, np.array([[1, 1, 1, 1, float("nan"), 2 / 3]]).T, equal_nan=True)


test_nan_to_zeros()
test_regress_y_space_on_x()

# from sklearn.utils.estimator_checks import parametrize_with_checks


# @parametrize_with_checks([LogisticRegression(), DecisionTreeRegressor()])
# def test_sklearn_compatible_estimator(estimator, check):
#     check(estimator)


class TPLS(BaseEstimator):
    """
    TPLS algorithm.

    Source: Garcia-Munoz, https://doi.org/10.1016/j.chemolab.2014.02.006, Chem.Intell.Lab.Sys. v133, p 49 to 62, 2014.
    Change of notation from the original paper:

    Paper           This code     Code input variable     Internal Numpy variable name (holds only NumPy values)
    =====           ========      ===================     ============================
    X^T             D             d_dataframes            d_mats
    X               D^T
    R               F             f_dataframes            f_mats
    Z               Z             z_matrix                z_mat
    Y               Y             y_matrix                y_mat

    Matrices in F, Z and Y must all have the same number of rows.
    Columns in F must be the same as the rows in D.

    Parameters
    ----------
    n_components : int
        A parameter used to specify the number of components.



    ---- Inputs ----

    D [d_dataframes] Database (dict) of dataframes, containing physical properties.

        D = { "Group A": dataframe of properties of group A materials. (columns contain properties, rows are materials),
              "Group B": dataframe of properties of group B materials. (columns contain properties, rows are materials),
              ...
            }

    F [f_dataframes] Formula matrices/ratio of materials, corresponding to the *rows* of D
                     (or columns of D after transposing):

        F = { "GroupA": dataframe of formula for group A used in each blend (one formula per row, columns are materials)
              "GroupB": dataframe of formula for group B used in each blend (one formula per row, columns are materials)
              ...
            }

    Z [z_matrix] Process conditions. One row per formula/blend; one column per condition.

    Y [y_matrix] Product characteristics (quality space; key performance indicators).
                 One row per formula/blend; one column per quality indicator.



    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    Example
    -------
    >>> from ___ import TPLS
    >>> import numpy as np
    >>> all_data = {"Z": ... , "D": ... , "F": ...}
    >>> y = np.zeros((100, ))
    >>> estimator = TPLS(n_components=2)
    >>> estimator.fit(all_data, y)

    # d_mats_ = {group_name: d_mat.values.astype(float) for group_name, d_mat in d_dataframes.items()}
    # f_mats_ = {group_name: f_mat.values.astype(float) for group_name, f_mat in f_dataframes.items()}
    # z_mat_ = z_matrix.values.astype(float)
    # y_mat_ = y_matrix.values.astype(float)

    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
    }

    def __init__(self, n_components: int):
        self.n_components = n_components

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, dict[str, pd.DataFrame] | pd.DataFrame], y: pd.DataFrame) -> "TPLS":
        """Fit the model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        X, y = self._validate_data(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X: dict[str, dict[str, pd.DataFrame] | pd.DataFrame]) -> dict:
        """Model inference on new data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return {}


def learn_center_and_scaling_parameters(y: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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
    centering = y.mean()
    scaling = y.std(ddof=1)
    scaling[scaling < epsqrt] = 1.0  # columns with little/no variance are left as-is.
    return centering, scaling


class TPLSpreprocess(TransformerMixin, BaseEstimator):
    """

    Pre-process the dataframes for TPLS models.


    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng()
    >>>
    >>> n_props_a, n_props_b = 6, 4
    >>> n_materials_a, n_materials_b = 12, 8
    >>> properties = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_materials_a, n_props_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_materials_b, n_props_b))),
    >>> }
    >>> n_formulas = 40
    >>> n_outputs = 3
    >>> n_conditions = 2
    >>> formulas = {
    >>>     "Group A": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_a))),
    >>>     "Group B": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_b))),
    >>> }
    >>> process_conditions = pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))
    >>> quality_indicators = pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))
    >>> all_data = {"Z": process_conditions, "D": properties, "F": formulas}
    >>> estimator = TPLSpreprocess()
    >>> estimator.fit(all_data, y=quality_indicators)
    """

    _parameter_constraints: typing.ClassVar = {}

    def __init__(self):
        pass

    def validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
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

        return check_array(
            df, accept_sparse=False, force_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, dict[str, pd.DataFrame]] | pd.DataFrame, y: pd.DataFrame) -> "TPLSpreprocess":
        """
        Fit/learn the preprocessing parameters from the training data.

        Parameters
        ----------
        X : {dictionary of dataframes}, keys that must be present: "D", "F", "Z"
            The training input samples.

        y : pd.DataFrame
            The quality indicators. Rows are the samples, columns are performance metrics, quality variables, outcomes.
            The number of rows in `y` must be the same as the number of rows in `F` and `Z`.

        Returns
        -------
        self : object
            Returns self.
        """
        self.preproc: dict[str, dict[str, pd.DataFrame] | pd.DataFrame] = {
            "D": {},
            "F": {},
            "Z": pd.DataFrame(),
            "Y": pd.DataFrame(),
        }

        assert "Z" in X, "The input data must contain a 'Z' key."
        assert "D" in X, "The input data must contain a 'D' key."
        assert "F" in X, "The input data must contain a 'F' key."
        self.validate_df(y)
        self.validate_df(X["Z"])
        for key in X["D"]:
            self.validate_df(X["D"][key])
            self.validate_df(X["F"][key])  # also ensure the keys in F are the same as in D

        # Learn the centering and scaling parameters
        self.preproc["Y"]["center"], self.preproc["Y"]["scale"] = learn_center_and_scaling_parameters(y)
        self.preproc["Z"]["center"], self.preproc["Z"]["scale"] = learn_center_and_scaling_parameters(X["Z"])
        for key, df_d in X["D"].items():
            self.preproc["D"][key] = {"block": np.sqrt(df_d.shape[1]), "center": pd.Series(), "scale": pd.Series()}
            self.preproc["F"][key] = {"block": float("nan"), "center": pd.Series(), "scale": pd.Series()}

            self.preproc["D"][key]["center"], self.preproc["D"][key]["scale"] = learn_center_and_scaling_parameters(
                df_d
            )
            self.preproc["F"][key]["center"], self.preproc["F"][key]["scale"] = learn_center_and_scaling_parameters(
                X["F"][key]
            )

        return self

    def transform(self, X: dict[str, dict[str, pd.DataFrame] | pd.DataFrame]) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Apply the transformation to the input data.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : dict[str, dict[pd.DataFrame]]
            The transformed input data, containing element-wise transformations applied to the values in the dataframes.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        X = self._validate_data(X, accept_sparse=True, reset=False)
        return X

    def _more_tags(self) -> dict:
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}


n_props_a, n_props_b = 6, 4
n_materials_a, n_materials_b = 12, 8
rng = np.random.default_rng()
properties = {
    "Group A": pd.DataFrame(rng.standard_normal((n_materials_a, n_props_a))),
    "Group B": pd.DataFrame(rng.standard_normal((n_materials_b, n_props_b))),
}
n_formulas = 40
n_outputs = 3
n_conditions = 2
formulas = {
    "Group A": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_a))),
    "Group B": pd.DataFrame(rng.standard_normal((n_formulas, n_materials_b))),
}
process_conditions = pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))
quality_indicators = pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))
all_data = {"Z": process_conditions, "D": properties, "F": formulas}
estimator = TPLSpreprocess()
estimator.fit(all_data, y=quality_indicators)
