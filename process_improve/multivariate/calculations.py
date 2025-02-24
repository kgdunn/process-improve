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
    >>> n_formulas = 40
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
    >>> process_conditions = pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))
    >>> quality_indicators = pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))
    >>> all_data = {"Z": process_conditions, "D": properties, "F": formulas}
    >>> estimator = TPLSpreprocess()
    >>> estimator.fit(all_data, y=quality_indicators)
    """

    _parameter_constraints: typing.ClassVar = {}

    def __init__(self):
        pass

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
        centering = y.mean()
        scaling = y.std(ddof=1)
        scaling[scaling < epsqrt] = 1.0  # columns with little/no variance are left as-is.
        return centering, scaling

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
            df, accept_sparse=False, ensure_all_finite="allow-nan", ensure_2d=True, allow_nd=False, ensure_min_samples=1
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, dict[str, pd.DataFrame]], y: None = None) -> "TPLSpreprocess":  # noqa: ARG002
        """
        Fit/learn the preprocessing parameters from the training data.

        Parameters
        ----------
        X : {dictionary of dataframes}, keys that must be present: "D", "F", "Z", and "Y"
            The training input samples. See documentation in the class definition for more information on each matrix.

        Returns
        -------
        self : object
            Returns self.
        """
        expected_blocks = {"D", "F", "Z", "Y"}
        assert set(X.keys()) == expected_blocks, f"Expected keys: {expected_blocks}, got: {set(X.keys())}"
        self.preproc_: dict[str, dict[str, dict[str, pd.Series]]] = {key: {} for key in expected_blocks}
        for block in expected_blocks:
            for key in X[block]:
                assert isinstance(X[block][key], pd.DataFrame), f"The 'X[{block}][{key}]' entries must be a DataFrame."

        self.validate_df(X["Y"]["Quality"])
        self.validate_df(X["Z"]["Conditions"])
        for key in X["D"]:
            self.validate_df(X["D"][key])
            self.validate_df(X["F"][key])  # this also ensures the keys in F are the same as in D

        # Learn the centering and scaling parameters
        self.preproc_["Y"]["Quality"] = {}
        self.preproc_["Y"]["Quality"]["center"], self.preproc_["Y"]["Quality"]["scale"] = (
            self._learn_center_and_scaling_parameters(X["Y"]["Quality"])
        )
        self.preproc_["Z"]["Conditions"] = {}
        self.preproc_["Z"]["Conditions"]["center"], self.preproc_["Z"]["Conditions"]["scale"] = (
            self._learn_center_and_scaling_parameters(X["Z"]["Conditions"])
        )
        for key, df_d in X["D"].items():
            self.preproc_["D"][key] = {}
            self.preproc_["F"][key] = {}
            self.preproc_["D"][key]["center"], self.preproc_["D"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(df_d)
            )
            self.preproc_["D"][key]["block"] = pd.Series([np.sqrt(df_d.shape[1])])
            self.preproc_["F"][key]["center"], self.preproc_["F"][key]["scale"] = (
                self._learn_center_and_scaling_parameters(X["F"][key])
            )

        return self

    def transform(self, X: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Apply the centering and scaling transformation to the input data.

        Parameters
        ----------
        X : {dictionary of dataframes}, keys that must be present: "D", "F", "Z", and "Y"
            The input data to be transformed

        Returns
        -------
        X_transformed : dict[str, dict[str, pd.DataFrame]]
            The transformed input data, containing element-wise transformations applied to the values in the dataframes.
        """
        check_is_fitted(self)
        X_transformed: dict[str, dict[str, pd.DataFrame]] = {"D": {}, "F": {}, "Z": {}, "Y": {}}
        for key, df_d in X["D"].items():
            X_transformed["D"][key] = (
                (df_d - self.preproc_["D"][key]["center"])
                / self.preproc_["D"][key]["scale"]
                / self.preproc_["D"][key]["block"][0]  # scalar!
            )
            X_transformed["F"][key] = (X["F"][key] - self.preproc_["F"][key]["center"]) / self.preproc_["F"][key][
                "scale"
            ]
        X_transformed["Z"]["Conditions"] = (
            pd.DataFrame(X["Z"]["Conditions"]) - self.preproc_["Z"]["Conditions"]["center"]
        ) / self.preproc_["Z"]["Conditions"]["scale"]
        X_transformed["Y"]["Quality"] = (
            pd.DataFrame(X["Y"]["Quality"]) - self.preproc_["Y"]["Quality"]["center"]
        ) / self.preproc_["Y"]["Quality"]["scale"]
        return X_transformed

    def _more_tags(self) -> dict:
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}


# -------
def load_tpls_example() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load example data for TPLS model.

    Data from: https://github.com/salvadorgarciamunoz/pyphi/tree/master/examples/JRPLS%20and%20TPLS
    """
    properties = {
        "Group 1": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group1.csv", sep=",", index_col=0, header=0
        ),
        "Group 2": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group2.csv", sep=",", index_col=0, header=0
        ),
        "Group 3": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group3.csv", sep=",", index_col=0, header=0
        ),
        "Group 4": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group4.csv", sep=",", index_col=0, header=0
        ),
        "Group 5": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group5.csv", sep=",", index_col=0, header=0
        ),
    }
    formulas = {
        "Group 1": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group1.csv", sep=",", index_col=0, header=0
        ),
        "Group 2": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group2.csv", sep=",", index_col=0, header=0
        ),
        "Group 3": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group3.csv", sep=",", index_col=0, header=0
        ),
        "Group 4": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group4.csv", sep=",", index_col=0, header=0
        ),
        "Group 5": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group5.csv", sep=",", index_col=0, header=0
        ),
    }
    process_conditions: dict[str, pd.DataFrame] = {
        "Conditions": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/process_conditions.csv", sep=",", index_col=0, header=0
        )
    }

    quality_indicators: dict[str, pd.DataFrame] = {
        "Quality": pd.read_csv(
            "process_improve/datasets/multivariate/tpls-pyphi/quality_indicators.csv", sep=",", index_col=0, header=0
        )
    }
    return {
        "Z": process_conditions,
        "D": properties,
        "F": formulas,
        "Y": quality_indicators,
    }


tpls_example = load_tpls_example()
estimator = TPLSpreprocess()
estimator.fit(tpls_example)
transformed_data = estimator.transform(tpls_example)


def test_tpls_preprocessing() -> None:
    """
    Test the `TPLSpreprocess` class using the example data.

    Test the centering and scaling of the dataframes in D, Z, F, and Y.

    """
    tpls_example = load_tpls_example()
    estimator = TPLSpreprocess()
    testing_df_dict = estimator.fit_transform(tpls_example)
    assert np.allclose(testing_df_dict["Z"]["Conditions"].mean(), 0.0)
    assert np.allclose(testing_df_dict["Z"]["Conditions"].std(), 1.0)
    assert np.allclose(testing_df_dict["Y"]["Quality"].mean(), 0.0)
    assert np.allclose(testing_df_dict["Y"]["Quality"].std(), 1.0)
    for key in tpls_example["D"]:
        assert np.allclose(testing_df_dict["D"][key].mean(), 0.0)
        assert np.allclose(testing_df_dict["D"][key].std(), 1 / estimator.preproc_["D"][key]["block"])
        assert np.allclose(testing_df_dict["F"][key].mean(), 0.0)
        assert np.allclose(testing_df_dict["F"][key].std(), 1.0)

    # Test the coefficients in the centering and scaling .preproc_ structure
    # Group 1, for D matrix:
    known_truth_d1m = np.array([99.85432099, 73.67901235, 3.07469136, 0.13950617, 2.09876543, 12.53703704, 41.58641975])
    known_truth_d1s = np.array([0.36883209, 1.80631852, 0.69231018, 0.09610943, 0.29927192, 1.99109474, 4.82828759])
    assert np.allclose(estimator.preproc_["D"]["Group 1"]["center"], known_truth_d1m)
    assert np.allclose(estimator.preproc_["D"]["Group 1"]["scale"], known_truth_d1s)
    assert np.allclose(estimator.preproc_["D"]["Group 1"]["center"], tpls_example["D"]["Group 1"].mean())
    assert np.allclose(estimator.preproc_["D"]["Group 1"]["scale"], tpls_example["D"]["Group 1"].std())

    # Group 4 in D has missing values. Test these.
    known_truth_d4m = np.array(
        [
            9.99055556e-01,
            1.15555556e-01,
            2.69444444e-01,
            2.14210526e01,
            1.07000000e01,
            2.04894737e02,
            5.82947368e01,
            9.86000000e01,
            4.88947368e00,
            3.43684211e00,
            7.03684211e00,
        ]
    )
    known_truth_d4s = np.array(
        [
            8.02365783e-04,
            7.04792186e-03,
            1.73110717e-02,
            3.64105890e00,
            2.11213425e00,
            2.18313579e00,
            6.27211409e00,
            1.84149698e00,
            6.57836255e-02,
            1.77045275e-01,
            3.68496233e-01,
        ]
    )
    assert np.allclose(estimator.preproc_["D"]["Group 4"]["center"], known_truth_d4m)
    assert np.allclose(estimator.preproc_["D"]["Group 4"]["scale"], known_truth_d4s)
    assert np.allclose(estimator.preproc_["D"]["Group 4"]["center"], tpls_example["D"]["Group 4"].mean())
    assert np.allclose(estimator.preproc_["D"]["Group 4"]["scale"], tpls_example["D"]["Group 4"].std())

    # Test the formula block, group 2:
    known_truth_f2m = np.array(
        [0.13333333, 0.0020127, 0.01904762, 0.00952381, 0.13282593, 0.1889709, 0.1047619, 0.17035596, 0.23916785]
    )
    known_truth_f2s = np.array(
        [0.34156503, 0.02062402, 0.13734798, 0.09759001, 0.33676188, 0.39173332, 0.3077152, 0.37647422, 0.4274989]
    )
    assert np.allclose(estimator.preproc_["F"]["Group 2"]["center"], known_truth_f2m)
    assert np.allclose(estimator.preproc_["F"]["Group 2"]["scale"], known_truth_f2s)
    assert np.allclose(estimator.preproc_["F"]["Group 2"]["center"], tpls_example["F"]["Group 2"].mean())
    assert np.allclose(estimator.preproc_["F"]["Group 2"]["scale"], tpls_example["F"]["Group 2"].std())

    # Test the `Conditions` (Z) block:
    known_truth_zm = np.array(
        [
            8.63809524e00,
            2.27764762e01,
            7.21470000e01,
            2.13978571e01,
            7.17852381e01,
            2.06402381e01,
            7.10553333e01,
            1.35438095e05,
            8.19047619e-01,
            1.80952381e-01,
        ]
    )
    known_truth_zs = np.array(
        [
            3.56835399e00,
            6.95963475e00,
            6.53039712e-01,
            6.27483262e00,
            1.29077224e00,
            3.59658120e00,
            1.45728018e00,
            1.89215897e03,
            3.86825154e-01,
            3.86825154e-01,
        ]
    )
    assert np.allclose(estimator.preproc_["Z"]["Conditions"]["center"], known_truth_zm)
    assert np.allclose(estimator.preproc_["Z"]["Conditions"]["scale"], known_truth_zs)
    assert np.allclose(estimator.preproc_["Z"]["Conditions"]["center"], tpls_example["Z"]["Conditions"].mean())
    assert np.allclose(estimator.preproc_["Z"]["Conditions"]["scale"], tpls_example["Z"]["Conditions"].std())

    # Test the `Quality` (Y) block:
    known_truth_ym = np.array([30.96834605, 3.328312, 57.01620571, 3.6485004, 79.34224822, 3.06157598])
    known_truth_ys = np.array([3.46989807, 0.96879553, 4.9740008, 1.47585478, 3.8676095, 1.18600054])
    assert np.allclose(estimator.preproc_["Y"]["Quality"]["center"], known_truth_ym)
    assert np.allclose(estimator.preproc_["Y"]["Quality"]["scale"], known_truth_ys)
    assert np.allclose(estimator.preproc_["Y"]["Quality"]["center"], tpls_example["Y"]["Quality"].mean())
    assert np.allclose(estimator.preproc_["Y"]["Quality"]["scale"], tpls_example["Y"]["Quality"].std())


test_tpls_preprocessing()


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
    >>> all_data = {"Z": ... , "D": ... , "F": ..., "Y": ...}
    >>> estimator = TPLS(n_components=2)
    >>> estimator.fit(all_data)

    # d_mats_ = {group_name: d_mat.values.astype(float) for group_name, d_mat in d_dataframes.items()}
    # f_mats_ = {group_name: f_mat.values.astype(float) for group_name, f_mat in f_dataframes.items()}
    # z_mat_ = z_matrix.values.astype(float)
    # y_mat_ = y_matrix.values.astype(float)

    """

    # # This is a dictionary allowing to define the type of parameters.
    # # It used to validate parameter within the `_fit_context` decorator.
    # _parameter_constraints: typing.ClassVar = {
    #     "n_components": [int],
    # }

    # def __init__(self, n_components: int):
    #     self.n_components = n_components

    # @_fit_context(prefer_skip_nested_validation=True)
    # def fit(self, X: dict[str, dict[str, pd.DataFrame] | pd.DataFrame], y: None = None) -> "TPLS":
    #     """Fit the model.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.

    #     Returns
    #     -------
    #     self : object
    #         Returns self.
    #     """
    #     X, y = self._validate_data(X, y, accept_sparse=False)
    #     self.is_fitted_ = True
    #     # `fit` should always return `self`
    #     return self

    # def predict(self, X: dict[str, dict[str, pd.DataFrame] | pd.DataFrame]) -> dict:
    #     """Model inference on new data.

    #     Parameters
    #     ----------
    #     X : {array-like, sparse matrix}, shape (n_samples, n_features)
    #         The training input samples.

    #     Returns
    #     -------
    #     y : ndarray, shape (n_samples,)
    #         Returns an array of ones.
    #     """
    #     # Check if fit had been called
    #     check_is_fitted(self)
    #     # We need to set reset=False because we don't want to overwrite `n_features_in_`
    #     # `feature_names_in_` but only check that the shape is consistent.
    #     X = self._validate_data(X, accept_sparse=True, reset=False)
    #     return {}
