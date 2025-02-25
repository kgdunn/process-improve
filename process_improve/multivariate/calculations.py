import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_array, check_is_fitted

epsqrt = np.sqrt(np.finfo(float).eps)


def nan_to_zeros(in_array: np.ndarray) -> np.ndarray:
    """Convert NaN to zero and return a NaN map."""

    nan_map = np.isnan(in_array)
    in_array[nan_map] = 0
    return in_array


def regress_a_space_on_b_row(a_space: np.ndarray, b_row: np.ndarray, a_space_present_map: np.ndarray) -> np.ndarray:
    """
    Project each row of `a_space` onto row vector `b_row`, to return a regression coefficient for every row in A.

    NOTE: Neither of these two inputs may have missing values. It is assumed you have replaced missing values by zero,
          and have a map of where the missing values were (more correctly, where the non-missing values are is given
          by `a_space_present_map`).

    NOTE: No checks are done on the incoming data to ensure consistency. That is the caller's responsibility. This
          function is called thousands of times, so that overhead is not acceptable.

    The `a_space_present_map` has `False` entries where `a_space` originally had NaN values.
    The `b_row` may never have missing values, and no map is provided for it. These row vectors are latent variable
    vectors, and therefore never have missing values.

    a_space             = [n_rows x j_cols]
    b_row               = [1      x j_cols]    # in other words, a row vector of `j_cols` entries
    a_space_present_map = [n_rows x j_cols]

    Returns               [n_rows x 1] = a_space * b_row^T  / ( b_row * b_row^T)
                                         (n x j) * (j x 1)  /  (1 x j)* (j x 1)  = n x 1
    """
    denom = np.tile(b_row, (a_space.shape[0], 1))  # tiles, row-by-row the `b_row` row vector, to create `n_rows`
    denominator = np.sum((denom * a_space_present_map) ** 2, axis=1).astype("float")
    denominator[denominator == 0] = np.nan
    return np.array((np.sum(a_space * denom, axis=1)) / denominator).reshape(-1, 1)


# ------- Tests -------


def test_nan_to_zeros() -> None:
    """Test the `nan_to_zeros` function."""
    in_array = np.array([[1, 2, np.nan], [4, 5, 6], [float("nan"), 8, 9]])
    out_array = nan_to_zeros(in_array)
    assert np.allclose(out_array, np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]]))


def test_regress_y_space_on_x() -> None:
    """Test the `regress_y_space_on_x` function."""
    x_space = np.array([[1, 2, 3, 4]])
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
    present_map = np.logical_not(np.isnan(y_space))
    y_space_filled = nan_to_zeros(y_space)
    regression_vector = regress_a_space_on_b_row(y_space_filled, x_space, present_map)
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
    >>> process_conditions = {"Conditions": pd.DataFrame(rng.standard_normal((n_formulas, n_conditions)))}
    >>> quality_indicators = pd.DataFrame(rng.standard_normal((n_formulas, n_outputs)))
    >>> all_data = {"Z": process_conditions, "D": properties, "F": formulas}
    >>> estimator = TPLSpreprocess()
    >>> estimator.fit(all_data, y=quality_indicators)
    """

    _parameter_constraints: typing.ClassVar = {}

    def __init__(self):
        super().__init__()

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

        self.is_fitted_ = True
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


def internal_pls_nipals_fit_one_pc(
    x_space: np.ndarray,
    y_space: np.ndarray,
    x_present_map: np.ndarray,
    y_present_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit a PLS model using the NIPALS algorithm."""
    max_iter: int = 500

    is_converged = False
    n_iter = 0
    u_i = y_space[:, [0]]
    while not is_converged:
        # Step 1. w_i = X'u / u'u. Regress the columns of X on u_i, and store the slope coeff in vectors w_i.
        w_i = regress_a_space_on_b_row(x_space.T, u_i.T, x_present_map.T)

        # Step 2. Normalize w to unit length.
        w_i = w_i / np.linalg.norm(w_i)

        # Step 3. t_i = Xw / w'w. Regress rows of X on w_i, and store slope coefficients in t_i.
        t_i = regress_a_space_on_b_row(x_space, w_i.T, x_present_map)

        # Step 4. q_i = Y't / t't. Regress columns of Y on t_i, and store slope coefficients in q_i.
        q_i = regress_a_space_on_b_row(y_space.T, t_i.T, y_present_map.T)

        # Step 5. u_new = Yq / q'q. Regress rows of Y on q_i, and store slope coefficients in u_new
        u_new = regress_a_space_on_b_row(y_space, q_i.T, y_present_map)

        if (abs(np.linalg.norm(u_i - u_new)) / np.linalg.norm(u_i)) < epsqrt:
            is_converged = True
        if n_iter > max_iter:
            is_converged = True

        n_iter += 1
        u_i = u_new

    # We have converged. Keep sign consistency. Fairly arbitrary rule, but ensures we report results consistently.
    if np.var(t_i[t_i < 0]) > np.var(t_i[t_i >= 0]):
        t_i = -t_i
        u_new = -u_new
        w_i = -w_i
        q_i = -q_i

    return dict(t_i=t_i, u_i=u_i, w_i=w_i, q_i=q_i)


class TPLS(BaseEstimator):
    """
    TPLS algorithm for T-shaped data structures.

    Source: Garcia-Munoz, https://doi.org/10.1016/j.chemolab.2014.02.006, Chem.Intell.Lab.Sys. v133, p 49 to 62, 2014.

    We change the notation from the original paper to avoid confusion with a generic "X" matrix, and match symbols
    that are more natural for our use.

    Paper           This code     Internal Numpy variable name (holds only NumPy values)
    =====           ========      ============================
    X^T             D             d_mats                            Database
    X               D^T
    R               F             f_mats                            Formula
    Z               Z             z_mats                            (Upstream) conditions
    Y               Y             y_mats                            Quality indicators

    Notes
    1. Matrices in F, Z and Y must all have the same number of rows.
    2. Columns in F must be the same as the rows in D.
    3. Conditions in Z may be missing (turning it into an L-shaped data structure).

    Parameters
    ----------
    n_components : int
        A parameter used to specify the number of components.

    Data structures in input `X` (a dictionary with 4 keys, as listed below)
    ------------------------------------------------------------------------

    D. Database of dataframes, containing properties.

        D = { "Group A": dataframe of properties of group A materials. (columns contain properties, rows are materials),
              "Group B": dataframe of properties of group B materials. (columns contain properties, rows are materials),
              ...
            }

    F. Formula matrices/ratio of materials, corresponding to the *rows* of D (or columns of D after transposing):

        F = { "GroupA": dataframe of formula for group A used in each blend (one formula per row, columns are materials)
              "GroupB": dataframe of formula for group B used in each blend (one formula per row, columns are materials)
              ...
            }

    Z. Process conditions. One row per formula/blend; one column per condition.

    Y. Product characteristics (quality space; key performance indicators). One row per formula/blend;
       one column per quality indicator.


    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    Example
    -------
    >>> from ___ import TPLS
    >>> import numpy as np
    >>> all_data = {"Z": ... , "D": ... , "F": ..., "Y": ...}  # see the example in the `TPLSpreprocess` class.
    >>> estimator = TPLS(n_components=2)
    >>> estimator.fit(all_data)
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
    }

    def __init__(self, n_components: int):
        assert n_components > 0, "Number of components must be positive."
        self.n_components = n_components
        self.tolerance_ = np.sqrt(np.finfo(float).eps)
        self.max_iterations_ = 500
        self.fitting_statistics_: dict[str, list] = {"iterations": [], "convergance_tolerance": []}

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, dict[str, pd.DataFrame]], y: None = None) -> "TPLS":  # noqa: ARG002
        """Fit the model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns self.
        """
        # Note: we assume the data have been pre-processed by the `TPLSpreprocess` class; so no data checks performed.
        assert isinstance(X, dict), "The input data must be a dictionary."
        assert set(X.keys()) == {"D", "F", "Z", "Y"}, "The input dictionary must have keys: D, F, Z, Y."

        group_keys = [str(key) for key in X["D"]]
        d_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["D"][key].values) for key in group_keys}
        # `key in X["D"]` is intentional in the line below, to ensure the keys in F are the same as in D.
        f_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["F"][key].values) for key in group_keys}
        z_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["Z"][key].values) for key in X["Z"]}  # only 1 key in Z
        y_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["Y"][key].values) for key in X["Y"]}  # only 1 key in Y

        self.d_mats = d_mats
        self.f_mats = f_mats
        self.z_mats = z_mats
        self.y_mats = y_mats  # corrected to assign y_mats

        # Create the missing value maps, except we store the opposite, i.e., not missing, since these are more useful.
        # We refer to these as `pmaps` in the code (present maps, as opposed to `mmap` or missing maps).
        self.not_na_d = {key: ~np.isnan(X["D"][key].values) for key in d_mats}
        self.not_na_f = {key: ~np.isnan(X["F"][key].values) for key in f_mats}
        self.not_na_z = {key: ~np.isnan(X["Z"][key].values) for key in z_mats}
        self.not_na_y = {key: ~np.isnan(X["Y"][key].values) for key in y_mats}

        self._fit_iterative_regressions()
        self.is_fitted_ = True

        return self

    def has_converged(
        self,
        starting_vector: np.ndarray,
        revised_vector: np.ndarray,
        iterations: int,
    ) -> bool:
        """
        Terminate the iterative algorithm when any one of these conditions is True.

        #. scores converge: the norm between two successive iterations is smaller than a tolerance
        #. maximum number of iterations is reached
        """
        delta_gap = float(
            np.linalg.norm(starting_vector - revised_vector, ord=None) / np.linalg.norm(starting_vector, ord=None)
        )
        converged = delta_gap < self.tolerance_
        max_iter = iterations >= self.max_iterations_
        return bool(np.any([max_iter, converged]))

    def _fit_iterative_regressions(self) -> None:
        """Fit the model via iterative regressions and store the model coefficients in the class instance."""

        # Formula matrix: assemble all not-na maps from blocks in F: make a single matrix.
        pmap_f = np.concatenate(list(self.not_na_f.values()), axis=1)

        for _pc_a in range(self.n_components):
            n_iter = 0
            # Follow the steps in the paper on page 54
            # Step 1: Select any column in Y as initial guess (they have all be scaled anyway)
            u_prior = np.zeros_like(list(self.y_mats.values())[0][:, [0]])
            u_i = list(self.y_mats.values())[0][:, [0]]

            while not self.has_converged(starting_vector=u_prior, revised_vector=u_i, iterations=n_iter):
                n_iter += 1
                u_prior = u_i.copy()
                # Step 2. h_i = F_i' u / u'u. Regress the columns of F on u_i, and store the slope coeff in vectors h_i
                h_i = {
                    key: regress_a_space_on_b_row(df_f.T, u_i.T, pmap_f.T)
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

                # Step 7: if there is a Condition matrix (non-empty Z block), regress columns of this on the initial u_i
                if len(self.z_mats) > 0:
                    w_i = {
                        key: regress_a_space_on_b_row(df_z.T, u_i.T, self.not_na_z[key].T)
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }

                    # Step 8: Normalize joint w to unit length.
                    w_i_normalized = {key: w / np.linalg.norm(w) for key, w in w_i.items()}

                    # Step 9: regress rows of Z on w_i, and store slope coefficients in t_z. There is an error in the
                    #        paper here, but in figure 4 it is clear what should be happening.
                    t_zb = {
                        key: regress_a_space_on_b_row(df_z, w_i_normalized[key].T, self.not_na_z[key])
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }
                    t_z = np.concatenate(list(t_zb.values()), axis=1)

                else:
                    t_z = np.zeros((t_f.shape[0], 0))  # empty matrix: in other words, no Z block

                # Step 10: Combine t_f and t_z to form a joint t matrix.
                t_combined = np.concatenate([t_f, t_z], axis=1)

                # Step 11: Build an inner PLS model: using the t_combined as the X matrix, and the Y (quality space)
                #          as the Y matrix.
                inner_pls = internal_pls_nipals_fit_one_pc(
                    x_space=t_combined,
                    y_space=np.array(list(self.y_mats.values())[0]),
                    x_present_map=np.ones(t_combined.shape).astype(bool),
                    y_present_map=np.array(list(self.not_na_y.values())[0]),
                )
                u_i = inner_pls["u_i"]
                t_i = inner_pls["t_i"]
                # wt_i = inner_pls["w_i"]
                q_i = inner_pls["q_i"]

            # Step 12. Converged. Now store information.
            delta_gap = float(np.linalg.norm(u_prior - u_i, ord=None) / np.linalg.norm(u_prior, ord=None))
            self.fitting_statistics_["iterations"].append(n_iter)
            self.fitting_statistics_["convergance_tolerance"].append(delta_gap)

            # Calculate and store the deflation vectors. See equation 7 on page 55.
            #
            # Step 13. p_i = F_i' t_i / t_i't_i. Regress the columns of F_i on t_i; store slope coeff in vectors p_i.
            # Note: the "t" vector is the t_i vector from the inner PLS model, marked as "Tt" in figure 4 of the paper.
            pf_i = {
                key: regress_a_space_on_b_row(df_f.T, t_i.T, pmap_f.T)
                for key, df_f, pmap_f in zip(
                    self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True
                )
            }
            # Step 13: Deflate the Z matrix with a loadings vector, pz
            pz_i = {
                key: regress_a_space_on_b_row(df_z.T, t_i.T, pmap_z.T)
                for key, df_z, pmap_z in zip(
                    self.z_mats.keys(), self.z_mats.values(), self.not_na_z.values(), strict=True
                )
            }

            # Step 13: v_i = D_i' r_i / r_i'r_i. Regress the rows of D_i (properties) on r_i; store slopes in v_i.
            v_i = {
                key: regress_a_space_on_b_row(df_d.T, r_i[key].T, pmap_d.T)
                for key, df_d, pmap_d in zip(
                    self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True
                )
            }


            START HERE AGAIN
            # Step 14. Do the actual deflation.
            for key in self.d_mats:
                # Two sets of matrices to deflate: properties D and formulas F.
                #self.d_mats[key] -= (r_i[key] @ v_i[key].T) * self.not_na_d[key]

                # Step to deflate F matrix
                self.f_mats[key] -= (t_i @ pf_i[key].T) * self.not_na_f[key]


            for key in self.z_mats:
                self.z_mats[key] -= (pz_i[key] @ w_i_normalized[key]) * self.not_na_z[key]

            for key in self.y_mats:
                self.y_mats[key] -= (t_i @ q_i) * self.not_na_y[key]


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


estimator = TPLSpreprocess()
transformed_data = estimator.fit_transform(load_tpls_example())

estimator = TPLS(n_components=2)
estimator.fit(transformed_data)
# estimator.predict(transformed_data)


def test_tpls_model_fitting() -> None:
    """Test the fitting process of the TPLS model to ensure it functions as expected."""
    estimator = TPLSpreprocess()
    transformed_data = estimator.fit_transform(load_tpls_example())
    n_components = 3
    estimator = TPLS(n_components=n_components)
    estimator.fit(transformed_data)
    # assert len(estimator.fitting_statistics_["iterations"]) == [10, 8, 27]
    # assert len(estimator.fitting_statistics_["convergance_tolerance"]) == [1, 2, 3]

    # assert speY_lim95 == 13.64726477217
    # assert speZ_lim95 == 12.4194708784079
    # assert T2[0:4] == [
    #     2.51977572,
    #     2.96430904,
    #     2.90972389,
    #     4.52220244,
    #     5.08398872,
    # ]

    # assert T2_lim99 == 12.504355909323642
    # assert T2_lim95 == 8.540762689459
    # # ["speX"]["Group 5"] == all zeros
    # assert ["speX"]["Group 3"] == [
    #     0.3406894831640213,
    #     0.044638334379333455,
    #     1.0657572477657882,
    #     0.05119160460432004,
    #     0.09905322804965565,
    #     0.05119160460432004,
    #     0.08187595280818297,
    #     0.255684525938675,
    #     0.255684525938675,
    #     0.6717294145551551,
    #     0.6717294145551551,
    #     0.1572203998509645,
    #     0.07587343697961875,
    #     0.10564880243978084,
    #     0.044638334379333455,
    #     0.1572203998509645,
    #     0.28550387731283433,
    #     0.7042886720057904,
    #     0.341878106545768,
    #     0.3751050324055338,
    #     0.3406894831640213,
    #     0.3751050324055338,
    # ]

    # assert speX_lim99 == [1.8825991434391316, 0.621101338945198, 1.2349010446786568, 2.1471656812020314, 0]

    # assert speY[0:4] == [5.60884167, 2.79520778, 1.61201577, 3.44436535]
    # assert speR["Group 1"] == [167.44354056, 132.23399455, 201.50643669, 198.14628337]
    # assert speR["Group 2"] == [48.02191422, 100.16264439, 47.73820238, 1.7668637]
    # assert speR["Group 3"] == [30.96973174, 31.45714235, 30.79004185, 4.45674311]
    # assert speR["Group 4"] == [31.25128561, 31.83840754, 31.03634115, 28.89802456]
    # assert speR["Group 5"] == [30.73602305, 31.60159978, 30.49536591, 97.95906999]

    # assert speR_lim99 == [
    #     261.7356580744748,
    #     79.36465502727141,
    #     82.83811194447107,
    #     76.97092754060112,
    #     61.467202764242906,
    # ]
    # assert speZ[-4:] == [2.79721437, 2.00803271, 10.77913002, 3.26386299]
