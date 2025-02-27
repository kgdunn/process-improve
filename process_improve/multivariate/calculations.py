import time
import typing
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.stats import chi2, f
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_array, check_is_fitted

epsqrt = np.sqrt(np.finfo(float).eps)

# DELETE THESE


def hotellings_t2_limit(conf_level: float = 0.95, n_components: int = 0, n_rows: int = 0) -> float:
    """Return the Hotelling's T2 value at the given level of confidence.

    Parameters
    ----------
    conf_level : float, optional
        Fractional confidence limit, less that 1.00; by default 0.95

    Returns
    -------
    float
        The Hotelling's T2 limit at the given level of confidence.
    """
    assert 0.0 < conf_level < 1.0
    assert n_rows > 0
    if n_components == n_rows:
        return float("inf")
    return (
        n_components
        * (n_rows - 1)
        * (n_rows + 1)
        / (n_rows * (n_rows - n_components))
        * f.isf((1 - conf_level), n_components, n_rows - n_components)
    )


def spe_calculation(spe_values: np.ndarray, conf_level: float = 0.95) -> float:
    """Return a limit for SPE (squared prediction error) at the given level of confidence.

    Parameters
    ----------
    spe_values : pd.Series
        The SPE values from the last component in the multivariate model.
    conf_level : [float], optional
        The confidence level, by default 0.95, i.e. the 95% confidence level.

    Returns
    -------
    float
        The limit, above which we judge observations in the model to have a different correlation
        structure than those values which were used to build the model.
    """
    assert conf_level > 0.0, "conf_level must be a value between (0.0, 1.0)"
    assert conf_level < 1.0, "conf_level must be a value between (0.0, 1.0)"

    # The limit is for the squares (i.e. the sum of the squared errors)
    # I.e. `spe_values` are square-rooted outside this function, so undo that.
    values = spe_values**2
    center_spe = float(values.mean())
    variance_spe = float(values.var(ddof=1))
    g = variance_spe / (2 * center_spe)
    h = (2 * (center_spe**2)) / variance_spe
    # Report square root again as SPE limit
    return np.sqrt(chi2.ppf(conf_level, h) * g)


# --------------


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

        for key in X["Y"]:
            self.validate_df(X["Y"][key])
        for key in X["Z"]:
            self.validate_df(X["Z"][key])
        for key in X["D"]:
            self.validate_df(X["D"][key])
            assert key in X["F"], f"Block/group name '{key}' in D must also be present in F."
            self.validate_df(X["F"][key])  # this also ensures the keys in F are the same as in D

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
        x_transformed : dict[str, dict[str, pd.DataFrame]]
            The transformed input data, containing element-wise transformations applied to the values in the dataframes.
        """
        check_is_fitted(self)
        x_transformed: dict[str, dict[str, pd.DataFrame]] = {"D": {}, "F": {}, "Z": {}, "Y": {}}
        for key, df_d in X["D"].items():
            x_transformed["D"][key] = (
                (df_d - self.preproc_["D"][key]["center"])
                / self.preproc_["D"][key]["scale"]
                / self.preproc_["D"][key]["block"][0]  # scalar!
            )
            x_transformed["F"][key] = (X["F"][key] - self.preproc_["F"][key]["center"]) / self.preproc_["F"][key][
                "scale"
            ]
        for key in X["Z"]:
            x_transformed["Z"][key] = (X["Z"][key] - self.preproc_["Z"][key]["center"]) / self.preproc_["Z"][key][
                "scale"
            ]
        for key in X["Y"]:
            x_transformed["Y"][key] = (X["Y"][key] - self.preproc_["Y"][key]["center"]) / self.preproc_["Y"][key][
                "scale"
            ]

        return x_transformed


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


# class Plot:
# def __init__(self):
#     pass


class Plot:
    """Make plots of estimators."""

    # _common_kinds = ("line", "bar", "barh", "kde", "density", "area", "hist", "box")
    # _series_kinds = ("pie",)
    # _dataframe_kinds = ("scatter", "hexbin")
    # _kind_aliases = {"density": "kde"}
    # _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, parent: BaseEstimator) -> None:
        self._parent = parent

    # def __call__(self, *args, **kwargs):
    #    plot_backend = do_stuff(kwargs.pop("backend", None))

    def scores(self, pc_horiz: int = 1, pc_vert: int = 2, **kwargs) -> go.Figure:  # noqa: ARG002
        """Generate a scores plot."""
        print(f"generate scores plot with {pc_horiz} horizontal and {pc_vert}")  # noqa: T201

        return go.Figure()


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
        self.n_substances = 0
        self.tolerance_ = np.sqrt(np.finfo(float).eps)
        self.max_iterations_ = 500
        self.fitting_statistics: dict[str, list] = {"iterations": [], "convergance_tolerance": [], "milliseconds": []}
        self.required_blocks_ = {"D", "F", "Z", "Y"}
        self.plot = Plot(self)

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
        assert set(X.keys()) == self.required_blocks_, "The input dictionary must have keys: D, F, Z, Y."

        group_keys = [str(key) for key in X["D"]]
        assert set(X["F"]) == set(group_keys), "The keys in F must match the keys in D."
        d_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["D"][key].values) for key in group_keys}
        f_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["F"][key].values) for key in group_keys}
        z_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["Z"][key].values) for key in X["Z"]}  # only 1 key in Z
        y_mats: dict[str, np.ndarray] = {key: nan_to_zeros(X["Y"][key].values) for key in X["Y"]}  # only 1 key in Y
        self.observation_names = X["F"][group_keys[0]].index
        # corrected to iterate over all group_keys
        self.property_names = {key: X["D"][key].index.to_list() for key in group_keys}

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

        # Empty model coefficients
        self.n_substances = sum(self.f_mats[key].shape[1] for key in group_keys)
        self.n_conditions = sum(self.z_mats[key].shape[1] for key in self.z_mats)
        self.n_outputs = sum(self.y_mats[key].shape[1] for key in self.y_mats)

        # Model performance
        # -----------------
        # 1. Prediction matrices (hat matrices: for example X^)
        self.hat: dict[str, dict[str, np.ndarray]] = {key: {} for key in self.required_blocks_}
        # tss: dict[str, dict[str, np.ndarray]] = {}  # total sum of squares
        # r2_b: dict[str, dict[str, np.ndarray]] = {}  # R2 per block
        # r2_col: dict[str, dict[str, np.ndarray]] = {}  # R2 per variable (column)

        # Model parameters
        # ----------------
        self.t_scores: pd.DataFrame = pd.DataFrame(index=self.observation_names)
        # self.u_scores: pd.DataFrame = pd.DataFrame()
        # self.w_scores: pd.DataFrame = pd.DataFrame()
        # self.q_scores: pd.DataFrame = pd.DataFrame()  # corrected from {}
        self.spe: dict[str, dict[str, pd.DataFrame]] = {key: {} for key in self.required_blocks_}
        self.spe_limit: dict[str, dict[str, Callable]] = {key: {} for key in self.required_blocks_}
        self.hotellings_t2: pd.DataFrame = pd.DataFrame()
        self.hotellings_t2_limit: Callable = hotellings_t2_limit

        self.is_fitted_ = False
        self._fit_iterative_regressions()
        self.is_fitted_ = True

        return self

    def _has_converged(self, starting_vector: np.ndarray, revised_vector: np.ndarray, iterations: int) -> bool:
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

    def _store_model_coefficients(self, pc_a: int, t_super_i: np.ndarray) -> None:
        """Store the model coefficients for later use."""
        self.t_scores = self.t_scores.join(pd.DataFrame(t_super_i, index=self.observation_names, columns=[f"PC{pc_a}"]))

    def _calculate_and_store_deflation_matrices(
        self,
        t_super_i: np.ndarray,
        q_super_i: np.ndarray,
        r_i: dict[str, np.ndarray],
    ) -> None:
        """
        Calculate and store the deflation matrices for the TPLS model.

        Deflate the matrices stored in the instance object.

        Returns the prediction matrices in a dictionary.
        """
        #
        # Step 13. p_i = F_i' t_i / t_i't_i. Regress the columns of F_i on t_i; store slope coeff in vectors p_i.
        # Note: the "t" vector is the t_i vector from the inner PLS model, marked as "Tt" in figure 4 of the paper.
        # It is the score column from the super score matrix regression onto Y.
        pf_i = {
            key: regress_a_space_on_b_row(df_f.T, t_super_i.T, pmap_f.T)
            for key, df_f, pmap_f in zip(self.f_mats.keys(), self.f_mats.values(), self.not_na_f.values(), strict=True)
        }
        # Step 13: Deflate the Z matrix with a loadings vector, pz_b (_b is for block)
        pz_b = {
            key: regress_a_space_on_b_row(df_z.T, t_super_i.T, pmap_z.T)
            for key, df_z, pmap_z in zip(self.z_mats.keys(), self.z_mats.values(), self.not_na_z.values(), strict=True)
        }

        # Step 13: v_i = D_i' r_i / r_i'r_i. Regress the rows of D_i (properties) on r_i; store slopes in v_i.
        v_i = {
            key: regress_a_space_on_b_row(df_d.T, r_i[key].T, pmap_d.T)
            for key, df_d, pmap_d in zip(self.d_mats.keys(), self.d_mats.values(), self.not_na_d.values(), strict=True)
        }

        # Step 14. Do the actual deflation.
        for key in self.d_mats:
            # Two sets of matrices to deflate: properties D and formulas F.
            self.hat["D"][key] = r_i[key] @ v_i[key].T
            self.d_mats[key] -= self.hat["D"][key] * self.not_na_d[key]

            # Step to deflate F matrix
            self.hat["F"][key] = t_super_i @ pf_i[key].T
            self.f_mats[key] -= self.hat["F"][key] * self.not_na_f[key]

        for key in self.z_mats:
            self.hat["Z"][key] = t_super_i @ pz_b[key].T
            self.z_mats[key] -= self.hat["Z"][key] * self.not_na_z[key]

        for key in self.y_mats:
            self.hat["Y"][key] = t_super_i @ q_super_i.T
            self.y_mats[key] -= self.hat["Y"][key] * self.not_na_y[key]

    def _update_performance_statistics(self) -> None:
        """Calculate and store the performance statistics of the model, such as R2, TSS, etc."""

    def _calculate_model_statistics_and_limits(self) -> None:
        """Calculate and store the model limits.

        Limits calculated:
        1. Hotelling's T2 limits
        2. Squared prediction error limits
        """

        # Calculate the Hotelling's T2 values, and limits
        variance_matrix = self.t_scores.T @ self.t_scores / self.t_scores.shape[0]
        self.hotellings_t2 = np.sum((self.t_scores.values @ np.linalg.inv(variance_matrix)) * self.t_scores, axis=1)
        self.hotellings_t2_limit = partial(
            hotellings_t2_limit, n_components=self.n_components, n_rows=self.hotellings_t2.shape[0]
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
        self.spe["D"] = {
            key: pd.DataFrame(
                np.sqrt(np.sum(np.square(self.d_mats[key]), axis=1, keepdims=True)),
                index=self.property_names[key],
                columns=column_name,
            )
            for key in self.d_mats
        }
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
                    w_i = {
                        key: regress_a_space_on_b_row(df_z.T, u_super_i.T, self.not_na_z[key].T)
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }

                    # Step 8: Normalize joint w to unit length. See MB-PLS by Westerhuis et al. 1998. This is normal.
                    w_i_normalized = {key: w / np.linalg.norm(w) for key, w in w_i.items()}

                    # Step 9: regress rows of Z on w_i, and store slope coefficients in t_z. There is an error in the
                    #        paper here, but in figure 4 it is clear what should be happening.
                    t_zb = {
                        key: regress_a_space_on_b_row(df_z, w_i_normalized[key].T, self.not_na_z[key])
                        for key, df_z in zip(self.z_mats.keys(), self.z_mats.values(), strict=True)
                    }
                    t_z = np.concatenate(list(t_zb.values()), axis=1)

                else:
                    # Step 7: No Z block. Take an empty matrix across to the the superblock.
                    t_z = np.zeros((t_f.shape[0], 0))  # empty matrix: in other words, no Z block

                # Step 10: Combine t_f and t_z to form a joint t matrix.
                t_combined = np.concatenate([t_f, t_z], axis=1)

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
                # wt_i = inner_pls["w_i"]

            # After convergance. Step 12: Now store information.
            delta_gap = float(np.linalg.norm(u_prior - u_super_i, ord=None) / np.linalg.norm(u_prior, ord=None))
            self.fitting_statistics["iterations"].append(n_iter)
            self.fitting_statistics["convergance_tolerance"].append(delta_gap)
            self.fitting_statistics["milliseconds"].append((time.time() - milliseconds_start) * 1000)

            # Store model coefficients

            # self.p_f_blocks
            # self.h_f_blocks
            # self.p_z_blocks
            # self.s_d_blocks
            # self.v_d_blocks
            # self.q_y_blocks

            self._store_model_coefficients(pc_a + 1, t_super_i=t_super_i)  # , q_super_i=q_super_i, r_i=r_i)

            # Calculate and store the deflation vectors. See equation 7 on page 55.
            self._calculate_and_store_deflation_matrices(t_super_i=t_super_i, q_super_i=q_super_i, r_i=r_i)

            # Update performance statistics
            self._update_performance_statistics()

        # Step 15: Calculate the final model limit
        self._calculate_model_statistics_and_limits()

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

fitted = TPLS(n_components=3)
fitted.fit(transformed_data)
plot_settings = dict(plotwidth=1000, title="Z Space")
fitted.plot.scores(pc_horiz=1, pc_vert=2, **plot_settings)
# fitted.plot.loadings("Z")
# fitted.plot.loadings("F")
# fitted.plot.loadings("D")
# fitted.plot.loadings("D", "Y")
# fitted.plot.vip()
# fitted.plot.r2()


# Predict a new blend
# rnew = {
#     "MAT1": [("A0129", 0.557949425), ("A0130", 0.442050575)],
#     "MAT2": [("Lac0003", 1)],
#     "MAT3": [("TLC018", 1)],
#     "MAT4": [("M0012", 1)],
#     "MAT5": [("CS0017", 1)],
# }
# znew = process[process["LotID"] == "L001"]
# znew = znew.values.reshape(-1)[1:].astype(float)
# # preds = phi.tpls_pred(rnew, znew, tplsobj)
# fitted.predict(transformed_data)


def test_tpls_model_fitting() -> None:
    """Test the fitting process of the TPLS model to ensure it functions as expected."""

    transformed_data = TPLSpreprocess().fit_transform(load_tpls_example())
    n_components = 3
    tpls_test = TPLS(n_components=n_components)
    tpls_test.fit(transformed_data)

    # Ensure model is fitted appropriately, with the expected number of iterations
    assert tpls_test.fitting_statistics["iterations"] == [11, 8, 26]
    assert all(tol < epsqrt for tol in tpls_test.fitting_statistics["convergance_tolerance"])

    # Model parameters tested
    assert np.allclose(tpls_test.hotellings_t2.iloc[0:5], [2.51977572, 2.96430904, 2.90972389, 4.52220244, 5.08398872])

    # Model l imits tested
    assert tpls_test.hotellings_t2_limit(0.95) == pytest.approx(8.318089340, rel=1e-6)
    assert tpls_test.hotellings_t2_limit(0.99) == pytest.approx(12.288844, rel=1e-6)

    assert np.square(tpls_test.spe["Y"]["Quality"].iloc[0:4].values) == pytest.approx(
        [5.60884167, 2.79520778, 1.61201577, 3.44436535], rel=1e-8
    )
    # Test the last 4 observations
    assert np.square(tpls_test.spe["Z"]["Conditions"].iloc[-4:].values) == pytest.approx(
        [2.79721437, 2.00803271, 10.77913002, 3.26386299], rel=1e-8
    )

    assert np.square(tpls_test.spe["F"]["Group 1"].iloc[0:4].values) == pytest.approx(
        [167.44354056, 132.23399455, 201.50643669, 198.14628337], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 2"].iloc[0:4].values) == pytest.approx(
        [48.02191422, 100.16264439, 47.73820238, 1.7668637], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 3"].iloc[0:4].values) == pytest.approx(
        [30.96973174, 31.45714235, 30.79004185, 4.45674311], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 4"].iloc[0:4].values) == pytest.approx(
        [31.25128561, 31.83840754, 31.03634115, 28.89802456], rel=1e-8
    )
    assert np.square(tpls_test.spe["F"]["Group 5"].iloc[0:4].values) == pytest.approx(
        [30.73602305, 31.60159978, 30.49536591, 97.95906999], rel=1e-8
    )
    assert np.square(tpls_test.spe["D"]["Group 3"].iloc[0:4].values) == pytest.approx(
        [0.340689483, 0.04463833, 1.06575724, 0.0511916], rel=1e-7
    )
    # Is this is all zero because there are only 3 columns of data, and we fit 3 components?
    assert np.square(tpls_test.spe["D"]["Group 5"].iloc[0:4].values) == pytest.approx([0, 0, 0, 0], rel=1e-8)

    # Test case uses a different method to calculate the chi2 value, so we use a different tolerance
    assert tpls_test.spe_limit["Y"]["Quality"](0.95) == pytest.approx(3.7078381486450, rel=1e-8)
    assert tpls_test.spe_limit["Y"]["Quality"](0.99) == pytest.approx(4.6381115504, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 1"](0.95) == pytest.approx(1.10682253690, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 2"](0.95) == pytest.approx(0.67468300994, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 3"](0.95) == pytest.approx(0.91134417, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 4"](0.95) == pytest.approx(1.0866787434, rel=1e-8)
    assert tpls_test.spe_limit["D"]["Group 5"](0.95) == pytest.approx(0, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 1"](0.99) == pytest.approx(16.180926707, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 2"](0.99) == pytest.approx(8.4753865788, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 3"](0.99) == pytest.approx(9.1176685583, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 4"](0.99) == pytest.approx(8.7773163687, rel=1e-8)
    assert tpls_test.spe_limit["F"]["Group 5"](0.99) == pytest.approx(7.8446720428, rel=1e-8)


test_tpls_model_fitting()
