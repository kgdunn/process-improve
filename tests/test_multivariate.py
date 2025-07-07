# (c) Kevin Dunn, 2010-2025. MIT License.

import pathlib

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from scipy.sparse import csr_matrix
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from process_improve.multivariate.methods import (
    PCA,
    PLS,
    TPLS,
    DataFrameDict,
    MCUVScaler,
    SpecificationWarning,
    center,
    epsqrt,
    nan_to_zeros,
    quick_regress,
    regress_a_space_on_b_row,
    scale,
    ssq,
)

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 20
pd.options.display.width = 200
pio.renderers.default = "browser"


def test_nan_to_zeros() -> None:
    """Test the `nan_to_zeros` function."""
    in_array = np.array([[1, 2, np.nan], [4, 5, 6], [float("nan"), 8, 9]])
    out_array = nan_to_zeros(in_array)
    assert pytest.approx(out_array) == np.array([[1, 2, 0], [4, 5, 6], [0, 8, 9]])


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
    assert pytest.approx(np.array([[1, 1, 1, 1, float("nan"), 2 / 3]]).T, nan_ok=True) == regression_vector


def test_pca_spe_limits() -> None:
    """Simulate data and see if SPE limit cuts off at 5%."""
    N = 1000
    repeats = 50
    outliers_95 = []
    outliers_99 = []
    for _ in range(repeats):
        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0, 0.0])

        # The desired covariance matrix.
        r = np.array([[5.20, -4.98, -1.00], [-4.98, 5.50, 2.94], [-1.00, 2.94, 2.77]])

        rng = np.random.default_rng()
        X = pd.DataFrame(rng.multivariate_normal(mu, r, size=N))
        scaler = MCUVScaler().fit(X)
        mcuv = scaler.fit_transform(X)

        A = 2
        pca = PCA(n_components=A).fit(mcuv)
        spe_limit_95 = pca.spe_limit(conf_level=0.95)
        spe_limit_99 = pca.spe_limit(conf_level=0.99)

        outliers_95.append((pca.squared_prediction_error.iloc[:, A - 1] > spe_limit_95).sum())
        outliers_99.append((pca.squared_prediction_error.iloc[:, A - 1] > spe_limit_99).sum())

    assert np.mean(outliers_95) == pytest.approx(0.05 * N, rel=0.1)
    assert np.mean(outliers_99) == pytest.approx(0.01 * N, rel=0.1)


def test_pca_foods() -> None:
    """Arrays with no variance should not be able to have variance extracted."""

    foods = pd.read_csv("https://openmv.net/file/food-texture.csv").drop(
        [
            "Unnamed: 0",
        ],
        axis=1,
    )
    scaler = MCUVScaler().fit(foods)
    foods_mcuv = scaler.fit_transform(foods)

    A = 2
    pca = PCA(n_components=A).fit(foods_mcuv)

    assert np.linalg.norm(
        np.diag(pca.x_scores.T @ pca.x_scores) / (pca.N - 1) - pca.explained_variance_
    ) == pytest.approx(0, abs=epsqrt)

    hotellings_t2_limit_95 = pca.hotellings_t2_limit(0.95)
    assert hotellings_t2_limit_95 == pytest.approx(6.64469, rel=1e-3)

    pca.spe_limit(conf_level=0.95)

    ellipse_x, ellipse_y = pca.ellipse_coordinates(score_horiz=1, score_vert=2, conf_level=0.95, n_points=100)
    assert ellipse_x[-1] == pytest.approx(4.48792, rel=1e-5)
    assert ellipse_y[-1] == pytest.approx(0, rel=1e-7)


@pytest.fixture
def fixture_kamyr_data_missing_value() -> pd.DataFrame:
    """Load the fixture."""
    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "multivariate"
    return pd.read_csv(
        folder / "kamyr.csv",
        index_col=None,
        header=None,
    )


def test_pca_missing_data(fixture_kamyr_data_missing_value: pd.DataFrame) -> None:
    """Testing PCA with the Kamyr data set."""
    X_mcuv = MCUVScaler().fit_transform(fixture_kamyr_data_missing_value)

    # Build the model
    A = 2
    pca = PCA(n_components=A)
    assert pca.missing_data_settings is None

    # Check that default missing data options were used
    model = pca.fit(X_mcuv)
    assert isinstance(model.missing_data_settings, dict)
    assert "md_tol" in model.missing_data_settings

    assert np.linalg.norm((model.loadings.T @ model.loadings) - np.eye(model.A)) == pytest.approx(0, abs=1e-2)


def test_pca_missing_data_as_numpy(fixture_kamyr_data_missing_value: pd.DataFrame) -> None:
    """Test the PCA model with missing data."""
    X_mcuv = MCUVScaler().fit_transform(fixture_kamyr_data_missing_value.values)

    # Build the model
    A = 2
    pca = PCA(n_components=A)
    assert pca.missing_data_settings is None

    # Check that default missing data options were used
    model = pca.fit(X_mcuv)
    assert isinstance(model.missing_data_settings, dict)
    assert "md_tol" in model.missing_data_settings

    assert np.linalg.norm((model.loadings.T @ model.loadings) - np.eye(model.A)) == pytest.approx(0, abs=1e-2)


@pytest.fixture
def fixture_mv_utilities() -> tuple[np.ndarray, np.ndarray]:
    """
    Multivariate methods depend on an internal regression and Sum of Squares
    calculations. This code tests those crucial steps.
    """
    x = np.asarray([1, 2, 3, 4, 5, 6]).reshape(6, 1)
    Y = np.asarray(
        [
            [1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, np.nan, 3, np.nan, 5, np.nan],
        ]
    )
    Y = Y.T
    return x, Y


def test_ssq(fixture_mv_utilities: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the sum-of-squares calculation."""
    x, _ = fixture_mv_utilities
    assert pytest.approx(ssq(x), abs=1e-9) == (1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6)


def test_quick_regress(fixture_mv_utilities: tuple[np.ndarray, np.ndarray]) -> None:
    """Test the quick_regress function."""
    x, Y = fixture_mv_utilities
    out = quick_regress(Y, x).ravel()
    assert pytest.approx(out[0], abs=1e-9) == 1
    assert pytest.approx(out[1], abs=1e-8) == 0.61538462
    assert pytest.approx(out[2], abs=1e-9) == 0

    # Checked against R: summary(lm(c(1,1,1,1,1,1) ~ seq(6) + 0))
    assert pytest.approx(out[3], abs=1e-6) == 0.23077

    # Checked against what is expected: (1 + 3^2 + 5^2)/(1 + 3^2 + 5^2)
    assert pytest.approx(out[4], abs=1e-14) == 1.0


@pytest.fixture
def fixture_tablet_spectra_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Verify the PCA model for the case of no missing data.

    # R code:
    # -------
    # Read large data file
    file <- 'http://openmv.net/file/tablet-spectra.csv'
    spectra <- read.csv(file, header = FALSE, row.names = 1)
    # Only extract 4 components, but
    # center and scale the data before
    # calculation the components
    model.pca <- prcomp(spectra,
                        center = TRUE,
                        scale =TRUE,
                        rank. = 4)
    summary(model.pca)

    Importance of first k=4 (out of 460) components:
                               PC1     PC2     PC3     PC4
    Standard deviation     21.8835 10.9748 3.60075 3.27081
    Proportion of Variance  0.7368  0.1853 0.01995 0.01646
    Cumulative Proportion   0.7368  0.9221 0.94200 0.95846


    # T' * T on the scores matrix T:
    t(model.pca$x) %*% model.pca$x

                  PC1          PC2           PC3          PC4
    PC1  2.198092e+05 6.885159e-11 -1.134026e-11 3.454659e-11
    PC2  6.885159e-11 5.528459e+04  2.042206e-10 5.821477e-11
    PC3 -1.134026e-11 2.042206e-10  5.951125e+03 7.815970e-13
    PC4  3.454659e-11 5.821477e-11  7.815970e-13 4.910481e+03
    """
    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "multivariate"
    spectra = pd.read_csv(
        folder / "tablet-spectra.csv",
        index_col=0,
        header=None,
    )

    # Ignoring values < 1E-8 (round them to zero) from the R output above.
    known_scores_covar = np.array(
        [
            [2.198092e05, 0, 0, 0],
            [0, 5.528459e04, 0, 0],
            [0, 0, 5.951125e03, 0],
            [0, 0, 0, 4.910481e03],
        ]
    )
    return spectra, known_scores_covar


def test_mcuv_centering(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Mean centering of the testing data."""

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)
    assert pytest.approx(np.max(np.abs(X_mcuv.mean(axis=0))), rel=1e-9) == 0.0


def test_mcuv_scaling(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Scaling by standard deviation."""

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)

    assert pytest.approx(np.min(np.abs(X_mcuv.std(axis=0))), 1e-10) == 1
    assert pytest.approx(X_mcuv.std(), 1e-10) == 1


def test_pca_tablet_spectra(fixture_tablet_spectra_data: tuple[pd.DataFrame, np.ndarray]) -> None:
    r"""
    Check PCA characteristics.

    1. model's loadings must be orthogonal if there are no missing data.
        P.T * P = I
    2. model's loadings must be of unit length (norm = 1.0)
        P.T * P = I
    3. model's scores must be orthogonal
        T.T * T is a diagonal matrix when there's no missing data
    4. each earlier score's variance must be >= variance of later score


    PCA models have the following properties:

    * :math:`p_i'p_j' = 0` for :math:`i\neq j`; i.e. :math:`p_i \perp p_j`
    * :math:`t_i't_j' = 0` for :math:`i\neq j`; i.e. :math:`t_i \perp t_j`
    * :math:`P'P = I_A` when extracting :math:`A` components
    * :math:`P_{all} \text{ is a } \min(N,K) \times \min(N,K)` matrix, for all components
    * :math:`T_{all} \text{ is a } \min(N,K) \times \min(N,K)` matrix, for all components
    (it is just a rearrangement of X)
    * :math:`\text{SVD}(X): UDV' = X` and :math:`V' = P'` and :math:`UD = T`
    """

    spectra, known_scores_covar = fixture_tablet_spectra_data

    # Number of components to calculate
    model = PCA(n_components=2)
    model.fit(scale(center(spectra)))

    # P'P = identity matrix of size A x A
    orthogonal_check = model.loadings.T @ model.loadings
    assert pytest.approx(np.linalg.norm(orthogonal_check - np.eye(model.A)), rel=1e-9) == 0.0

    # Check the R2 value against the R software output
    assert model.R2cum[1] == pytest.approx(0.7368, rel=1e-3)
    assert model.R2cum[2] == pytest.approx(0.9221, rel=1e-2)

    # Unit length: actually checked above, via subtraction with I matrix.
    # Check if scores are orthogonal
    scores_covar = pd.DataFrame(model.x_scores.T @ model.x_scores).astype(float)
    for i in range(model.A):
        for j in range(model.A):
            # Technically not need, but more explict this way.
            if i == j:
                assert scores_covar.iloc[i, j] == pytest.approx(known_scores_covar[i, j], rel=1e-2)
            else:
                assert scores_covar.iloc[i, j] == pytest.approx(known_scores_covar[i, j], abs=1e-4)

                # if i >= 1: Creates unnecessary type errors and not such a useful check.
                #    assert scores_covar.iloc[j, j] > scores_covar.iloc[i, i]

    # Check the model against an SVD: this raw data set has no missing
    # data, so the SVD should be faster and more accurate than NIPALS
    autoscaled_X = scale(center(spectra))
    u, s, v = np.linalg.svd(autoscaled_X)

    loadings_delta = np.linalg.norm(np.abs(v[0 : model.A, :]) - np.abs(model.loadings.T))
    assert loadings_delta == pytest.approx(0, abs=1e-8)

    # It is not possible, it seems, to get the scores to match the SVD
    # scores. Numerical error?


def test_pca_errors_no_variance_to_start() -> None:
    """Arrays with no variance should seem to work, but should have no variability explained."""
    K, N, A = 17, 12, 5
    data = pd.DataFrame(np.zeros((N, K)))
    model = PCA(n_components=A)
    # with pytest.raises(RuntimeError):
    model.fit(data)
    assert np.sum(model.x_scores.values) == pytest.approx(0, abs=epsqrt)
    assert model.R2cum.sum() == pytest.approx(0, abs=epsqrt)
    assert np.isnan(model.R2cum[A - 1])


def test_pca_invalid_calls() -> None:
    """Tests various invalid calls, and corresponding error messages."""
    K, N, A = 4, 3, 5
    rng = np.random.default_rng()
    data = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, K)))
    model = PCA(n_components=A)
    with pytest.warns(
        SpecificationWarning,
        match=r"The requested number of components is more than can be computed from data(.*)",
    ):
        model.fit(data)
    data.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError, match="Tolerance must exceed machine precision"):
        _ = PCA(n_components=A, missing_data_settings=dict(md_method="nipals", md_tol=0)).fit(data)

    with pytest.raises(AssertionError, match=r"Missing data method is not recognized(.*)"):
        _ = PCA(n_components=A, missing_data_settings={"md_method": "SCP"}).fit(data)

    # TODO: replace with a check to ensure the data is in a DataFrame.
    # from scipy.sparse import csr_matrix
    # sparse_data = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    # with pytest.raises(TypeError, match="This PCA class does not support sparse input."):
    #     model = PCA(n_components=2)
    #     model.fit(sparse_data)


def test_pca_no_more_variance() -> None:
    """Create a rank 2 matrix and it should fail on the 3rd component."""

    K = 17
    N = 12
    A = 3
    rng = np.random.default_rng()
    T = rng.uniform(low=-1, high=1, size=(N, 2))
    P = rng.uniform(low=-1, high=1, size=(K, 2))
    X = T @ P.T
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0, ddof=0)
    _ = pd.DataFrame((X - meanX) / stdX)
    _ = PCA(n_components=A)

    # with pytest.raises(RuntimeError):
    #    m.fit(X)
    # TODO: check that the m.R2[2] (3rd PC is zero.)


def test_pca_columns_with_no_variance() -> None:
    """Create a column with no variance. That column's loadings should be 0."""
    K = 14
    N = 29
    A = 4
    rng = np.random.default_rng()
    cols_with_no_variance = [10, 3]
    T = rng.uniform(low=-1, high=1, size=(N, A))
    P = rng.uniform(low=-1, high=1, size=(K, A))
    X = T @ P.T
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0, ddof=0)
    x_to_fit = pd.DataFrame((X - meanX) / stdX)
    x_to_fit.iloc[:, cols_with_no_variance] = 0

    m = PCA(n_components=2)
    m.fit(x_to_fit)

    # `loadings` is a K by A matrix.  Check sum of loadings in rows with
    # no variance must be zero
    assert np.sum(np.abs(m.loadings.iloc[cols_with_no_variance, :].values)) == pytest.approx(0, abs=1e-14)
    # The loadings must still be orthonormal though:
    assert np.sum(np.identity(m.A) - m.loadings.values.T @ m.loadings.values) == pytest.approx(0, abs=1e-14)

    # Are scores orthogonal?
    covmatrix = m.x_scores.T @ m.x_scores
    covmatrix - np.diag(np.diag(covmatrix))
    assert (np.sum(np.abs(covmatrix - np.diag(np.diag(covmatrix))))).values == pytest.approx(0, abs=1e-6)


@pytest.fixture
def fixture_pca_pca_wold_etal_paper() -> pd.DataFrame:
    """
    Return data from the PCA paper by Wold, Esbensen and Geladi, 1987.

    Principal Component Analysis, Chemometrics and Intelligent Laboratory
    Systems, v 2, p37-52; http://dx.doi.org/10.1016/0169-7439(87)80084-9
    """
    return pd.DataFrame(np.array([[3, 4, 2, 2], [4, 3, 4, 3], [5.0, 5, 6, 4]]))


def test_pca_wold_centering(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Checks the centering step."""
    _, centering = center(fixture_pca_pca_wold_etal_paper, extra_output=True)
    assert centering == pytest.approx([4, 4, 4, 3], rel=1e-8)


def test_pca_wold_scaling(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Checks the scaling step. Page 40 of the above paper."""

    _, scaling = scale(center(fixture_pca_pca_wold_etal_paper), extra_output=True, ddof=1)
    assert scaling == pytest.approx([1, 1, 0.5, 1])


def test_pca_wold_model_results(fixture_pca_pca_wold_etal_paper: pd.DataFrame) -> None:
    """Check if the PCA model matches the results in the paper."""

    X_preproc = scale(center(fixture_pca_pca_wold_etal_paper))
    pca_1 = PCA(n_components=1)
    pca_1.fit(X_preproc.copy())

    # TODO: complete these tests

    # The remaining sum of squares, on page 43
    # SS_X = np.sum(pca_1["residuals"].values ** 2, axis=0)
    # self.assertTrue(
    #     np.all(compare_entries(SS_X, np.array([0.0551, 1.189, 0.0551, 0.0551]), 3))
    # )

    # # The residuals after 1 component
    # self.assertTrue(
    #     np.all(compare_entries(SS_X, np.array([0.0551, 1.189, 0.0551, 0.0551]), 3))
    # )

    # # With 2 components, the loadings are, page 40
    # P.T = [ 0.5410, 0.3493,  0.5410,  0.5410],
    #      [-0.2017, 0.9370, -0.2017, -0.2017]
    X_preproc = scale(center(fixture_pca_pca_wold_etal_paper))
    pca_2 = PCA(n_components=2)
    pca_2.fit(X_preproc)
    assert np.abs(pca_2.loadings.values[:, 0]) == pytest.approx([0.5410, 0.3493, 0.5410, 0.5410], abs=1e-4)
    assert np.abs(pca_2.loadings.values[:, 1]) == pytest.approx([0.2017, 0.9370, 0.2017, 0.2017], abs=1e-4)

    # Scores. The scaling is off here by a constant factor of 0.8165
    # assert np.all(pca_2.x_scores["1"] == pytest.approx([-1.6229, -0.3493, 1.9723], rel=1e-3))
    # assert np.all(pca_2.x_scores["2"] == pytest.approx([0.6051, -0.9370, 0.3319], rel=1e-4))

    # R2 values, given on page 43
    assert pca_2.R2.values == pytest.approx([0.831, 0.169], rel=1e-2)

    # SS values, on page 43
    # SS_X = np.sum(X_preproc ** 2, axis=0)
    # assert SS_X == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)

    # Testing data:
    # X_test = Block(np.array([[3, 4, 3, 4], [1, 2, 3, 4.0]]))
    # X.preprocess(X_test)
    # compare_entries(X_test.data, np.array([[-1, 0, -0.5, 1],
    # [-3, -2, -0.5, 1]])
    # #testing = PCA_model.apply(X_test)
    # compare_entries(testing.T, np.array([[-0.2075, 0.1009],
    # [-2.0511, -1.3698]])


def test_pls_properties_todo() -> None:
    """
    Complete this later.

    TODO:
    diag(T.T * T) related to S
    W.T * W = I for PLS only
    P.T * W: ones on diagonal, zeros below diagonal
    W.T * R: ones on diagonal, zeros below diagonal
    R.T * P = ID
    """


@pytest.mark.skip(reason="API still has to be improved to handle this case")
def test_pls_invalid_calls() -> None:
    """Tests various invalid calls, and corresponding error messages."""
    K, N, M, A = 4, 3, 2, 5
    rng = np.random.default_rng()
    data_x = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, K)))
    data_y = pd.DataFrame(rng.uniform(low=-1, high=1, size=(N, M)))
    with pytest.raises(ValueError, match="Tolerance `tol`` must be between 1E-16 and 1.0"):
        _ = PLS(n_components=A, tol=0)

    # The `method` input parameter does not exist anymore
    with pytest.raises(ValueError, match="Method 'SVDS' is not known."):
        _ = PLS(n_components=A, method="SVDS")

    with pytest.raises(ValueError, match="Missing data method 'SCP' is not known."):
        _ = PLS(n_components=A, md_method="SCP")

    model = PLS(n_components=A)
    with pytest.warns(SpecificationWarning, match=r"The requested number of components is (.*)"):
        model.fit(data_x, data_y)

    sparse_data = csr_matrix([[1, 2], [0, 3], [4, 5]])
    with pytest.raises(TypeError, match="This PLS class does not support sparse input."):
        PLS(n_components=2).fit(data_x, sparse_data)


@pytest.fixture
def fixture_pls_model_simca_1_component() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test simple model against Simca-P, version 14.1.

    Testing on 28 June 2020.

    When X and y are mean centered and scaled, the model should provide the loadings as listed here

    TODO: test against R

    X = matrix(c(41.1187, 21.2833, 21.1523,  0.2446, -0.0044, -0.131,  1.12,
                 41.7755, 22.0978, 21.1653,  0.3598,  0.1622, -0.9325, 1.01,
                 41.2568, 21.4873, 20.7407,  0.2536,  0.1635, -0.7467, 0.97,
                 41.5469, 22.2043, 20.4518,  0.6317,  0.1997, -1.7525, 0.83,
                 40.0234, 23.7399, 21.978,  -0.0534, -0.0158, -1.7619, 0.93,
                 39.9203, 21.9997, 21.5859, -0.1811,  0.089,  -0.4138, 1.02,
                 42.1886, 21.4891, 20.4427,  0.686,   0.1124, -1.0464, 0.91,
                 42.1454, 20.3803, 18.2327,  0.6607,  0.1291, -2.1476, 0.70,
                 42.272,  18.9725, 18.3763,  0.561,   0.0453, -0.5962, 1.26,
                 41.49,   18.603,  17.9978,  0.4872,  0.1198, -0.6052, 1.05,
                 41.5306, 19.1558, 18.2172,  0.6233,  0.1789, -0.9386, 0.95), nrow=11, byrow=T)

    data = data.frame(X)
    colnames(data) <- c("A","B","C", "D", "E", "F", "y")

    library(pls)
    model = plsr(y~., 1, data=data, method="simpls")
    plot(model)

    self.expected_y_predicted = [
        1.17475,
        0.930441,
        0.979066,
        0.773083,
        0.945719,
        1.10583,
        0.929441,
        0.796271,
        1.09379,
        1.0635,
        0.958111,
        ]
    """
    data: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    data["X"] = pd.DataFrame(
        np.array(
            [
                [
                    41.1187,
                    21.2833,
                    21.1523,
                    0.2446,
                    -0.0044,
                    -0.131,
                ],
                [
                    41.7755,
                    22.0978,
                    21.1653,
                    0.3598,
                    0.1622,
                    -0.9325,
                ],
                [
                    41.2568,
                    21.4873,
                    20.7407,
                    0.2536,
                    0.1635,
                    -0.7467,
                ],
                [
                    41.5469,
                    22.2043,
                    20.4518,
                    0.6317,
                    0.1997,
                    -1.7525,
                ],
                [
                    40.0234,
                    23.7399,
                    21.978,
                    -0.0534,
                    -0.0158,
                    -1.7619,
                ],
                [
                    39.9203,
                    21.9997,
                    21.5859,
                    -0.1811,
                    0.089,
                    -0.4138,
                ],
                [
                    42.1886,
                    21.4891,
                    20.4427,
                    0.686,
                    0.1124,
                    -1.0464,
                ],
                [
                    42.1454,
                    20.3803,
                    18.2327,
                    0.6607,
                    0.1291,
                    -2.1476,
                ],
                [
                    42.272,
                    18.9725,
                    18.3763,
                    0.561,
                    0.0453,
                    -0.5962,
                ],
                [
                    41.49,
                    18.603,
                    17.9978,
                    0.4872,
                    0.1198,
                    -0.6052,
                ],
                [
                    41.5306,
                    19.1558,
                    18.2172,
                    0.6233,
                    0.1789,
                    -0.9386,
                ],
            ]
        )
    )

    data["y"] = pd.DataFrame(np.array([1.12, 1.01, 0.97, 0.83, 0.93, 1.02, 0.91, 0.7, 1.26, 1.05, 0.95]))
    data["expected_y_predicted"] = np.array(
        [
            1.17475,
            0.930441,
            0.979066,
            0.773083,
            0.945719,
            1.10583,
            0.929441,
            0.796271,
            1.09379,
            1.0635,
            0.958111,
        ]
    )
    data["loadings_P1"] = np.array(
        [
            -0.2650725,
            -0.2165038,
            0.08547913,
            -0.3954746,
            -0.4935882,
            0.7541404,
        ]
    )
    data["loadings_r1"] = np.array(
        [
            -0.04766187,
            -0.3137862,
            0.004006641,
            -0.238001,
            -0.4430451,
            0.8039384,
        ]
    )
    data["loadings_y_c1"] = 0.713365
    data["SDt"] = 1.19833
    data["R2X"] = 0.261641
    data["R2Y"] = 0.730769
    data["t1"] = np.array(
        [
            1.889566,
            -0.4481195,
            0.0171578,
            -1.953837,
            -0.3019302,
            1.230112,
            -0.4576912,
            -1.731961,
            1.114923,
            0.8251334,
            -0.1833536,
        ]
    )
    data["Tsq"] = np.array(
        [
            2.48638,
            0.1398399,
            0.0002050064,
            2.658398,
            0.0634829,
            1.053738,
            0.1458776,
            2.08891,
            0.8656327,
            0.4741239,
            0.02341113,
        ]
    )
    data["DModX"] = np.array(
        [
            0.367926,
            1.01727,
            0.970395,
            0.635592,
            2.36596,
            0.449567,
            0.645429,
            1.12458,
            0.520623,
            0.384443,
            0.764301,
        ]
    )
    data["Xavg"] = np.array([41.38802, 21.03755, 20.03097, 0.3884909, 0.1072455, -1.006582])
    data["Xws"] = 1 / np.array(
        [
            1.259059,
            0.628138,
            0.6594034,
            3.379028,
            13.8272,
            1.589986,
        ]
    )
    data["Yavg"] = 0.9772727
    data["Yws"] = 1 / 6.826007  # Simca-P uses inverse standard deviation
    data["A"] = 1
    data["conf"] = 0.95
    return data


def test_pls_compare_sklearn_1_component(fixture_pls_model_simca_1_component: dict) -> None:
    """Test PLS with 1 component."""
    data = fixture_pls_model_simca_1_component

    plsmodel = PLSRegression(n_components=data["A"], scale=True)
    plsmodel.fit(data["X"], data["y"])

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert plsmodel._x_mean == pytest.approx(data["Xavg"], abs=1e-5)
    assert plsmodel._x_std == pytest.approx(data["Xws"], abs=1e-6)
    assert plsmodel._y_mean == pytest.approx(data["Yavg"], abs=1e-7)
    assert plsmodel._y_std == pytest.approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    T = plsmodel.x_scores_
    P = plsmodel.x_loadings_
    assert T.ravel() == pytest.approx(data["t1"], abs=1e-5)
    assert np.std(T, ddof=1) == pytest.approx(data["SDt"], rel=1e-5)
    assert data["loadings_P1"].ravel() == pytest.approx(P.ravel(), rel=1e-5)
    assert data["loadings_r1"] == pytest.approx(plsmodel.x_weights_.ravel(), rel=1e-4)

    # Check the model's predictions
    t1_predict, y_pp = plsmodel.transform(data["X"], data["y"])
    assert data["t1"] == pytest.approx(t1_predict.ravel(), abs=1e-5)
    # assert y_pp == pytest.approx((data["y"] - data["Yavg"]) / data["Yws"], abs=1e-6)

    # Manually make the PLS prediction
    # X_check = data["X"].copy()
    # X_check_mcuv = (X_check - plsmodel._x_mean) / plsmodel._x_std
    # t1_predict_manually = X_check_mcuv @ plsmodel.x_weights_

    # TODO: fix the rest of this test. Not sure what the purpose of this test is anyway.

    # # Simca's C:
    # N = data["X"].shape[0]
    # simca_C = (y_pp.reshape(1, N) @ t1_predict) / (t1_predict.T @ t1_predict)
    # # assert simca_C == pytest.approx(data["loadings_y_c1"], 1e-6)
    # assert t1_predict_manually.values.ravel() == pytest.approx(t1_predict.ravel(), 1e-9)

    # # Deflate the X's:
    # X_check_mcuv -= t1_predict_manually @ plsmodel.x_loadings_.T
    # y_hat = t1_predict_manually @ simca_C
    # y_hat_rawunits = y_hat * plsmodel._y_std + plsmodel._y_mean
    # assert data["expected_y_predicted"] == pytest.approx(y_hat_rawunits.values.ravel(), abs=1e-5)

    # prediction_error = data["y"].values - y_hat_rawunits.values
    # R2_y = (data["y"].var(ddof=1) - prediction_error.var(ddof=1)) / data["y"].var(ddof=1)
    # assert R2_y == pytest.approx(data["R2Y"], abs=1e-6)


def test_pls_compare_model_api(
    fixture_pls_model_simca_1_component: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Test two variants of the PLS model."""
    data = fixture_pls_model_simca_1_component
    assert isinstance(data["X"], pd.DataFrame)
    assert isinstance(data["y"], pd.DataFrame)
    assert isinstance(data["A"], int)
    plsmodel = PLS(n_components=int(data["A"]))
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(np.array(data["y"]))

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert X_mcuv.center_.values == pytest.approx(data["Xavg"], abs=1e-5)
    assert X_mcuv.scale_.values == pytest.approx(data["Xws"], abs=1e-6)
    assert Y_mcuv.center_.values == pytest.approx(data["Yavg"], abs=1e-7)
    assert Y_mcuv.scale_.values == pytest.approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(np.array(data["y"])))
    assert data["SDt"] == pytest.approx(np.std(plsmodel.x_scores, ddof=1), abs=1e-5)
    assert data["t1"] == pytest.approx(plsmodel.x_scores.values.ravel(), abs=1e-5)
    assert data["loadings_P1"] == pytest.approx(plsmodel.x_loadings.values.ravel(), abs=1e-5)
    assert data["loadings_r1"] == pytest.approx(plsmodel.x_weights.values.ravel(), abs=1e-6)

    assert np.array(data["expected_y_predicted"]).ravel() == pytest.approx(
        Y_mcuv.inverse_transform(plsmodel.predictions).values.ravel(), abs=1e-5
    )
    assert data["R2Y"] == pytest.approx(plsmodel.R2cum, abs=1e-6)

    # Check the model's predictions
    state = plsmodel.predict(X_mcuv.transform(data["X"]))
    assert plsmodel.squared_prediction_error.values.ravel() == pytest.approx(
        state.squared_prediction_error.values, abs=1e-9
    )
    assert data["t1"] == pytest.approx(state.x_scores.values.ravel(), abs=1e-5)
    assert data["Tsq"] == pytest.approx(state.hotellings_t2.values.ravel(), abs=1e-5)
    assert data["expected_y_predicted"] == pytest.approx(Y_mcuv.inverse_transform(state.y_hat).values.ravel(), abs=1e-5)


@pytest.fixture
def fixture_pls_simca_2_components() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test simple model against Simca-P, version 14.1.

    Testing on 02 July 2020.
    No missing data

    When X and y are mean centered and scaled, the model should provide the loadings listed here.
    """
    out: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    out["X"] = pd.DataFrame(
        np.array(
            [
                [
                    1.27472,
                    0.897732,
                    -0.193397,
                ],
                [
                    1.27472,
                    -1.04697,
                    0.264243,
                ],
                [
                    0.00166722,
                    1.26739,
                    1.06862,
                ],
                [
                    0.00166722,
                    -0.0826556,
                    -1.45344,
                ],
                [
                    0.00166722,
                    -1.46484,
                    1.91932,
                ],
                [
                    -1.27516,
                    0.849516,
                    -0.326239,
                ],
                [
                    -1.27516,
                    -1.06304,
                    0.317718,
                ],
                [
                    -0.000590006,
                    1.26739,
                    1.06862,
                ],
                [
                    -0.000590006,
                    -0.0826556,
                    -1.45344,
                ],
                [
                    -0.000590006,
                    -1.09519,
                    0.427109,
                ],
                [
                    -1.27516,
                    0.849516,
                    -0.326239,
                ],
                [
                    -1.27516,
                    -1.06304,
                    0.317718,
                ],
                [
                    1.27398,
                    0.897732,
                    -0.193397,
                ],
                [
                    1.27398,
                    -0.130872,
                    -1.4372,
                ],
            ]
        )
    )

    out["y"] = pd.DataFrame(
        np.array(
            [
                -0.0862851,
                -1.60162,
                0.823439,
                0.242033,
                -1.64304,
                1.59583,
                -0.301604,
                0.877623,
                0.274155,
                -0.967692,
                1.47491,
                -0.194163,
                0.097352,
                -0.590925,
            ]
        )
    )
    out["expected_y_predicted"] = np.array(
        [
            0.04587483,
            -1.671657,
            0.8337691,
            0.2326966,
            -1.622544,
            1.520105,
            -0.209406,
            0.8350853,
            0.2340128,
            -1.002176,
            1.520105,
            -0.209406,
            0.04630876,
            -0.552768,
        ]
    )
    out["loadings_P"] = np.array([[-0.3799977, -0.7815778], [0.8737038, -0.2803103], [-0.3314019, 0.55731]])
    out["loadings_W"] = np.array([[-0.4839311, -0.7837874], [0.8361799, -0.2829775], [-0.2580969, 0.5528119]])  # W
    out["loadings_C"] = np.array([1.019404, 0.1058565])
    out["SDt"] = np.array([0.9724739, 1.098932])
    out["R2X"] = np.array(
        [
            0.3207782,
            0.4025633,
        ]
    )  # cumulative: 32%, then 72% for second component
    out["R2Y"] = np.array([0.9827625, 0.01353244])
    out["T"] = np.array(
        [
            [0.1837029, -1.335702],
            [-1.560534, -0.7636986],
            [0.7831483, 0.334647],
            [0.3052059, -0.7409231],
            [-1.721048, 1.246014],
            [1.411638, 0.7658994],
            [-0.3538088, 1.428992],
            [0.7842407, 0.3365611],
            [0.3062983, -0.7390091],
            [-1.025724, 0.4104719],
            [1.411638, 0.7658994],
            [-0.3538088, 1.428992],
            [0.184063, -1.335071],
            [-0.3550123, -1.803074],
        ]
    )
    # TODO: test against this still
    out["Tsq"] = np.array(
        [
            1.513014,
            3.05803,
            0.7412658,
            0.5530728,
            4.417653,
            2.592866,
            1.823269,
            0.74414,
            0.5514336,
            1.252029,
            2.592866,
            1.823269,
            1.511758,
            2.825334,
        ]
    )

    # TODO: test against this still
    out["DModX"] = np.array(
        [
            0.8796755,
            0.2482767,
            1.641307,
            1.350485,
            0.9410046,
            0.4101084,
            0.856736,
            1.640294,
            1.351499,
            0.2035393,
            0.4101084,
            0.856736,
            0.8793415,
            0.7906867,
        ]
    )
    out["A"] = 2
    return out


def test_pls_sklearn_2_components(fixture_pls_simca_2_components: dict) -> None:
    """Test the Scikit model against the Simca-P model."""
    data = fixture_pls_simca_2_components

    plsmodel = PLSRegression(n_components=data["A"], scale=False)

    X_mcuv = MCUVScaler().fit_transform(data["X"])
    Y_mcuv = MCUVScaler().fit_transform(data["y"])

    plsmodel.fit(X_mcuv, Y_mcuv)

    # Extract the model parameters
    assert np.abs(data["T"]) == pytest.approx(np.abs(plsmodel.x_scores_), abs=1e-5)
    assert np.std(plsmodel.x_scores_, ddof=1, axis=0) == pytest.approx(data["SDt"], abs=1e-6)
    assert np.abs(data["loadings_P"]) == pytest.approx(np.abs(plsmodel.x_loadings_), abs=1e-5)
    assert np.abs(data["loadings_W"]) == pytest.approx(np.abs(plsmodel.x_weights_), abs=1e-5)


def test_pls_compare_api(fixture_pls_simca_2_components: dict) -> None:
    """Test PLS comparison between two different methods."""
    data = fixture_pls_simca_2_components

    plsmodel = PLS(n_components=data["A"])

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["y"]))

    # Extract the model parameters
    assert data["SDt"] == pytest.approx(np.std(plsmodel.x_scores, ddof=1, axis=0), abs=1e-6)
    assert np.abs(data["T"]) == pytest.approx(np.abs(plsmodel.x_scores), abs=1e-5)
    assert np.abs(data["loadings_P"]) == pytest.approx(np.abs(plsmodel.x_loadings), abs=1e-5)
    assert np.abs(data["loadings_W"]) == pytest.approx(np.abs(plsmodel.x_weights), abs=1e-5)
    assert Y_mcuv.inverse_transform(plsmodel.predictions).values == pytest.approx(
        data["expected_y_predicted"].reshape(-1, 1), abs=1e-5
    )
    assert sum(data["R2Y"]) == pytest.approx(plsmodel.R2cum.values[-1], abs=1e-7)

    # Check the model's predictions
    state = plsmodel.predict(X_mcuv.transform(data["X"]))
    # TODO: a check on SPE vs Simca-P. Here we are doing a check between the SPE from the
    # model building, to model-using, but not against an external library.
    assert plsmodel.squared_prediction_error.iloc[:, -1].values == pytest.approx(
        state.squared_prediction_error, abs=1e-10
    )
    assert data["Tsq"] == pytest.approx(state.hotellings_t2, abs=1e-5)
    assert data["expected_y_predicted"] == pytest.approx(Y_mcuv.inverse_transform(state.y_hat).values.ravel(), abs=1e-5)
    assert np.abs(data["T"]) == pytest.approx(np.abs(state.x_scores), abs=1e-5)


@pytest.fixture
def fixture_pls_ldpe_example() -> dict[str, pd.DataFrame | np.ndarray | float | int]:
    """
    Test PLS example with no missing data.

    Source: https://openmv.net/info/ldpe

    Data from a low-density polyethylene production process.
    There are 14 process variables and 5 quality variables (last 5 columns).
    More details: http://dx.doi.org/10.1002/aic.690400509
    The first 50 observations are from common-cause (normal) operation, while the last 4 show a
    process fault developing: the impurity level in the ethylene feed in both zones is increasing.

    Tin: inlet temperature to zone 1 of the reactor [K]
    Tmax1: maximum temperature along zone 1 [K]
    Tout1: outlet temperature from zone 1 [K]
    Tmax2: maximum temperature along zone 2 [K]
    Tout2: outlet temperature from zone 2 [K]
    Tcin1: temperature of inlet coolant to zone [K]
    Tcin2: temperature of inlet coolant to zone 2 [K]
    z1: percentage along zone 1 where Tmax1 occurs [%]
    z2: percentage along zone 2 where Tmax2 occurs [%]
    Fi1: flow rate of initiators to zone 1 [g/s]
    Fi2: flow rate of initiators to zone 2 [g/s]np.abs(state.scores)
    Fs1: flow rate of solvent to zone 1 [% of ethylene]
    Fs2: flow rate of solvent to zone 2 [% of ethylene]
    Press: pressure in the reactor [atm]
    ------
    Conv: quality variable: cumulative conversion
    Mn: quality variable: number average molecular weight
    Mw: quality variable: weight average molecular weight
    LCB: quality variable: long chain branching per 1000 C atoms
    SCB: quality variable: short chain branching per 1000 C atoms

    N = 54
    K = 14
    M = 5
    A = 6
    """
    out: dict[str, pd.DataFrame | np.ndarray | float | int] = {}
    folder = pathlib.Path(__file__).parents[1] / "process_improve" / "datasets" / "multivariate"
    values = pd.read_csv(
        folder / "LDPE" / "LDPE.csv",
        index_col=0,
    )
    out["expected_T"] = pd.read_csv(folder / "LDPE" / "T.csv", header=None)
    out["expected_W"] = pd.read_csv(folder / "LDPE" / "W.csv", header=None)
    out["expected_P"] = pd.read_csv(folder / "LDPE" / "P.csv", header=None)
    out["expected_C"] = pd.read_csv(folder / "LDPE" / "C.csv", header=None)
    out["expected_U"] = pd.read_csv(folder / "LDPE" / "U.csv", header=None)
    out["expected_hotellings_t2_a3"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A3.csv",
        header=None,
    )
    out["expected_hotellings_t2_a6"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A6.csv",
        header=None,
    )
    out["expected_yhat_a6"] = pd.read_csv(
        folder / "LDPE" / "Yhat_A6.csv",
        header=None,
    )
    out["expected_sd_t"] = np.array([1.872539, 1.440642, 1.216218, 1.141096, 1.059435, 0.9459715])
    out["expected_t2_lim_95_a6"] = 15.2017
    out["expected_t2_lim_99_a6"] = 21.2239
    out["X"] = np.array(values.iloc[:, :14])
    out["Y"] = np.array(values.iloc[:, 14:])
    out["A"] = 6
    assert isinstance(out["X"], np.ndarray | pd.DataFrame)
    assert out["X"].shape == (54, 14)
    assert isinstance(out["Y"], np.ndarray | pd.DataFrame)
    assert out["Y"].shape == (54, 5)

    return out


def test_pls_simca_ldpe(fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int]) -> None:
    """Unit test for LDPE case study.

    Parameters
    ----------
    PLS_model_SIMCA_LDPE_example : dict
        Dictionary of raw data and expected outputs from the PLS model.
    """
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["Y"], np.ndarray)
    assert isinstance(data["A"], int)
    plsmodel = PLS(n_components=data["A"])
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["Y"]))

    # Can only get these to very loosely match
    assert data["expected_t2_lim_95_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.95), rel=1e-1)
    assert data["expected_t2_lim_99_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.99), rel=1e-1)

    assert np.mean(np.abs(np.array(data["expected_T"])) - np.abs(plsmodel.x_scores.values)) == pytest.approx(
        0, abs=1e-4
    )
    assert np.mean(np.abs(np.array(data["expected_P"])) - np.abs(plsmodel.x_loadings.values)) == pytest.approx(
        0, abs=1e-5
    )
    assert np.mean(np.abs(np.array(data["expected_W"])) - np.abs(plsmodel.x_weights.values)) == pytest.approx(
        0, abs=1e-6
    )
    assert np.mean(np.abs(np.array(data["expected_C"])) - np.abs(plsmodel.y_loadings.values)) == pytest.approx(
        0, abs=1e-6
    )
    assert np.mean(np.abs(np.array(data["expected_U"])) - np.abs(plsmodel.y_scores.values)) == pytest.approx(
        0, abs=1e-5
    )
    assert np.mean(
        np.array(data["expected_hotellings_t2_a3"]).ravel() - plsmodel.hotellings_t2.iloc[:, 2].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_hotellings_t2_a6"]).ravel() - plsmodel.hotellings_t2.iloc[:, 5].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_sd_t"]).ravel() - plsmodel.scaling_factor_for_scores.values.ravel()
    ) == pytest.approx(0, abs=1e-5)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(np.abs(Y_mcuv.inverse_transform(plsmodel.predictions) - np.array(data["expected_yhat_a6"])))
            / Y_mcuv.center_
        )
    ) == pytest.approx(0, abs=1e-2)


def test_pls_simca_ldpe_missing_data(
    fixture_pls_ldpe_example: dict[str, pd.DataFrame | np.ndarray | float | int],
) -> None:
    """Unit test for LDPE case study.
    From visual inspection, observation 12 has low influence in the model.
    Set 1 value in this observation to missing and check that the results are similar to the
    full-data case: "test_PLS_SIMCA_LDPE",
    the only differences are that the tolerances are slightly relaxed.

    """
    data = fixture_pls_ldpe_example
    assert isinstance(data["X"], np.ndarray)
    assert isinstance(data["Y"], np.ndarray)
    assert isinstance(data["A"], int)
    data["X"][11, 0] = float("nan")
    plsmodel = PLS(n_components=int(data["A"]), missing_data_settings=dict(md_method="scp"))
    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel = plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(pd.DataFrame(data["Y"])))
    # Can only get these to very loosely match
    assert data["expected_t2_lim_95_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.95), rel=1e-1)
    assert data["expected_t2_lim_99_a6"] == pytest.approx(plsmodel.hotellings_t2_limit(0.99), rel=1e-1)

    assert np.mean(np.abs(np.array(data["expected_T"])) - np.abs(plsmodel.x_scores.values)) == pytest.approx(
        0, abs=1e-2
    )
    assert np.mean(np.abs(np.array(data["expected_P"])) - np.abs(plsmodel.x_loadings.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_W"])) - np.abs(plsmodel.x_weights.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_C"])) - np.abs(plsmodel.y_loadings.values)) == pytest.approx(
        0, abs=1e-3
    )
    assert np.mean(np.abs(np.array(data["expected_U"])) - np.abs(plsmodel.y_scores.values)) == pytest.approx(
        0, abs=5e-1
    )
    assert np.mean(
        np.array(data["expected_hotellings_t2_a3"]).ravel() - plsmodel.hotellings_t2.iloc[:, 2].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_hotellings_t2_a6"]).ravel() - plsmodel.hotellings_t2.iloc[:, 5].values.ravel()
    ) == pytest.approx(0, abs=1e-6)
    assert np.mean(
        np.array(data["expected_sd_t"]).ravel() - plsmodel.scaling_factor_for_scores.values.ravel()
    ) == pytest.approx(0, abs=1e-2)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(np.abs(Y_mcuv.inverse_transform(plsmodel.predictions) - np.array(data["expected_yhat_a6"])))
            / Y_mcuv.center_
        )
    ) == pytest.approx(0, abs=0.5)


# ---- TPLS models ----
@pytest.fixture
def fixture_tpls_example() -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load example data for TPLS model.

    Data from: https://github.com/salvadorgarciamunoz/pyphi/tree/master/examples/JRPLS%20and%20TPLS
    """
    properties = {
        "Group 1": pd.read_csv(  # 162 x 7
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group1.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 2": pd.read_csv(  # 9 x 6
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group2.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 3": pd.read_csv(  # 22 x 6
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group3.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 4": pd.read_csv(  # 19 x 11
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group4.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 5": pd.read_csv(  # 18 x 3
            "process_improve/datasets/multivariate/tpls-pyphi/properties_Group5.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
    }
    formulas = {
        "Group 1": pd.read_csv(  # 105 x 162
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group1.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 2": pd.read_csv(  # 105 x 9
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group2.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 3": pd.read_csv(  # 105 x 22
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group3.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 4": pd.read_csv(  # 105 x 19
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group4.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
        "Group 5": pd.read_csv(  # 105 x 18
            "process_improve/datasets/multivariate/tpls-pyphi/formulas_Group5.csv", sep=",", index_col=0, header=0
        ).astype("float64"),
    }
    process_conditions: dict[str, pd.DataFrame] = {
        "Conditions": pd.read_csv(  # 105 x 10
            "process_improve/datasets/multivariate/tpls-pyphi/process_conditions.csv", sep=",", index_col=0, header=0
        ).astype("float64")
    }

    quality_indicators: dict[str, pd.DataFrame] = {
        "Quality": pd.read_csv(  # 105 x 6
            "process_improve/datasets/multivariate/tpls-pyphi/quality_indicators.csv", sep=",", index_col=0, header=0
        ).astype("float64")
    }
    return {
        "Z": process_conditions,
        "D": properties,
        "F": formulas,
        "Y": quality_indicators,
    }


def test_tpls_model_fitting(fixture_tpls_example: dict) -> None:
    """
    Test the fitting process of the TPLS model to ensure it functions as expected.

    Tests here include tests for pre-processing and model fitting.
    """

    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # Test the coefficients in the centering and scaling self.preproc_ structure. Group 1, for D matrix pre-processing:
    known_truth_d1m = np.array([99.85432099, 73.67901235, 3.07469136, 0.13950617, 2.09876543, 12.53703704, 41.58641975])
    known_truth_d1s = np.array([0.36883209, 1.80631852, 0.69231018, 0.09610943, 0.29927192, 1.99109474, 4.82828759])
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["center"]) == known_truth_d1m
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["scale"]) == known_truth_d1s
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["center"]) == d_matrix["Group 1"].mean()
    assert pytest.approx(tpls_test.preproc_["D"]["Group 1"]["scale"]) == d_matrix["Group 1"].std()

    # Group 4 in D has missing values pre-processing. Test these.
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
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["center"]) == known_truth_d4m
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["scale"]) == known_truth_d4s
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["center"]) == d_matrix["Group 4"].mean()
    assert pytest.approx(tpls_test.preproc_["D"]["Group 4"]["scale"]) == d_matrix["Group 4"].std()

    # Test the formula block, group 2 pre-processing
    known_truth_f2m = np.array(
        [0.13333333, 0.0020127, 0.01904762, 0.00952381, 0.13282593, 0.1889709, 0.1047619, 0.17035596, 0.23916785]
    )
    known_truth_f2s = np.array(
        [0.34156503, 0.02062402, 0.13734798, 0.09759001, 0.33676188, 0.39173332, 0.3077152, 0.37647422, 0.4274989]
    )
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["center"]) == known_truth_f2m
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["scale"]) == known_truth_f2s
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["center"]) == fixture_tpls_example["F"]["Group 2"].mean()
    assert pytest.approx(tpls_test.preproc_["F"]["Group 2"]["scale"]) == fixture_tpls_example["F"]["Group 2"].std()

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
    assert pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["center"]) == known_truth_zm
    assert pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["scale"]) == known_truth_zs
    assert (
        pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["center"]) == fixture_tpls_example["Z"]["Conditions"].mean()
    )
    assert (
        pytest.approx(tpls_test.preproc_["Z"]["Conditions"]["scale"]) == fixture_tpls_example["Z"]["Conditions"].std()
    )

    # Test the `Quality` (Y) block pre-processing
    known_truth_ym = np.array([30.96834605, 3.328312, 57.01620571, 3.6485004, 79.34224822, 3.06157598])
    known_truth_ys = np.array([3.46989807, 0.96879553, 4.9740008, 1.47585478, 3.8676095, 1.18600054])
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["center"]) == known_truth_ym
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["scale"]) == known_truth_ys
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["center"]) == fixture_tpls_example["Y"]["Quality"].mean()
    assert pytest.approx(tpls_test.preproc_["Y"]["Quality"]["scale"]) == fixture_tpls_example["Y"]["Quality"].std()

    # Test various R2 values for columns in blocks
    # For the last component, for the Z block:
    assert pytest.approx(tpls_test.r2_frac[-1]["Z"]["Conditions"].round(3)) == np.array(
        [0.411, 0.0, 0.004, 0.262, 0.049, 0.001, 0.028, 0.001, 0.015, 0.015]
    ).astype(np.float64)
    # For the D-block
    assert pytest.approx(tpls_test.r2_frac[-1]["D"]["Group 1"].round(3)) == np.array(
        [0.285, 0.054, 0.437, 0.001, 0.021, 0.143, 0.138]
    ).astype(np.float64)
    # For the F-block
    assert pytest.approx(tpls_test.r2_frac[-1]["F"]["Group 2"].round(3)) == np.array(
        [0.047, 0.0, 0.011, 0.0, 0.047, 0.025, 0.0, 0.003, 0.052]
    ).astype(np.float64)
    # For the Y-block after 1 component, 2 components, and 3 components:
    assert pytest.approx(tpls_test.r2_frac[1]["Y"]["Quality"].round(3)) == np.array(
        [0.005, 0.079, 0.288, 0.329, 0.332, 0.153]
    ).astype(np.float64)
    assert pytest.approx(tpls_test.r2_frac[2]["Y"]["Quality"].round(3)) == np.array(
        [0.135, 0.0, 0.039, 0.023, 0.007, 0.051]
    ).astype(np.float64)
    assert pytest.approx(tpls_test.r2_frac[-1]["Y"]["Quality"].round(3)) == np.array(
        [0.017, 0.073, 0.018, 0.0, 0.005, 0.0]
    ).astype(np.float64)


def test_tpls_model_plots(fixture_tpls_example: dict) -> None:
    """Test the plotting functionality of the TPLS model."""

    n_components = 3
    tpls_test = TPLS(n_components=n_components, d_matrix=fixture_tpls_example.pop("D"))
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # TODO: perform various assertions on the model's Plotly plots
    assert tpls_test.plot.scores() is not None
    # assert tpls_test.plot.loadings() is not None


def test_tpls_model_predictions(fixture_tpls_example: dict) -> None:  # noqa: PLR0915
    """Test the prediction process of the TPLS model to ensure it functions as expected."""
    n_components = 3
    d_matrix = fixture_tpls_example.pop("D")
    tpls_test = TPLS(n_components=n_components, d_matrix=d_matrix)
    tpls_test.fit(DataFrameDict(fixture_tpls_example))

    # Test the model's predictions. Use the first few samples of the data as a testing data point.
    testing_samples = ["L001", "L002", "L003", "L004"]
    new_observation_raw = {
        "Z": {"Conditions": fixture_tpls_example["Z"]["Conditions"].loc[testing_samples]},
        "F": {key: val.loc[testing_samples] for key, val in fixture_tpls_example["F"].items()},
    }

    # OK, now use these to make predictions
    predictions = tpls_test.predict(DataFrameDict(new_observation_raw))
    expected_keys = ["hat", "t_scores_super", "hotellings_t2", "spe"]
    assert set(expected_keys) == set(predictions.keys())

    # Test the predicted values are what were expected:
    assert pytest.approx(predictions["hat"]["Quality"].iloc[0, :]) == np.array(
        [33.09434113, 3.15224246, 58.76423934, 3.29991258, 79.90201127, 2.67042528]
    )
    # Compare the predictions to what is stored in the training data
    assert pytest.approx(predictions["hat"]["Quality"].iloc[0, :]) == np.array(
        tpls_test.hat["Quality"].iloc[0, :].values
    )

    # Test that the SPE_z values:
    assert pytest.approx([predictions["spe"]["Z"][key].values[0, -1] for key in predictions["spe"]["Z"]]) == [
        1.87201925670
    ]

    # Test that the SPE_f values are correct:
    assert pytest.approx([predictions["spe"]["F"][key].values[0, -1] for key in predictions["spe"]["F"]]) == [
        12.9399977,
        6.92978457,
        5.56504553,
        5.59028493,
        5.54400785,
    ]
    # Check that the Hotelling's T2 matches what would have been calculated by the model.
    assert pytest.approx(tpls_test.hotellings_t2.loc[testing_samples]) == predictions["hotellings_t2"].values[
        :, -1
    ].reshape(-1, 1)

    # from sklearn.model_selection import cross_val_score

    # scores = cross_val_score(tpls_test, transformed_data, None, cv=5)
    # Ensure model is fitted appropriately, with the expected number of iterations
    assert tpls_test.fitting_statistics["iterations"] == [11, 8, 26]
    assert all(tol < epsqrt for tol in tpls_test.fitting_statistics["convergance_tolerance"])

    # Model parameters tested
    assert pytest.approx(tpls_test.hotellings_t2.iloc[0:5].values.ravel()) == np.array(
        [
            2.51977572,
            2.96430904,
            2.90972389,
            4.52220244,
            5.08398872,
        ]
    )

    # Model limits tested
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

    # Test building the model without the "Z" block.
    fixture_tpls_example.pop("Z")
    n_components = 3
    tpls_test_no_z = TPLS(n_components=n_components, d_matrix=d_matrix).fit(DataFrameDict(fixture_tpls_example))
    # Test the model's predictions. Use the first few samples of the data as a testing data point.
    testing_samples = ["L001", "L002", "L003", "L004"]
    new_observation_raw_no_z = DataFrameDict(
        {
            "F": {key: val.loc[testing_samples] for key, val in fixture_tpls_example["F"].items()},
        }
    )

    # OK, now use these to make predictions
    predictions = tpls_test_no_z.predict(new_observation_raw_no_z)
    expected_keys = ["hat", "t_scores_super", "hotellings_t2", "spe"]
    assert set(expected_keys) == set(predictions.keys())
    assert predictions["spe"]["Z"] == {}
    assert predictions["spe"]["F"] is not None
    assert predictions.spe is not None  # test the `Bunch` functionality


def test_tpls_cross_validation(fixture_tpls_example: dict) -> None:
    """Test the prediction process of the TPLS model to ensure it functions as expected."""
    n_components = 3
    full_model = TPLS(n_components=n_components, d_matrix=fixture_tpls_example.pop("D"))
    _ = cross_val_score(
        estimator=full_model,
        X=DataFrameDict(fixture_tpls_example),
        cv=5,
        scoring="r2",
        n_jobs=1,
    )
    # TODO: tests on the output


def manual_cross_validation(tpls_model: TPLS, full_datadict: dict, cv: int = 5, scoring: str = "r2") -> np.ndarray:
    """Perform manual cross-validation for a TPLS model."""
    kfold = KFold(n_splits=cv, shuffle=True)
    n_samples = len(full_datadict)
    scores = []

    for train_idx, test_idx in kfold.split(range(n_samples)):
        training_datadict = {}
        testing_datadict = {}

        training_datadict = full_datadict[train_idx]
        testing_datadict = full_datadict[test_idx]

        # Fit and predict
        tpls_model.fit(training_datadict)
        tpls_model.display_results()
        inference = tpls_model.predict(testing_datadict)

        # Calculate score
        if scoring == "r2":
            score = r2_score(testing_datadict["Y"]["Quality"], inference.hat["Quality"])
        elif scoring == "mse":
            score = -mean_squared_error(testing_datadict["Y"]["Quality"], inference.hat["Quality"])
        else:
            score = tpls_model.score(testing_datadict, inference.hat["Quality"])

        scores.append(score)

    return np.array(scores)


# n_components = 3
# data = fixture_tpls_example()
# full_model = TPLS(n_components=n_components, d_matrix=data.pop("D"))
# full_model.fit(DataFrameDict(data))

# data = fixture_tpls_example()
# cv_results = cross_val_score(
#     estimator=TPLS(n_components=n_components, d_matrix=data.pop("D")),
#     X=DataFrameDict(data),
#     cv=10,
#     # scoring={"score": scorer},
#     n_jobs=-1,
#     verbose=1,
# )
# print(cv_results)
# fixture_tpls_example = fixture_tpls_example()
# test_tpls_model_predictions(fixture_tpls_example)
