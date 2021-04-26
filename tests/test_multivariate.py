# (c) Kevin Dunn, 2019-2021. MIT License.

import pathlib


import pytest
from pytest import approx
import pandas as pd
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from process_improve.multivariate.methods import (
    epsqrt,
    center,
    scale,
    quick_regress,
    ssq,
    PCA,
    MCUVScaler,
    PLS,
    SpecificationWarning,
)


def test_PCA_SPE_limits():
    """
    Simulate data and see if SPE limit cuts off at 5%.
    """
    N = 1000
    repeats = 50
    outliers_95 = []
    outliers_99 = []
    for k in range(repeats):

        # The desired mean values of the sample.
        mu = np.array([0.0, 0.0, 0.0])

        # The desired covariance matrix.
        r = np.array([[5.20, -4.98, -1.00], [-4.98, 5.50, 2.94], [-1.00, 2.94, 2.77]])

        X = pd.DataFrame(np.random.multivariate_normal(mu, r, size=N))
        scaler = MCUVScaler().fit(X)
        mcuv = scaler.fit_transform(X)

        A = 2
        pca = PCA(n_components=A).fit(mcuv)
        SPE_limit_95 = pca.SPE_limit(0.95)
        SPE_limit_99 = pca.SPE_limit(0.99)

        outliers_95.append(
            (pca.squared_prediction_error.iloc[:, A - 1] > SPE_limit_95).sum()
        )
        outliers_99.append(
            (pca.squared_prediction_error.iloc[:, A - 1] > SPE_limit_99).sum()
        )

    assert np.mean(outliers_95) == approx(0.05 * N, rel=0.1)
    assert np.mean(outliers_99) == approx(0.01 * N, rel=0.1)


def test_PCA_foods():
    """
    Arrays with no variance should not be able to have variance extracted.
    """

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
    ) == approx(0, abs=epsqrt)

    T2_limit_95 = pca.T2_limit(0.95)
    assert T2_limit_95 == approx(6.64469, rel=1e-3)

    pca.SPE_limit(0.95)

    ellipse_x, ellipse_y = pca.ellipse_coordinates(1, 2, 0.95, 100)
    assert ellipse_x[-1] == approx(4.48792, rel=1e-5)
    assert ellipse_y[-1] == approx(0, rel=1e-7)


@pytest.fixture
def fixture_kamyr_data_missing_value():
    folder = (
        pathlib.Path(__file__).parents[1]
        / "process_improve"
        / "datasets"
        / "multivariate"
    )
    return pd.read_csv(
        folder / "kamyr.csv",
        index_col=None,
        header=None,
    )


def test_PCA_missing_data(fixture_kamyr_data_missing_value):

    X_mcuv = MCUVScaler().fit_transform(fixture_kamyr_data_missing_value)

    # Build the model
    A = 2
    pca = PCA(n_components=A)
    assert pca.missing_data_settings is None

    # Check that default missing data options were used
    model = pca.fit(X_mcuv)
    assert isinstance(model.missing_data_settings, dict)
    assert "md_tol" in model.missing_data_settings

    assert np.linalg.norm(
        (model.loadings.T @ model.loadings) - np.eye(model.A)
    ) == approx(0, abs=1e-2)


@pytest.fixture
def fixture_mv_utilities():
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
            [1, np.NaN, 3, np.NaN, 5, np.NaN],
        ]
    )
    Y = Y.T
    return x, Y


def test_ssq(fixture_mv_utilities):
    x, _ = fixture_mv_utilities
    assert (1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6) == approx(ssq(x), abs=1e-9)


def test_quick_regress(fixture_mv_utilities):
    x, Y = fixture_mv_utilities
    out = quick_regress(Y, x).ravel()
    assert 1 == approx(out[0], abs=1e-9)
    assert 0.61538462 == approx(out[1], abs=1e-8)
    assert 0 == approx(out[2], abs=1e-9)

    # Checked against R: summary(lm(c(1,1,1,1,1,1) ~ seq(6) + 0))
    assert 0.23077 == approx(out[3], abs=1e-6)

    # Checked against what is expected: (1 + 3^2 + 5^2)/(1 + 3^2 + 5^2)
    assert 1.0 == approx(out[4], abs=1e-14)


@pytest.fixture
def fixture_tablet_spectra_data():
    """
    Verifies the PCA model for the case of no missing data.
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
    folder = (
        pathlib.Path(__file__).parents[1]
        / "process_improve"
        / "datasets"
        / "multivariate"
    )
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


def test_MCUV_centering(fixture_tablet_spectra_data):
    """
    Mean centering of the testing data.
    """

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)
    assert 0.0 == approx(np.max(np.abs(X_mcuv.mean(axis=0))), rel=1e-9)


def test_MCUV_scaling(fixture_tablet_spectra_data):
    """Scaling by standard deviation."""

    spectra, _ = fixture_tablet_spectra_data
    X_mcuv = MCUVScaler().fit_transform(spectra)

    assert 1 == approx(np.min(np.abs(X_mcuv.std(axis=0))), 1e-10)
    assert 1 == approx(X_mcuv.std(), 1e-10)


def test_PCA_tablet_spectra(fixture_tablet_spectra_data):
    r"""
    PCA characteristics:

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
    assert 0.0 == approx(np.linalg.norm(orthogonal_check - np.eye(model.A)), rel=1e-9)

    # Check the R2 value against the R software output
    assert model.R2cum[0] == approx(0.7368, rel=1e-3)
    assert model.R2cum[1] == approx(0.9221, rel=1e-2)

    # Unit length: actually checked above, via subtraction with I matrix.
    # Check if scores are orthogonal
    scores_covar = model.x_scores.T @ model.x_scores
    for i in range(model.A):
        for j in range(model.A):

            # Technically not need, but more explict this way.
            if i == j:
                assert scores_covar.iloc[i, j] == approx(
                    known_scores_covar[i, j], rel=1e-2
                )
            else:
                assert scores_covar.iloc[i, j] == approx(
                    known_scores_covar[i, j], abs=1e-4
                )

                if i >= 1:
                    assert scores_covar.iloc[j, j] > scores_covar.iloc[i, i]

    # Check the model against an SVD: this raw data set has no missing
    # data, so the SVD should be faster and more accurate than NIPALS
    autoscaled_X = scale(center(spectra))
    u, s, v = np.linalg.svd(autoscaled_X)

    loadings_delta = np.linalg.norm(
        np.abs(v[0 : model.A, :]) - np.abs(model.loadings.T)
    )
    assert loadings_delta == approx(0, abs=1e-8)

    # It is not possible, it seems, to get the scores to match the SVD
    # scores. Numerical error?


def test_PCA_errors_no_variance_to_start():
    """
    Arrays with no variance should seem to work, but should have no variability explained.
    """
    K, N, A = 17, 12, 5
    data = pd.DataFrame(np.zeros((N, K)))
    model = PCA(n_components=A)
    # with pytest.raises(RuntimeError):
    model.fit(data)
    assert np.sum(model.x_scores.values, axis=None) == approx(0, abs=epsqrt)
    assert model.R2cum.sum() == approx(0, abs=epsqrt)
    assert np.isnan(model.R2cum[A - 1])


def test_PCA_invalid_calls():
    """
    Tests various invalid calls, and corresponding error messages.
    """
    K, N, A = 4, 3, 5
    data = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N, K)))
    with pytest.warns(
        SpecificationWarning,
        match=r"The requested number of components is more than can be computed from data(.*)",
    ):
        model = PCA(n_components=A)
        model.fit(data)

    data.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError, match="Tolerance must exceed machine precision"):
        _ = PCA(
            n_components=A, missing_data_settings=dict(md_method="nipals", md_tol=0)
        ).fit(data)

    with pytest.raises(
        AssertionError, match=r"Missing data method is not recognized(.*)"
    ):
        _ = PCA(n_components=A, missing_data_settings={"md_method": "SCP"}).fit(data)

    # TODO: replace with a check to ensure the data is in a DataFrame.
    # from scipy.sparse import csr_matrix
    # sparse_data = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
    # with pytest.raises(TypeError, match="This PCA class does not support sparse input."):
    #     model = PCA(n_components=2)
    #     model.fit(sparse_data)


def test_PCA_no_more_variance():
    """
    Create a rank 2 matrix and it should fail on the 3rd component.
    """

    K = 17
    N = 12
    A = 3
    T = np.random.uniform(low=-1, high=1, size=(N, 2))
    P = np.random.uniform(low=-1, high=1, size=(K, 2))
    X = T @ P.T
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0, ddof=0)
    X = pd.DataFrame((X - meanX) / stdX)
    _ = PCA(n_components=A)

    # with pytest.raises(RuntimeError):
    #    m.fit(X)
    # TODO: check that the m.R2[2] (3rd PC is zero.)


def test_PCA_columns_with_no_variance():
    """
    Create a column with no variance. That column's loadings should be 0.
    """
    K = 14
    N = 29
    A = 4
    cols_with_no_variance = [10, 3]
    T = np.random.uniform(low=-1, high=1, size=(N, A))
    P = np.random.uniform(low=-1, high=1, size=(K, A))
    X = T @ P.T
    meanX = X.mean(axis=0)
    stdX = X.std(axis=0, ddof=0)
    X = pd.DataFrame((X - meanX) / stdX)
    X.iloc[:, cols_with_no_variance] = 0

    m = PCA(n_components=2)
    m.fit(X)

    # `loadings` is a K by A matrix.  Check sum of loadings in rows with
    # no variance must be zero
    assert np.sum(np.abs(m.loadings.iloc[cols_with_no_variance, :].values)) == approx(
        0, abs=1e-14
    )
    # The loadings must still be orthonormal though:
    assert np.sum(np.identity(m.A) - m.loadings.values.T @ m.loadings.values) == approx(
        0, abs=1e-14
    )

    # Are scores orthogonal?
    covmatrix = m.x_scores.T @ m.x_scores
    covmatrix - np.diag(np.diag(covmatrix))
    (np.sum(np.abs(covmatrix - np.diag(np.diag(covmatrix))))).values == approx(
        0, abs=1e-6
    )


@pytest.fixture
def fixture_pca_PCA_Wold_etal_paper():
    """
    From the PCA paper by Wold, Esbensen and Geladi, 1987
    Principal Component Analysis, Chemometrics and Intelligent Laboratory
    Systems, v 2, p37-52; http://dx.doi.org/10.1016/0169-7439(87)80084-9
    """
    return pd.DataFrame(np.array([[3, 4, 2, 2], [4, 3, 4, 3], [5.0, 5, 6, 4]]))


def test_PCA_Wold_centering(fixture_pca_PCA_Wold_etal_paper):
    """
    Checks the centering step
    """
    out, centering = center(fixture_pca_PCA_Wold_etal_paper, extra_output=True)
    assert centering == approx([4, 4, 4, 3], rel=1e-8)


def test_PCA_Wold_scaling(fixture_pca_PCA_Wold_etal_paper):
    """
    Checks the scaling step. Page 40 of the above paper.
    """

    out, scaling = scale(
        center(fixture_pca_PCA_Wold_etal_paper), extra_output=True, ddof=1
    )
    assert scaling == approx([1, 1, 0.5, 1])


def test_PCA_Wold_model_results(fixture_pca_PCA_Wold_etal_paper):
    """
    Checks if the PCA model matches the results in the paper.
    """

    X_preproc = scale(center(fixture_pca_PCA_Wold_etal_paper))
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
    X_preproc = scale(center(fixture_pca_PCA_Wold_etal_paper))
    pca_2 = PCA(n_components=2)
    pca_2.fit(X_preproc)
    assert np.abs(pca_2.loadings.values[:, 0]) == approx(
        [0.5410, 0.3493, 0.5410, 0.5410], abs=1e-4
    )
    assert np.abs(pca_2.loadings.values[:, 1]) == approx(
        [0.2017, 0.9370, 0.2017, 0.2017], abs=1e-4
    )

    # Scores. The scaling is off here by a constant factor of 0.8165
    # assert np.all(pca_2.x_scores["1"] == approx([-1.6229, -0.3493, 1.9723], rel=1e-3))
    # assert np.all(pca_2.x_scores["2"] == approx([0.6051, -0.9370, 0.3319], rel=1e-4))

    # R2 values, given on page 43
    assert pca_2.R2.values == approx([0.831, 0.169], rel=1e-2)

    # SS values, on page 43
    # SS_X = np.sum(X_preproc ** 2, axis=0)
    # assert SS_X == approx([0.0, 0.0, 0.0, 0.0], abs=1e-9)

    # Testing data:
    # X_test = Block(np.array([[3, 4, 3, 4], [1, 2, 3, 4.0]]))
    # X.preprocess(X_test)
    # compare_entries(X_test.data, np.array([[-1, 0, -0.5, 1],
    # [-3, -2, -0.5, 1]])
    # #testing = PCA_model.apply(X_test)
    # compare_entries(testing.T, np.array([[-0.2075, 0.1009],
    # [-2.0511, -1.3698]])


def test_PLS_properties_TODO():
    """
    TODO:
    diag(T.T * T) related to S
    W.T * W = I for PLS only
    P.T * W: ones on diagonal, zeros below diagonal
    W.T * R: ones on diagonal, zeros below diagonal
    R.T * P = ID
    """
    pass


@pytest.mark.skip(reason="API still has to be improved to handle this case")
def test_PLS_invalid_calls():
    """
    Tests various invalid calls, and corresponding error messages.
    """
    K, N, M, A = 4, 3, 2, 5
    dataX = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N, K)))
    dataY = pd.DataFrame(np.random.uniform(low=-1, high=1, size=(N, M)))
    with pytest.raises(
        ValueError, match="Tolerance `tol`` must be between 1E-16 and 1.0"
    ):
        _ = PLS(n_components=A, tol=0)

    with pytest.raises(ValueError, match="Method 'SVDS' is not known."):
        _ = PLS(n_components=A, method="SVDS")

    with pytest.raises(ValueError, match="Missing data method 'SCP' is not known."):
        _ = PLS(n_components=A, md_method="SCP")

    with pytest.warns(
        SpecificationWarning, match=r"The requested number of components is (.*)"
    ):
        model = PLS(
            n_components=A,
        )
        model.fit(dataX, dataY)

    from scipy.sparse import csr_matrix

    sparse_data = csr_matrix([[1, 2], [0, 3], [4, 5]])
    with pytest.raises(
        TypeError, match="This PLS class does not support sparse input."
    ):
        model = PLS(n_components=2)
        model.fit(dataX, sparse_data)


@pytest.fixture
def fixture_PLS_model_SIMCA_1_component():
    """
    Simple model tested against Simca-P, version 14.1.
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
    data = {}
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

    data["y"] = pd.DataFrame(
        np.array([1.12, 1.01, 0.97, 0.83, 0.93, 1.02, 0.91, 0.7, 1.26, 1.05, 0.95])
    )
    data["expected_y_predicted"] = [
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
    data["Xavg"] = np.array(
        [41.38802, 21.03755, 20.03097, 0.3884909, 0.1072455, -1.006582]
    )
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


def test_PLS_compare_sklearn_1_component(fixture_PLS_model_SIMCA_1_component):

    data = fixture_PLS_model_SIMCA_1_component

    plsmodel = PLSRegression(n_components=data["A"], scale="True")
    plsmodel.fit(data["X"], data["y"])

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert plsmodel._x_mean == approx(data["Xavg"], abs=1e-5)
    assert plsmodel._x_std == approx(data["Xws"], abs=1e-6)
    assert plsmodel._y_mean == approx(data["Yavg"], abs=1e-7)
    assert plsmodel._y_std == approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    T = plsmodel.x_scores_
    P = plsmodel.x_loadings_
    assert T.ravel() == approx(data["t1"], abs=1e-5)
    assert np.std(T, ddof=1) == approx(data["SDt"], rel=1e-5)
    assert data["loadings_P1"].ravel() == approx(P.ravel(), rel=1e-5)
    assert data["loadings_r1"] == approx(plsmodel.x_weights_.ravel(), rel=1e-4)

    # Check the model's predictions
    t1_predict, y_pp = plsmodel.transform(data["X"], data["y"])
    assert data["t1"] == approx(t1_predict.ravel(), abs=1e-5)
    # assert y_pp == approx((data["y"] - data["Yavg"]) / data["Yws"], abs=1e-6)

    # Manually make the PLS prediction
    # X_check = data["X"].copy()
    # X_check_mcuv = (X_check - plsmodel._x_mean) / plsmodel._x_std
    # t1_predict_manually = X_check_mcuv @ plsmodel.x_weights_

    # TODO: fix the rest of this test. Not sure what the purpose of this test is anyway.

    # # Simca's C:
    # N = data["X"].shape[0]
    # simca_C = (y_pp.reshape(1, N) @ t1_predict) / (t1_predict.T @ t1_predict)
    # # assert simca_C == approx(data["loadings_y_c1"], 1e-6)
    # assert t1_predict_manually.values.ravel() == approx(t1_predict.ravel(), 1e-9)

    # # Deflate the X's:
    # X_check_mcuv -= t1_predict_manually @ plsmodel.x_loadings_.T
    # y_hat = t1_predict_manually @ simca_C
    # y_hat_rawunits = y_hat * plsmodel._y_std + plsmodel._y_mean
    # assert data["expected_y_predicted"] == approx(y_hat_rawunits.values.ravel(), abs=1e-5)

    # prediction_error = data["y"].values - y_hat_rawunits.values
    # R2_y = (data["y"].var(ddof=1) - prediction_error.var(ddof=1)) / data["y"].var(ddof=1)
    # assert R2_y == approx(data["R2Y"], abs=1e-6)


def test_PLS_compare_model_api(fixture_PLS_model_SIMCA_1_component):

    data = fixture_PLS_model_SIMCA_1_component
    plsmodel = PLS(n_components=data["A"])

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["y"])

    # Check the pre-processing: sig figs have been taken as high as possible.
    assert X_mcuv.center_.values == approx(data["Xavg"], abs=1e-5)
    assert X_mcuv.scale_.values == approx(data["Xws"], abs=1e-6)
    assert Y_mcuv.center_.values == approx(data["Yavg"], abs=1e-7)
    assert Y_mcuv.scale_.values == approx(data["Yws"], abs=1e-8)

    # Extract the model parameters
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["y"]))
    assert data["SDt"] == approx(np.std(plsmodel.x_scores, ddof=1), abs=1e-5)
    assert data["t1"] == approx(plsmodel.x_scores.values.ravel(), abs=1e-5)
    assert data["loadings_P1"] == approx(plsmodel.x_loadings.values.ravel(), abs=1e-5)
    assert data["loadings_r1"] == approx(plsmodel.x_weights.values.ravel(), abs=1e-6)

    assert data["expected_y_predicted"] == approx(
        Y_mcuv.inverse_transform(plsmodel.predictions).values.ravel(), abs=1e-5
    )
    assert data["R2Y"] == approx(plsmodel.R2cum, abs=1e-6)

    # Check the model's predictions
    state = plsmodel.predict(X_mcuv.transform(data["X"]))
    assert plsmodel.squared_prediction_error.values.ravel() == approx(
        state.squared_prediction_error.values, abs=1e-9
    )
    assert data["t1"] == approx(state.x_scores.values.ravel(), abs=1e-5)
    assert data["Tsq"] == approx(state.Hotellings_T2.values.ravel(), abs=1e-5)
    assert data["expected_y_predicted"] == approx(
        Y_mcuv.inverse_transform(state.y_hat).values.ravel(), abs=1e-5
    )


@pytest.fixture
def fixture_PLS_SIMCA_2_components():
    """
    Simple model tested against Simca-P, version 14.1.
    Testing on 02 July 2020.
    No missing data

    When X and y are mean centered and scaled, the model should provide the loadings listed here.
    """
    out = {}
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
    out["expected_y_predicted"] = [
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
    out["loadings_P"] = np.array(
        [[-0.3799977, -0.7815778], [0.8737038, -0.2803103], [-0.3314019, 0.55731]]
    )
    out["loadings_W"] = np.array(  # W
        [[-0.4839311, -0.7837874], [0.8361799, -0.2829775], [-0.2580969, 0.5528119]]
    )
    out["loadings_C"] = [1.019404, 0.1058565]
    out["SDt"] = [0.9724739, 1.098932]

    out["R2X"] = [
        0.3207782,
        0.4025633,
    ]  # cumulative: 32%, then 72% for second component
    out["R2Y"] = [0.9827625, 0.01353244]
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


def test_PLS_sklearn_2_components(fixture_PLS_SIMCA_2_components):

    data = fixture_PLS_SIMCA_2_components

    plsmodel = PLSRegression(n_components=data["A"], scale=False)

    X_mcuv = MCUVScaler().fit_transform(data["X"])
    Y_mcuv = MCUVScaler().fit_transform(data["y"])

    plsmodel.fit(X_mcuv, Y_mcuv)

    # Extract the model parameters
    assert np.abs(data["T"]) == approx(np.abs(plsmodel.x_scores_), abs=1e-5)
    assert np.std(plsmodel.x_scores_, ddof=1, axis=0) == approx(data["SDt"], abs=1e-6)
    assert np.abs(data["loadings_P"]) == approx(np.abs(plsmodel.x_loadings_), abs=1e-5)
    assert np.abs(data["loadings_W"]) == approx(np.abs(plsmodel.x_weights_), abs=1e-5)


def test_PLS_compare_API(fixture_PLS_SIMCA_2_components):
    data = fixture_PLS_SIMCA_2_components

    plsmodel = PLS(n_components=data["A"])

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["y"]))

    # Extract the model parameters
    assert data["SDt"] == approx(np.std(plsmodel.x_scores, ddof=1, axis=0), abs=1e-6)
    assert np.abs(data["T"]) == approx(np.abs(plsmodel.x_scores), abs=1e-5)
    assert np.abs(data["loadings_P"]) == approx(np.abs(plsmodel.x_loadings), abs=1e-5)
    assert np.abs(data["loadings_W"]) == approx(np.abs(plsmodel.x_weights), abs=1e-5)
    assert Y_mcuv.inverse_transform(plsmodel.predictions).values == approx(
        data["expected_y_predicted"], abs=1e-5
    )
    assert sum(data["R2Y"]) == approx(plsmodel.R2cum.values[-1], abs=1e-7)

    # Check the model's predictions
    state = plsmodel.predict(X_mcuv.transform(data["X"]))
    # TODO: a check on SPE vs Simca-P. Here we are doing a check between the SPE from the
    # model building, to model-using, but not against an external library.
    assert plsmodel.squared_prediction_error.iloc[:, -1].values == approx(
        state.squared_prediction_error, abs=1e-10
    )
    assert data["Tsq"] == approx(state.Hotellings_T2, abs=1e-5)
    assert data["expected_y_predicted"] == approx(
        Y_mcuv.inverse_transform(state.y_hat).values.ravel(), abs=1e-5
    )
    assert np.abs(data["T"]) == approx(np.abs(state.x_scores), abs=1e-5)


@pytest.fixture
def fixture_PLS_LDPE_example():
    """
    No missing data.
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
    out = {}
    folder = (
        pathlib.Path(__file__).parents[1]
        / "process_improve"
        / "datasets"
        / "multivariate"
    )
    values = pd.read_csv(
        folder / "LDPE" / "LDPE.csv",
        index_col=0,
    )
    out["expected_T"] = pd.read_csv(folder / "LDPE" / "T.csv", header=None)
    out["expected_P"] = pd.read_csv(folder / "LDPE" / "P.csv", header=None)
    out["expected_W"] = pd.read_csv(folder / "LDPE" / "W.csv", header=None)
    out["expected_C"] = pd.read_csv(folder / "LDPE" / "C.csv", header=None)
    out["expected_U"] = pd.read_csv(folder / "LDPE" / "U.csv", header=None)
    out["expected_Hotellings_T2_A3"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A3.csv",
        header=None,
    )
    out["expected_Hotellings_T2_A6"] = pd.read_csv(
        folder / "LDPE" / "Hotellings_T2_A6.csv",
        header=None,
    )
    out["expected_Yhat_A6"] = pd.read_csv(
        folder / "LDPE" / "Yhat_A6.csv",
        header=None,
    )
    out["expected_SD_t"] = np.array(
        [1.872539, 1.440642, 1.216218, 1.141096, 1.059435, 0.9459715]
    )
    out["expected_T2_lim_95_A6"] = 15.2017
    out["expected_T2_lim_99_A6"] = 21.2239
    out["X"] = values.iloc[:, :14]
    out["Y"] = values.iloc[:, 14:]
    assert out["X"].shape == approx([54, 14])
    assert out["Y"].shape == approx([54, 5])
    out["A"] = 6
    return out


def test_PLS_SIMCA_LDPE(fixture_PLS_LDPE_example):
    """Unit test for LDPE case study.

    Parameters
    ----------
    PLS_model_SIMCA_LDPE_example : dict
        Dictionary of raw data and expected outputs from the PLS model.
    """
    data = fixture_PLS_LDPE_example
    plsmodel = PLS(n_components=data["A"])

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["Y"]))

    # Can only get these to very loosely match
    assert data["expected_T2_lim_95_A6"] == approx(plsmodel.T2_limit(0.95), rel=1e-1)
    assert data["expected_T2_lim_99_A6"] == approx(plsmodel.T2_limit(0.99), rel=1e-1)

    assert np.mean(
        np.abs(data["expected_T"].values) - np.abs(plsmodel.x_scores.values)
    ) == approx(0, abs=1e-4)
    assert np.mean(
        np.abs(data["expected_P"].values) - np.abs(plsmodel.x_loadings.values)
    ) == approx(0, abs=1e-5)
    assert np.mean(
        np.abs(data["expected_W"].values) - np.abs(plsmodel.x_weights.values)
    ) == approx(0, abs=1e-6)
    assert np.mean(
        np.abs(data["expected_C"].values) - np.abs(plsmodel.y_loadings.values)
    ) == approx(0, abs=1e-6)
    assert np.mean(
        np.abs(data["expected_U"].values) - np.abs(plsmodel.y_scores.values)
    ) == approx(0, abs=1e-5)
    assert np.mean(
        data["expected_Hotellings_T2_A3"].values.ravel()
        - plsmodel.Hotellings_T2.iloc[:, 2].values.ravel()
    ) == approx(0, abs=1e-6)
    assert np.mean(
        data["expected_Hotellings_T2_A6"].values.ravel()
        - plsmodel.Hotellings_T2.iloc[:, 5].values.ravel()
    ) == approx(0, abs=1e-6)
    assert np.mean(
        data["expected_SD_t"].ravel()
        - plsmodel.scaling_factor_for_scores.values.ravel()
    ) == approx(0, abs=1e-5)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(
                np.abs(
                    Y_mcuv.inverse_transform(plsmodel.y_predicted)
                    - data["expected_Yhat_A6"].values
                )
            )
            / Y_mcuv.center_
        )
    ) == approx(0, abs=1e-2)


def test_PLS_SIMCA_LDPE_missing_data(fixture_PLS_LDPE_example):
    """Unit test for LDPE case study.
    From visual inspection, observation 12 has low influence in the model.
    Set 1 value in this observation to missing and check that the results are similar to the
    full-data case: "test_PLS_SIMCA_LDPE",
    the only differences are that the tolerances are slightly relaxed.

    """
    data = fixture_PLS_LDPE_example
    data["X"].iloc[11, 0] = np.NaN
    plsmodel = PLS(n_components=data["A"], missing_data_settings=dict(md_method="scp"))

    X_mcuv = MCUVScaler().fit(data["X"])
    Y_mcuv = MCUVScaler().fit(data["Y"])
    plsmodel = plsmodel.fit(X_mcuv.transform(data["X"]), Y_mcuv.transform(data["Y"]))
    # Can only get these to very loosely match
    assert data["expected_T2_lim_95_A6"] == approx(plsmodel.T2_limit(0.95), rel=1e-1)
    assert data["expected_T2_lim_99_A6"] == approx(plsmodel.T2_limit(0.99), rel=1e-1)

    assert np.mean(
        np.abs(data["expected_T"].values) - np.abs(plsmodel.x_scores.values)
    ) == approx(0, abs=1e-2)
    assert np.mean(
        np.abs(data["expected_P"].values) - np.abs(plsmodel.x_loadings.values)
    ) == approx(0, abs=1e-3)
    assert np.mean(
        np.abs(data["expected_W"].values) - np.abs(plsmodel.x_weights.values)
    ) == approx(0, abs=1e-3)
    assert np.mean(
        np.abs(data["expected_C"].values) - np.abs(plsmodel.y_loadings.values)
    ) == approx(0, abs=1e-3)
    assert np.mean(
        np.abs(data["expected_U"].values) - np.abs(plsmodel.y_scores.values)
    ) == approx(0, abs=5e-1)
    assert np.mean(
        data["expected_Hotellings_T2_A3"].values.ravel()
        - plsmodel.Hotellings_T2.iloc[:, 2].values.ravel()
    ) == approx(0, abs=1e-6)
    assert np.mean(
        data["expected_Hotellings_T2_A6"].values.ravel()
        - plsmodel.Hotellings_T2.iloc[:, 5].values.ravel()
    ) == approx(0, abs=1e-6)
    assert np.mean(
        data["expected_SD_t"].ravel()
        - plsmodel.scaling_factor_for_scores.values.ravel()
    ) == approx(0, abs=1e-2)

    # Absolute sum of the deviations, accounting for the fact that each column in Y has quite
    # different range/scaling.
    assert np.sum(
        np.abs(
            np.sum(
                np.abs(
                    Y_mcuv.inverse_transform(plsmodel.y_predicted)
                    - data["expected_Yhat_A6"].values
                )
            )
            / Y_mcuv.center_
        )
    ) == approx(0, abs=0.5)
