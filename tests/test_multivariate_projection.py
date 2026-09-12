"""Tests for missing-data score estimation: PCA.project / PLS.project and the operators."""

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._projection import (
    PROJECTION_METHODS,
    coerce_observed_mask,
    operator_for_pattern,
)


@pytest.fixture
def correlated_data() -> pd.DataFrame:
    """Highly correlated 8-column data with a known 2-dimensional structure."""
    rng = np.random.default_rng(7)
    n = 120
    t1 = rng.standard_normal(n) * 3.0
    t2 = rng.standard_normal(n) * 1.5
    columns = {}
    for j in range(8):
        columns[f"x{j}"] = 0.9 * np.cos(j) * t1 + 0.8 * np.sin(1 + j) * t2 + 0.05 * rng.standard_normal(n)
    return pd.DataFrame(columns)


@pytest.fixture
def fitted_pca(correlated_data: pd.DataFrame) -> PCA:
    scaled = MCUVScaler().fit_transform(correlated_data)
    return PCA(n_components=2).fit(scaled)


@pytest.fixture
def fitted_pls(correlated_data: pd.DataFrame) -> PLS:
    rng = np.random.default_rng(11)
    beta = rng.uniform(0.5, 1.5, size=correlated_data.shape[1])
    y = pd.DataFrame({"y": correlated_data.to_numpy() @ beta + 0.05 * rng.standard_normal(len(correlated_data))})
    return PLS(n_components=2).fit(correlated_data, y)


def test_pca_project_complete_rows_bitwise_equal_transform(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """On complete rows, project() returns exactly the transform() scores."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    projected = fitted_pca.project(scaled)
    expected = fitted_pca.transform(scaled)
    assert (projected.scores.to_numpy() == expected.to_numpy()).all()
    diagnostics = fitted_pca.diagnose(scaled)
    np.testing.assert_allclose(projected.spe.to_numpy(), diagnostics.spe.to_numpy(), rtol=1e-12)
    assert (projected.condition_number == 1.0).all()
    assert (projected.n_observed == correlated_data.shape[1]).all()


def test_pls_project_complete_rows_bitwise_equal_transform(fitted_pls: PLS, correlated_data: pd.DataFrame) -> None:
    """On complete rows, PLS.project() returns exactly the transform() scores and diagnose() y_hat."""
    projected = fitted_pls.project(correlated_data)
    expected = fitted_pls.transform(correlated_data)
    assert (projected.scores.to_numpy() == expected.to_numpy()).all()
    diagnostics = fitted_pls.diagnose(correlated_data)
    np.testing.assert_allclose(projected.y_hat.to_numpy(), diagnostics.y_hat.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(projected.spe.to_numpy(), diagnostics.spe.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("method", PROJECTION_METHODS)
def test_pca_project_recovers_hidden_scores(method: str, fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """With 3 of 8 correlated columns hidden, every estimator stays close to the truth."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    true_scores = fitted_pca.transform(scaled).to_numpy()
    hidden = scaled.copy()
    hidden.iloc[:, [1, 4, 6]] = np.nan
    estimate = fitted_pca.project(hidden, method=method)
    assert (estimate.n_observed == 5).all()
    # The columns are highly redundant, so the estimate correlates strongly
    # with the complete-data scores for every method.
    for a in range(2):
        corr = np.corrcoef(estimate.scores.to_numpy()[:, a], true_scores[:, a])[0, 1]
        assert corr > 0.95, f"{method} component {a + 1}: correlation {corr:.3f}"


def test_tsr_beats_scp_on_influential_missing(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """TSR gives a lower score-estimation error than SCP (Arteaga and Ferrer's ranking)."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    true_scores = fitted_pca.transform(scaled).to_numpy()
    hidden = scaled.copy()
    hidden.iloc[:, [0, 2, 3, 5]] = np.nan
    errors = {}
    for method in ("tsr", "scp"):
        estimate = fitted_pca.project(hidden, method=method).scores.to_numpy()
        errors[method] = float(np.mean((estimate - true_scores) ** 2))
    assert errors["tsr"] <= errors["scp"]


def test_project_matches_manual_operator(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """project() on an incomplete row equals applying projection_matrix by hand."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    hidden = scaled.copy()
    hidden.iloc[:, [2, 5]] = np.nan
    projected = fitted_pca.project(hidden, method="tsr")
    observed_columns = [c for i, c in enumerate(scaled.columns) if i not in (2, 5)]
    op = fitted_pca.projection_matrix(observed_columns, method="tsr")
    manual = scaled[observed_columns].to_numpy() @ op.matrix.to_numpy().T
    np.testing.assert_allclose(projected.scores.to_numpy(), manual, rtol=1e-10)
    assert list(op.matrix.columns) == observed_columns


def test_projection_matrix_boolean_mask_and_labels_agree(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """A boolean mask and the equivalent label list give the same operator."""
    mask = np.ones(correlated_data.shape[1], dtype=bool)
    mask[[1, 3]] = False
    labels = [c for c, keep in zip(correlated_data.columns, mask, strict=True) if keep]
    by_mask = fitted_pca.projection_matrix(mask, method="pmp")
    by_labels = fitted_pca.projection_matrix(labels, method="pmp")
    np.testing.assert_array_equal(by_mask.matrix.to_numpy(), by_labels.matrix.to_numpy())


def test_ridge_improves_conditioning(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """Adding ridge reduces the reported condition number of the inverted matrix."""
    mask = np.zeros(correlated_data.shape[1], dtype=bool)
    mask[:2] = True  # only two observed columns: poorly determined
    plain = fitted_pca.projection_matrix(mask, method="pmp", ridge=0.0)
    ridged = fitted_pca.projection_matrix(mask, method="pmp", ridge=0.5)
    assert ridged.condition_number < plain.condition_number


def test_pls_project_predicts_with_missing_columns(fitted_pls: PLS, correlated_data: pd.DataFrame) -> None:
    """Hiding redundant columns barely moves the PLS prediction."""
    complete = fitted_pls.diagnose(correlated_data).y_hat.to_numpy().ravel()
    hidden = correlated_data.copy()
    hidden.iloc[:, [1, 6]] = np.nan
    partial = fitted_pls.project(hidden, method="tsr").y_hat.to_numpy().ravel()
    corr = np.corrcoef(complete, partial)[0, 1]
    assert corr > 0.99


def test_all_nan_row_rejected(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """A row with no observed features is rejected with a clear message."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    hidden = scaled.iloc[:3].copy()
    hidden.iloc[1] = np.nan
    with pytest.raises(ValueError, match="no observed features"):
        fitted_pca.project(hidden)


def test_unknown_method_rejected(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """An unknown estimator name is rejected."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    with pytest.raises(ValueError, match="method must be one of"):
        fitted_pca.project(scaled, method="nonsense")


def test_negative_ridge_rejected(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """A negative ridge is rejected."""
    mask = np.ones(correlated_data.shape[1], dtype=bool)
    mask[0] = False
    with pytest.raises(ValueError, match="ridge must be non-negative"):
        fitted_pca.projection_matrix(mask, ridge=-0.1)


def test_coerce_observed_mask_errors() -> None:
    """Bad masks and unknown labels are rejected with actionable messages."""
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="length 3"):
        coerce_observed_mask(np.array([True, False]), names)
    with pytest.raises(ValueError, match="not model features"):
        coerce_observed_mask(["a", "zzz"], names)
    mask = coerce_observed_mask(["b", "c"], names)
    assert mask.tolist() == [False, True, True]


def test_operator_for_pattern_empty_mask_rejected() -> None:
    """An all-False observed mask is rejected at the operator level."""
    P = np.ones((4, 2))
    with pytest.raises(ValueError, match="observed mask is all-False"):
        operator_for_pattern(P, P, np.ones(2), np.zeros(4, dtype=bool))


def test_scp_matches_adaptive_kernel_shape(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """SCP via the operator equals the sequential deflation computed by hand."""
    scaled = MCUVScaler().fit_transform(correlated_data)
    mask = np.ones(correlated_data.shape[1], dtype=bool)
    mask[[0, 7]] = False
    op = fitted_pca.projection_matrix(mask, method="scp")
    row = scaled.iloc[5].to_numpy()
    z_obs = row[mask]
    # Hand-computed sequential SCP with deflation over the observed entries.
    loadings = fitted_pca.loadings_.to_numpy()
    deflate = z_obs.copy()
    expected = np.zeros(2)
    for a in range(2):
        p_a = loadings[mask, a]
        expected[a] = float(deflate @ p_a) / float(p_a @ p_a)
        deflate = deflate - expected[a] * p_a
    np.testing.assert_allclose(op.matrix.to_numpy() @ z_obs, expected, rtol=1e-12)


def test_tsr_is_the_regression_of_scores_on_trimmed_scores(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """TSR is what its name says: the model's scores regressed on the trimmed scores.

    Arteaga and Ferrer (2002) write the operator in closed form,
    ``Theta P*' G* [G*' S** G*]^-1 G*'``, with ``S**`` the covariance of the
    observed columns in the training data. That is algebraically the
    least-squares fit of the training scores on the trimmed scores, which is
    computed here from the data as the reference.
    """
    scaled = MCUVScaler().fit_transform(correlated_data)
    mask = np.zeros(correlated_data.shape[1], dtype=bool)
    mask[:3] = True

    operator = fitted_pca.projection_matrix(mask, method="tsr").matrix.to_numpy()

    loadings = fitted_pca.loadings_.to_numpy()
    trimmed = scaled.to_numpy()[:, mask] @ loadings[mask, :]
    coefficients = np.linalg.lstsq(trimmed, fitted_pca.scores_.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(operator, (loadings[mask, :] @ coefficients).T, rtol=1e-8, atol=1e-10)


def test_tsr_is_not_projection_to_the_model_plane(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """Dropping the residual term from the TSR inner matrix turns it into PMP.

    For a PCA the guide is the loading matrix, so ``G*' P* Theta P*' G*`` alone
    cancels down to ``(P*'P*)^-1 P*'``: the two operators were bitwise equal,
    and one of the three estimators was a relabelling of another.
    """
    mask = np.zeros(correlated_data.shape[1], dtype=bool)
    mask[:3] = True
    tsr = fitted_pca.projection_matrix(mask, method="tsr").matrix.to_numpy()
    pmp = fitted_pca.projection_matrix(mask, method="pmp").matrix.to_numpy()
    assert not np.allclose(tsr, pmp, atol=1e-6)


def test_tsr_without_training_residuals_warns_and_falls_back(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """Called bare, the operator says it is giving PMP rather than doing so quietly."""
    mask = np.zeros(correlated_data.shape[1], dtype=bool)
    mask[:3] = True
    loadings = fitted_pca.loadings_.to_numpy()
    variances = np.asarray(fitted_pca.explained_variance_, dtype=float)

    with pytest.warns(SpecificationWarning, match="not .*trimmed score regression"):
        bare = operator_for_pattern(loadings, loadings, variances, mask, method="tsr")
    pmp = operator_for_pattern(loadings, loadings, variances, mask, method="pmp")
    np.testing.assert_allclose(bare.matrix, pmp.matrix, atol=1e-10)


def test_tsr_shrinks_where_the_observed_columns_say_little(fitted_pca: PCA, correlated_data: pd.DataFrame) -> None:
    """With few columns observed TSR stays inside the training spread; PMP does not.

    Columns 0 and 3 point almost the same way in the loading plane, so they
    pin the first score and say little about the second. The regression has
    seen that and shrinks the second score toward the mean row; PMP fits the
    plane to whatever it is given and overshoots. Lower error for TSR is the
    ranking Arteaga and Ferrer report.
    """
    scaled = MCUVScaler().fit_transform(correlated_data)
    hidden = scaled.copy()
    hidden.iloc[:, [1, 2, 4, 5, 6, 7]] = np.nan  # columns 0 and 3 left

    true_scores = fitted_pca.scores_.to_numpy()
    training_spread = true_scores.std(axis=0, ddof=1)
    tsr = fitted_pca.project(hidden, method="tsr").scores.to_numpy()
    pmp = fitted_pca.project(hidden, method="pmp").scores.to_numpy()

    assert (tsr.std(axis=0, ddof=1) <= training_spread * 1.01).all()
    assert (pmp.std(axis=0, ddof=1) / training_spread).max() > 1.2
    assert np.mean((tsr - true_scores) ** 2) < np.mean((pmp - true_scores) ** 2)


def test_pls_tsr_is_the_regression_of_scores_on_trimmed_scores(fitted_pls: PLS, correlated_data: pd.DataFrame) -> None:
    """The same identity holds for PLS, where the guide is the direct weights."""
    mask = np.zeros(correlated_data.shape[1], dtype=bool)
    mask[:3] = True

    operator = fitted_pls.projection_matrix(mask, method="tsr").matrix.to_numpy()

    guide = fitted_pls.direct_weights_.to_numpy()
    scaled = fitted_pls._x_scaler.transform(correlated_data).to_numpy()
    trimmed = scaled[:, mask] @ guide[mask, :]
    coefficients = np.linalg.lstsq(trimmed, fitted_pls.scores_.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(operator, (guide[mask, :] @ coefficients).T, rtol=1e-7, atol=1e-9)
