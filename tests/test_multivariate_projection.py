"""Tests for missing-data score estimation: PCA.project / PLS.project and the operators."""

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._projection import (
    PROJECTION_METHODS,
    coerce_observed_mask,
    operator_for_pattern,
    project_rows,
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
    message = "At least one feature must be observed to estimate scores; the observed mask is all-False."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        operator_for_pattern(P, P, np.ones(2), np.zeros(4, dtype=bool))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"observed": np.ones(3, dtype=bool)},
            r"^observed must be a boolean mask over the 4 features; got shape \(3,\)\.$",
        ),
        (
            {"x_weights": np.ones((3, 2))},
            r"^x_weights must have the shape of x_loadings \(4, 2\); got \(3, 2\)\.$",
        ),
        (
            {"method": "tsr", "x_residuals": np.ones((10, 3))},
            r"^x_residuals must be \(n_train, 4\) over the model's features; got shape \(10, 3\)\.$",
        ),
    ],
    ids=["mask-of-another-length", "weights-of-another-shape", "residuals-over-other-features"],
)
def test_operator_for_pattern_refuses_inputs_of_the_wrong_shape(arguments: dict, message: str) -> None:
    """The mask, weights and residuals must all be laid out over the model's 4 features."""
    loadings = np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.3], [0.1, 0.9]])
    call = {
        "x_loadings": loadings,
        "guide": loadings,
        "score_variances": np.ones(2),
        "observed": np.ones(4, dtype=bool),
        **arguments,
    }
    with pytest.raises(ValueError, match=message):
        operator_for_pattern(**call)


def test_scp_cannot_estimate_a_score_whose_weights_vanish_where_observed() -> None:
    """A component whose weights are zero on every observed feature gets a zero row and an infinite condition."""
    loadings = np.array([[1.0, 0.5], [0.5, 1.0], [0.2, 0.3], [0.1, 0.9]])
    weights = loadings.copy()
    weights[[0, 1], 1] = 0.0  # the second component's weight lives only on the unobserved features
    observed = np.array([True, True, False, False])
    op = operator_for_pattern(loadings, loadings, np.ones(2), observed, method="scp", x_weights=weights)
    assert op.condition_number == np.inf
    np.testing.assert_array_equal(op.matrix[1], np.zeros(2))
    # The first component's weights are observed, so its row is a real, finite estimator.
    assert np.all(np.isfinite(op.matrix[0]))
    assert np.any(op.matrix[0] != 0)


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


def test_coerce_observed_mask_names_at_most_five_unknown_labels() -> None:
    """The message quotes the first five unknown labels, not all of them."""
    with pytest.raises(
        ValueError, match=r"^observed contains labels that are not model features: \['p', 'q', 'r', 's', 't'\]\.$"
    ):
        coerce_observed_mask(["a", "p", "q", "r", "s", "t", "u"], pd.Index(["a", "b"]))
    np.testing.assert_array_equal(coerce_observed_mask(np.array([False, True]), pd.Index(["a", "b"])), [False, True])


# A two-component model over three features, small enough to work every operator by hand.
_LOADINGS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) / np.array([np.sqrt(2.0), np.sqrt(2.0)])
_VARIANCES = np.array([4.0, 1.0])


def test_pmp_and_tsr_operators_with_a_ridge_by_hand() -> None:
    """PMP is ``(P*'P* + rI)^-1 P*'``; TSR is ``Theta P*'G* (G*'S**G* + rI)^-1 G*'``, worked with ``np.linalg.solve``.

    With the first two features observed, ``P*`` is the identity over sqrt(2), so the PMP Gram is
    ``I/2 + rI``. One training row of residuals ``e`` gives ``S**`` the term ``G*'e'eG* / max(1, 1 - 1)``.
    """
    observed = np.array([True, True, False])
    p_obs = _LOADINGS[observed]
    ridge = 0.25
    pmp = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed, method="pmp", ridge=ridge)
    np.testing.assert_allclose(pmp.matrix, np.linalg.solve(np.eye(2) / 2 + ridge * np.eye(2), p_obs.T), rtol=1e-14)
    assert pmp.condition_number == pytest.approx(1.0)
    assert pmp.method == "pmp"

    residuals = np.array([[0.3, -0.1, 0.2]])
    tsr = operator_for_pattern(
        _LOADINGS, _LOADINGS, _VARIANCES, observed, method="tsr", ridge=ridge, x_residuals=residuals
    )
    theta = np.diag(_VARIANCES)
    projected = residuals[:, observed] @ p_obs
    inner = p_obs.T @ p_obs @ theta @ p_obs.T @ p_obs + ridge * np.eye(2) + projected.T @ projected
    np.testing.assert_allclose(tsr.matrix, theta @ p_obs.T @ p_obs @ np.linalg.solve(inner, p_obs.T), rtol=1e-12)
    assert tsr.condition_number == pytest.approx(np.linalg.cond(inner), rel=1e-12)
    assert tsr.method == "tsr"


def test_tsr_divides_the_residual_spread_by_n_minus_1() -> None:
    """Three training rows: the residual half of the inner matrix is ``G*'E*'E*G* / 2``."""
    observed = np.array([True, True, False])
    p_obs = _LOADINGS[observed]
    residuals = np.array([[0.3, -0.1, 0.2], [0.0, 0.4, -0.2], [-0.2, 0.1, 0.5]])
    tsr = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed, x_residuals=residuals)
    theta = np.diag(_VARIANCES)
    projected = residuals[:, observed] @ p_obs
    inner = p_obs.T @ p_obs @ theta @ p_obs.T @ p_obs + projected.T @ projected / 2
    np.testing.assert_allclose(tsr.matrix, theta @ p_obs.T @ p_obs @ np.linalg.solve(inner, p_obs.T), rtol=1e-12)


def test_a_pattern_too_sparse_to_invert_names_the_matrix() -> None:
    """One observed feature cannot pin two scores: both inverting estimators refuse, naming the matrix."""
    observed = np.array([False, False, True])
    with pytest.raises(np.linalg.LinAlgError, match=r"^\(P_observed' @ P_observed\) is singular"):
        operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed, method="pmp")
    with pytest.raises(np.linalg.LinAlgError, match=r"^\(TSR inner matrix\) is singular"):
        operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed, x_residuals=np.zeros((4, 3)))


def test_scp_by_hand_with_its_condition_number() -> None:
    """With features 1 and 3 observed, the second weight keeps half its squared length, so the inflation is 2.

    Component 1: ``w* = (1, 1)/sqrt(2)``, so its row is ``w*' / (w*'w*) = (1, 1)/sqrt(2)``. Deflating
    by ``p* w*'/(w*'w*)`` leaves ``D = I - p* r1``, and component 2's row is ``w2*' D / (w2*'w2*)``.
    """
    observed = np.array([True, False, True])
    op = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed, method="scp")
    w = _LOADINGS[observed]
    row_1 = w[:, 0] / (w[:, 0] @ w[:, 0])
    deflate = np.eye(2) - np.outer(w[:, 0], row_1)
    row_2 = w[:, 1] @ deflate / (w[:, 1] @ w[:, 1])
    np.testing.assert_allclose(op.matrix, np.vstack([row_1, row_2]), rtol=1e-14)
    assert op.condition_number == pytest.approx(2.0, rel=1e-14)
    assert op.method == "scp"
    # Every feature observed: no weight loses any length, and the inflation is exactly 1.
    everything = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, np.ones(3, dtype=bool), method="scp")
    assert everything.condition_number == 1.0


def test_a_mask_of_zeros_and_ones_and_a_single_observed_feature() -> None:
    """An integer 0/1 mask is read as a boolean mask, and one observed feature is enough for SCP."""
    as_integers = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, np.array([1, 0, 1]), method="scp")
    as_booleans = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, np.array([True, False, True]), method="scp")
    np.testing.assert_array_equal(as_integers.matrix, as_booleans.matrix)
    single = operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, np.array([False, False, True]), method="scp")
    assert single.matrix.shape == (2, 1)


def test_tsr_without_residuals_warns_in_full_and_from_the_caller() -> None:
    observed = np.array([True, True, False])
    with pytest.warns(SpecificationWarning) as record:
        operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, observed)
    assert str(record[0].message) == (
        "method='tsr' was asked for without the training residuals (x_residuals), so its inner matrix keeps only "
        "the model-plane term. That is the projection-to-the-model-plane estimator, not trimmed score regression. "
        "Pass the fitted model's residual block, or ask for method='pmp' explicitly."
    )
    assert pathlib.Path(record[0].filename).name != "_projection.py"


def test_residuals_must_be_a_matrix() -> None:
    with pytest.raises(
        ValueError, match=r"^x_residuals must be \(n_train, 3\) over the model's features; got shape \(3,\)\.$"
    ):
        operator_for_pattern(_LOADINGS, _LOADINGS, _VARIANCES, np.ones(3, dtype=bool), x_residuals=np.zeros(3))


def test_project_rows_by_hand() -> None:
    """A complete row, two rows sharing one pattern and a row with another, scored, with SPE over what is observed."""
    x = np.array([[0.5, -1.0, 2.0], [1.0, np.nan, 0.5], [-2.0, np.nan, 1.0], [np.nan, 3.0, -1.0]])
    weights = np.array([[1.0, 0.2], [0.0, 1.0], [0.5, -1.0]])
    # A ridge of None leaves the argument out, to check the default of 0.
    for method, x_weights, ridge in (
        ("scp", None, None),
        ("scp", weights, None),
        ("pmp", None, None),
        ("pmp", None, 0.5),
    ):
        given = {} if ridge is None else {"ridge": ridge}
        result = project_rows(_LOADINGS, _LOADINGS, _VARIANCES, x, method=method, x_weights=x_weights, **given)
        assert result.n_observed.dtype.kind == "i"
        for row in (1, 2, 3):
            observed = ~np.isnan(x[row])
            op = operator_for_pattern(
                _LOADINGS, _LOADINGS, _VARIANCES, observed, method=method, ridge=ridge or 0.0, x_weights=x_weights
            )
            z = x[row, observed]
            scores = op.matrix @ z
            np.testing.assert_allclose(result.scores[row], scores, rtol=1e-14)
            residual = z - _LOADINGS[observed] @ scores
            assert result.spe[row] == pytest.approx(np.sqrt(residual @ residual), rel=1e-14)
            assert result.condition_number[row] == op.condition_number
            assert result.n_observed[row] == observed.sum()
    # The complete row takes the plain path: scores z P, all of z in its SPE, and condition 1.
    scores = x[0] @ _LOADINGS
    np.testing.assert_allclose(result.scores[0], scores, rtol=1e-14)
    residual = x[0] - _LOADINGS @ scores
    assert result.spe[0] == pytest.approx(np.sqrt(residual @ residual), rel=1e-14)
    assert result.condition_number[0] == 1.0
    assert result.n_observed[0] == 3


def test_project_rows_names_the_first_row_with_nothing_observed() -> None:
    x = np.array([[0.5, -1.0, 2.0], [np.nan] * 3, [1.0, 2.0, np.nan], [np.nan] * 3])
    with pytest.raises(
        ValueError, match=r"^Row 1 has no observed features \(all-NaN\); scores cannot be estimated for it\.$"
    ):
        project_rows(_LOADINGS, _LOADINGS, _VARIANCES, x)
