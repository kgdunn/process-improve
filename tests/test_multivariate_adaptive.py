# (c) Kevin Dunn, 2010-2026. MIT License.
"""Tests for the adaptive (recursive) PCA and PLS estimators.

Covers, for both estimators:

* initial-fit correctness - the kernel-initialised model reproduces the batch PCA loadings
  and the batch PLS beta coefficients on synthetic and real (LDPE) data;
* the Krzanowski subspace distance metric bounds and sign invariance;
* the injection term semantics (``gamma = 0`` recovers the plain recursive
  update; ``gamma > 0`` improves conditioning under rank-poor operation);
* numerical stability of the norm-rescaled kernel;
* drift tracking versus a frozen model;
* the infrequently-sampled response path for PLS;
* sklearn estimator conventions (``get_params`` / ``set_params`` / ``clone``).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from process_improve.multivariate import PCA, PLS, AdaptivePCA, AdaptivePLS, MCUVScaler
from process_improve.multivariate._adaptive import (
    _kernel_pca,
    _kernel_pls,
    _subspace_distance,
)
from process_improve.multivariate._nipals import _sign_align

DATASETS = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_pca_data() -> pd.DataFrame:
    """Return a correlated, full-rank block for PCA initialisation and streaming tests."""
    rng = np.random.default_rng(42)
    n, k, a = 250, 8, 3
    loadings = rng.standard_normal((k, a))
    scores = rng.standard_normal((n, a))
    values = scores @ loadings.T + 0.1 * rng.standard_normal((n, k))
    return pd.DataFrame(values, columns=[f"v{i}" for i in range(k)])


@pytest.fixture
def synthetic_pls_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a synthetic single-response regression block for PLS initialisation tests."""
    rng = np.random.default_rng(7)
    n, k = 180, 6
    beta = rng.standard_normal((k, 1))
    X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"x{i}" for i in range(k)])
    Y = pd.DataFrame(X.to_numpy() @ beta + 0.1 * rng.standard_normal((n, 1)), columns=["y"])
    return X, Y


@pytest.fixture
def ldpe_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real LDPE dataset (54 x 14 predictors, 5 responses).

    The first 50 rows are common-cause; the last 4 show a developing fault.
    """
    values = pd.read_csv(DATASETS / "LDPE" / "LDPE.csv", index_col=0)
    X = values.iloc[:, :14]
    Y = values.iloc[:, 14:]
    return X, Y


# --------------------------------------------------------------------------- #
# Kernel-helper unit tests
# --------------------------------------------------------------------------- #


def test_kernel_pca_matches_svd(synthetic_pca_data: pd.DataFrame) -> None:
    """``_kernel_pca`` eigenvectors match the SVD loadings of the scaled data."""
    scaled = MCUVScaler().fit_transform(synthetic_pca_data)
    Xs = scaled.to_numpy()
    loadings, eigenvalues = _kernel_pca(Xs.T @ Xs, n_components=3)
    # Batch PCA on the same scaled data (AdaptivePCA scales internally).
    batch = PCA(n_components=3).fit(scaled)
    np.testing.assert_allclose(np.abs(loadings), np.abs(batch.loadings_.to_numpy()), atol=1e-8)
    np.testing.assert_allclose(eigenvalues / (len(Xs) - 1), batch.explained_variance_, rtol=1e-8)


def test_kernel_pls_reconstructs_beta(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """``_kernel_pls`` yields the same scaled beta as the batch NIPALS PLS."""
    X, Y = synthetic_pls_data
    xs = MCUVScaler().fit(X)
    ys = MCUVScaler().fit(Y)
    Xs = xs.transform(X).to_numpy()
    Ys = ys.transform(Y).to_numpy()
    _, _, direct, y_load, _ = _kernel_pls(Xs.T @ Xs, Xs.T @ Ys, n_components=3)
    beta_scaled = direct @ y_load.T
    beta_raw = beta_scaled * (ys.scale_.to_numpy()[np.newaxis, :] / xs.scale_.to_numpy()[:, np.newaxis])
    batch = PLS(n_components=3, scale=True).fit(X, Y)
    np.testing.assert_allclose(beta_raw, batch.beta_coefficients_.to_numpy(), atol=1e-9)


def test_kernel_pls_first_weight_with_two_responses_is_the_leading_singular_vector_of_xty() -> None:
    """With several responses, the first weight is the direction ``w`` that maximises ``|Y'X w|``.

    That is the leading left singular vector of ``X'Y``, taken here from numpy's SVD rather than
    from the eigenproblem of ``(X'Y)'X'Y`` that ``_kernel_pls`` solves.
    """
    rng = np.random.default_rng(5)
    x = rng.standard_normal((30, 4))
    y = x @ rng.standard_normal((4, 2)) + 0.1 * rng.standard_normal((30, 2))
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    weights, *_ = _kernel_pls(x.T @ x, x.T @ y, n_components=1)
    leading = np.linalg.svd(x.T @ y)[0][:, 0]
    np.testing.assert_allclose(weights[:, 0] * np.sign(weights[:, 0] @ leading), leading, rtol=0, atol=1e-12)


def test_sign_align_flips_only_the_columns_that_point_against_the_previous_ones() -> None:
    """A column flips when its inner product with the previous one is negative, and only then."""
    previous = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    # Inner products with ``previous``: -2 (flip), 0.5 (positive, though below 1), 0 (orthogonal), 3.
    current = np.array([[-2.0, 0.5, 0.0, 3.0], [1.0, 1.0, 7.0, -1.0]])
    np.testing.assert_array_equal(_sign_align(current, previous), [-1.0, 1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("kernel_xx", "kernel_xy"),
    [(np.eye(2), np.zeros((2, 1))), (np.zeros((2, 2)), np.ones((2, 1)))],
    ids=["no-covariance-with-y", "no-spread-in-x"],
)
def test_kernel_pls_extracts_nothing_when_nothing_is_left(kernel_xx: np.ndarray, kernel_xy: np.ndarray) -> None:
    """With no X'Y covariance, or no X'X spread along it, no component is extracted: every output stays zero."""
    for part in _kernel_pls(kernel_xx, kernel_xy, n_components=1):
        np.testing.assert_array_equal(part, np.zeros_like(part))


# --------------------------------------------------------------------------- #
# Initial-fit correctness
# --------------------------------------------------------------------------- #


def test_adaptive_pca_initial_matches_batch(synthetic_pca_data: pd.DataFrame) -> None:
    """AdaptivePCA initialises an i=0 model identical to batch PCA on the scaled data."""
    model = AdaptivePCA(n_components=3).fit(synthetic_pca_data)
    batch = PCA(n_components=3).fit(MCUVScaler().fit_transform(synthetic_pca_data))
    np.testing.assert_allclose(np.abs(model.loadings0_), np.abs(batch.loadings_.to_numpy()), atol=1e-8)
    np.testing.assert_allclose(model.explained_variance_, batch.explained_variance_, rtol=1e-8)


def test_adaptive_pls_initial_matches_batch_ldpe(ldpe_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """AdaptivePLS initialises beta identical to the batch PLS on the real LDPE data."""
    X, Y = ldpe_data
    model = AdaptivePLS(n_components=6).fit(X, Y)
    batch = PLS(n_components=6, scale=True).fit(X, Y)
    # rtol relaxed from the assert_allclose default of 1e-7 to 1e-6: the
    # scale-relative NIPALS convergence criterion (#504) stops the batch fit
    # once successive score iterations agree to epsqrt (~1.49e-8) *relative*
    # accuracy, and the kernel-vs-NIPALS agreement on this fixture then sits
    # just past 1e-7 relative (1.07e-7 on one of 70 elements). Both solvers
    # are converged to their own criteria; 1e-6 keeps the comparison tight
    # without asserting more agreement than the stopping rule guarantees.
    np.testing.assert_allclose(
        model.beta_coefficients_.to_numpy(), batch.beta_coefficients_.to_numpy(), rtol=1e-6, atol=1e-7
    )
    # Predictions with the freshly-initialised model equal the batch predictions.
    np.testing.assert_allclose(model.predict(X).to_numpy(), batch.predictions_.to_numpy(), atol=1e-6)


def test_adaptive_pls_initial_matches_batch_synthetic(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """AdaptivePLS initialises beta to machine precision on well-conditioned synthetic data."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3).fit(X, Y)
    batch = PLS(n_components=3, scale=True).fit(X, Y)
    np.testing.assert_allclose(model.beta_coefficients_.to_numpy(), batch.beta_coefficients_.to_numpy(), atol=1e-12)


# --------------------------------------------------------------------------- #
# Distance metric
# --------------------------------------------------------------------------- #


def test_distance_metric_bounds_and_sign_invariance() -> None:
    """The distance is A for identical spaces, 0 for orthogonal, sign-invariant."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    p = q[:, :3]
    assert _subspace_distance(p, p) == pytest.approx(3.0)
    # Sign flips of any column leave the metric unchanged.
    flipped = p.copy()
    flipped[:, 1] *= -1
    assert _subspace_distance(p, flipped) == pytest.approx(3.0)
    # An orthogonal complement gives zero overlap.
    other = q[:, 3:6]
    assert _subspace_distance(p, other) == pytest.approx(0.0, abs=1e-12)


def test_distance_stays_at_a_without_updates(synthetic_pca_data: pd.DataFrame) -> None:
    """With no kernel adaptation the distance stays exactly at A."""
    model = AdaptivePCA(n_components=3, forgetting_factor=0.0, gamma=0.0).fit(synthetic_pca_data)
    model.update(synthetic_pca_data.iloc[0].to_numpy())
    assert model.distance_.iloc[-1] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Injection term (gamma) semantics
# --------------------------------------------------------------------------- #


def test_gamma_zero_recovers_plain_recursion(synthetic_pca_data: pd.DataFrame) -> None:
    """gamma=0 makes the kernel update the plain (1-mu) X'X + mu x x' recursion."""
    stream = synthetic_pca_data.iloc[:20]
    # Freeze the EWMA preprocessing (lambda = alpha = 0) so the scaled observation
    # is a fixed function of the raw row and the recursion is reproducible here.
    model = AdaptivePCA(
        n_components=3,
        forgetting_factor=0.1,
        gamma=0.0,
        lambda_center=0.0,
        alpha_scale=0.0,
        update_when_out_of_control=True,
    ).fit(synthetic_pca_data)
    kernel = model.XtX0_.copy()
    # The re-scale holds the nuclear norm (= trace for this PSD kernel) constant.
    norm0 = float(np.trace(kernel))
    mu = 0.1
    for _, row in stream.iterrows():
        x = np.nan_to_num((row.to_numpy() - model.mx_) / model.sx_)
        model.update(row.to_numpy())
        kernel = (1 - mu) * kernel + mu * np.outer(x, x)
        kernel *= norm0 / float(np.trace(kernel))
    np.testing.assert_allclose(model.XtX_, kernel, atol=1e-8)


def test_gamma_improves_conditioning_under_quiet_operation() -> None:
    """Under rank-poor excitation, gamma>0 keeps X'X better conditioned than gamma=0."""
    rng = np.random.default_rng(5)
    # A well-excited training block, then a "quiet" stream living in a 2-D subspace.
    n, k = 200, 6
    train = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"v{i}" for i in range(k)])
    basis = rng.standard_normal((2, k))
    quiet = pd.DataFrame(
        rng.standard_normal((400, 2)) @ basis + 0.01 * rng.standard_normal((400, k)),
        columns=train.columns,
    )
    kwargs = dict(n_components=3, forgetting_factor=0.05, update_when_out_of_control=True)
    m0 = AdaptivePCA(gamma=0.0, **kwargs).fit(train)
    mg = AdaptivePCA(gamma=0.2, **kwargs).fit(train)
    for _, row in quiet.iterrows():
        m0.update(row.to_numpy())
        mg.update(row.to_numpy())
    assert np.linalg.cond(mg.XtX_) < np.linalg.cond(m0.XtX_)


# --------------------------------------------------------------------------- #
# Numerical stability
# --------------------------------------------------------------------------- #


def test_kernel_norm_stays_constant(synthetic_pca_data: pd.DataFrame) -> None:
    """The norm-rescaling holds the nuclear norm (trace) of X'X at its training value over many updates."""
    model = AdaptivePCA(n_components=3, forgetting_factor=0.1, gamma=0.1, update_when_out_of_control=True).fit(
        synthetic_pca_data
    )
    norm0 = float(np.trace(model.XtX0_))
    rng = np.random.default_rng(1)
    for _ in range(500):
        model.update(synthetic_pca_data.iloc[rng.integers(len(synthetic_pca_data))].to_numpy())
    assert float(np.trace(model.XtX_)) == pytest.approx(norm0, rel=1e-6)


# --------------------------------------------------------------------------- #
# Drift tracking
# --------------------------------------------------------------------------- #


def test_adaptive_tracks_mean_drift_where_frozen_alarms() -> None:
    """A slow drift of the operating point alarms a frozen model far more than an adaptive one.

    The stream keeps the training correlation structure but slowly moves the
    operating point along the first latent direction. An adaptive model tracks
    the moving centre (and re-learns the kernel from the in-control observations),
    so it raises substantially fewer alarms than a model frozen at the training data.
    """
    rng = np.random.default_rng(11)
    n, k, a = 300, 5, 2
    noise = 0.3
    loadings = rng.standard_normal((k, a))
    train = pd.DataFrame(
        rng.standard_normal((n, a)) @ loadings.T + noise * rng.standard_normal((n, k)),
        columns=[f"v{i}" for i in range(k)],
    )
    steps = 400
    drift = np.linspace(0, 5.0, steps)[:, None] * loadings[:, 0][None, :]
    stream = pd.DataFrame(
        rng.standard_normal((steps, a)) @ loadings.T + noise * rng.standard_normal((steps, k)) + drift,
        columns=train.columns,
    )
    adaptive = AdaptivePCA(n_components=a, forgetting_factor=0.05, gamma=0.1, lambda_center=0.25).fit(train)
    frozen = AdaptivePCA(n_components=a, forgetting_factor=0.0, gamma=0.0, lambda_center=0.0).fit(train)
    adaptive_alarms = sum(not adaptive.update(r.to_numpy()).in_control for _, r in stream.iterrows())
    frozen_alarms = sum(not frozen.update(r.to_numpy()).in_control for _, r in stream.iterrows())
    # The frozen model raises clearly more alarms on a drift it never learns.
    assert frozen_alarms > 1.5 * adaptive_alarms
    assert adaptive_alarms < steps // 2


# --------------------------------------------------------------------------- #
# Streaming interface / infrequent-Y
# --------------------------------------------------------------------------- #


def test_partial_fit_accumulates_history(synthetic_pca_data: pd.DataFrame) -> None:
    """partial_fit records one history row per streamed observation."""
    train = synthetic_pca_data.iloc[:150]
    stream = synthetic_pca_data.iloc[150:]
    model = AdaptivePCA(n_components=3).fit(train)
    model.partial_fit(stream)
    assert model.scores_.shape == (len(stream), 3)
    assert model.spe_.shape[0] == len(stream)
    assert model.hotellings_t2_.shape[0] == len(stream)
    assert len(model.distance_) == len(stream)


def test_adaptive_pls_infrequent_y(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The X-space adapts every step; the regression only when a response arrives."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3, forgetting_factor=0.05, gamma=0.1).fit(X, Y)
    xty_initial = model.XtY_.copy()
    rng = np.random.default_rng(2)
    n_x_only = 0
    for i in range(60):
        row = X.iloc[rng.integers(len(X))].to_numpy()
        if i % 10 == 0:
            model.update(row, y_row=Y.iloc[i % len(Y)].to_numpy())
        else:
            before = model.XtY_.copy()
            model.update(row, y_row=None)
            # X'Y untouched by an X-only observation.
            np.testing.assert_array_equal(model.XtY_, before)
            n_x_only += 1
    assert n_x_only > 0
    # X'Y did change at the lab points.
    assert not np.allclose(model.XtY_, xty_initial)
    assert model.predictions_.shape[0] == 60


def test_out_of_control_does_not_update_by_default(synthetic_pca_data: pd.DataFrame) -> None:
    """A wild observation is flagged and, by default, does not move the model."""
    model = AdaptivePCA(n_components=3, forgetting_factor=0.1, gamma=0.1).fit(synthetic_pca_data)
    kernel_before = model.XtX_.copy()
    wild = synthetic_pca_data.iloc[0].to_numpy() + 50.0
    out = model.update(wild)
    assert not out.in_control
    assert not out.updated
    np.testing.assert_array_equal(model.XtX_, kernel_before)


# --------------------------------------------------------------------------- #
# sklearn conventions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("estimator", [AdaptivePCA(2), AdaptivePLS(2)])
def test_get_set_params_roundtrip(estimator: AdaptivePCA | AdaptivePLS) -> None:
    """get_params / set_params / clone behave per the sklearn contract."""
    params = estimator.get_params()
    assert params["n_components"] == 2
    estimator.set_params(gamma=0.33, forgetting_factor=0.07)
    assert estimator.get_params()["gamma"] == 0.33
    fresh = clone(estimator)
    assert fresh.get_params()["gamma"] == 0.33
    assert fresh.get_params()["forgetting_factor"] == 0.07


def test_invalid_parameters_raise(synthetic_pca_data: pd.DataFrame) -> None:
    """Out-of-range tuning constants are rejected at fit time."""
    with pytest.raises(ValueError, match="forgetting_factor"):
        AdaptivePCA(n_components=2, forgetting_factor=1.5).fit(synthetic_pca_data)
    with pytest.raises(ValueError, match="gamma"):
        AdaptivePCA(n_components=2, gamma=-0.1).fit(synthetic_pca_data)
    with pytest.raises(ValueError, match="lambda_center"):
        AdaptivePCA(n_components=2, lambda_center=2.0).fit(synthetic_pca_data)


# --------------------------------------------------------------------------- #
# transform / predict / property views
# --------------------------------------------------------------------------- #


def test_transform_and_predict_match_initial(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Predict and transform on the freshly-initialised model equal the batch model."""
    X, Y = synthetic_pls_data
    ad = AdaptivePLS(n_components=3).fit(X, Y)
    batch = PLS(n_components=3, scale=True).fit(X, Y)
    np.testing.assert_allclose(ad.predict(X).to_numpy(), batch.predictions_.to_numpy(), atol=1e-9)
    # Scores line up with the batch X-scores up to a per-component sign.
    np.testing.assert_allclose(np.abs(ad.transform(X).to_numpy()), np.abs(batch.scores_.to_numpy()), atol=1e-7)


def test_pca_transform_shape(synthetic_pca_data: pd.DataFrame) -> None:
    """AdaptivePCA.transform returns one score row per input row."""
    ad = AdaptivePCA(n_components=3).fit(synthetic_pca_data)
    scores = ad.transform(synthetic_pca_data.iloc[:10])
    assert scores.shape == (10, 3)


def test_property_views_have_expected_shapes(ldpe_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The current-parameter DataFrame views carry the right labels and shapes."""
    X, Y = ldpe_data
    ad = AdaptivePLS(n_components=4).fit(X, Y)
    assert ad.x_weights_.shape == (14, 4)
    assert ad.x_loadings_.shape == (14, 4)
    assert ad.direct_weights_.shape == (14, 4)
    assert ad.beta_coefficients_.shape == (14, 5)
    assert list(ad.beta_coefficients_.columns) == list(Y.columns)
    pca = AdaptivePCA(n_components=2).fit(X)
    assert pca.loadings_.shape == (14, 2)


def test_missing_data_projection_is_finite(synthetic_pca_data: pd.DataFrame) -> None:
    """A partially-missing observation projects through single-component projection."""
    ad = AdaptivePCA(n_components=3).fit(synthetic_pca_data)
    row = synthetic_pca_data.iloc[0].to_numpy().copy()
    row[1] = np.nan
    row[4] = np.nan
    out = ad.update(row)
    assert np.all(np.isfinite(out.scores))
    assert np.isfinite(out.spe)
    assert np.isfinite(out.hotellings_t2)


def test_all_missing_row_is_skipped(synthetic_pca_data: pd.DataFrame) -> None:
    """An all-missing observation never updates the model and is out of control."""
    ad = AdaptivePCA(n_components=3, update_when_out_of_control=True).fit(synthetic_pca_data)
    kernel_before = ad.XtX_.copy()
    out = ad.update(np.full(synthetic_pca_data.shape[1], np.nan))
    assert not out.in_control
    assert not out.updated
    np.testing.assert_array_equal(ad.XtX_, kernel_before)


def test_partial_fit_pls_with_response(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """partial_fit threads a matching Y block and records a prediction per row."""
    X, Y = synthetic_pls_data
    ad = AdaptivePLS(n_components=3, forgetting_factor=0.02).fit(X.iloc[:100], Y.iloc[:100])
    ad.partial_fit(X.iloc[100:], Y.iloc[100:])
    assert ad.predictions_.shape[0] == len(X) - 100
    assert ad.scores_.shape == (len(X) - 100, 3)


def test_envelope_tracking_reduces_drift_bias() -> None:
    """With continuous tracking, a soft-sensor's prediction bias stays small under drift.

    A single-response regression whose input distribution slowly shifts induces a
    growing prediction bias in a frozen model. An adaptive model that tracks the
    operating envelope keeps the late-period bias far smaller.
    """
    rng = np.random.default_rng(19)
    k = 5
    beta = rng.standard_normal((k, 1))

    def make_block(n: int, shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = rng.standard_normal((n, k)) + shift
        X = pd.DataFrame(base, columns=[f"x{i}" for i in range(k)])
        y = X.to_numpy() @ beta + 0.1 * rng.standard_normal((n, 1))
        return X, pd.DataFrame(y, columns=["y"])

    X0, Y0 = make_block(200, shift=0.0)
    frozen = AdaptivePLS(n_components=3, forgetting_factor=0.0, gamma=0.0, lambda_center=0.0).fit(X0, Y0)
    adaptive = AdaptivePLS(
        n_components=3,
        forgetting_factor=0.01,
        gamma=0.1,
        lambda_center=0.03,
        alpha_scale=0.02,
        lambda_center_y=0.05,
        update_when_out_of_control=True,
    ).fit(X0, Y0)

    frozen_err, adaptive_err = [], []
    for step in range(300):
        Xb, Yb = make_block(1, shift=3.0 * step / 300)  # slowly drift the operating point
        x = Xb.iloc[0].to_numpy()
        y_true = float(Yb.iloc[0, 0])
        frozen_err.append(float(frozen.update(x, y_row=np.array([y_true])).prediction[0]) - y_true)
        adaptive_err.append(float(adaptive.update(x, y_row=np.array([y_true])).prediction[0]) - y_true)

    late = slice(200, None)
    assert abs(np.mean(adaptive_err[late])) < abs(np.mean(frozen_err[late]))


# --------------------------------------------------------------------------- #
# Adaptation diagnostics (preprocessing vs kernel channels, state drift)
# --------------------------------------------------------------------------- #


def test_prediction_channels_sum_to_adaptive(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Check static + preprocessing + kernel == adaptive == the streamed prediction."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(
        n_components=3,
        forgetting_factor=0.05,
        gamma=0.1,
        lambda_center=0.03,
        alpha_scale=0.02,
        lambda_center_y=0.05,
        update_when_out_of_control=True,
    ).fit(X.iloc[:90], Y.iloc[:90])
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    ch = model.prediction_channels_
    np.testing.assert_allclose(ch["adaptive"], ch["static"] + ch["preprocessing"] + ch["kernel"], atol=1e-9)
    np.testing.assert_allclose(ch["adaptive"].to_numpy(), model.predictions_.to_numpy().ravel(), atol=1e-9)


def test_decompose_prediction_matches_predict(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The snapshot decomposition's adaptive column equals predict() on the same rows."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3, forgetting_factor=0.05, gamma=0.1, update_when_out_of_control=True).fit(
        X.iloc[:90], Y.iloc[:90]
    )
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    probe = X.iloc[:5]
    decomposed = model.decompose_prediction(probe)
    np.testing.assert_allclose(decomposed["adaptive"].to_numpy(), model.predict(probe).to_numpy().ravel(), atol=1e-9)


def test_frozen_model_has_zero_drift_and_channels(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """With all forgetting factors and gamma at zero, nothing drifts and both channels are zero."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(
        n_components=3,
        forgetting_factor=0.0,
        gamma=0.0,
        lambda_center=0.0,
        alpha_scale=0.0,
        lambda_center_y=0.0,
        alpha_scale_y=0.0,
        update_when_out_of_control=True,
    ).fit(X.iloc[:90], Y.iloc[:90])
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    assert float(model.center_shift_.abs().max()) == pytest.approx(0.0, abs=1e-9)
    assert float(model.scale_shift_.abs().max()) == pytest.approx(0.0, abs=1e-9)
    assert float(model.beta_shift_.abs().max()) == pytest.approx(0.0, abs=1e-9)
    ch = model.prediction_channels_
    assert float(ch["preprocessing"].abs().max()) == pytest.approx(0.0, abs=1e-9)
    assert float(ch["kernel"].abs().max()) == pytest.approx(0.0, abs=1e-9)


def test_center_shift_grows_under_drift() -> None:
    """A drifting operating point makes center_shift_ increase from ~0."""
    rng = np.random.default_rng(3)
    n, k = 150, 5
    base = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"v{i}" for i in range(k)])
    model = AdaptivePCA(
        n_components=2,
        forgetting_factor=0.05,
        gamma=0.1,
        lambda_center=0.05,
        alpha_scale=0.02,
        update_when_out_of_control=True,
    ).fit(base)
    for step in range(200):
        row = rng.standard_normal(k) + 4.0 * step / 200  # slide the operating point
        model.update(row)
    cs = model.center_shift_
    assert float(cs.iloc[0]) < 0.5
    assert float(cs.iloc[-1]) > float(cs.iloc[0]) + 1.0


def test_injection_and_kernel_norm_zero_when_not_updating(synthetic_pca_data: pd.DataFrame) -> None:
    """Observations that do not update the model record zero injection / kernel-update norm."""
    model = AdaptivePCA(n_components=3, forgetting_factor=0.05, gamma=0.1, update_when_out_of_control=False).fit(
        synthetic_pca_data
    )
    # A wildly out-of-control row: not learned from, so its diagnostics are zero.
    model.update(synthetic_pca_data.iloc[0].to_numpy() + 50.0)
    assert float(model.injection_ratio_.iloc[-1]) == 0.0
    assert float(model.kernel_update_norm_.iloc[-1]) == 0.0


def test_adaptation_plot_builds(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], synthetic_pca_data: pd.DataFrame
) -> None:
    """adaptation_plot returns a Plotly figure with the channels panel only for PLS."""
    X, Y = synthetic_pls_data
    pls = AdaptivePLS(n_components=3, forgetting_factor=0.05, update_when_out_of_control=True).fit(
        X.iloc[:90], Y.iloc[:90]
    )
    pls.partial_fit(X.iloc[90:], Y.iloc[90:])
    fig = pls.adaptation_plot()
    assert len(fig.data) == 7  # 3 channel + 2 preprocessing + 2 kernel traces

    pca = AdaptivePCA(n_components=2, forgetting_factor=0.05, update_when_out_of_control=True).fit(synthetic_pca_data)
    pca.partial_fit(synthetic_pca_data)
    fig_pca = pca.adaptation_plot()
    assert len(fig_pca.data) == 3  # 2 preprocessing + 1 kernel trace (no channels)


def test_adaptive_models_expose_hotellings_t2_limit(
    synthetic_pca_data: pd.DataFrame, synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """The inherited limit method needs ``n_components_``; both adaptive fits must set it.

    Regression test for #542: ``hotellings_t2_limit`` raised ``AttributeError`` on a
    fitted ``AdaptivePLS`` because ``fit`` stored the count only in ``n_components``.
    The value must also agree with the limit the ``update`` Bunch reports.
    """
    pca = AdaptivePCA(n_components=2).fit(synthetic_pca_data)
    assert pca.n_components_ == 2
    limit_pca = float(pca.hotellings_t2_limit(conf_level=0.99))
    assert np.isfinite(limit_pca)
    assert limit_pca > 0

    X, Y = synthetic_pls_data
    pls = AdaptivePLS(n_components=2, forgetting_factor=0, lambda_center=0, alpha_scale=0, conf_level=0.99).fit(X, Y)
    assert pls.n_components_ == 2
    limit_pls = float(pls.hotellings_t2_limit(conf_level=0.99))
    probe = pls.update(X.iloc[0].to_numpy())
    assert limit_pls == pytest.approx(float(probe.hotellings_t2_limit))


# --------------------------------------------------------------------------- #
# Guards shared by both estimators, and the paths beside them
# --------------------------------------------------------------------------- #


def _fit_either(
    cls: type[AdaptivePCA | AdaptivePLS], X: pd.DataFrame, Y: pd.DataFrame, **params: object
) -> AdaptivePCA | AdaptivePLS:
    """Fit either estimator on the same data: AdaptivePCA takes no Y."""
    model = cls(n_components=2, **params)
    return model.fit(X) if cls is AdaptivePCA else model.fit(X, Y)


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"forgetting_factor": 1.5}, r"^forgetting_factor must lie in \[0, 1\]; got 1\.5\.$"),
        ({"gamma": -0.1}, r"^gamma must lie in \[0, 1\]; got -0\.1\.$"),
    ],
    ids=["forgetting-factor-above-1", "gamma-below-0"],
)
def test_both_estimators_refuse_tuning_constants_outside_0_1(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type, params: dict, message: str
) -> None:
    """The forgetting factor and the injection weight are fractions, checked at fit time."""
    X, Y = synthetic_pls_data
    with pytest.raises(ValueError, match=message):
        _fit_either(cls, X, Y, **params)


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_an_update_that_empties_the_kernel_leaves_it_at_zero(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type
) -> None:
    """With a forgetting factor of 1 an observation at the centre empties the kernel.

    The kernel is then left at zero rather than rescaled by its zero norm, so the model
    stays finite.
    """
    X, Y = synthetic_pls_data
    model = _fit_either(cls, X, Y, forgetting_factor=1.0, update_when_out_of_control=True)
    centre = (model.mx_.copy(),) if cls is AdaptivePCA else (model.mx_.copy(), model.my_.copy())
    assert model.update(*centre).updated
    np.testing.assert_array_equal(model.XtX_, np.zeros_like(model.XtX_))
    if cls is AdaptivePLS:
        np.testing.assert_array_equal(model.XtY_, np.zeros_like(model.XtY_))
    assert np.isfinite(model.x_loadings_ if cls is AdaptivePLS else model.loadings_).all(axis=None)


def test_a_fixed_spe_limit_stays_at_the_training_limit(synthetic_pca_data: pd.DataFrame) -> None:
    """adaptive_spe_limit=False keeps the training limit; the default refits it as rows stream in."""
    training, stream = synthetic_pca_data.iloc[:200], synthetic_pca_data.iloc[200:].to_numpy()
    fixed = AdaptivePCA(n_components=3, forgetting_factor=0.05, adaptive_spe_limit=False).fit(training)
    rolling = AdaptivePCA(n_components=3, forgetting_factor=0.05).fit(training)

    fixed_limits = {fixed.update(row).spe_limit for row in stream}
    rolling_limits = {rolling.update(row).spe_limit for row in stream}
    assert len(fixed_limits) == 1
    assert len(rolling_limits) > 1


def test_pls_missing_data_projection_is_finite(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A partially missing observation projects, and predicts, through single-component projection."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3).fit(X, Y)
    row = X.iloc[0].to_numpy().copy()
    row[1] = np.nan
    out = model.update(row)
    assert np.all(np.isfinite(out.scores))
    assert np.all(np.isfinite(out.prediction))
    assert np.isfinite(out.spe)
    assert np.isfinite(out.hotellings_t2)


def test_a_missing_response_is_treated_as_no_response(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A NaN in y_row holds the regression side, exactly as passing no response does."""
    X, Y = synthetic_pls_data
    with_nan = AdaptivePLS(n_components=2, forgetting_factor=0.05).fit(X, Y)
    without = AdaptivePLS(n_components=2, forgetting_factor=0.05).fit(X, Y)
    row = X.iloc[3].to_numpy()
    assert with_nan.update(row, y_row=[np.nan]).updated
    assert without.update(row).updated
    for attribute in ("XtX_", "XtY_", "my_", "sy_"):
        np.testing.assert_array_equal(getattr(with_nan, attribute), getattr(without, attribute))


def test_prediction_channels_for_several_responses(ldpe_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """With several responses the channels are keyed by (target, channel), and still add up per target."""
    X, Y = ldpe_data
    model = AdaptivePLS(n_components=3, forgetting_factor=0.05, update_when_out_of_control=True).fit(
        X.iloc[:40], Y.iloc[:40]
    )
    model.partial_fit(X.iloc[40:], Y.iloc[40:])
    channels = model.prediction_channels_
    assert channels.columns.names == ["target", "channel"]
    assert list(channels.columns.get_level_values("target").unique()) == list(Y.columns)
    for target in Y.columns:
        parts = channels[target]
        np.testing.assert_allclose(
            parts["adaptive"], parts["static"] + parts["preprocessing"] + parts["kernel"], atol=1e-9
        )
        np.testing.assert_allclose(parts["adaptive"].to_numpy(), model.predictions_[target].to_numpy(), atol=1e-9)


def test_adaptation_plot_blanks_the_channels_outside_valid(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Rows marked invalid are left out of the channel traces, and only those rows."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3, forgetting_factor=0.05, update_when_out_of_control=True).fit(
        X.iloc[:90], Y.iloc[:90]
    )
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    valid = np.ones(90, dtype=bool)
    valid[:10] = False
    fig = model.adaptation_plot(valid=valid)
    for trace in fig.data[:3]:  # preprocessing, kernel and their total
        values = np.asarray(trace.y, dtype=float)
        assert np.isnan(values[:10]).all(), trace.name
        assert np.isfinite(values[10:]).all(), trace.name


# --------------------------------------------------------------------------- #
# The moving centre and scale
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_a_missing_value_moves_neither_its_centre_nor_its_scale(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type
) -> None:
    """Every other tag takes one EWMA step, by hand: s = sqrt((1 - alpha) s^2 + alpha (x - m)^2), m += lambda (x - m).

    The missing tag keeps its centre and scale (#1029). They used to become NaN, and stayed
    NaN, so the tag was treated as missing in every later observation.
    """
    X, Y = synthetic_pls_data
    model = _fit_either(cls, X, Y, lambda_center=0.25, alpha_scale=0.5, update_when_out_of_control=True)
    centre, scale = model.mx_.copy(), model.sx_.copy()
    row = X.iloc[0].to_numpy().copy()
    row[1] = np.nan
    assert model.update(row).updated
    expected_scale = np.sqrt(0.5 * scale**2 + 0.5 * (row - centre) ** 2)
    expected_centre = 0.75 * centre + 0.25 * row
    expected_scale[1], expected_centre[1] = scale[1], centre[1]
    np.testing.assert_allclose(model.sx_, expected_scale, rtol=1e-15)
    np.testing.assert_allclose(model.mx_, expected_centre, rtol=1e-15)


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_a_scale_is_set_to_1_only_when_it_is_0(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type
) -> None:
    """With a scale factor of 1 each new scale is the tag's deviation from the centre, in any units.

    A deviation of 0 has no spread to divide by, and its scale is 1, as MCUVScaler scales a
    column of equal values. One of 1e-12 is kept: the scale used to be set to 1 below 1.5e-8,
    which took a tag in small units for a constant one (#1029).
    """
    X, Y = synthetic_pls_data
    params = {"alpha_scale": 1.0} if cls is AdaptivePCA else {"alpha_scale": 1.0, "alpha_scale_y": 1.0}
    model = _fit_either(cls, X, Y, update_when_out_of_control=True, **params)
    centre = model.mx_.copy()
    row = centre + np.r_[0.0, 1e-12, -0.5, np.zeros(X.shape[1] - 3)]
    deviation = np.abs(row - centre)  # 1e-12 and 0.5 to rounding, 0 exactly
    if cls is AdaptivePCA:
        model.update(row)
    else:
        y_centre = model.my_.copy()
        model.update(row, y_row=y_centre + 1e-12)
        np.testing.assert_allclose(model.sy_, np.abs(y_centre + 1e-12 - y_centre), rtol=1e-15)
    np.testing.assert_allclose(model.sx_, np.where(deviation > 0, deviation, 1.0), rtol=1e-15)
    assert model.sx_[0] == 1.0


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_a_tag_in_small_units_adapts_as_it_does_in_large_ones(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type
) -> None:
    """Scaled by its moving sd, a tag 1e9 times smaller streams to the same scores and statistics.

    Its sd, near 1e-9, was below the 1.5e-8 that was taken for a scale collapsed to zero, so
    after the first update it was divided by 1 and dropped out of the model (#1029); so was
    the response of AdaptivePLS.
    """
    X, Y = synthetic_pls_data
    small_x, small_y = X.assign(x0=X["x0"] * 1e-9), Y * 1e-9
    params = {"forgetting_factor": 0.05, "alpha_scale": 0.05, "update_when_out_of_control": True}
    reference = _fit_either(cls, X.iloc[:90], Y.iloc[:90], **params)
    small = _fit_either(cls, small_x.iloc[:90], small_y.iloc[:90], **params)
    reference.partial_fit(X.iloc[90:], None if cls is AdaptivePCA else Y.iloc[90:])
    small.partial_fit(small_x.iloc[90:], None if cls is AdaptivePCA else small_y.iloc[90:])
    units = np.r_[1e-9, np.ones(X.shape[1] - 1)]
    np.testing.assert_allclose(small.sx_, reference.sx_ * units, rtol=1e-9)
    for history in ("scores_", "hotellings_t2_", "spe_"):
        np.testing.assert_allclose(getattr(small, history), getattr(reference, history), rtol=1e-8, atol=1e-10)
    if cls is AdaptivePLS:
        np.testing.assert_allclose(small.sy_, reference.sy_ * 1e-9, rtol=1e-9)
        np.testing.assert_allclose(small.predictions_ * 1e9, reference.predictions_, rtol=1e-8)


# --------------------------------------------------------------------------- #
# By hand: one update, the rolling SPE limit, the kernel helpers
# --------------------------------------------------------------------------- #


def _signed(vectors: np.ndarray) -> np.ndarray:
    """Flip each column whose largest-magnitude entry is negative (Wold, Esbensen and Geladi, 1987)."""
    largest = vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])]
    return vectors * np.sign(largest)


def _pca_by_svd(X: pd.DataFrame, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the autoscaled block (sd with ddof=1), signed loadings, score variances and SPE, from numpy's SVD."""
    scaled = ((X - X.mean()) / X.std()).to_numpy()
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    loadings = _signed(right[:n_components].T)
    residual = scaled - scaled @ loadings @ loadings.T
    return scaled, loadings, singular[:n_components] ** 2 / (len(X) - 1), np.sqrt((residual**2).sum(axis=1))


def _spe_limit_by_hand(spe: np.ndarray, conf_level: float) -> float:
    """Box's (1954) chi-squared limit g * chi2(h), matched to the mean and variance of the squared SPE."""
    squared = spe**2
    mean, variance = squared.mean(), squared.var(ddof=1)
    return float(np.sqrt(variance / (2 * mean) * stats.chi2.ppf(conf_level, 2 * mean**2 / variance)))


def test_kernel_pca_keeps_the_leading_eigenvectors_signed_and_clips_noise_below_zero() -> None:
    """Eigenvectors in descending order of eigenvalue, each with its largest-magnitude entry positive.

    The kernel is built from known eigenvectors: the leading one, (-0.6, 0.8, 0), is signed by its
    0.8, not by its first entry. An eigenvalue below zero, rounding noise in a kernel X'X, is 0,
    and an eigenvalue below 1 is kept as it is.
    """
    vectors = np.array([[-0.6, 0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    kernel = vectors @ np.diag([3.0, 0.5, -1e-16]) @ vectors.T
    loadings, values = _kernel_pca(kernel, n_components=3)
    np.testing.assert_allclose(loadings, vectors, rtol=0, atol=1e-12)
    np.testing.assert_allclose(values, [3.0, 0.5, 0.0], rtol=1e-12)
    assert values[2] == 0.0


def test_one_update_of_adaptive_pca_by_hand(synthetic_pca_data: pd.DataFrame) -> None:
    """One in-control observation, then one out of control, from the method's equations and numpy.

    The first is monitored with the training model, then moves the centre and scale (equations 1a
    and 2a), and the kernel by (1 - mu) X'X + mu x x' + f X'X_0 with f = gamma mu ||x||^2 / trace(X'X_0),
    rescaled to the training trace; x is scaled with the moved vectors. The loadings are the
    kernel's eigenvectors, signed to agree with the ones before. The second is not learned from:
    the model and its drift stay put, and its injection and kernel change are 0.
    """
    X = synthetic_pca_data
    mu, gamma, lam, alpha = 0.1, 0.3, 0.2, 0.4
    model = AdaptivePCA(n_components=3, forgetting_factor=mu, gamma=gamma, lambda_center=lam, alpha_scale=alpha)
    model.fit(X)
    scaled, loadings, variances, _ = _pca_by_svd(X, 3)
    centre, scale = X.mean().to_numpy(), X.std().to_numpy()
    kernel = scaled.T @ scaled
    x = X.iloc[5].to_numpy()

    out = model.update(x, label="first")
    scores = (x - centre) / scale @ loadings
    np.testing.assert_allclose(out.scores, scores, rtol=1e-9)
    assert out.hotellings_t2 == pytest.approx(float(np.sum(scores**2 / variances)), rel=1e-9)
    assert out.spe == pytest.approx(float(np.linalg.norm((x - centre) / scale - loadings @ scores)), rel=1e-9)
    assert out.in_control
    assert out.updated

    new_scale = np.sqrt((1 - alpha) * scale**2 + alpha * (x - centre) ** 2)
    new_centre = centre + lam * (x - centre)
    np.testing.assert_allclose(model.sx_, new_scale, rtol=1e-14)
    np.testing.assert_allclose(model.mx_, new_centre, rtol=1e-14)
    x_new = (x - new_centre) / new_scale
    injection = gamma * mu * float(x_new @ x_new) / np.trace(kernel)
    moved = (1 - mu) * kernel + mu * np.outer(x_new, x_new) + injection * kernel
    moved *= np.trace(kernel) / np.trace(moved)
    np.testing.assert_allclose(model.XtX_, moved, rtol=1e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(moved)
    new_loadings = eigenvectors[:, ::-1][:, :3]
    new_loadings *= np.sign(np.sum(new_loadings * loadings, axis=0))
    np.testing.assert_allclose(model.loadings_.to_numpy(), new_loadings, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.explained_variance_, eigenvalues[::-1][:3] / (len(X) - 1), rtol=1e-12)
    np.testing.assert_allclose(model.scaling_factor_for_scores_, np.sqrt(eigenvalues[::-1][:3] / (len(X) - 1)))
    overlap = float(np.sum((loadings.T @ new_loadings) ** 2))

    wild = model.update(x + 50.0)
    assert not wild.in_control
    assert not wild.updated
    np.testing.assert_allclose(model.XtX_, moved, rtol=1e-12)
    assert list(model.distance_.index) == ["first", 1]
    np.testing.assert_allclose(model.distance_, [overlap, overlap], rtol=1e-12)
    np.testing.assert_allclose(model.injection_ratio_, [injection, 0.0], rtol=1e-12)
    change = np.linalg.norm(moved - kernel) / np.linalg.norm(kernel)
    np.testing.assert_allclose(model.kernel_update_norm_, [change, 0.0], rtol=1e-9)
    drift = np.linalg.norm((new_centre - centre) / scale)
    np.testing.assert_allclose(model.center_shift_, [drift, drift], rtol=1e-12)
    spread = np.mean(np.abs(new_scale / scale - 1))
    np.testing.assert_allclose(model.scale_shift_, [spread, spread], rtol=1e-12)


def test_the_rolling_spe_limit_by_hand(synthetic_pca_data: pd.DataFrame) -> None:
    """The SPE limit is Box's, fitted to the last spe_limit_window squared SPEs once there are 20.

    Nineteen training rows leave the window one short of max(20, A + 2): the first update is held
    to the training limit, which comes from the same 19. Its in-control SPE makes 20, and the
    second update's limit is fitted to all 20. With a window of 25 over 30 training rows, only the
    last 25 count.
    """
    train = synthetic_pca_data.iloc[:19]
    _, _, _, spe = _pca_by_svd(train, 2)
    model = AdaptivePCA(n_components=2, conf_level=0.9).fit(train)
    first = model.update(train.iloc[0].to_numpy())
    assert first.in_control
    assert first.spe == pytest.approx(spe[0], rel=1e-9)
    assert first.spe_limit == pytest.approx(_spe_limit_by_hand(spe, 0.9), rel=1e-9)
    second = model.update(train.iloc[1].to_numpy())
    assert second.spe_limit == pytest.approx(_spe_limit_by_hand(np.r_[spe, spe[0]], 0.9), rel=1e-9)

    train = synthetic_pca_data.iloc[:30]
    _, _, _, spe = _pca_by_svd(train, 2)
    model = AdaptivePCA(n_components=2, spe_limit_window=25).fit(train)
    assert model.update(train.iloc[0].to_numpy()).spe_limit == pytest.approx(
        _spe_limit_by_hand(spe[-25:], 0.95), rel=1e-9
    )


# --------------------------------------------------------------------------- #
# Streaming and the guards on it
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_partial_fit_is_update_row_by_row(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type) -> None:
    """A labelled frame keeps its labels, an array is numbered from 0; the response rows go with them."""
    X, Y = synthetic_pls_data
    params = {"forgetting_factor": 0.05, "update_when_out_of_control": True}
    by_rows, by_frame, by_array = (_fit_either(cls, X.iloc[:90], Y.iloc[:90], **params) for _ in range(3))
    stream_x = X.iloc[90:100].set_axis([f"r{i}" for i in range(10)])
    stream_y = Y.iloc[90:100] if cls is AdaptivePLS else None
    for pos, label in enumerate(stream_x.index):
        y_row = None if stream_y is None else stream_y.to_numpy()[pos]
        by_rows.update(stream_x.to_numpy()[pos], y_row=y_row, label=label)
    by_frame.partial_fit(stream_x, stream_y)
    by_array.partial_fit(stream_x.to_numpy(), None if stream_y is None else stream_y.to_numpy())
    assert list(by_frame.spe_.index) == list(stream_x.index)
    assert list(by_array.spe_.index) == list(range(10))
    for model in (by_frame, by_array):
        np.testing.assert_array_equal(model.spe_.to_numpy(), by_rows.spe_.to_numpy())
        np.testing.assert_array_equal(model.XtX_, by_rows.XtX_)
        if cls is AdaptivePLS:
            np.testing.assert_array_equal(model.XtY_, by_rows.XtY_)
    if cls is AdaptivePLS:
        assert not np.array_equal(by_rows.XtY_, by_rows.XtY0_)


@pytest.mark.parametrize("cls", [AdaptivePCA, AdaptivePLS])
def test_a_model_whose_fit_was_refused_is_not_fitted(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type
) -> None:
    """A refused fit sets a few attributes before its check, but leaves the model unfitted for every method."""
    X, Y = synthetic_pls_data
    model = cls(n_components=2, gamma=2.0)
    with pytest.raises(ValueError, match=r"^gamma must lie in \[0, 1\]; got 2\.0\.$"):
        model.fit(X) if cls is AdaptivePCA else model.fit(X, Y)
    assert model.n_components_ == 2
    calls = [lambda: model.partial_fit(X.iloc[:0]), lambda: model.update(X.iloc[0]), lambda: model.transform(X)]
    if cls is AdaptivePLS:
        calls += [lambda: model.predict(X), lambda: model.decompose_prediction(X)]
    for call in calls:
        with pytest.raises(NotFittedError):
            call()


@pytest.mark.parametrize(
    "value", [-0.01, 1.01, np.nan, [0.5, 0.5, -0.1, 0.5, 0.5, 0.5]], ids=["below-0", "above-1", "nan", "one-of-six"]
)
@pytest.mark.parametrize(
    ("cls", "name"),
    [
        (AdaptivePCA, "lambda_center"),
        (AdaptivePCA, "alpha_scale"),
        (AdaptivePLS, "lambda_center"),
        (AdaptivePLS, "alpha_scale"),
        (AdaptivePLS, "lambda_center_y"),
        (AdaptivePLS, "alpha_scale_y"),
    ],
)
def test_an_ewma_factor_outside_0_1_is_refused(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame], cls: type, name: str, value: object
) -> None:
    """Each factor, or each entry of a vector of them, is a fraction; a NaN is refused too (#1029).

    A NaN factor used to pass, and made every centre it moved NaN.
    """
    X, Y = synthetic_pls_data
    if name.endswith("_y") and isinstance(value, list):
        value = [-0.1]  # one per response, and there is one
    with pytest.raises(ValueError, match=rf"^{name} must lie in \[0, 1\]; got values outside that range\.$"):
        _fit_either(cls, X, Y, **{name: value})


def test_ewma_factors_of_0_and_1_are_kept_as_one_float_per_variable(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The bounds are allowed; a scalar is repeated for every variable, as a float."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=2, lambda_center=0, alpha_scale=1, lambda_center_y=[1], alpha_scale_y=0)
    model.fit(X, Y)
    for name, expected in [
        ("lambda_center_", np.zeros(6)),
        ("alpha_scale_", np.ones(6)),
        ("lambda_center_y_", np.ones(1)),
        ("alpha_scale_y_", np.zeros(1)),
    ]:
        assert getattr(model, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(model, name), expected)
