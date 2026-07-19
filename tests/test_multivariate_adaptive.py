# (c) Kevin Dunn, 2010-2026. MIT License.
"""Tests for the adaptive (recursive) PCA and PLS estimators.

Covers, for both estimators:

* seed correctness - the kernel-seeded model reproduces the batch PCA loadings
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
from sklearn.base import clone

from process_improve.multivariate import PCA, PLS, AdaptivePCA, AdaptivePLS, MCUVScaler
from process_improve.multivariate._adaptive import (
    _kernel_pca,
    _kernel_pls,
    _subspace_distance,
)

DATASETS = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_pca_data() -> pd.DataFrame:
    """Return a correlated, full-rank block for PCA seeding and streaming tests."""
    rng = np.random.default_rng(42)
    n, k, a = 250, 8, 3
    loadings = rng.standard_normal((k, a))
    scores = rng.standard_normal((n, a))
    values = scores @ loadings.T + 0.1 * rng.standard_normal((n, k))
    return pd.DataFrame(values, columns=[f"v{i}" for i in range(k)])


@pytest.fixture
def synthetic_pls_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a synthetic single-response regression block for PLS seeding tests."""
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


# --------------------------------------------------------------------------- #
# Seed correctness
# --------------------------------------------------------------------------- #


def test_adaptive_pca_seed_matches_batch(synthetic_pca_data: pd.DataFrame) -> None:
    """AdaptivePCA seeds an i=0 model identical to batch PCA on the scaled data."""
    model = AdaptivePCA(n_components=3).fit(synthetic_pca_data)
    batch = PCA(n_components=3).fit(MCUVScaler().fit_transform(synthetic_pca_data))
    np.testing.assert_allclose(np.abs(model.loadings0_), np.abs(batch.loadings_.to_numpy()), atol=1e-8)
    np.testing.assert_allclose(model.explained_variance_, batch.explained_variance_, rtol=1e-8)


def test_adaptive_pls_seed_matches_batch_ldpe(ldpe_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """AdaptivePLS seeds beta identical to the batch PLS on the real LDPE data."""
    X, Y = ldpe_data
    model = AdaptivePLS(n_components=6).fit(X, Y)
    batch = PLS(n_components=6, scale=True).fit(X, Y)
    np.testing.assert_allclose(
        model.beta_coefficients_.to_numpy(), batch.beta_coefficients_.to_numpy(), atol=1e-7
    )
    # Predictions with the freshly-seeded model equal the batch predictions.
    np.testing.assert_allclose(
        model.predict(X).to_numpy(), batch.predictions_.to_numpy(), atol=1e-6
    )


def test_adaptive_pls_seed_matches_batch_synthetic(
    synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """AdaptivePLS seeds beta to machine precision on well-conditioned synthetic data."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(n_components=3).fit(X, Y)
    batch = PLS(n_components=3, scale=True).fit(X, Y)
    np.testing.assert_allclose(
        model.beta_coefficients_.to_numpy(), batch.beta_coefficients_.to_numpy(), atol=1e-12
    )


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
    """The norm-rescaling holds the nuclear norm (trace) of X'X at its seed value over many updates."""
    model = AdaptivePCA(
        n_components=3, forgetting_factor=0.1, gamma=0.1, update_when_out_of_control=True
    ).fit(synthetic_pca_data)
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
    so it raises substantially fewer alarms than a model frozen at the seed.
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
    xty_seed = model.XtY_.copy()
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
    assert not np.allclose(model.XtY_, xty_seed)
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


def test_transform_and_predict_match_seed(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Predict and transform on the freshly-seeded model equal the batch model."""
    X, Y = synthetic_pls_data
    ad = AdaptivePLS(n_components=3).fit(X, Y)
    batch = PLS(n_components=3, scale=True).fit(X, Y)
    np.testing.assert_allclose(ad.predict(X).to_numpy(), batch.predictions_.to_numpy(), atol=1e-9)
    # Scores line up with the batch X-scores up to a per-component sign.
    np.testing.assert_allclose(
        np.abs(ad.transform(X).to_numpy()), np.abs(batch.scores_.to_numpy()), atol=1e-7
    )


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
        n_components=3, forgetting_factor=0.05, gamma=0.1, lambda_center=0.03,
        alpha_scale=0.02, lambda_center_y=0.05, update_when_out_of_control=True,
    ).fit(X.iloc[:90], Y.iloc[:90])
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    ch = model.prediction_channels_
    np.testing.assert_allclose(ch["adaptive"], ch["static"] + ch["preprocessing"] + ch["kernel"], atol=1e-9)
    np.testing.assert_allclose(ch["adaptive"].to_numpy(), model.predictions_.to_numpy().ravel(), atol=1e-9)


def test_decompose_prediction_matches_predict(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The snapshot decomposition's adaptive column equals predict() on the same rows."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(
        n_components=3, forgetting_factor=0.05, gamma=0.1, update_when_out_of_control=True
    ).fit(X.iloc[:90], Y.iloc[:90])
    model.partial_fit(X.iloc[90:], Y.iloc[90:])
    probe = X.iloc[:5]
    decomposed = model.decompose_prediction(probe)
    np.testing.assert_allclose(decomposed["adaptive"].to_numpy(), model.predict(probe).to_numpy().ravel(), atol=1e-9)


def test_frozen_model_has_zero_drift_and_channels(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """With all forgetting factors and gamma at zero, nothing drifts and both channels are zero."""
    X, Y = synthetic_pls_data
    model = AdaptivePLS(
        n_components=3, forgetting_factor=0.0, gamma=0.0, lambda_center=0.0, alpha_scale=0.0,
        lambda_center_y=0.0, alpha_scale_y=0.0, update_when_out_of_control=True,
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
        n_components=2, forgetting_factor=0.05, gamma=0.1, lambda_center=0.05,
        alpha_scale=0.02, update_when_out_of_control=True,
    ).fit(base)
    for step in range(200):
        row = rng.standard_normal(k) + 4.0 * step / 200  # slide the operating point
        model.update(row)
    cs = model.center_shift_
    assert float(cs.iloc[0]) < 0.5
    assert float(cs.iloc[-1]) > float(cs.iloc[0]) + 1.0


def test_injection_and_kernel_norm_zero_when_not_updating(synthetic_pca_data: pd.DataFrame) -> None:
    """Observations that do not update the model record zero injection / kernel-update norm."""
    model = AdaptivePCA(
        n_components=3, forgetting_factor=0.05, gamma=0.1, update_when_out_of_control=False
    ).fit(synthetic_pca_data)
    # A wildly out-of-control row: not learned from, so its diagnostics are zero.
    model.update(synthetic_pca_data.iloc[0].to_numpy() + 50.0)
    assert float(model.injection_ratio_.iloc[-1]) == 0.0
    assert float(model.kernel_update_norm_.iloc[-1]) == 0.0


def test_adaptation_plot_builds(synthetic_pls_data: tuple[pd.DataFrame, pd.DataFrame],
                                synthetic_pca_data: pd.DataFrame) -> None:
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
