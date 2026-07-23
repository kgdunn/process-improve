# (c) Kevin Dunn, 2010-2026. MIT License.
"""Tests for the O-PLS estimator and the equivalence with PLS.

The central results checked here come from García-Carrión et al., "On the
equivalence between null space and orthogonal space in latent variable
regression modeling", Journal of Chemometrics, 39 (2025): e70057. An
A-component PLS model and an O-PLS(1; A - 1) model give identical predictions,
and the O-PLS orthogonal space is the same linear space as the PLS null space.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import OPLS, PLS, MCUVScaler


@pytest.fixture
def synthetic_single_response() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a correlated-X, single-response dataset with a known coefficient vector."""
    rng = np.random.default_rng(7)
    n_samples, n_features = 60, 6
    base = rng.standard_normal((n_samples, 2))
    loadings = rng.standard_normal((2, n_features - 2))
    tail = base @ loadings + 0.1 * rng.standard_normal((n_samples, n_features - 2))
    X = pd.DataFrame(np.hstack([base, tail]), columns=[f"x{i}" for i in range(n_features)])
    beta = rng.standard_normal((n_features, 1))
    y = pd.DataFrame(X.to_numpy() @ beta + 0.2 * rng.standard_normal((n_samples, 1)), columns=["y"])
    return X, y


@pytest.fixture
def cheese() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cheddar-cheese: 30 samples, 3 predictors, single response (Taste)."""
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    path = folder / "cheddar-cheese.csv"
    if not path.exists():
        pytest.skip("cheddar-cheese.csv fixture not present")
    data = pd.read_csv(path)
    return data[["Acetic", "H2S", "Lactic"]], data[["Taste"]]


@pytest.mark.parametrize("n_total", [2, 3, 4])
def test_opls_predictions_match_pls(synthetic_single_response: tuple[pd.DataFrame, pd.DataFrame], n_total: int) -> None:
    """O-PLS(1; A-1) reproduces the predictions of an A-component PLS model."""
    X, y = synthetic_single_response
    pls = PLS(n_components=n_total).fit(X, y)
    opls = OPLS(n_orthogonal_components=n_total - 1).fit(X, y)
    assert opls.predict(X).to_numpy() == pytest.approx(pls.predict(X).to_numpy(), abs=1e-8)
    assert opls.beta_coefficients_.to_numpy() == pytest.approx(pls.beta_coefficients_.to_numpy(), abs=1e-8)


def test_opls_orthogonal_scores_are_y_orthogonal(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Every orthogonal score is uncorrelated with the (centered) response."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    y_centered = y.to_numpy() - y.to_numpy().mean()
    projection = opls.orthogonal_scores_.to_numpy().T @ y_centered
    assert np.abs(projection).max() == pytest.approx(0.0, abs=1e-8)


def test_opls_predict_round_trips_through_invert(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """invert() then predict() recovers the desired response (single division)."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    for target in (15.0, 25.0, 40.0):
        result = opls.invert(target)
        assert result.y_hat == pytest.approx(target, abs=1e-6)
        y_back = float(opls.predict(result.x_new.to_frame().T).iloc[0, 0])
        assert y_back == pytest.approx(target, abs=1e-6)
        assert result.orthogonal_space_dimension == 1


def test_orthogonal_space_equals_null_space(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The O-PLS orthogonal space and the PLS null space span the same subspace.

    This is the central result of García-Carrión et al. (2025): reconstruct
    both bases into the input space and confirm they point the same way.
    """
    X, y = cheese
    pls = PLS(n_components=2).fit(X, y)
    opls = OPLS(n_orthogonal_components=1).fit(X, y)

    target = 25.0
    ns_scores = pls.invert(target).null_space_basis.to_numpy()  # (A, A-1) in score space
    ns_input = ns_scores.T @ pls.x_loadings_.to_numpy().T  # (A-1, K) reconstructed to input space
    os_input = opls.invert(target).orthogonal_space_basis.to_numpy().T  # (Ao, K)

    # 1-D subspaces here: compare directions via absolute cosine.
    ns_dir = ns_input.ravel()
    os_dir = os_input.ravel()
    cosine = np.abs(ns_dir @ os_dir) / (np.linalg.norm(ns_dir) * np.linalg.norm(os_dir))
    assert cosine == pytest.approx(1.0, abs=1e-8)


def test_opls_correct_removes_orthogonal_variation(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The corrected X has no remaining projection on the orthogonal weights."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    x_filtered = opls.correct(X).to_numpy()
    # No systematic variation left along the orthogonal weight directions.
    residual_on_ortho = x_filtered @ opls.orthogonal_weights_.to_numpy()
    assert np.abs(residual_on_ortho).max() == pytest.approx(0.0, abs=1e-8)


def test_opls_scores_shape_and_labels(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """scores_ is [predictive | orthogonal] with informative labels."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=2).fit(X, y)
    assert list(opls.scores_.columns) == ["t_predictive", "t_orthogonal_1", "t_orthogonal_2"]
    assert opls.scores_.shape == (X.shape[0], 3)
    assert opls.n_components == 3


def test_opls_transform_matches_predictive_scores(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """transform() on the training data reproduces the fitted predictive scores."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    assert opls.transform(X).to_numpy() == pytest.approx(opls.predictive_scores_.to_numpy(), abs=1e-9)


def test_opls_rejects_multi_response() -> None:
    """A two-column response is rejected with a clear message."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((20, 4)), columns=[f"x{i}" for i in range(4)])
    Y = pd.DataFrame(rng.standard_normal((20, 2)), columns=["y0", "y1"])
    with pytest.raises(ValueError, match="single response"):
        OPLS(n_orthogonal_components=1).fit(X, Y)


def test_opls_scale_false_matches_prescaled(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """scale=False on pre-scaled data equals scale=True on raw data (predictions)."""
    X, y = cheese
    opls_scaled = OPLS(n_orthogonal_components=1).fit(X, y)
    Xs = MCUVScaler().fit_transform(X)
    ys = MCUVScaler().fit_transform(y)
    opls_prescaled = OPLS(n_orthogonal_components=1, scale=False).fit(Xs, ys)
    # Predictions on the scaled model are on the scaled-y scale; compare the
    # standardized predictive scores instead, which are scale-free.
    assert opls_prescaled.predictive_scores_.to_numpy() == pytest.approx(
        opls_scaled.predictive_scores_.to_numpy(), abs=1e-8
    )


def test_opls_accepts_1d_and_array_inputs(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A 1-D y (Series and ndarray) and a plain-array X are accepted."""
    X, y = cheese
    y_series = y.iloc[:, 0]
    from_frame = OPLS(n_orthogonal_components=1).fit(X, y).predict(X).to_numpy()
    from_series = OPLS(n_orthogonal_components=1).fit(X, y_series).predict(X).to_numpy()
    from_array = OPLS(n_orthogonal_components=1).fit(X.to_numpy(), y.to_numpy().ravel()).predict(X.to_numpy())
    assert from_series == pytest.approx(from_frame, abs=1e-8)
    assert from_array.to_numpy() == pytest.approx(from_frame, abs=1e-8)


def test_opls_feature_names_out(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """get_feature_names_out returns the component labels."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    assert list(opls.get_feature_names_out()) == ["t_predictive", "t_orthogonal_1"]


def test_opls_negative_components_raises(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """A negative orthogonal-component count is rejected."""
    X, y = cheese
    with pytest.raises(ValueError, match=">= 0"):
        OPLS(n_orthogonal_components=-1).fit(X, y)


def test_opls_zero_orthogonal_components(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """O-PLS(1; 0) is a one-component model whose invert has no orthogonal space."""
    X, y = cheese
    pls = PLS(n_components=1).fit(X, y)
    opls = OPLS(n_orthogonal_components=0).fit(X, y)
    assert opls.predict(X).to_numpy() == pytest.approx(pls.predict(X).to_numpy(), abs=1e-8)
    assert opls.invert(20.0).orthogonal_space_dimension == 0


def test_opls_invert_input_forms_and_errors(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Invert accepts scalar/Series/DataFrame and rejects bad targets."""
    X, y = cheese
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    base = opls.invert(25.0).x_new.to_numpy()
    assert opls.invert(pd.Series([25.0])).x_new.to_numpy() == pytest.approx(base)
    assert opls.invert(pd.DataFrame({"y": [25.0]})).x_new.to_numpy() == pytest.approx(base)
    with pytest.raises(ValueError, match="single desired response"):
        opls.invert(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="finite"):
        opls.invert(np.inf)


def test_opls_reproduces_paper_case_study_1() -> None:
    """On the paper's Case Study 1, O-PLS(1;1) and PLS(2) predictions agree."""
    X = pd.DataFrame(
        [
            [5.43, 7.54, 125.64, 58.51, 50.49],
            [5.43, 15.97, 126.20, 258.48, 74.44],
            [99.23, 7.54, 9893.38, 59.29, 737.15],
            [99.23, 15.97, 9765.16, 254.11, 1576.28],
            [52.33, 11.76, 2787.64, 139.21, 538.76],
            [52.33, 11.76, 2849.95, 135.67, 630.73],
        ],
        columns=["x1", "x2", "x3", "x4", "x5"],
    )
    y = pd.DataFrame({"y": [61.85, 278.99, 307.89, 436.4, 266.08, 260.52]})
    pls = PLS(n_components=2).fit(X, y)
    opls = OPLS(n_orthogonal_components=1).fit(X, y)
    assert opls.predict(X).to_numpy() == pytest.approx(pls.predict(X).to_numpy(), abs=1e-6)
    # Both inversions reach the same design for a desired response.
    yy = 168.23
    pls_y_hat = float(pls.predict(pls.invert(yy).x_new.to_frame().T).iloc[0, 0])
    assert opls.invert(yy).y_hat == pytest.approx(pls_y_hat, abs=1e-6)
