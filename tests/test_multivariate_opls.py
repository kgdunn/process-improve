# (c) Kevin Dunn, 2010-2026. MIT License.
"""Tests for the O-PLS estimator and the equivalence with PLS.

The central results checked here come from García-Carrión et al., "On the
equivalence between null space and orthogonal space in latent variable
regression modeling", Journal of Chemometrics, 39 (2025): e70057. An
A-component PLS model and an O-PLS(1; A - 1) model give identical predictions,
and the O-PLS orthogonal space is the same linear space as the PLS null space.
"""

from __future__ import annotations

import functools
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._common import NotEnoughVarianceError, SpecificationWarning
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
    assert opls.n_components_ == 3


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


def test_opls_scale_false_end_to_end_and_2d_array_y(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Exercise the unscaled code paths: predict/transform/invert and a 2-D array Y."""
    X, y = cheese
    Xs = MCUVScaler().fit_transform(X)
    ys = MCUVScaler().fit_transform(y)
    # Y supplied as a 2-D ndarray (neither DataFrame nor 1-D) exercises that branch.
    opls = OPLS(n_orthogonal_components=1, scale=False).fit(Xs, ys.to_numpy())

    # predict / transform on the unscaled model.
    assert opls.predict(Xs).shape == (X.shape[0], 1)
    assert opls.transform(Xs).to_numpy() == pytest.approx(opls.predictive_scores_.to_numpy(), abs=1e-9)

    # invert on the unscaled model round-trips on the scaled-y scale.
    target_scaled = 0.5
    result = opls.invert(target_scaled)
    assert result.y_hat == pytest.approx(target_scaled, abs=1e-6)
    y_back = float(opls.predict(result.x_new.to_frame().T).iloc[0, 0])
    assert y_back == pytest.approx(target_scaled, abs=1e-6)


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


@pytest.mark.parametrize(("name", "value"), [("tol", 1e-3), ("max_iter", 7)])
def test_the_removed_inert_parameters_are_refused(name: str, value: float) -> None:
    """``max_iter`` and ``tol`` never reached the closed-form algorithm and were removed in 2.0 (#588)."""
    assert name not in OPLS(n_orthogonal_components=2).get_params()
    # The removed keyword goes in through a dict, so a static analyser does not flag the call.
    with pytest.raises(TypeError, match=name):
        OPLS(n_orthogonal_components=2, **{name: value})


@pytest.fixture
def labelled_cheese(cheese: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cheese data with labelled rows, so every output's labels can be checked."""
    X, y = cheese
    rows = [f"cheese{i}" for i in range(len(X))]
    return X.set_axis(rows), y.set_axis(rows)


class TestComponentCount:
    """O-PLS(1; Ao) has 1 + Ao components, which the data must be able to carry, as for PCA and PLS."""

    def test_more_components_than_columns_are_capped_with_plss_warning(
        self, cheese: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Three columns carry three components: O-PLS(1; 3) becomes O-PLS(1; 2), which is PLS with three.

        Before the cap, the fourth component took the last of X's variance from the predictive one,
        and the predictions were 27 units off.
        """
        X, y = cheese
        with pytest.warns(SpecificationWarning) as record:
            opls = OPLS(n_orthogonal_components=3).fit(X, y)
        assert str(record[0].message) == (
            "The requested number of components is more than can be computed from data. The maximum number of "
            "components is the minimum of either the number of rows (30) or the number of columns (3)."
        )
        assert pathlib.Path(record[0].filename).name != "_opls.py"
        assert opls.n_components_ == 3
        assert list(opls.orthogonal_scores_.columns) == ["o1", "o2"]
        np.testing.assert_allclose(opls.predict(X), PLS(n_components=3).fit(X, y).predict(X), atol=1e-8)
        assert opls.invert(25.0).orthogonal_space_dimension == 2

    def test_as_many_components_as_columns_need_no_warning(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert OPLS(n_orthogonal_components=2).fit(X, y).n_components_ == 3

    def test_more_components_than_the_centred_rows_carry_are_refused(self) -> None:
        """Five centred rows have rank 4, so a fifth component finds no variance left, as PLS's does."""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(5, 8)))
        y = pd.DataFrame(X.to_numpy() @ rng.normal(size=8))
        message = (
            r"^There is no variance left in the data array for X: cannot compute any more components beyond "
            r"component 4\.$"
        )
        with pytest.raises(NotEnoughVarianceError, match=message):
            PLS(n_components=5).fit(X, y)
        with pytest.raises(NotEnoughVarianceError, match=message):
            OPLS(n_orthogonal_components=4).fit(X, y)
        np.testing.assert_allclose(
            OPLS(n_orthogonal_components=3).fit(X, y).predict(X), PLS(n_components=4).fit(X, y).predict(X), atol=1e-8
        )

    def test_data_in_small_units_keep_their_components(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """No variance left is judged against X's own size, so a millionth of the units fits the same model."""
        X, y = cheese
        Xs, ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(y)
        unit = OPLS(n_orthogonal_components=2, scale=False).fit(Xs, ys)
        tiny = OPLS(n_orthogonal_components=2, scale=False).fit(Xs * 1e-6, ys * 1e-6)
        np.testing.assert_allclose(tiny.scores_, unit.scores_ * 1e-6, rtol=1e-8)

    def test_an_x_with_no_variance_has_no_component(self) -> None:
        X = pd.DataFrame(np.zeros((4, 2)))
        with pytest.raises(NotEnoughVarianceError, match=r"beyond component 0\.$"):
            OPLS(n_orthogonal_components=1, scale=False).fit(X, pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]}))

    def test_two_samples_are_enough(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """With two rows the variance of a score divides its sum of squares by 2 - 1 = 1."""
        X, y = cheese
        opls = OPLS(n_orthogonal_components=0).fit(X.iloc[:2], y.iloc[:2])
        scores = opls.predictive_scores_.to_numpy().ravel()
        np.testing.assert_allclose(opls.explained_variance_, [scores @ scores], rtol=1e-12)
        for scale in (True, False):
            with pytest.raises(ValueError, match="a minimum of 2 is required"):
                OPLS(n_orthogonal_components=0, scale=scale).fit(X.iloc[:1], y.iloc[:1])

    def test_an_ill_conditioned_filter_names_its_matrix(
        self, cheese: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P_o' W_o is unit upper triangular, so no data reaches the limit; a limit of 1 shows what the error names."""
        from process_improve import _linalg
        from process_improve.multivariate import _opls

        monkeypatch.setattr(_opls, "safe_inverse", functools.partial(_linalg.safe_inverse, cond_limit=1.0))
        X, y = cheese
        with pytest.raises(np.linalg.LinAlgError, match=r"^\(P_o' @ W_o\) is singular or ill-conditioned"):
            OPLS(n_orthogonal_components=2).fit(X, y)


class TestFittedValuesByHand:
    """Each fitted attribute from its definition, on the scaled cheese data, or from PLS of the same size."""

    def test_the_weights_scores_and_loadings(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        opls = OPLS(n_orthogonal_components=2).fit(X, y)
        Xs = MCUVScaler().fit_transform(X).to_numpy()
        ys = MCUVScaler().fit_transform(y).to_numpy().ravel()

        w_p = Xs.T @ ys / np.linalg.norm(Xs.T @ ys)
        np.testing.assert_allclose(opls.predictive_weights_, w_p, rtol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(opls.orthogonal_weights_, axis=0), [1.0, 1.0], rtol=1e-12)

        filtered = Xs - opls.orthogonal_scores_.to_numpy() @ opls.orthogonal_loadings_.to_numpy().T
        np.testing.assert_allclose(opls.x_filtered_, filtered, atol=1e-12)
        np.testing.assert_allclose(opls.correct(X), filtered, atol=1e-12)
        t_p = opls.predictive_scores_.to_numpy().ravel()
        np.testing.assert_allclose(t_p, filtered @ w_p, atol=1e-12)
        np.testing.assert_allclose(opls.predictive_loadings_, filtered.T @ t_p / (t_p @ t_p), rtol=1e-12)
        assert opls.y_loadings_ == pytest.approx(ys @ t_p / (t_p @ t_p), rel=1e-12)

    def test_the_score_variances_spe_and_t2(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """SPE and the final T2 are those of PLS with as many components; T2 starts at (n - 1) t^2 / t't."""
        X, y = cheese
        opls = OPLS(n_orthogonal_components=1).fit(X, y)
        pls = PLS(n_components=2).fit(X, y)
        scores = opls.scores_.to_numpy()
        np.testing.assert_allclose(opls.explained_variance_, np.var(scores, axis=0, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(opls.scaling_factor_for_scores_, np.std(scores, axis=0, ddof=1), rtol=1e-10)

        spe = opls.spe_.to_numpy()
        np.testing.assert_allclose(spe[:, 0], pls.spe_.to_numpy()[:, -1], rtol=1e-9)
        np.testing.assert_array_equal(spe[:, 1], spe[:, 0])

        t2 = opls.hotellings_t2_.to_numpy()
        t_p = scores[:, 0]
        np.testing.assert_allclose(t2[:, 0], (len(t_p) - 1) * t_p**2 / (t_p @ t_p), rtol=1e-10)
        np.testing.assert_allclose(t2[:, 1], pls.hotellings_t2_.to_numpy()[:, -1], rtol=1e-9)

    def test_unscaled_coefficients_are_plss(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Without scaling, beta stays in the units of the data handed in, as PLS's does."""
        X, y = cheese
        Xs, ys = MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(y)
        opls = OPLS(n_orthogonal_components=1, scale=False).fit(Xs, ys)
        pls = PLS(n_components=2, scale=False).fit(Xs, ys)
        np.testing.assert_allclose(opls.beta_coefficients_, pls.beta_coefficients_, atol=1e-10)
        predicted = opls.predict(Xs)
        assert list(predicted.columns) == ["Taste"]
        np.testing.assert_allclose(predicted, Xs.to_numpy() @ opls.beta_coefficients_.to_numpy(), atol=1e-10)

    def test_the_counts_and_the_fitted_feature_names(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        opls = OPLS(n_orthogonal_components=1).fit(X, y)
        assert (opls.n_samples_, opls.n_targets_, opls.n_features_in_) == (30, 1, 3)
        assert list(opls.feature_names_in_) == ["Acetic", "H2S", "Lactic"]
        names = opls.get_feature_names_out()
        assert names.dtype == object
        assert list(names) == ["t_predictive", "t_orthogonal_1"]


class TestLabels:
    def test_every_output_is_labelled_by_row_feature_and_component(
        self, labelled_cheese: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        X, y = labelled_cheese
        opls = OPLS(n_orthogonal_components=2).fit(X, y)
        rows, features = list(X.index), ["Acetic", "H2S", "Lactic"]
        components = ["t_predictive", "t_orthogonal_1", "t_orthogonal_2"]
        for frame, index, columns in (
            (opls.predictive_scores_, rows, ["t_predictive"]),
            (opls.orthogonal_scores_, rows, ["o1", "o2"]),
            (opls.orthogonal_weights_, features, ["o1", "o2"]),
            (opls.orthogonal_loadings_, features, ["o1", "o2"]),
            (opls.beta_coefficients_, features, ["Taste"]),
            (opls.x_filtered_, rows, features),
            (opls.scores_, rows, components),
            (opls.spe_, rows, components),
            (opls.hotellings_t2_, rows, components),
            (opls.transform(X), rows, ["t_predictive"]),
            (opls.correct(X), rows, features),
            (opls.predict(X), rows, ["Taste"]),
        ):
            assert (list(frame.index), list(frame.columns)) == (index, columns)
        for series, index, name in (
            (opls.predictive_weights_, features, "w_predictive"),
            (opls.predictive_loadings_, features, "p_predictive"),
            (opls.scaling_factor_for_scores_, components, "Standard deviation per score"),
            (opls.invert(25.0).x_new, features, "x_new"),
        ):
            assert (list(series.index), series.name) == (index, name)

    def test_an_array_x_is_labelled_by_position(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        opls = OPLS(n_orthogonal_components=1).fit(X.to_numpy(), y)
        assert list(opls.scores_.index) == list(range(30))
        assert list(opls.beta_coefficients_.index) == [0, 1, 2]
        assert list(opls.transform(X.to_numpy()[:2]).index) == [0, 1]

    def test_a_response_is_read_by_position_and_named_by_its_series(
        self, labelled_cheese: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """A Series with its own row numbers fits as the labelled frame does."""
        X, y = labelled_cheese
        from_frame = OPLS(n_orthogonal_components=1).fit(X, y).predict(X)
        from_series = OPLS(n_orthogonal_components=1).fit(X, y["Taste"].reset_index(drop=True)).predict(X)
        pd.testing.assert_frame_equal(from_series, from_frame)

    def test_a_response_held_as_objects_is_read_as_numbers(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        np.testing.assert_array_equal(
            OPLS(n_orthogonal_components=1).fit(X, y.astype(object)).predict(X),
            OPLS(n_orthogonal_components=1).fit(X, y).predict(X),
        )

    def test_named_data_meets_no_feature_name_warning(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for scale in (True, False):
                opls = OPLS(n_orthogonal_components=1, scale=scale).fit(X, y)
                opls.predict(X)
                opls.correct(X)
                opls.invert(25.0)


class TestInputsAndMessages:
    def test_the_messages_in_full(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        with pytest.raises(ValueError, match=r"^n_orthogonal_components must be >= 0\.$"):
            OPLS(n_orthogonal_components=-1).fit(X, y)
        opls = OPLS(n_orthogonal_components=1).fit(X, y)
        with pytest.raises(ValueError, match=r"^OPLS\.invert expects a single desired response value\.$"):
            opls.invert(np.array([1.0, 2.0]))
        for target in (np.inf, None):
            with pytest.raises(ValueError, match=r"^y_desired must be a finite value\.$"):
                opls.invert(target)  # type: ignore[arg-type]

    def test_the_predictive_score_of_an_inversion_is_its_design_s_score(
        self, cheese: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        X, y = cheese
        opls = OPLS(n_orthogonal_components=1).fit(X, y)
        result = opls.invert(25.0)
        assert result.predictive_score == pytest.approx(float(opls.transform(result.x_new.to_frame().T).iloc[0, 0]))

    def test_missing_and_sparse_data_are_refused(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        from scipy import sparse

        X, y = cheese
        gappy = X.copy()
        gappy.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input X contains NaN"):
            OPLS(n_orthogonal_components=1).fit(gappy, y)
        opls = OPLS(n_orthogonal_components=1).fit(X, y)
        with pytest.raises(ValueError, match="Input X contains NaN"):
            opls.predict(gappy)
        with pytest.raises(TypeError, match="Sparse data was passed"):
            OPLS(n_orthogonal_components=1).fit(sparse.csr_matrix(X.to_numpy()), y)
        with pytest.raises(TypeError, match="Sparse data was passed"):
            opls.predict(sparse.csr_matrix(X.to_numpy()))

    def test_a_different_number_of_features_is_refused(self, cheese: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        X, y = cheese
        opls = OPLS(n_orthogonal_components=1).fit(X.to_numpy(), y)
        with pytest.raises(ValueError, match="X has 2 features, but OPLS is expecting 3 features"):
            opls.predict(X.to_numpy()[:, :2])

    def test_the_settings_and_tags(self) -> None:
        assert OPLS(n_orthogonal_components=2).get_params() == {
            "n_orthogonal_components": 2,
            "scale": True,
            "copy": True,
        }
        assert OPLS(n_orthogonal_components=2, copy=False).get_params()["copy"] is False
        tags = OPLS(n_orthogonal_components=2).__sklearn_tags__()
        assert (tags.target_tags.required, tags.target_tags.single_output) == (True, True)
