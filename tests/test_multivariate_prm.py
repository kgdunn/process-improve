"""Partial Robust M-regression and its primitives (#191).

The tests are written around the property that justifies the method: a few bad
rows should move the fit a little, where they move an ordinary least-squares
PLS a lot. A test that only checks "PRM runs and returns coefficients" would
pass just as well against plain PLS, so each case here contrasts the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from sklearn.base import clone, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from process_improve.multivariate._preprocessing import _WeightedMCUVScaler
from process_improve.multivariate._robust import fair_weights, l1_median
from process_improve.multivariate.methods import PLS, PRM, MCUVScaler


def _cost(centre: np.ndarray, X: np.ndarray) -> float:
    """Total Euclidean distance from ``centre`` to every row: what the L1 median minimises."""
    return float(np.linalg.norm(X - centre, axis=1).sum())


class TestFairWeights:
    def test_weight_is_one_at_zero_and_a_quarter_at_the_cutoff(self) -> None:
        """The cutoff is a scale, not a threshold: it names where the weight reaches 1/4."""
        weights = fair_weights(np.array([0.0, 4.0]), cutoff=4.0)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.25)

    def test_never_reaches_zero(self) -> None:
        """Why Fair and not a hard cutoff: nothing is ever switched fully off.

        A weight that can hit exactly zero lets an observation leave and re-enter
        the fit between iterations, and the reweighting loop cycles instead of
        settling.
        """
        assert fair_weights(np.array([1e6]), cutoff=4.0)[0] > 0

    def test_decreasing_and_sign_blind(self) -> None:
        distances = np.array([0.0, 0.5, 1.0, 5.0, 50.0])
        weights = fair_weights(distances, cutoff=4.0)
        assert np.all(np.diff(weights) < 0)
        np.testing.assert_allclose(fair_weights(-distances, cutoff=4.0), weights)

    def test_shape_is_preserved(self) -> None:
        assert fair_weights(np.zeros((3, 2)), cutoff=4.0).shape == (3, 2)

    @pytest.mark.parametrize("cutoff", [0.0, -1.0, np.nan, np.inf])
    def test_bad_cutoff_rejected(self, cutoff: float) -> None:
        with pytest.raises(ValueError, match="cutoff must be a positive finite number"):
            fair_weights(np.array([1.0]), cutoff=cutoff)

    def test_non_finite_distances_rejected(self) -> None:
        with pytest.raises(ValueError, match="distances must be finite"):
            fair_weights(np.array([1.0, np.nan]), cutoff=4.0)


class TestL1Median:
    def test_matches_the_optimiser_it_is_a_shortcut_for(self) -> None:
        """The definition is an argmin, so check against a general-purpose argmin.

        Contaminated and clean data both, because the iteration's awkward case is
        a badly spread cloud rather than a nice one.
        """
        rng = np.random.default_rng(0)
        worst = 0.0
        for trial in range(40):
            n_samples, n_features = int(rng.integers(3, 30)), int(rng.integers(1, 5))
            X = rng.normal(size=(n_samples, n_features))
            if trial % 2:
                bad = max(1, n_samples // 4)
                X[:bad] += rng.normal(scale=50, size=(bad, n_features))
            found = _cost(l1_median(X), X)
            reference = _cost(
                minimize(
                    _cost,
                    np.median(X, axis=0),
                    args=(X,),
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
                ).x,
                X,
            )
            worst = max(worst, abs(found - reference) / max(reference, 1e-12))
        assert worst < 1e-6

    def test_centre_of_a_symmetric_cloud(self) -> None:
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(l1_median(corners), [0.5, 0.5], atol=1e-8)

    def test_one_dimensional_case_is_the_ordinary_median(self) -> None:
        values = np.array([[1.0], [2.0], [10.0]])
        np.testing.assert_allclose(l1_median(values), [np.median(values)])

    def test_iterate_landing_on_a_data_point(self) -> None:
        """The case plain Weiszfeld divides by zero on; Vardi-Zhang is why it is here.

        The origin is both the answer and one of the rows, so the very first
        iteration has a row at distance zero.
        """
        star = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        np.testing.assert_allclose(l1_median(star), [0.0, 0.0], atol=1e-8)

    def test_degenerate_inputs(self) -> None:
        np.testing.assert_allclose(l1_median(np.array([[3.0, 7.0]])), [3.0, 7.0])
        np.testing.assert_allclose(l1_median(np.tile([2.0, 5.0], (6, 1))), [2.0, 5.0])

    def test_one_wild_row_barely_moves_it(self) -> None:
        """The reason PRM measures leverage against this and not the mean."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(21, 3))
        clean = l1_median(X)
        X[0] = 1e6
        assert np.linalg.norm(l1_median(X) - clean) < 1.0
        assert np.linalg.norm(X.mean(axis=0) - clean) > 1e4

    def test_rotation_equivariant(self) -> None:
        """Unlike the coordinate-wise median, which is why the scores use this one."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 2))
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        np.testing.assert_allclose(l1_median(X @ rotation.T), rotation @ l1_median(X), atol=1e-6)

    def test_bad_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            l1_median(np.zeros(5))
        with pytest.raises(ValueError, match="at least one row"):
            l1_median(np.zeros((0, 3)))
        with pytest.raises(ValueError, match="must be finite"):
            l1_median(np.array([[1.0, np.nan]]))


# ---------------------------------------------------------------------------
# PRM itself
# ---------------------------------------------------------------------------

TRUE_BETA = np.array([2.0, -1.0])
N_SAMPLES, N_FEATURES, N_BAD = 60, 8, 9


def _dataset(seed: int, *, n_bad: int = 0, kind: str = "vertical") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a rank-2 X with a linear y, optionally contaminated in one of two ways.

    The two kinds are the two ways a least-squares fit is broken, and they are
    broken by different halves of PRM's weight:

    ``vertical``
        An ordinary position in X with a y that does not follow the
        relationship. The residual weight is what catches these.
    ``leverage``
        Far out along the real score directions, where least squares gives an
        observation the most influence of all, *and* with a y that contradicts
        the relationship. Scattering such rows randomly in X instead would not
        do: measured over ten seeds that leaves PLS no worse on average than
        the clean fit, so a test built on it would be testing nothing.
    """
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(N_SAMPLES, 2))
    loadings = rng.normal(size=(N_FEATURES, 2))
    X = scores @ loadings.T + rng.normal(scale=0.1, size=(N_SAMPLES, N_FEATURES))
    y = scores @ TRUE_BETA + rng.normal(scale=0.1, size=N_SAMPLES)

    if n_bad and kind == "vertical":
        y[:n_bad] += 25.0
    elif n_bad:
        far = rng.normal(loc=8.0, scale=1.0, size=(n_bad, 2))
        X[:n_bad] = far @ loadings.T
        y[:n_bad] = -(far @ TRUE_BETA)

    columns = [f"x{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=columns), pd.DataFrame({"y": y})


def _rmse_on(model: PLS, X: pd.DataFrame, y: pd.DataFrame) -> float:
    """Prediction error on data the model never saw, and that is not contaminated."""
    predicted = np.asarray(model.predict(X), dtype=float).ravel()
    return float(np.sqrt(np.mean((predicted - y["y"].to_numpy()) ** 2)))


@pytest.fixture
def clean_test_set() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _dataset(seed=500)


class TestPRMResistsContamination:
    """The property the method exists for, and the only reason to prefer it to PLS."""

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["vertical", "leverage"])
    def test_recovers_the_model_that_the_clean_data_would_have_given(
        self, kind: str, clean_test_set: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Scored on clean held-out data, PRM on dirty data should match PLS on clean data.

        Stated as a ratio against the clean-data fit rather than an absolute
        RMSE, so the threshold does not silently encode this fixture's noise
        level. Ten seeds, because a single one can flatter either estimator.
        """
        X_test, y_test = clean_test_set
        damage = []
        for seed in range(10):
            target = _rmse_on(PLS(n_components=2).fit(*_dataset(seed)), X_test, y_test)
            dirty = _dataset(seed, n_bad=N_BAD, kind=kind)
            damage.append(_rmse_on(PRM(n_components=2).fit(*dirty), X_test, y_test) / target)
        assert max(damage) < 1.25, f"worst-case damage {max(damage):.2f}x"
        assert float(np.median(damage)) < 1.1

    def test_where_pls_is_the_one_that_breaks(self, clean_test_set: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """The contrast, without which the test above could pass on an easy fixture.

        Only the vertical case is asserted on: scattered leverage points happen
        to leave PLS roughly where it was, so requiring PLS to break there would
        be asserting something that is not true.
        """
        X_test, y_test = clean_test_set
        damage = []
        for seed in range(10):
            target = _rmse_on(PLS(n_components=2).fit(*_dataset(seed)), X_test, y_test)
            dirty = _dataset(seed, n_bad=N_BAD, kind="vertical")
            damage.append(_rmse_on(PLS(n_components=2).fit(*dirty), X_test, y_test) / target)
        assert float(np.median(damage)) > 1.3, "the fixture does not actually break PLS"

    @pytest.mark.parametrize("kind", ["vertical", "leverage"])
    def test_the_planted_rows_are_the_ones_downweighted(self, kind: str) -> None:
        """Not just a better fit: the right rows have to be the ones held responsible."""
        X, y = _dataset(seed=0, n_bad=N_BAD, kind=kind)
        model = PRM(n_components=2).fit(X, y)
        weights = model.robust_weights_
        assert weights[:N_BAD].mean() * 10 < weights[N_BAD:].mean()
        assert weights.min() > 0.0, "the Fair function must never switch a row fully off"

    def test_outlier_summary_names_them(self) -> None:
        X, y = _dataset(seed=0, n_bad=N_BAD, kind="vertical")
        flagged = PRM(n_components=2).fit(X, y).outlier_summary(threshold=0.1)
        assert set(flagged.index) == set(range(N_BAD))
        assert flagged["weight"].is_monotonic_increasing


class TestPRMCostsNothingOnCleanData:
    def test_matches_pls_when_there_is_nothing_to_resist(
        self, clean_test_set: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """A robust estimator that gave up real accuracy on clean data would not be worth using."""
        X_test, y_test = clean_test_set
        X, y = _dataset(seed=0)
        ordinary = _rmse_on(PLS(n_components=2).fit(X, y), X_test, y_test)
        robust = _rmse_on(PRM(n_components=2).fit(X, y), X_test, y_test)
        assert robust < ordinary * 1.05

    def test_equal_weights_reproduce_the_base_class_scaling(self) -> None:
        """PRM's one override has to be a no-op when the weights are uniform.

        Otherwise the override would be changing the model for reasons unrelated
        to robustness, and any difference measured against PLS would be
        confounded.
        """
        X, y = _dataset(seed=0)
        model = PRM(n_components=2)
        weighted, _ = model._make_scalers(X, y, np.ones(N_SAMPLES))
        plain, _ = PLS(n_components=2)._make_scalers(X, y, np.ones(N_SAMPLES))
        pd.testing.assert_series_equal(weighted.center_, plain.center_)
        pd.testing.assert_series_equal(weighted.scale_, plain.scale_)


class TestPRMConvergenceReporting:
    def test_converges_and_says_so(self) -> None:
        model = PRM(n_components=2).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.weights_converged_
        assert model.weight_shift_ <= model.weight_tol
        assert 1 <= model.n_weight_iter_ <= model.max_weight_iter

    def test_a_limit_cycle_stops_early_rather_than_running_out_the_budget(self) -> None:
        """Reweighting can settle into a cycle instead of a point; that is not a reason to spin.

        The fixture is rows scattered far in X with an unrelated y, which is the
        shape that makes the leverage weight chase its own tail: suppressing a
        row contracts the score cloud, which raises everyone's standardised
        distance, which changes the fit again. Each iteration is a full PLS fit,
        so stopping matters.
        """
        rng = np.random.default_rng(0)
        X, y = _dataset(seed=0)
        X.iloc[:N_BAD] = rng.normal(scale=12, size=(N_BAD, N_FEATURES))
        y.iloc[:N_BAD] = rng.normal(scale=1, size=N_BAD).reshape(-1, 1)

        model = PRM(n_components=2, max_weight_iter=100).fit(X, y)
        assert not model.weights_converged_
        assert model.n_weight_iter_ < 40, "gave up far too late"
        # The point of reporting the shift: 'not converged' here still means the
        # weights are stable to a few percent, which a bare flag cannot say.
        assert model.weight_shift_ < 0.2
        assert model.robust_weights_[:N_BAD].mean() < model.robust_weights_[N_BAD:].mean()

    def test_a_tiny_budget_is_honoured(self) -> None:
        model = PRM(n_components=2, max_weight_iter=1).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.n_weight_iter_ == 1
        assert not model.weights_converged_


class TestPRMApi:
    def test_is_an_sklearn_regressor(self) -> None:
        X, y = _dataset(seed=0)
        model = PRM(n_components=2)
        assert is_regressor(model)
        assert clone(model).get_params() == model.get_params()
        assert model.fit(X, y) is model

    def test_inherits_the_pls_diagnostics(self) -> None:
        """Subclassing is the point: a resistant fit should not cost the surrounding toolkit."""
        model = PRM(n_components=2).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.vip().shape == (N_FEATURES,)
        assert model.spe_.shape == (N_SAMPLES, 2)
        assert model.scores_.shape == (N_SAMPLES, 2)
        assert np.isfinite(model.hotellings_t2_limit(0.95))

    def test_composes_in_a_pipeline_and_cross_validates(self) -> None:
        X, y = _dataset(seed=0)
        pipeline = Pipeline([("scale", MCUVScaler()), ("model", PRM(n_components=2))])
        assert np.asarray(pipeline.fit(X, y).predict(X)).shape == (N_SAMPLES, 1)
        assert np.all(cross_val_score(PRM(n_components=2), X, y, cv=3) > 0.9)

    @pytest.mark.parametrize("as_type", ["frame", "series", "ndarray"])
    def test_accepts_the_usual_target_shapes(self, as_type: str) -> None:
        X, y = _dataset(seed=0)
        target = {"frame": y, "series": y["y"], "ndarray": y.to_numpy()}[as_type]
        assert PRM(n_components=2).fit(X, target).predict(X).shape == (N_SAMPLES, 1)

    def test_multi_target(self) -> None:
        """The paper is written for a single y; a row's residual becomes its norm over targets."""
        X, y = _dataset(seed=0)
        two = pd.DataFrame({"a": y["y"], "b": y["y"] * -0.5 + 1.0})
        assert PRM(n_components=2).fit(X, two).predict(X).shape == (N_SAMPLES, 2)

    def test_a_zero_prior_weight_row_is_dropped_not_merely_silenced(self) -> None:
        """sample_weight expresses what the data cannot: a run already known to be bad.

        PLS documents a zero weight as equivalent to removing the row, so PRM
        has to honour that all the way through. It is not automatic: the robust
        weights are built from a median residual scale and a median score
        distance, and a zeroed row left in those medians would still be steering
        the fit while appearing to have no weight. The row picked here is an
        ordinary one that PRM would otherwise keep, so the comparison is about
        the prior weight and not about the model's own opinion of the row.
        """
        X, y = _dataset(seed=0)
        prior = np.ones(N_SAMPLES)
        prior[0] = 0.0

        kept = PRM(n_components=2).fit(X, y)
        zeroed = PRM(n_components=2).fit(X, y, sample_weight=prior)
        dropped = PRM(n_components=2).fit(X.iloc[1:], y.iloc[1:])

        assert kept.robust_weights_[0] > 0.1, "the fixture row must be one the model would keep"
        beta = lambda model: np.asarray(model.beta_coefficients_, dtype=float).ravel()  # noqa: E731
        to_dropped = float(np.linalg.norm(beta(zeroed) - beta(dropped)))
        to_kept = float(np.linalg.norm(beta(zeroed) - beta(kept)))
        assert to_dropped < to_kept / 3, f"zeroed fit sits {to_dropped:.5f} from dropped, {to_kept:.5f} from kept"

    def test_prior_weights_are_validated(self) -> None:
        X, y = _dataset(seed=0)
        with pytest.raises(ValueError, match="expected 60 to match X"):
            PRM(n_components=2).fit(X, y, sample_weight=np.ones(5))
        with pytest.raises(ValueError, match="at least one row with a positive weight"):
            PRM(n_components=2).fit(X, y, sample_weight=np.zeros(N_SAMPLES))


class TestPRMRejectsWhatItCannotHandle:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"cutoff": 0.0}, "cutoff must be a positive finite number"),
            ({"cutoff": np.inf}, "cutoff must be a positive finite number"),
            ({"max_weight_iter": 0}, "max_weight_iter must be at least 1"),
            ({"weight_tol": -1.0}, "weight_tol must be a positive finite number"),
        ],
    )
    def test_bad_settings(self, kwargs: dict, message: str) -> None:
        X, y = _dataset(seed=0)
        with pytest.raises(ValueError, match=message):
            PRM(n_components=2, **kwargs).fit(X, y)

    def test_missing_data_is_refused_with_a_reason_and_an_alternative(self) -> None:
        """Silently threading NaN through would downweight a row for being incomplete."""
        X, y = _dataset(seed=0)
        X.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="does not support missing data"):
            PRM(n_components=2).fit(X, y)
        # The message has to name the way out, not just the refusal.
        with pytest.raises(ValueError, match="Use PLS"):
            PRM(n_components=2).fit(X, y)

    def test_outlier_summary_before_fit(self) -> None:
        with pytest.raises(AttributeError, match="not fitted yet"):
            PRM(n_components=2).outlier_summary()


class TestWeightedScaler:
    """PRM's centring override, tested where it is easier to see than through a fit."""

    def test_uniform_weights_reproduce_mcuvscaler_exactly(self) -> None:
        """Including ddof=1, which the naive weighted variance gets wrong by sqrt((n-1)/n)."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
        plain = MCUVScaler().fit(X)
        for weight in (None, 1.0, 3.7):
            weights = None if weight is None else np.full(20, weight)
            weighted = _WeightedMCUVScaler().fit(X, sample_weight=weights)
            pd.testing.assert_series_equal(weighted.center_, plain.center_)
            pd.testing.assert_series_equal(weighted.scale_, plain.scale_)

    def test_a_zero_weight_row_behaves_exactly_like_a_dropped_row(self) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
        contaminated = X.copy()
        contaminated.iloc[0] = 500.0
        weights = np.ones(20)
        weights[0] = 0.0

        weighted = _WeightedMCUVScaler().fit(contaminated, sample_weight=weights)
        dropped = MCUVScaler().fit(X.iloc[1:])
        pd.testing.assert_series_equal(weighted.center_, dropped.center_)
        pd.testing.assert_series_equal(weighted.scale_, dropped.scale_)

    def test_a_wild_row_moves_it_only_as_far_as_its_weight_allows(self) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
        contaminated = X.copy()
        contaminated.iloc[0] = 500.0
        weights = np.ones(20)
        weights[0] = 1e-4

        plain = MCUVScaler().fit(contaminated)
        weighted = _WeightedMCUVScaler().fit(contaminated, sample_weight=weights)
        clean = MCUVScaler().fit(X)
        assert np.abs(weighted.center_ - clean.center_).max() < np.abs(plain.center_ - clean.center_).max()

    def test_constant_and_missing_columns_survive(self) -> None:
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 2)), columns=list("ab"))
        X["constant"] = 4.0
        X.iloc[3, 1] = np.nan
        scaler = _WeightedMCUVScaler().fit(X, sample_weight=rng.uniform(0.1, 1.0, 20))
        assert scaler.scale_["constant"] == pytest.approx(1.0)
        assert np.all(np.isfinite(scaler.scale_))

    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (np.ones(5), "expected 20 to match X"),
            (np.full(20, np.nan), "must be finite"),
            (-np.ones(20), "must be non-negative"),
            (np.zeros(20), "must not sum to zero"),
        ],
    )
    def test_bad_weights_rejected(self, weights: np.ndarray, message: str) -> None:
        X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3)), columns=list("abc"))
        with pytest.raises(ValueError, match=message):
            _WeightedMCUVScaler().fit(X, sample_weight=weights)
