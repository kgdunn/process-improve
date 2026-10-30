"""Partial Robust M-regression and its primitives (#191).

The tests are written around the property that justifies the method: a few bad
rows should move the fit a little, where they move an ordinary least-squares
PLS a lot. A test that only checks "PRM runs and returns coefficients" would
pass just as well against plain PLS, so each case here contrasts the two.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from sklearn.base import clone, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from process_improve.multivariate._common import epsqrt
from process_improve.multivariate._prm import _column_medians, _residual_weights, _robust_scaler
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
        with pytest.raises(ValueError, match=r"^distances must be finite \(no NaN / inf\)\.$"):
            fair_weights(np.array([1.0, np.nan]), cutoff=4.0)

    def test_a_cutoff_below_one_is_a_cutoff_like_any_other(self) -> None:
        """At |z| = 1 and c = 0.5 the weight is (1 + 2) ** -2 = 1/9."""
        assert fair_weights(np.array([1.0]), cutoff=0.5)[0] == pytest.approx(1.0 / 9.0, rel=1e-15)

    def test_distances_held_as_objects_are_read_as_numbers(self) -> None:
        weights = fair_weights(np.array([0, 4], dtype=object), cutoff=4.0)
        np.testing.assert_array_equal(weights, [1.0, 0.25])


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
        with pytest.raises(ValueError, match=r"^X must be 2-dimensional \(n_samples, n_features\); got 1 dimension"):
            l1_median(np.zeros(5))
        with pytest.raises(ValueError, match=r"^X must be 2-dimensional .*; got 3 dimension"):
            l1_median(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError, match=r"^X must have at least one row\.$"):
            l1_median(np.zeros((0, 3)))
        with pytest.raises(ValueError, match=r"^X must be finite \(no NaN / inf\)\.$"):
            l1_median(np.array([[1.0, np.nan]]))


class TestL1MedianByHand:
    """One Vardi-Zhang step worked by hand, and the edges of the tolerance.

    The five rows (0, 0), (1, 0), (2, 0), (0, 1) and (0, -1) have coordinate-wise median (0, 0),
    the first row, so the iteration starts on a row. The other four are at distances 1, 2, 1, 1:
    the Weiszfeld point is ((1, 0) + (2, 0) / 2 + (0, 1) + (0, -1)) / (1 + 1/2 + 1 + 1) = (4/7, 0).
    Their unit vectors sum to (2, 0), a pull of 2 against the one row on the estimate, so the step
    is damped by 1/2 to (2/7, 0).
    """

    ROWS = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    def test_one_step(self) -> None:
        np.testing.assert_allclose(l1_median(self.ROWS, max_iter=1), [2.0 / 7.0, 0.0], rtol=1e-15)
        # Numbers held as objects are read as numbers.
        np.testing.assert_allclose(l1_median(self.ROWS.astype(object), max_iter=1), [2.0 / 7.0, 0.0], rtol=1e-15)

    def test_no_step_leaves_the_coordinatewise_median(self) -> None:
        np.testing.assert_array_equal(l1_median(self.ROWS + np.array([10.0, 20.0]), max_iter=0), [10.0, 20.0])

    def test_the_last_step_is_kept_when_it_is_short_enough(self) -> None:
        """Away from the origin, a first step of 2/7 is below a tolerance of 0.5: it is taken, then the loop stops."""
        np.testing.assert_allclose(l1_median(self.ROWS + 10.0, tol=0.5), [10.0 + 2.0 / 7.0, 10.0], rtol=1e-15)

    def test_each_use_of_the_tolerance_includes_its_edge(self) -> None:
        """A step, a pull and a distance equal to ``tol`` each count as within it."""
        step = 0.5 * (2.0 / 3.5)  # the first step, 2/7, exactly as the iteration computes it
        np.testing.assert_array_equal(l1_median(self.ROWS, tol=step), [step, 0.0])
        # Scaled by 10 the pull is still 2: with tol = 2 the rows on the estimate hold it there.
        np.testing.assert_array_equal(l1_median(10.0 * self.ROWS, tol=2.0), [0.0, 0.0])
        # With tol = 1, the three rows at distance 1 sit on the estimate, and the one left pulls by 1.
        np.testing.assert_array_equal(l1_median(self.ROWS, tol=1.0), [0.0, 0.0])

    def test_the_default_budget_is_200_iterations(self) -> None:
        """Six rows whose median the iteration is still approaching after 200 steps, 1e-6 at a time."""
        rows = np.array([[0.8, 0.9], [-0.7, 0.2], [-0.8, 2.3], [-0.7, -0.5], [-1.1, -0.3], [-0.0, 0.8]])
        np.testing.assert_array_equal(l1_median(rows), l1_median(rows, max_iter=200))
        assert not np.array_equal(l1_median(rows, max_iter=200), l1_median(rows, max_iter=201))


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


class TestPRMCostsLittleOnCleanData:
    def test_about_as_accurate_as_pls_when_there_is_nothing_to_resist(
        self, clean_test_set: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """A robust estimator that gave up real accuracy on clean data would not be worth using.

        The Fair weights fall below 1 for ordinary rows too, so on any one seed PRM
        can land either side of PLS; over ten seeds it is as accurate.
        """
        X_test, y_test = clean_test_set
        ratios = []
        for seed in range(10):
            X, y = _dataset(seed)
            ordinary = _rmse_on(PLS(n_components=2).fit(X, y), X_test, y_test)
            ratios.append(_rmse_on(PRM(n_components=2).fit(X, y), X_test, y_test) / ordinary)
        assert float(np.median(ratios)) < 1.05


# chemometrics::prm (R package chemometrics 1.4.4) on the cheddar cheese data, with its one
# departure from the paper undone: `dt <- T - mt` replaced by `dt <- sweep(T, 2, mt)`, since R
# recycles the vector mt down the columns of T cell by cell. With one component the two agree,
# but pcaPP's l1median stops with an error on a single column, so a = 2 and 3 are the cases.
CHEMOMETRICS_CHEESE = {
    2: {
        "coefficients": [4.82091875346975, 3.88207238660843, 9.53461287401522],
        "intercept": -39.6390573529007,
        "weights": [0.269362106370723, 0.800427328850199, 0.345650545048036, 0.309268357969527, 0.490720999890499],
    },
    3: {
        "coefficients": [-1.42316987147143, 4.11081084179467, 19.31562817829933],
        "intercept": -21.6374492638998,
        "weights": [0.269135922215876, 0.727340338820908, 0.294374481244887, 0.295731913301512, 0.581942924597089],
    },
}


@pytest.fixture
def cheese() -> tuple[pd.DataFrame, pd.Series]:
    from importlib import resources

    with (
        resources.files("process_improve.datasets.multivariate")
        .joinpath("cheddar-cheese.csv")
        .open(encoding="utf-8") as f
    ):
        data = pd.read_csv(f, index_col=0)
    return data[["Acetic", "H2S", "Lactic"]], data["Taste"]


class TestPRMIsThePublishedAlgorithm:
    @pytest.mark.dataset
    @pytest.mark.parametrize("n_components", [2, 3])
    def test_scale_false_reproduces_the_authors_r_code(
        self, cheese: tuple[pd.DataFrame, pd.Series], n_components: int
    ) -> None:
        """Coefficients, intercept and weights as chemometrics::prm gives them, to its 1% stopping rule's echo."""
        X, y = cheese
        model = PRM(n_components=n_components, scale=False).fit(X, y)
        expected = CHEMOMETRICS_CHEESE[n_components]
        np.testing.assert_allclose(model.beta_coefficients_.to_numpy().ravel(), expected["coefficients"], atol=1e-6)
        intercept = float(model.predict(X.iloc[[0]] * 0.0).iloc[0, 0])
        assert intercept == pytest.approx(expected["intercept"], abs=1e-6)
        np.testing.assert_allclose(model.robust_weights_[:5], expected["weights"], atol=1e-7)
        assert model.weights_converged_

    def test_the_intercept_is_the_median_residual(self) -> None:
        """The paper's intercept: half the fitted rows lie above the model and half below."""
        X, y = _dataset(seed=0, n_bad=N_BAD)
        model = PRM(n_components=2).fit(X, y)
        residuals = y["y"].to_numpy() - np.asarray(model.predict(X), dtype=float).ravel()
        assert float(np.median(residuals)) == pytest.approx(0.0, abs=1e-10)

    def test_scale_makes_the_model_blind_to_each_columns_units(self) -> None:
        """scale=True divides each column by its robust spread, so a change of units changes nothing.

        With scale=False a column in larger units dominates the distances, as in the paper.
        """
        X, y = _dataset(seed=0, n_bad=N_BAD)
        units = np.linspace(1.0, 1000.0, N_FEATURES)
        for scale, same in ((True, True), (False, False)):
            plain = np.asarray(PRM(n_components=2, scale=scale).fit(X, y).predict(X), dtype=float)
            rescaled = PRM(n_components=2, scale=scale).fit(X * units, y * 50.0).predict(X * units)
            assert np.allclose(np.asarray(rescaled, dtype=float) / 50.0, plain) == same


class TestPRMConvergenceReporting:
    def test_converges_and_says_so(self) -> None:
        model = PRM(n_components=2).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.weights_converged_
        assert 1 <= model.n_weight_iter_ < 10
        assert model.weight_shift_ < 0.5

    def test_a_tighter_tolerance_iterates_longer_and_moves_the_fit_little(self) -> None:
        """coefficient_tol is the paper's stopping rule; tightening it buys iterations, not a different model."""
        data = _dataset(seed=0, n_bad=N_BAD)
        default = PRM(n_components=2).fit(*data)
        tight = PRM(n_components=2, coefficient_tol=1e-8).fit(*data)
        assert tight.n_weight_iter_ > default.n_weight_iter_
        beta = lambda model: np.asarray(model.beta_coefficients_, dtype=float).ravel()  # noqa: E731
        assert np.linalg.norm(beta(tight) - beta(default)) < 0.05 * np.linalg.norm(beta(tight))

    def test_a_tiny_budget_is_honoured(self) -> None:
        model = PRM(n_components=2, max_weight_iter=1).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.n_weight_iter_ == 1
        assert not model.weights_converged_

    def test_weight_tol_is_deprecated_and_has_no_effect(self) -> None:
        data = _dataset(seed=0, n_bad=N_BAD)
        with pytest.warns(DeprecationWarning, match="weight_tol is deprecated since 2.2.2"):
            old = PRM(n_components=2, weight_tol=1e-4).fit(*data)
        np.testing.assert_array_equal(old.robust_weights_, PRM(n_components=2).fit(*data).robust_weights_)


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
            ({"coefficient_tol": -1.0}, "coefficient_tol must be a positive finite number"),
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


class TestRobustScalingByHand:
    """The centre and spread PRM fixes once, worked by hand."""

    # Column a: 1, 2, 4, 7, 11 has median 4 and absolute deviations 3, 2, 0, 3, 7, whose median is 3.
    # Column b: four zeros and a 5 has median 0 and a median absolute deviation of 0.
    BLOCK = pd.DataFrame({"a": [1.0, 2.0, 4.0, 7.0, 11.0], "b": [0.0, 0.0, 0.0, 0.0, 5.0]})

    def test_column_medians_and_spreads(self) -> None:
        scaler = _robust_scaler(self.BLOCK, np.ones(5, dtype=bool), scale=True, centre=_column_medians)
        np.testing.assert_allclose(scaler.center_, [4.0, 0.0], rtol=1e-15)
        np.testing.assert_allclose(scaler.scale_, [3 * 1.4826, 1.0], rtol=1e-15)
        assert list(scaler.center_.index) == list(scaler.scale_.index) == ["a", "b"]

    def test_without_scaling_the_spread_is_one(self) -> None:
        scaler = _robust_scaler(self.BLOCK, np.ones(5, dtype=bool), scale=False, centre=_column_medians)
        np.testing.assert_allclose(scaler.center_, [4.0, 0.0], rtol=1e-15)
        np.testing.assert_array_equal(scaler.scale_, [1.0, 1.0])

    def test_only_the_active_rows_count(self) -> None:
        """Without the last row, column a is 1, 2, 4, 7: median 3, deviations 2, 1, 1, 4, their median 1.5."""
        active = np.array([True, True, True, True, False])
        scaler = _robust_scaler(self.BLOCK, active, scale=True, centre=_column_medians)
        np.testing.assert_allclose(scaler.center_, [3.0, 0.0], rtol=1e-15)
        np.testing.assert_allclose(scaler.scale_, [1.5 * 1.4826, 1.0], rtol=1e-15)

    def test_a_spread_of_exactly_epsqrt_counts_as_none(self) -> None:
        """Deviations of v = epsqrt / 1.4826 scale to exactly epsqrt, which is not above it, so the spread is 1."""
        v = epsqrt / 1.4826
        assert v * 1.4826 == epsqrt
        block = pd.DataFrame({"tiny": [-v, -v, 0.0, v, v]})
        scaler = _robust_scaler(block, np.ones(5, dtype=bool), scale=True, centre=_column_medians)
        np.testing.assert_array_equal(scaler.scale_, [1.0])

    def test_x_is_centred_at_the_l1_median_in_scaled_units(self) -> None:
        rng = np.random.default_rng(3)
        block = pd.DataFrame(rng.normal(size=(15, 2)) * [1.0, 40.0])
        scaler = _robust_scaler(block, np.ones(15, dtype=bool), scale=True, centre=l1_median)
        scaled = block.to_numpy() / scaler.scale_.to_numpy()
        best = minimize(
            _cost, np.median(scaled, axis=0), args=(scaled,), method="Nelder-Mead", options={"xatol": 1e-10}
        )
        np.testing.assert_allclose(scaler.center_.to_numpy() / scaler.scale_.to_numpy(), best.x, atol=1e-6)


class TestRobustWeightsByHand:
    def test_each_target_is_standardised_alone(self) -> None:
        """Two targets: 0, 1, 2, 3, 10 (median 2, median absolute deviation 1) and 5, 6, 8, 10, 0 (6 and 2).

        The standardised residuals are (-2, -1, 0, 1, 8) and (-0.5, 0, 1, 2, -3); a row's distance is
        their norm, and its weight the Fair function's 1 / (1 + d / 4)^2.
        """
        residuals = np.array([[0.0, 5.0], [1.0, 6.0], [2.0, 8.0], [3.0, 10.0], [10.0, 0.0]])
        distances = np.hypot([-2.0, -1.0, 0.0, 1.0, 8.0], [-0.5, 0.0, 1.0, 2.0, -3.0])
        weights = _residual_weights(residuals, np.ones(5, dtype=bool), 4.0)
        np.testing.assert_allclose(weights, 1.0 / (1.0 + distances / 4.0) ** 2, rtol=1e-14)


class TestPRMScaling:
    def test_scale_true_is_scale_false_on_blocks_divided_by_their_robust_spreads(self) -> None:
        """Each column of X and of Y over 1.4826 times its median absolute deviation (1 where that is 0).

        Two targets of different spreads, and an X column whose median absolute deviation is 0. One
        reweighting each, so the comparison is of one weighted fit, not of where two loops stop.
        """
        X, y = _dataset(seed=4)
        X["steps"] = np.r_[np.zeros(N_SAMPLES - 5), np.arange(1.0, 6.0)]
        Y = pd.DataFrame({"a": y["y"], "b": 30.0 * y["y"] + np.random.default_rng(4).normal(size=N_SAMPLES)})

        def spreads(block: pd.DataFrame) -> np.ndarray:
            values = block.to_numpy()
            mad = 1.4826 * np.median(np.abs(values - np.median(values, axis=0)), axis=0)
            return np.where(mad > 0, mad, 1.0)

        sx, sy = spreads(X), spreads(Y)
        assert sx[-1] == 1.0
        scaled = PRM(n_components=2, max_weight_iter=1).fit(X, Y)
        by_hand = PRM(n_components=2, max_weight_iter=1, scale=False).fit(X / sx, Y / sy)
        np.testing.assert_allclose(scaled.scores_, by_hand.scores_, atol=1e-9)
        np.testing.assert_allclose(scaled.predict(X), by_hand.predict(X / sx) * sy, rtol=1e-9)

    def test_the_intercept_of_each_target_is_its_median_residual(self) -> None:
        X, y = _dataset(seed=0, n_bad=N_BAD)
        Y = pd.DataFrame({"a": y["y"], "b": -2.0 * y["y"] + 7.0})
        model = PRM(n_components=2).fit(X, Y)
        residuals = Y.to_numpy() - model.predict(X).to_numpy()
        np.testing.assert_allclose(np.median(residuals, axis=0), [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(model.predictions_, model.predict(X), rtol=1e-12)


class TestPRMSettingsInFull:
    def test_the_defaults_and_settings_reach_pls(self) -> None:
        assert PRM(n_components=2).get_params() == {
            "n_components": 2,
            "cutoff": 4.0,
            "max_weight_iter": 100,
            "coefficient_tol": 0.01,
            "weight_tol": None,
            "scale": True,
            "max_iter": 1000,
            "tol": epsqrt,
            "copy": True,
            "missing_data_settings": None,
            "warn_on_uncentred": True,
        }
        settings = {"max_iter": 7, "tol": 1e-3, "copy": False, "missing_data_settings": {"md_tol": 1e-3}}
        model = PRM(n_components=2, warn_on_uncentred=False, **settings)
        assert {name: model.get_params()[name] for name in settings} == settings
        assert model.get_params()["warn_on_uncentred"] is False

    @pytest.mark.parametrize("cutoff", [0.0, -1.0, np.inf])
    def test_the_settings_are_checked_before_the_data(self, cutoff: float) -> None:
        X, y = _dataset(seed=0)
        X.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match=rf"^cutoff must be a positive finite number; got {cutoff!r}\.$"):
            PRM(n_components=2, cutoff=cutoff).fit(X, y)

    def test_a_cutoff_of_one_is_allowed(self) -> None:
        assert np.isfinite(PRM(n_components=2, cutoff=1.0).fit(*_dataset(seed=0)).robust_weights_).all()

    def test_a_zero_coefficient_tol_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"^coefficient_tol must be a positive finite number; got 0\.0\.$"):
            PRM(n_components=2, coefficient_tol=0.0).fit(*_dataset(seed=0))

    def test_the_messages_in_full(self) -> None:
        X, y = _dataset(seed=0)
        with pytest.raises(ValueError, match=r"^sample_weight has 5 entries; expected 60 to match X / Y\.$"):
            PRM(n_components=2).fit(X, y, sample_weight=np.ones(5))
        with pytest.raises(ValueError, match=r"^sample_weight must leave at least one row with a positive weight\.$"):
            PRM(n_components=2).fit(X, y, sample_weight=np.zeros(N_SAMPLES))
        with pytest.raises(AttributeError, match=r"^This PRM instance is not fitted yet; call 'fit' first\.$"):
            PRM(n_components=2).outlier_summary()
        y.iloc[3, 0] = np.nan
        with pytest.raises(
            ValueError,
            match=r"^PRM does not support missing data: a row with missing cells has residual and leverage "
            r"distances that are not comparable with a complete row's, so it would be downweighted for being "
            r"incomplete rather than for being an outlier\. Use PLS, which threads missing data through NIPALS, "
            r"or impute first\.$",
        ):
            PRM(n_components=2).fit(X, y)

    def test_none_in_a_block_held_as_objects_is_missing(self) -> None:
        X, y = _dataset(seed=0)
        X = X.astype(object)
        X.iloc[0, 0] = None
        with pytest.raises(ValueError, match=r"^PRM does not support missing data"):
            PRM(n_components=2).fit(X, y)

    def test_weights_held_as_objects_are_read_as_numbers(self) -> None:
        X, y = _dataset(seed=0)
        as_objects = PRM(n_components=2).fit(X, y, sample_weight=np.ones(N_SAMPLES, dtype=object))
        np.testing.assert_array_equal(as_objects.robust_weights_, PRM(n_components=2).fit(X, y).robust_weights_)

    def test_the_deprecation_in_full_and_from_the_caller(self) -> None:
        with pytest.warns(DeprecationWarning, match="^process_improve") as record:
            PRM(n_components=2, weight_tol=1e-4).fit(*_dataset(seed=0))
        assert str(record[0].message) == (
            "process_improve.multivariate.methods.PRM.weight_tol is deprecated since 2.2.2 and will be removed in "
            "3.0; use coefficient_tol instead. weight_tol has no effect: PRM stops, as Serneels et al. (2005) do, "
            "when the norm of its regression on the unit-length scores changes by less than coefficient_tol."
        )
        assert pathlib.Path(record[0].filename).name != "_prm.py"


class TestPRMLoopInFull:
    def test_the_first_iteration_never_stops_the_loop(self) -> None:
        """The first change is from an infinite norm, so even a huge tolerance stops at the second iteration."""
        model = PRM(n_components=2, coefficient_tol=1e9).fit(*_dataset(seed=0))
        assert model.n_weight_iter_ == 2
        assert model.weights_converged_ is True

    def test_a_budget_of_one_reports_no_convergence(self) -> None:
        model = PRM(n_components=2, max_weight_iter=1).fit(*_dataset(seed=0, n_bad=N_BAD))
        assert model.weights_converged_ is False

    def test_a_weight_that_underflows_to_zero_is_floored(self) -> None:
        """A row 1e100 out in X and in y: two Fair weights near (4 / 1e100)^2 multiply to 0, which becomes 1e-6."""
        X, y = _dataset(seed=0)
        X.iloc[5] = 1e100 * np.sign(X.iloc[0].to_numpy())
        y.iloc[5, 0] = -1e100
        weights = PRM(n_components=2, max_weight_iter=1).fit(X, y).robust_weights_
        assert weights[5] == 1e-6
        assert (weights[np.arange(N_SAMPLES) != 5] > 0.01).all()


class TestOutlierSummaryInFull:
    def test_labelled_as_the_training_rows_with_a_threshold_of_one_tenth(self) -> None:
        X, y = _dataset(seed=0, n_bad=N_BAD, kind="vertical")
        labels = [f"run{i}" for i in range(N_SAMPLES)]
        X.index = y.index = labels
        model = PRM(n_components=2).fit(X, y)
        flagged = model.outlier_summary()
        pd.testing.assert_frame_equal(flagged, model.outlier_summary(threshold=0.1))
        assert set(flagged.index) == set(labels[:N_BAD])

    def test_a_weight_equal_to_the_threshold_is_not_below_it(self) -> None:
        model = PRM(n_components=2).fit(*_dataset(seed=0, n_bad=N_BAD))
        weakest = float(model.robust_weights_.min())
        assert model.outlier_summary(threshold=weakest).empty
