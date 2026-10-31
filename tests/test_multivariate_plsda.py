"""PLS discriminant analysis (#375).

`PLSDA` is a classification layer over `PLS`, so these tests split into three concerns:
the encoding and decision rules that are new here, the classifier diagnostics, and the
promise that everything `PLS` already offers still works on the subclass.
"""

from __future__ import annotations

import inspect
import itertools
import pathlib
import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import softmax
from scipy.stats import norm
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold, check_cv, cross_val_score
from sklearn.pipeline import Pipeline

from process_improve.multivariate._plsda import _bayes_threshold, _gaussian_log_density, _mean_and_sd
from process_improve.multivariate.methods import (
    PLS,
    PLSDA,
    MCUVScaler,
    NotEnoughVarianceError,
    SpecificationWarning,
    confusion_matrix_plot,
)


def _separable(
    n_per_class: int = 25,
    n_features: int = 6,
    separation: float = 3.0,
    classes: tuple[str, ...] = ("low", "high"),
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Gaussian clusters, one per class, displaced along a random direction.

    ``separation=0`` gives classes that are pure noise, which is what the permutation
    test has to be able to tell from the rest.
    """
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((len(classes), n_features))
    blocks, labels = [], []
    for position, name in enumerate(classes):
        blocks.append(rng.standard_normal((n_per_class, n_features)) + separation * directions[position])
        labels += [name] * n_per_class
    frame = pd.DataFrame(np.vstack(blocks), columns=[f"x{i}" for i in range(n_features)])
    return frame, np.asarray(labels)


# ---------------------------------------------------------------------------
# Fitting, prediction, and the two decision rules
# ---------------------------------------------------------------------------


def test_plsda_fits_a_binary_problem() -> None:
    """The basic contract: labels in, labels out, with indicator scores in between."""
    X, y = _separable()
    model = PLSDA(n_components=2).fit(X, y)

    assert list(model.classes_) == ["high", "low"]  # np.unique sorts
    assert model.n_classes_ == 2
    assert model.accuracy_ == pytest.approx(1.0)

    predicted = model.predict(X)
    assert predicted.shape == y.shape
    assert set(predicted) <= set(model.classes_)
    assert (predicted == y).all()

    scores = model.decision_function(X)
    assert scores.shape == (len(y), 2)
    assert list(scores.columns) == list(model.classes_)
    # The indicator for the true class is the larger one, by construction of "max".
    assert (scores.to_numpy().argmax(axis=1) == (y == "low").astype(int)).all()


def test_plsda_fits_a_multiclass_problem() -> None:
    """Three classes: one indicator column each, and posteriors that sum to 1."""
    X, y = _separable(classes=("low", "mid", "high"), n_per_class=20, seed=1)
    model = PLSDA(n_components=3).fit(X, y)

    assert model.n_classes_ == 3
    assert model.decision_function(X).shape == (len(y), 3)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(y)))
    assert (proba >= 0).all()
    # argmax of the posterior recovers the labels on well-separated data.
    assert (model.classes_[proba.argmax(axis=1)] == y).all()


def test_plsda_confusion_and_rates_on_a_held_out_split() -> None:
    """`confusion()` on held-out rows, checked against the counts by hand."""
    X, y = _separable(n_per_class=30, separation=1.2, seed=2)
    train = np.zeros(len(y), dtype=bool)
    train[::2] = True  # every other row, so both classes are in both halves
    model = PLSDA(n_components=2).fit(X[train], y[train])

    held_out = model.confusion(X[~train], y[~train])
    matrix = held_out.matrix.to_numpy()
    assert matrix.sum() == (~train).sum()
    assert list(held_out.matrix.index) == list(model.classes_)
    assert held_out.accuracy == pytest.approx(np.trace(matrix) / matrix.sum())

    # Sensitivity is the diagonal over the row total; specificity is everything else
    # classified as "not this class" over everything that is not this class.
    for position, klass in enumerate(model.classes_):
        actual = matrix[position].sum()
        assert held_out.sensitivity[klass] == pytest.approx(matrix[position, position] / actual)
        negatives = matrix.sum() - actual
        true_negative = matrix.sum() - actual - (matrix[:, position].sum() - matrix[position, position])
        assert held_out.specificity[klass] == pytest.approx(true_negative / negatives)


def test_plsda_confusion_rejects_an_unseen_label() -> None:
    """A label the model never saw is an error, not a silently dropped row."""
    X, y = _separable()
    model = PLSDA(n_components=2).fit(X, y)
    intruder = np.where(np.arange(len(y)) == 0, "unheard-of", y)
    with pytest.raises(ValueError, match=r"labels the model was not fitted on.*unheard-of"):
        model.confusion(X, intruder)


# ---------------------------------------------------------------------------
# The Bayesian decision layer
# ---------------------------------------------------------------------------


def test_bayes_threshold_lands_where_the_densities_cross() -> None:
    """The threshold solves prior_in * N_in(t) == (1 - prior_in) * N_out(t)."""
    # Equal spreads and equal priors: the crossing is the midpoint, exactly.
    assert _bayes_threshold(mean_in=1.0, sd_in=0.2, mean_out=0.0, sd_out=0.2, prior_in=0.5) == pytest.approx(0.5)

    # A more common class gets a lower bar: it takes less evidence to be assigned to it.
    common = _bayes_threshold(mean_in=1.0, sd_in=0.2, mean_out=0.0, sd_out=0.2, prior_in=0.9)
    rare = _bayes_threshold(mean_in=1.0, sd_in=0.2, mean_out=0.0, sd_out=0.2, prior_in=0.1)
    assert common < 0.5 < rare

    # Unequal spreads: check the defining equation directly rather than a closed form.
    mean_in, sd_in, mean_out, sd_out, prior = 1.0, 0.10, 0.0, 0.35, 0.3
    threshold = _bayes_threshold(mean_in, sd_in, mean_out, sd_out, prior)
    assert mean_out < threshold < mean_in

    def density(t: float, mean: float, sd: float) -> float:
        return float(np.exp(-0.5 * ((t - mean) / sd) ** 2) / sd)

    assert prior * density(threshold, mean_in, sd_in) == pytest.approx(
        (1 - prior) * density(threshold, mean_out, sd_out), rel=1e-9
    )

    # A prior so small that neither crossing falls between the means: the midpoint stands in.
    assert _bayes_threshold(mean_in=1.0, sd_in=0.1, mean_out=0.0, sd_out=1.0, prior_in=1e-6) == pytest.approx(0.5)


def test_plsda_bayes_rule_handles_a_class_with_one_member() -> None:
    """A lone member has no spread, so the floor stands in for it, and it is still classified."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        np.vstack([rng.standard_normal((12, 4)), rng.standard_normal((12, 4)) + 3, rng.standard_normal((1, 4)) - 3])
    )
    y = np.array(["a"] * 12 + ["b"] * 12 + ["solo"])
    model = PLSDA(n_components=2, decision_rule="bayes").fit(X, y)

    assert model.class_statistics_.loc["solo", "sd_in"] == pytest.approx(np.sqrt(np.finfo(float).eps))
    assert np.isfinite(model.thresholds_).all()
    assert list(model.predict(X.iloc[[24]])) == ["solo"]


def test_plsda_bayes_rule_recovers_the_rare_class() -> None:
    """The point of the "bayes" rule: "max" cannot see that one class is rare.

    With a 1:9 split the rare class's indicator column is pulled toward zero, because
    nine rows in ten want it there. Argmax then hands almost everything to the common
    class: on the fixture below it misses six of the eight rare samples, for a rare-class
    sensitivity of 0.25, while looking respectable at 92.5% overall accuracy. That is the
    classic accuracy trap on unbalanced data.

    The Bayesian rule reads each column against its own in-class and out-of-class spread
    rather than against the other columns, so a value that is high *for the rare column*
    counts even though it is numerically small. It recovers seven of the eight, at the
    cost of three false alarms, and comes out ahead on accuracy too.
    """
    rng = np.random.default_rng(7)
    n_rare, n_common, n_features = 8, 72, 5
    direction = rng.standard_normal(n_features)
    X = pd.DataFrame(
        np.vstack(
            [
                rng.standard_normal((n_rare, n_features)) + 1.1 * direction,
                rng.standard_normal((n_common, n_features)),
            ]
        ),
        columns=[f"x{i}" for i in range(n_features)],
    )
    y = np.array(["rare"] * n_rare + ["common"] * n_common)

    by_max = PLSDA(n_components=2, decision_rule="max").fit(X, y)
    by_bayes = PLSDA(n_components=2, decision_rule="bayes").fit(X, y)

    # Both fitted the same PLS model: only the rule differs, so the indicators match.
    np.testing.assert_allclose(by_max.decision_function(X).to_numpy(), by_bayes.decision_function(X).to_numpy())
    # The empirical prior is 1:9, and the Bayesian rule uses it.
    assert by_bayes.priors_ == pytest.approx([n_common / len(y), n_rare / len(y)])
    # The rare class is found far more often, which is the whole point.
    assert by_max.sensitivity_["rare"] < 0.5
    assert by_bayes.sensitivity_["rare"] > 0.75
    # And not by trading away so many false alarms that the model is worse overall.
    assert by_bayes.accuracy_ > by_max.accuracy_


def test_plsda_priors_argument() -> None:
    """`priors` accepts the two names and an explicit vector, and rejects nonsense."""
    X, y = _separable(n_per_class=10, classes=("a", "b", "c"), seed=3)
    assert PLSDA(n_components=2, priors="uniform").fit(X, y).priors_ == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert PLSDA(n_components=2, priors=[0.2, 0.3, 0.5]).fit(X, y).priors_ == pytest.approx([0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match="priors must be 'empirical', 'uniform', or an array"):
        PLSDA(n_components=2, priors="whatever").fit(X, y)
    with pytest.raises(ValueError, match="priors has 2 entries but there are 3 classes"):
        PLSDA(n_components=2, priors=[0.5, 0.5]).fit(X, y)
    with pytest.raises(ValueError, match="priors must be positive and sum to 1"):
        PLSDA(n_components=2, priors=[0.2, 0.2, 0.2]).fit(X, y)


# ---------------------------------------------------------------------------
# Validation and the permutation test
# ---------------------------------------------------------------------------


def test_plsda_rejects_input_it_cannot_use() -> None:
    """Each rejection names what is wrong with the call."""
    X, y = _separable(n_per_class=10)
    with pytest.raises(ValueError, match=r"decision_rule must be one of"):
        PLSDA(n_components=2, decision_rule="vote").fit(X, y)
    with pytest.raises(ValueError, match=r"at least 2 classes to discriminate"):
        PLSDA(n_components=2).fit(X, np.array(["same"] * len(y)))
    with pytest.raises(ValueError, match=r"X has 20 rows but y has 5 labels"):
        PLSDA(n_components=2).fit(X, y[:5])
    # A missing label, as None or NaN, names no class: it used to fail on a KeyError or on sorting.
    for missing in (None, np.nan):
        gappy = y.astype(object)
        gappy[[3, 17]] = missing
        with pytest.raises(ValueError, match=r"^y has no label in 2 of its rows; every row needs a class\.$"):
            PLSDA(n_components=2).fit(X, gappy)
        with pytest.raises(ValueError, match=r"^y has no label in 2 of its rows; every row needs a class\.$"):
            PLSDA.compare_cv_criteria(X, gappy)
    numbers = np.where(y == "low", 0.0, 1.0)
    numbers[0] = np.nan
    with pytest.raises(ValueError, match=r"^y has no label in 1 of its rows"):
        PLSDA(n_components=2).fit(X, numbers)


@pytest.mark.slow
def test_plsda_permutation_test_tells_signal_from_chance() -> None:
    """Separable data is significant; the same shape of noise is not.

    This is the test the issue asks for, and it is the one PLS-DA most needs: training
    accuracy is 1.0 on both datasets below, so nothing about the fitted model
    distinguishes them. Only the cross-validated null does.
    """
    signal_x, signal_y = _separable(n_per_class=20, n_features=5, separation=3.0, seed=4)
    noise_x, noise_y = _separable(n_per_class=20, n_features=5, separation=0.0, seed=4)

    signal = PLSDA(n_components=2).fit(signal_x, signal_y)
    noise = PLSDA(n_components=2).fit(noise_x, noise_y)

    signal_result = signal.permutation_test(signal_x, signal_y, n_permutations=19, cv=4, random_state=0)
    noise_result = noise.permutation_test(noise_x, noise_y, n_permutations=19, cv=4, random_state=0)

    assert signal_result.p_value <= 0.05
    assert signal_result.observed > noise_result.observed
    assert noise_result.p_value > 0.05
    # The null sits near chance for a two-class problem in both cases; it is the observed
    # value that moves, which is what makes the comparison meaningful.
    assert noise_result.null.mean() == pytest.approx(0.5, abs=0.15)
    assert signal_result.n_permutations + signal_result.n_failed == 19
    # Floor of the p-value with the +1 correction on both sides.
    assert signal_result.p_value >= 1.0 / (1 + signal_result.n_permutations)


@pytest.mark.slow
def test_plsda_permutation_test_is_reproducible() -> None:
    """The same `random_state` gives the same null, a different one does not."""
    X, y = _separable(n_per_class=15, n_features=4, seed=5)
    model = PLSDA(n_components=2).fit(X, y)
    first = model.permutation_test(X, y, n_permutations=9, cv=3, random_state=42)
    again = model.permutation_test(X, y, n_permutations=9, cv=3, random_state=42)
    other = model.permutation_test(X, y, n_permutations=9, cv=3, random_state=7)

    np.testing.assert_allclose(first.null, again.null)
    assert first.p_value == again.p_value
    assert not np.array_equal(first.null, other.null)


def test_plsda_permutation_test_without_cross_validation_is_the_fast_path() -> None:
    """`cv=None` scores on the training data; documented as quick and weak."""
    X, y = _separable(n_per_class=12, n_features=4, seed=6)
    model = PLSDA(n_components=2).fit(X, y)
    result = model.permutation_test(X, y, n_permutations=5, cv=None, random_state=0)
    assert result.observed == pytest.approx(model.accuracy_)
    assert result.null.size == 5
    # A model that memorises the real labels memorises shuffled ones too, which is exactly
    # why this path is not the default: the null sits far above chance.
    assert result.null.mean() > 0.5


def _null_fits_that_fail(monkeypatch: pytest.MonkeyPatch, every: int) -> None:
    """Make every `every`-th null fit raise NotEnoughVarianceError, as a collapsed shuffle does.

    Call 0 is the observed fit, which always succeeds; with ``every=1`` every null fit fails.
    """
    original = PLSDA._clone_unfitted
    calls = itertools.count()

    def clone_that_may_not_fit(self: PLSDA) -> PLSDA:
        fresh = original(self)
        call = next(calls)
        if call > 0 and call % every == 0:

            def refuse(*_args: object, **_kwargs: object) -> PLSDA:
                raise NotEnoughVarianceError("simulated: nothing left to fit")

            fresh.fit = refuse  # type: ignore[method-assign]
        return fresh

    monkeypatch.setattr(PLSDA, "_clone_unfitted", clone_that_may_not_fit)


def test_plsda_permutation_test_drops_draws_that_cannot_be_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shuffle that cannot be fitted is dropped from the null, not scored as zero, and counted."""
    X, y = _separable(n_per_class=12, n_features=4, seed=6)
    model = PLSDA(n_components=2).fit(X, y)
    _null_fits_that_fail(monkeypatch, every=2)
    with pytest.warns(
        SpecificationWarning,
        match=r"^5 of 10 permutations could not be fitted at n_components=2 and were dropped; "
        r"the p-value is over the remaining 5\.",
    ):
        result = model.permutation_test(X, y, n_permutations=10, cv=None, random_state=0)
    assert (result.n_permutations, result.n_failed) == (5, 5)
    assert result.null.size == 5


def test_plsda_permutation_test_refuses_when_no_draw_can_be_fitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every shuffle unfittable there is no null to compare against, and the error says what to change."""
    X, y = _separable(n_per_class=12, n_features=4, seed=6)
    model = PLSDA(n_components=2).fit(X, y)
    _null_fits_that_fail(monkeypatch, every=1)
    with pytest.raises(
        RuntimeError,
        match=r"^Every one of the 10 permutations failed to fit at n_components=2\. There is no null distribution "
        r"to compare against; refit with fewer components, or with more samples per fold\.$",
    ):
        model.permutation_test(X, y, n_permutations=10, cv=None, random_state=0)


def test_plsda_score_weights_each_sample() -> None:
    """Weighted accuracy counts each sample by its weight: all on the right ones gives 1, on the wrong ones 0."""
    X, y = _separable(n_per_class=20, separation=1.0, seed=13)
    model = PLSDA(n_components=2).fit(X, y)
    right = model.predict(X) == y
    assert 0 < right.sum() < len(y)  # some of each, so the weights have something to choose

    assert model.score(X, y, sample_weight=np.ones(len(y))) == pytest.approx(model.score(X, y))
    assert model.score(X, y, sample_weight=right.astype(float)) == pytest.approx(1.0)
    assert model.score(X, y, sample_weight=(~right).astype(float)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------


def test_plsda_roc_auc() -> None:
    """AUC comes from the continuous indicator, so the decision rule cannot change it."""
    # Deliberately only partly separable, so the AUC is strictly inside (0.5, 1) and the
    # assertions below are about the number rather than about a saturated 1.0.
    X, y = _separable(n_per_class=25, separation=0.45, seed=8)
    by_max = PLSDA(n_components=2).fit(X, y)
    by_bayes = PLSDA(n_components=2, decision_rule="bayes").fit(X, y)
    auc = by_max.roc_auc(X, y)
    assert 0.5 < auc < 1.0
    assert auc == pytest.approx(by_bayes.roc_auc(X, y))

    # Either class may be named positive and the answer is the same, which is not a
    # coincidence: the one-hot columns sum to 1 in every row, and PLS predictions are
    # affine in the scores, so for two classes the predicted columns sum to exactly 1 as
    # well. One indicator is therefore the exact mirror of the other, and mirroring both
    # the score and the label leaves the ranking, and the AUC, untouched.
    scores = by_max.decision_function(X)
    assert scores.sum(axis=1).to_numpy() == pytest.approx(np.ones(len(y)))
    assert by_max.roc_auc(X, y, positive_class=by_max.classes_[0]) == pytest.approx(
        by_max.roc_auc(X, y, positive_class=by_max.classes_[1])
    )


def test_plsda_roc_auc_needs_a_positive_class_when_multiclass() -> None:
    """With three classes there is no default positive class to assume."""
    X, y = _separable(n_per_class=15, classes=("a", "b", "c"), seed=9)
    model = PLSDA(n_components=2).fit(X, y)
    with pytest.raises(ValueError, match=r"roc_auc needs `positive_class` when there are 3 classes"):
        model.roc_auc(X, y)
    assert 0.5 < model.roc_auc(X, y, positive_class="b") <= 1.0
    with pytest.raises(ValueError, match=r"positive_class 'z' is not one of"):
        model.roc_auc(X, y, positive_class="z")


# ---------------------------------------------------------------------------
# sklearn contract, and everything inherited from PLS
# ---------------------------------------------------------------------------


def test_plsda_satisfies_the_sklearn_classifier_contract() -> None:
    """A classifier, not a regressor, despite inheriting PLS's RegressorMixin."""
    model = PLSDA(n_components=2, decision_rule="bayes", priors="uniform")
    assert is_classifier(model)
    assert not is_regressor(model)

    assert clone(model).get_params()["decision_rule"] == "bayes"
    assert model.set_params(n_components=3).n_components == 3
    assert set(model.get_params()) >= {"n_components", "decision_rule", "priors", "scale", "max_iter"}

    X, y = _separable(n_per_class=20, seed=10)
    pipe = Pipeline([("scale", MCUVScaler()), ("da", PLSDA(n_components=2))])
    scores = cross_val_score(pipe, X, y, cv=4)
    assert scores.shape == (4,)
    assert scores.min() > 0.5  # separable data; anything at chance means the wiring broke
    # cross_val_score with no `scoring=` goes through `score`, which must be accuracy and
    # not the R2 that PLS.score would return for the same call.
    assert pipe.fit(X, y).score(X, y) == pytest.approx(1.0)


def test_plsda_constructor_forwards_every_pls_parameter() -> None:
    """`PLSDA.__init__` must leave nothing `PLS.__init__` owns unset.

    It names the base explicitly (`PLS.__init__(self, ...)`) rather than going through
    `super()`, because `ClassifierMixin` sits in front of `PLS` in the MRO and defines no
    `__init__`, so the two forms are the same call and only the explicit one says which
    constructor runs. The risk that trades for is a parameter added to `PLS` and not
    forwarded here, so this reads the base signature rather than a list that would go
    stale the moment `PLS` grows an argument.
    """

    def unlike(default: object) -> object:
        """Build a value that is definitely not `default`, so a dropped forward cannot hide.

        Comparing against PLS's own defaults would not do: a parameter that PLSDA stops
        forwarding falls back to exactly that default, so the check would pass on the
        bug it exists to catch. `bool` is tested before `int` because it is one.
        """
        if isinstance(default, bool):
            return not default
        if isinstance(default, int):
            return default + 7
        if isinstance(default, float):
            return default * 10.0 + 1.0
        if isinstance(default, str):
            return default + "-changed"
        if default is None:
            return {"md_method": "pmp"}
        raise AssertionError(f"extend `unlike` to cover {type(default).__name__}")

    empty = inspect.Parameter.empty
    asked: dict[str, object] = {
        name: 3 if parameter.default is empty else unlike(parameter.default)
        for name, parameter in inspect.signature(PLS.__init__).parameters.items()
        if name != "self" and parameter.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert asked, "PLS.__init__ takes no parameters; this test is checking nothing"

    model = PLSDA(**asked)  # a TypeError here means PLSDA no longer accepts one of them
    for name, expected in asked.items():
        assert getattr(model, name, None) == expected, f"PLSDA.__init__ did not forward {name!r} to PLS"


def test_plsda_keeps_every_pls_diagnostic() -> None:
    """The whole point of subclassing: the latent-variable view is still there."""
    X, y = _separable(n_per_class=20, n_features=7, seed=11)
    model = PLSDA(n_components=3).fit(X, y)

    assert isinstance(model, PLS)
    assert model.scores_.shape == (len(y), 3)
    assert model.x_loadings_.shape == (7, 3)
    assert list(model.x_loadings_.index) == list(X.columns)
    assert model.vip().shape == (7,)
    assert np.isfinite(model.hotellings_t2_.to_numpy()).all()
    assert model.spe_limit(0.95) > 0
    assert 0 < float(model.r2_cumulative_.iloc[-1]) <= 1.0

    # The Y-block here is the class indicators, so the explained-variance plot must say
    # "Y-variance" as it does for PLS, not "X-variance" as an exact-name check would give.
    assert "Y-variance" in model.explained_variance_plot().layout.title.text


def test_plsda_pickles_and_keeps_its_decision_layer() -> None:
    """A fitted classifier survives a round trip, decision layer included."""
    import pickle

    X, y = _separable(n_per_class=15, seed=12)
    model = PLSDA(n_components=2, decision_rule="bayes").fit(X, y)
    reloaded = pickle.loads(pickle.dumps(model))  # noqa: S301 - our own object
    np.testing.assert_array_equal(reloaded.predict(X), model.predict(X))
    np.testing.assert_allclose(reloaded.predict_proba(X), model.predict_proba(X))
    pd.testing.assert_frame_equal(reloaded.class_statistics_, model.class_statistics_)


# ---------------------------------------------------------------------------
# Plot and real data
# ---------------------------------------------------------------------------


def test_confusion_matrix_plot() -> None:
    """The heat map carries the counts, or the row fractions when asked."""
    X, y = _separable(n_per_class=20, separation=1.0, seed=13)
    model = PLSDA(n_components=2).fit(X, y)

    fig = model.confusion_matrix_plot()
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    np.testing.assert_allclose(np.asarray(fig.data[0].z), model.confusion_matrix_.to_numpy())
    assert fig.layout.yaxis.autorange == "reversed"  # printed top-to-bottom

    normalised = model.confusion_matrix_plot(settings={"normalize": True})
    assert np.asarray(normalised.data[0].z).sum(axis=1) == pytest.approx(np.ones(model.n_classes_))

    with pytest.raises(ValueError, match="Model is not fitted"):
        confusion_matrix_plot(PLSDA(n_components=2))


def test_confusion_matrix_plot_of_a_supplied_matrix() -> None:
    """A matrix passed in, say a held-out one, is drawn as given, onto the figure passed in.

    The model need not be fitted then, and `show_values=False` leaves the cells unlabelled.
    """
    go = pytest.importorskip("plotly.graph_objects")
    held_out = pd.DataFrame([[5, 1], [2, 7]], index=["high", "low"], columns=["high", "low"])
    canvas = go.Figure()
    fig = confusion_matrix_plot(PLSDA(n_components=2), held_out, settings={"show_values": False}, fig=canvas)
    assert fig is canvas
    np.testing.assert_allclose(np.asarray(fig.data[0].z), held_out.to_numpy())
    assert fig.data[0].text is None


@pytest.mark.dataset
def test_plsda_on_the_cheddar_cheese_data() -> None:
    """A real dataset: predict whether a cheese tastes above or below the median.

    Thirty cheeses, three chemical measurements, and a tasting score. Splitting the score
    at its median turns the package's own regression example into the classification
    question a process engineer actually asks: is this batch a good one, and which
    measurement says so.
    """
    folder = pathlib.Path("src/process_improve/datasets/multivariate")
    path = folder / "cheddar-cheese.csv"
    if not path.exists():
        pytest.skip("cheddar-cheese.csv fixture not present")
    data = pd.read_csv(path, index_col=0)
    X = data[["Acetic", "H2S", "Lactic"]]
    y = np.where(data["Taste"] > data["Taste"].median(), "tasty", "bland")

    model = PLSDA(n_components=2).fit(X, y)
    assert list(model.classes_) == ["bland", "tasty"]
    # Real, noisy, correlated data: not perfectly separable, but clearly better than
    # chance, and the AUC says the ranking is sound.
    assert 0.7 < model.accuracy_ < 1.0
    assert model.roc_auc(X, y) > 0.85
    # H2S is the strongest single predictor of taste in this dataset, which the regression
    # write-up also finds; VIP should rank it top.
    assert model.vip().idxmax() == "H2S"


# ---------------------------------------------------------------------------
# Checked by hand: each number against a calculation that does not use PLSDA
# ---------------------------------------------------------------------------

_FLOOR = float(np.sqrt(np.finfo(float).eps))


def _assert_near(actual: object, expected: object, rtol: float = 1e-7) -> None:
    """Assert agreement to ``rtol`` of each value, or of the largest one for a value near 0.

    Two iterative fits, ours and sklearn's, agree to their convergence tolerances; the
    absolute floor is that tolerance scaled to the largest value compared.
    """
    expected = np.asarray(expected, dtype=float)
    finite = np.abs(expected[np.isfinite(expected)])
    floor = rtol * (float(finite.max()) if finite.size else 0.0)
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=rtol, atol=floor)


def _sklearn_indicators(x_train: np.ndarray, labels: np.ndarray, x_new: np.ndarray, n_components: int) -> np.ndarray:
    """Predicted one-hot indicators from sklearn's PLSRegression, autoscaled as PLSDA is, converged tightly."""
    classes = np.unique(labels)
    indicators = (labels[:, None] == classes[None, :]).astype(float)
    model = PLSRegression(n_components=n_components, scale=True, tol=1e-14, max_iter=10_000)
    return model.fit(np.asarray(x_train), indicators).predict(np.asarray(x_new))


def _crossing(mean_in: float, sd_in: float, mean_out: float, sd_out: float, prior: float) -> float:
    """Where prior * N_in(t) == (1 - prior) * N_out(t) between the means, by bracketing; else the midpoint."""
    sd_in, sd_out = max(sd_in, _FLOOR), max(sd_out, _FLOOR)

    def excess(t: float) -> float:
        return np.log(prior) + norm.logpdf(t, mean_in, sd_in) - np.log1p(-prior) - norm.logpdf(t, mean_out, sd_out)

    lo, hi = sorted((mean_in, mean_out))
    if excess(lo) * excess(hi) > 0:  # no crossing between the means
        return 0.5 * (lo + hi)
    return float(brentq(excess, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))


def _class_statistics_by_hand(fitted: np.ndarray, labels: np.ndarray, priors: np.ndarray) -> pd.DataFrame:
    """Fit the Gaussians of the decision layer to the in-sample indicators: sample sd, 0 for one member."""
    classes = np.unique(labels)
    rows = []
    for position, klass in enumerate(classes):
        inside = labels == klass
        member, other = fitted[inside, position], fitted[~inside, position]
        sd_in = float(np.std(member, ddof=1)) if member.size > 1 else 0.0
        sd_out = float(np.std(other, ddof=1)) if other.size > 1 else 0.0
        rows.append(
            {
                "mean_in": member.mean(),
                "sd_in": max(sd_in, _FLOOR),
                "mean_out": other.mean(),
                "sd_out": max(sd_out, _FLOOR),
                "prior": priors[position],
                "threshold": _crossing(member.mean(), sd_in, other.mean(), sd_out, priors[position]),
            }
        )
    return pd.DataFrame(rows, index=pd.Index(classes, name="class"))


def _log_evidence_by_hand(values: np.ndarray, stats: pd.DataFrame) -> np.ndarray:
    """log(prior) plus the log ratio of the in-class to the out-of-class normal density, per column."""
    return np.column_stack(
        [
            np.log(row.prior)
            + norm.logpdf(values[:, g], row.mean_in, row.sd_in)
            - norm.logpdf(values[:, g], row.mean_out, row.sd_out)
            for g, row in enumerate(stats.itertuples())
        ]
    )


def _confusion_by_hand(truth: np.ndarray, predicted: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    counts = np.array([[int(np.sum((truth == a) & (predicted == b))) for b in classes] for a in classes])
    return pd.DataFrame(counts, index=pd.Index(classes, name="true"), columns=pd.Index(classes, name="predicted"))


def _three_classes(seed: int = 21) -> tuple[pd.DataFrame, np.ndarray]:
    """Three overlapping classes of 7, 6 and 2 rows, in no sorted order, with rows labelled by name."""
    rng = np.random.default_rng(seed)
    sizes = {"c": 7, "a": 6, "b": 2}
    centres = {"a": [1.0, 0.0, 0.5, 0.0], "b": [0.0, 1.5, 0.0, -0.5], "c": [-1.0, -0.5, 0.0, 0.5]}
    blocks = [rng.standard_normal((n, 4)) + centres[name] for name, n in sizes.items()]
    labels = np.concatenate([[name] * n for name, n in sizes.items()])
    order = rng.permutation(labels.size)
    X = pd.DataFrame(
        np.vstack(blocks)[order], columns=["p", "q", "r", "s"], index=[f"row{i}" for i in range(labels.size)]
    )
    return X, labels[order]


@pytest.mark.parametrize("rule", ["max", "bayes"])
def test_plsda_fit_matches_sklearn_and_the_decision_layer_by_hand(rule: str) -> None:
    """Every fitted number, from sklearn's PLSRegression on the one-hot labels and the formulas written out."""
    X, labels = _three_classes()
    model = PLSDA(n_components=2, decision_rule=rule).fit(X, labels)
    classes = np.array(["a", "b", "c"])
    fitted = _sklearn_indicators(X, labels, X, 2)
    priors = np.array([6, 2, 7]) / 15

    np.testing.assert_array_equal(model.classes_, classes)
    assert model.n_classes_ == 3
    _assert_near(model.priors_, priors, rtol=1e-15)
    # The one-hot block PLS is fitted to: a column per class, named for it, rows as X's.
    assert model.y_loadings_.index.tolist() == ["a", "b", "c"]
    assert model.predictions_.index.equals(X.index)
    assert model.y_scores_.index.equals(X.index)
    _assert_near(model.predictions_, fitted, rtol=1e-6)

    stats = _class_statistics_by_hand(fitted, labels, priors)
    pd.testing.assert_frame_equal(model.class_statistics_, stats, rtol=1e-6)
    pd.testing.assert_series_equal(model.thresholds_, stats["threshold"], rtol=1e-6)
    # Class b has two members: its spread is their sample sd, not the floor of a lone member.
    assert model.class_statistics_.loc["b", "sd_in"] > 100 * _FLOOR

    evidence = _log_evidence_by_hand(fitted, stats)
    by_max, by_bayes = classes[fitted.argmax(axis=1)], classes[evidence.argmax(axis=1)]
    assert (by_max != by_bayes).any()  # the two rules disagree somewhere on this fixture
    predicted = by_max if rule == "max" else by_bayes

    pd.testing.assert_frame_equal(model.confusion_matrix_, _confusion_by_hand(labels, predicted, classes))
    assert model.accuracy_ == pytest.approx((predicted == labels).mean(), abs=1e-15)
    actual = np.array([np.sum(labels == k) for k in classes])
    hits = np.array([np.sum((labels == k) & (predicted == k)) for k in classes])
    false_alarms = np.array([np.sum((labels != k) & (predicted == k)) for k in classes])
    index = pd.Index(classes, name="class")
    pd.testing.assert_series_equal(model.sensitivity_, pd.Series(hits / actual, index=index, name="sensitivity"))
    expected_specificity = (len(labels) - actual - false_alarms) / (len(labels) - actual)
    pd.testing.assert_series_equal(model.specificity_, pd.Series(expected_specificity, index=index, name="specificity"))

    scores = model.decision_function(X)
    pd.testing.assert_index_equal(scores.columns, pd.Index(classes, name="class"))
    assert scores.index.equals(X.index)
    _assert_near(scores, fitted, rtol=1e-6)
    np.testing.assert_array_equal(model.predict(X), predicted)
    _assert_near(model.predict_proba(X), softmax(evidence, axis=1), rtol=1e-5)


def test_plsda_predict_proba_survives_evidence_that_would_overflow() -> None:
    """A row far out along a class gives log evidence past 709, where exp overflows to inf.

    Each row's evidence is shifted by its own largest value before exp, so that row is one-hot,
    and the ordinary row beside it keeps its own posteriors.
    """
    X, y = _separable(n_per_class=12, n_features=3, separation=2.0, seed=31)
    model = PLSDA(n_components=2).fit(X, y)
    far = X.iloc[[0, 1]].copy()
    far.iloc[0] = 60.0 * X.iloc[0]
    stats = _class_statistics_by_hand(_sklearn_indicators(X, y, X, 2), y, np.array([0.5, 0.5]))
    evidence = _log_evidence_by_hand(_sklearn_indicators(X, y, far, 2), stats)
    assert np.abs(evidence[0]).min() > np.log(np.finfo(float).max)  # exp of either is inf or 0
    proba = model.predict_proba(far)
    assert np.isfinite(proba).all()
    _assert_near(proba, softmax(evidence, axis=1), rtol=1e-5)


def test_gaussian_log_density_and_the_class_spread_by_hand() -> None:
    """The log density drops only the constant 0.5 log(2 pi); one value has a spread of 0."""
    x = np.array([-1.5, 0.25, 4.0])
    _assert_near(_gaussian_log_density(x, 0.75, 1.6), norm.logpdf(x, 0.75, 1.6) + 0.5 * np.log(2 * np.pi), rtol=1e-14)
    assert _mean_and_sd(np.array([2.5])) == (2.5, 0.0)
    assert _mean_and_sd(np.array([1.0, 4.0])) == pytest.approx((2.5, np.sqrt(4.5)), rel=1e-15)
    assert _mean_and_sd(np.array([1.0, 2.0, 6.0])) == pytest.approx((3.0, np.sqrt(7.0)), rel=1e-15)


@pytest.mark.parametrize(
    ("mean_in", "sd_in", "mean_out", "sd_out", "prior"),
    [
        (3.0, 0.2, 1.5, 0.6, 0.3),  # the in-class spread is the smaller: the first root is between
        (3.0, 0.6, 1.5, 0.2, 0.3),  # the larger: the second root is between
        (1.5, 0.3, 3.0, 0.5, 0.7),  # the class sits below the rest
        (-2.0, 0.4, 0.5, 0.25, 0.55),
        (1.0, 0.5, 0.0, 1.0, 0.2),
    ],
)
def test_bayes_threshold_is_the_crossing_of_the_weighted_densities(
    mean_in: float, sd_in: float, mean_out: float, sd_out: float, prior: float
) -> None:
    assert _bayes_threshold(mean_in, sd_in, mean_out, sd_out, prior) == pytest.approx(
        _crossing(mean_in, sd_in, mean_out, sd_out, prior), abs=1e-12
    )


def test_bayes_threshold_at_its_edges() -> None:
    """The midpoint, the linear root and the floor, each checked by hand."""
    # Equal spreads: one root, t = midpoint + sd^2 ln((1 - p) / p) / (m_in - m_out).
    expected = 2.0 + 0.36 * np.log(0.7 / 0.3) / 2.0
    assert _bayes_threshold(3.0, 0.6, 1.0, 0.6, 0.3) == pytest.approx(expected, rel=1e-14)
    # Equal spreads and equal means: nothing separates the classes, so the midpoint, the mean.
    assert _bayes_threshold(2.5, 0.4, 2.5, 0.4, 0.3) == 2.5
    # A prior so small the weighted densities never cross: the midpoint, and no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _bayes_threshold(3.0, 0.1, 1.0, 1.0, 1e-6) == 2.0
        assert _bayes_threshold(1.0, 0.1, 3.0, 1.0, 1e-6) == 2.0
    # A spread of 0 is read as the floor sqrt(eps), as a lone class member's is.
    assert _bayes_threshold(1.0, 0.0, 0.0, 0.3, 0.4) == _bayes_threshold(1.0, _FLOOR, 0.0, 0.3, 0.4)
    assert _bayes_threshold(1.0, 0.3, 0.0, 0.0, 0.4) == _bayes_threshold(1.0, 0.3, 0.0, _FLOOR, 0.4)


def test_bayes_threshold_takes_the_linear_root_while_the_quadratic_term_is_rounding() -> None:
    """|quad| <= sqrt(eps) * max(|lin|, 1) is read as equal spreads, and the root is -const / lin.

    With spreads equal to 1e-10, lin = 1e4 and quad = -1e-6: the quadratic formula would lose
    1.5e-7 of the crossing to cancellation, the linear root keeps it to 3e-11.
    """
    near_equal = (1.0, 0.01, 0.0, 0.01 * (1 + 1e-10), 0.3)
    assert _bayes_threshold(*near_equal) == pytest.approx(_crossing(*near_equal), abs=1e-9)
    # With lin near 1 and quad = 1.5 sqrt(eps), past the bound, the quadratic root is the crossing
    # to 1e-9; the linear root would be 1.9e-7 away from it.
    past_bound = (3.0, 1.0, 2.0, float(np.sqrt(1 / (2 * (0.5 + 1.5 * _FLOOR)))), 0.4)
    assert _bayes_threshold(*past_bound) == pytest.approx(_crossing(*past_bound), abs=2e-8)
    # At exactly the bound: an sd_out whose quad is 2**-26 to the bit, with |lin| < 1. The rule
    # includes the bound, so the root is -const / lin, which the quadratic formula misses by 1.3e-7.
    sd_out = 0.9999999850988391
    a, b = 1.0 / (2.0 * sd_out**2), 0.5
    assert a - b == 2.0**-26
    lin = 2.0 * (b * 1.5 - a * 1.0)
    const = a * 1.0 - b * 2.25 + np.log(sd_out) + np.log(0.4 / 0.6)
    assert _bayes_threshold(1.5, 1.0, 1.0, sd_out, 0.4) == pytest.approx(-const / lin, rel=1e-15)
    # |lin| of exactly sqrt(eps) still reads as no slope at all, so the midpoint, not -const / lin = 2.7e7.
    assert _bayes_threshold(2.0**-26, 1.0, 0.0, 1.0, 0.4) == 2.0**-27


def test_bayes_threshold_counts_a_crossing_exactly_on_a_mean_as_between() -> None:
    """With sd ratio 2, equal priors and the means sqrt(ln(2) / 2) apart, the crossing is a mean.

    delta squares to ln(2) / 2 to the bit, so the constant term is 0 exactly and one root is 0.
    """
    delta = np.float64(0.5887050112577373)
    assert delta * delta == np.log(2.0) / 2
    assert _bayes_threshold(delta, 0.5, np.float64(0.0), 1.0, 0.5) == 0.0  # at mean_out, the lower mean
    assert _bayes_threshold(np.float64(0.0), 1.0, -delta, 0.5, 0.5) == 0.0  # at mean_in, the upper mean


def test_plsda_scores_and_rates_on_held_out_rows_by_hand() -> None:
    """Weighted accuracy, and the rates of a class absent from the rows and of the only class in them."""
    X, labels = _three_classes()
    model = PLSDA(n_components=2).fit(X, labels)
    predicted = np.array(["a", "b", "c"])[_sklearn_indicators(X, labels, X, 2).argmax(axis=1)]
    weights = np.linspace(0.5, 2.0, len(labels))
    assert model.score(X, labels, sample_weight=weights) == pytest.approx(
        np.sum(weights * (predicted == labels)) / weights.sum(), rel=1e-14
    )

    only_a = labels == "a"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 0 / 0 is the NaN of a rate with no rows to count
        held_out = model.confusion(X[only_a], labels[only_a])
    classes = np.array(["a", "b", "c"])
    pd.testing.assert_frame_equal(held_out.matrix, _confusion_by_hand(labels[only_a], predicted[only_a], classes))
    index = pd.Index(classes, name="class")
    a_hits = np.mean(predicted[only_a] == "a")
    pd.testing.assert_series_equal(
        held_out.sensitivity, pd.Series([a_hits, np.nan, np.nan], index=index, name="sensitivity")
    )
    b_kept = np.mean(predicted[only_a] != "b")
    c_kept = np.mean(predicted[only_a] != "c")
    pd.testing.assert_series_equal(
        held_out.specificity, pd.Series([np.nan, b_kept, c_kept], index=index, name="specificity")
    )
    assert held_out.accuracy == pytest.approx(a_hits, abs=1e-15)
    # One row against two labels: refused, not truncated to the one row.
    with pytest.raises(ValueError, match="zip"):
        model.confusion(X.iloc[:1], labels[:2])


def test_plsda_roc_auc_by_hand() -> None:
    """The AUC is the share of (positive, negative) pairs the indicator ranks the right way round."""
    X, y = _separable(n_per_class=15, separation=0.5, seed=17)
    model = PLSDA(n_components=2).fit(X, y)
    column = _sklearn_indicators(X, y, X, 2)[:, 1]  # classes_[1], "low", is the default positive class
    positive, negative = column[y == "low"], column[y == "high"]
    pairs = positive[:, None] - negative[None, :]
    auc = float(np.mean(pairs > 0) + 0.5 * np.mean(pairs == 0))
    assert 0.5 < auc < 1.0
    assert model.roc_auc(X, y) == pytest.approx(auc, abs=1e-15)

    X3, y3 = _separable(n_per_class=10, classes=("a", "b", "c"), seed=9)
    three = PLSDA(n_components=2).fit(X3, y3)
    with pytest.raises(
        ValueError,
        match=r"^roc_auc needs `positive_class` when there are 3 classes: "
        r"the result is one class against all the others\.$",
    ):
        three.roc_auc(X3, y3)


def _cv_accuracy_by_hand(X: np.ndarray, labels: np.ndarray, splits: list, n_components: int) -> float:
    correct = 0
    for train, test in splits:
        indicators = _sklearn_indicators(X[train], labels[train], X[test], n_components)
        correct += int(np.sum(np.unique(labels[train])[indicators.argmax(axis=1)] == labels[test]))
    return correct / len(labels)


@pytest.mark.slow
def test_plsda_permutation_test_at_its_defaults_by_hand() -> None:
    """99 shuffles, 5 stratified folds each: the observed accuracy, the null and the p-value by hand.

    The accuracies are multiples of 1/16, so some shuffles tie the observed one, and a tie counts
    against it.
    """
    X, y = _separable(n_per_class=8, n_features=3, separation=0.6, seed=41)
    values = X.to_numpy()
    model = PLSDA(n_components=1).fit(X, y)
    result = model.permutation_test(X, y, random_state=5)

    def cv_accuracy(target: np.ndarray) -> float:
        return _cv_accuracy_by_hand(values, target, list(StratifiedKFold(5).split(values, target)), 1)

    rng = np.random.default_rng(5)
    null = np.array([cv_accuracy(rng.permutation(y)) for _ in range(99)])
    observed = cv_accuracy(y)
    assert (null == observed).any()  # ties, which count as at least as good
    assert result.observed == observed
    np.testing.assert_array_equal(result.null, null)
    assert result.p_value == (1 + np.sum(null >= observed)) / 100
    assert (result.n_permutations, result.n_failed) == (99, 0)


def test_plsda_permutation_test_with_its_other_splitters_by_hand() -> None:
    """Three stratified folds, a KFold given as a splitter, an array X, and no cross-validation."""
    X, y = _separable(n_per_class=6, n_features=3, separation=0.8, seed=43)
    values = X.to_numpy()
    model = PLSDA(n_components=1).fit(X, y)

    def by_hand(splitter: object, seed: int) -> tuple[float, np.ndarray]:
        def accuracy(target: np.ndarray) -> float:
            return _cv_accuracy_by_hand(values, target, list(splitter.split(values, target)), 1)

        rng = np.random.default_rng(seed)
        null = np.array([accuracy(rng.permutation(y)) for _ in range(4)])
        return accuracy(y), null

    for cv, splitter in [
        (3, StratifiedKFold(3)),
        (KFold(3, shuffle=True, random_state=2), KFold(3, shuffle=True, random_state=2)),
    ]:
        observed, null = by_hand(splitter, seed=1)
        for data in (X, values):
            result = model.permutation_test(data, y, n_permutations=4, cv=cv, random_state=1)
            assert result.observed == observed
            np.testing.assert_array_equal(result.null, null)

    # cv=None: the training accuracy of a fit to all rows, for the labels and for each shuffle.
    rng = np.random.default_rng(3)

    def training_accuracy(target: np.ndarray) -> float:
        return float(
            np.mean(np.unique(target)[_sklearn_indicators(values, target, values, 1).argmax(axis=1)] == target)
        )

    null = np.array([training_accuracy(rng.permutation(y)) for _ in range(4)])
    result = model.permutation_test(values, y, n_permutations=4, cv=None, random_state=3)
    assert result.observed == training_accuracy(y)
    np.testing.assert_array_equal(result.null, null)


def test_plsda_sklearn_tags_and_defaults_are_a_classifiers_on_pls_settings() -> None:
    """No regressor tags left behind, one output, and every PLS setting defaulting as PLS's does."""
    tags = PLSDA(n_components=2).__sklearn_tags__()
    assert tags.estimator_type == "classifier"
    assert tags.regressor_tags is None
    assert tags.target_tags.multi_output is False
    params = PLSDA(n_components=2).get_params()
    for name, parameter in inspect.signature(PLS.__init__).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            assert params[name] == parameter.default, name


def test_plsda_refuses_every_question_until_fitted() -> None:
    X, y = _separable(n_per_class=5)
    model = PLSDA(n_components=1)
    for ask in (
        model.decision_function,
        model.predict,
        model.predict_proba,
        lambda rows: model.confusion(rows, y),
        lambda rows: model.roc_auc(rows, y),
        lambda rows: model.permutation_test(rows, y, n_permutations=1),
    ):
        with pytest.raises(NotFittedError, match="This PLSDA instance is not fitted yet"):
            ask(X)


def test_plsda_fits_with_sample_weights_as_pls_does_on_the_indicators() -> None:
    X, labels = _three_classes()
    weights = np.linspace(0.2, 3.0, len(labels))
    indicators = pd.DataFrame((labels[:, None] == np.array(["a", "b", "c"])).astype(float), index=X.index)
    weighted = PLSDA(n_components=2).fit(X, labels, sample_weight=weights)
    reference = PLS(n_components=2).fit(X, indicators, sample_weight=weights)
    _assert_near(weighted.predictions_, reference.predictions_, rtol=1e-9)
    assert not np.allclose(weighted.predictions_, PLSDA(n_components=2).fit(X, labels).predictions_)


def test_plsda_priors_by_hand() -> None:
    """Exact fractions are read as floats; a prior of 0 or below is refused even when they sum to 1."""
    X, y = _separable(n_per_class=10, classes=("a", "b", "c"), seed=3)
    model = PLSDA(n_components=2, priors=[Fraction(1, 6), Fraction(1, 3), Fraction(1, 2)]).fit(X, y)
    assert model.priors_.dtype == np.float64
    np.testing.assert_array_equal(model.priors_, [1 / 6, 1 / 3, 1 / 2])
    with pytest.raises(ValueError, match=r"^priors must be positive and sum to 1; got \[0\.0, 0\.5, 0\.5\]\.$"):
        PLSDA(n_components=2, priors=[0.0, 0.5, 0.5]).fit(X, y)
    with pytest.raises(ValueError, match=r"^priors must be positive and sum to 1; got \[-0\.5, 0\.75, 0\.75\]\.$"):
        PLSDA(n_components=2, priors=[-0.5, 0.75, 0.75]).fit(X, y)


def _indicator_frame(labels: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Build the one-hot class indicators, a column per sorted class named as text."""
    classes = np.unique(labels)
    return pd.DataFrame(
        (labels[:, None] == classes[None, :]).astype(float), index=index, columns=[str(c) for c in classes]
    )


def _assert_same_splits(got: list, expected: list) -> None:
    assert len(got) == len(expected)
    for (train, test), (train_expected, test_expected) in zip(got, expected, strict=True):
        np.testing.assert_array_equal(train, train_expected)
        np.testing.assert_array_equal(test, test_expected)


def test_plsda_cross_validation_runs_pls_on_the_class_indicators() -> None:
    """PLS's cross-validation methods take PLSDA's labels and run on the one-hot indicators it fits.

    They used to read the labels as the numeric response: text labels could not be read as
    floats, and labels 0 and 1 gave a shape error against the two indicator columns.
    """
    X, y = _separable(n_per_class=8, n_features=4, separation=1.0, classes=("a", "b", "c"), seed=51)
    indicators = _indicator_frame(y, X.index)
    settings = {"max_components": 2, "n_permutations": 9, "n_cv_permutations": 0, "random_state": 0}

    got, expected = PLSDA.compare_cv_criteria(X, y, **settings), PLS.compare_cv_criteria(X, indicators, **settings)
    pd.testing.assert_frame_equal(got.table, expected.table)
    pd.testing.assert_frame_equal(got.recommendations, expected.recommendations)
    _assert_same_splits(got.cv_splits, expected.cv_splits)
    assert type(got.global_model) is PLS

    pv = PLSDA.pseudo_validation_set(X, y, n_components=2, random_state=0)
    pd.testing.assert_frame_equal(pv.Y_pv, indicators)
    pd.testing.assert_frame_equal(
        pv.X_pv, PLS.pseudo_validation_set(X, indicators, n_components=2, random_state=0).X_pv
    )

    selection = {"max_components": 3, "n_repeats": 2, "random_state": 0}
    chosen = PLSDA.select_n_components(X, y, **selection)
    reference = PLS.select_n_components(X, indicators, **selection)
    assert chosen.n_components == reference.n_components
    pd.testing.assert_frame_equal(chosen.rmsecv, reference.rmsecv)

    nesting = {"max_components": 2, "inner_cv": 3, "n_inner_repeats": 1, "random_state": 0}
    pd.testing.assert_series_equal(PLSDA.nested_cv(X, y, **nesting).q2y, PLS.nested_cv(X, indicators, **nesting).q2y)

    resampling = {"cv": 3, "random_state": 0, "show_progress": False}
    classifier = PLSDA(n_components=2, decision_rule="bayes", priors="uniform", scale=True)
    resampled = classifier.cross_validate(X, y, **resampling)
    expected_resampling = PLS(n_components=2).fit(X, indicators).cross_validate(X, indicators, **resampling)
    pd.testing.assert_frame_equal(resampled.beta_mean, expected_resampling.beta_mean)
    pd.testing.assert_series_equal(resampled.q_squared, expected_resampling.q_squared)
    # This model's own PLS settings reach the fit: one component, not PLS's two above.
    one = PLSDA(n_components=1).cross_validate(X, y, **resampling)
    pd.testing.assert_frame_equal(
        one.beta_mean, PLS(n_components=1).fit(X, indicators).cross_validate(X, indicators, **resampling).beta_mean
    )

    with pytest.raises(ValueError, match=r"^X has 24 rows but y has 23 labels\.$"):
        PLSDA.compare_cv_criteria(X, y[:-1], **settings)


def test_plsda_cross_validation_gives_a_splitter_the_labels() -> None:
    """A stratified splitter splits by class; given the indicators it refuses them as multilabel.

    Labels 0 and 1, X as an array: the indicator columns are "0" and "1", rows numbered from 0.
    """
    X, y = _separable(n_per_class=9, n_features=4, separation=1.0, seed=52)
    values, labels = X.to_numpy(), (y == "low").astype(int)
    indicators = _indicator_frame(labels, pd.RangeIndex(len(labels)))
    assert indicators.columns.tolist() == ["0", "1"]
    splitter = StratifiedKFold(3, shuffle=True, random_state=4)
    splits = list(splitter.split(values, labels))
    settings = {"max_components": 2, "n_permutations": 9, "n_cv_permutations": 0, "random_state": 0}

    got = PLSDA.compare_cv_criteria(values, labels, cv=splitter, **settings)
    _assert_same_splits(got.cv_splits, splits)
    pd.testing.assert_frame_equal(got.table, PLS.compare_cv_criteria(values, indicators, cv=splits, **settings).table)

    pv = PLSDA.pseudo_validation_set(values, labels, n_components=1, cv=splitter, random_state=0)
    _assert_same_splits(pv.cv_splits, splits)
    pd.testing.assert_frame_equal(pv.Y_pv, indicators)

    chosen = PLSDA.select_n_components(values, labels, max_components=2, cv=splitter, random_state=0)
    reference = PLS.select_n_components(values, indicators, max_components=2, cv=splits, random_state=0)
    pd.testing.assert_frame_equal(chosen.rmsecv, reference.rmsecv)

    nesting = {"max_components": 2, "inner_cv": 3, "n_inner_repeats": 1, "random_state": 0}
    nested = PLSDA.nested_cv(values, labels, outer_cv=splitter, **nesting)
    reference_nested = PLS.nested_cv(values, indicators, outer_cv=check_cv(splits), **nesting)
    pd.testing.assert_series_equal(nested.q2y, reference_nested.q2y)
