"""PLS discriminant analysis (#375).

`PLSDA` is a classification layer over `PLS`, so these tests split into three concerns:
the encoding and decision rules that are new here, the classifier diagnostics, and the
promise that everything `PLS` already offers still works on the subclass.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from process_improve.multivariate._plsda import _bayes_threshold
from process_improve.multivariate.methods import PLS, PLSDA, MCUVScaler, confusion_matrix_plot


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
