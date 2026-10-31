# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""PLS discriminant analysis (#375): a classification layer over :class:`PLS`.

PLS-DA is ordinary PLS regression against a class-indicator ``Y``. Everything that
makes PLS attractive on process data carries over unchanged (many correlated
variables, more variables than samples, missing values, interpretable loadings, VIP,
contributions, Hotelling's T2 and SPE); what this module adds is the encoding of the
labels, the decision rule that turns predicted indicator values back into labels, and
the diagnostics a classifier is judged by.

Nothing here reimplements the PLS math: :class:`PLSDA` subclasses :class:`PLS`, so the
NIPALS fit, the scores, the loadings and all thirteen convenience methods are inherited.
"""

from __future__ import annotations

import typing
import warnings

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, check_cv
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from process_improve._random import check_random_state

from ._common import DataMatrix, NotEnoughVarianceError, SpecificationWarning, _model_method, epsqrt
from ._pls import PLS
from .plots import confusion_matrix_plot as _confusion_matrix_plot

#: Floor for a class-conditional standard deviation. A class with one training member
#: has no spread at all, and a class whose members all score identically has none
#: either; both would divide by zero in the Gaussian below.
_SD_FLOOR = float(np.sqrt(np.finfo(float).eps))


def _gaussian_log_density(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """Log of the normal density, without the constant that cancels in a ratio."""
    return -0.5 * ((x - mean) / sd) ** 2 - np.log(sd)


def _mean_and_sd(values: np.ndarray) -> tuple[float, float]:
    """Mean and sample standard deviation, with a single value giving a spread of zero."""
    if values.size < 2:  # one observation has a mean but no spread
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def _bayes_threshold(mean_in: float, sd_in: float, mean_out: float, sd_out: float, prior_in: float) -> float:
    """Where the in-class and out-of-class densities cross, weighted by the priors.

    This is the classical PLS-DA decision threshold (Perez, Sanchez & Fernandez-Pierna,
    2009): model the predicted indicator values of the members of a class as
    ``N(mean_in, sd_in)`` and of everything else as ``N(mean_out, sd_out)``, then put the
    threshold where ``prior_in * N_in(t) == (1 - prior_in) * N_out(t)``. Taking logs
    turns that into a quadratic in ``t``::

        (A - B) t^2 + 2 (B m_in - A m_out) t + (A m_out^2 - B m_in^2 + K) = 0

    with ``A = 1 / (2 sd_out^2)``, ``B = 1 / (2 sd_in^2)`` and
    ``K = ln(sd_out / sd_in) + ln(prior_in / (1 - prior_in))``.

    Equal spreads collapse the quadratic term and leave one root. Unequal spreads give
    two, only one of which sits between the two means; that is the crossing that
    separates them, and the other is the spurious one far out in a tail. If neither root
    lands between the means (which needs the two distributions to be almost on top of
    each other) the midpoint is returned, which is the equal-variance, equal-prior answer.

    Parameters
    ----------
    mean_in, sd_in : float
        Mean and standard deviation of the predicted indicator for members of the class.
    mean_out, sd_out : float
        The same for every sample not in the class.
    prior_in : float
        Prior probability of the class; ``0 < prior_in < 1``.

    Returns
    -------
    threshold : float
        The indicator value above which a sample is assigned to the class.
    """
    sd_in, sd_out = max(sd_in, _SD_FLOOR), max(sd_out, _SD_FLOOR)
    lo, hi = sorted((mean_in, mean_out))
    midpoint = 0.5 * (mean_in + mean_out)

    a = 1.0 / (2.0 * sd_out**2)
    b = 1.0 / (2.0 * sd_in**2)
    k = np.log(sd_out / sd_in) + np.log(prior_in / (1.0 - prior_in))
    quad, lin, const = a - b, 2.0 * (b * mean_in - a * mean_out), a * mean_out**2 - b * mean_in**2 + k

    if abs(quad) <= epsqrt * max(abs(lin), 1.0):  # equal spreads: one linear root
        return midpoint if abs(lin) <= epsqrt else float(-const / lin)

    # A negative discriminant means the densities never cross: its NaN roots lie between
    # no means, and the midpoint stands in.
    with np.errstate(invalid="ignore"):
        root = np.sqrt(lin**2 - 4.0 * quad * const)
    candidates = [(-lin + root) / (2.0 * quad), (-lin - root) / (2.0 * quad)]
    between = [t for t in candidates if lo <= t <= hi]
    return float(between[0]) if between else midpoint


class PLSDA(ClassifierMixin, PLS):
    """PLS discriminant analysis: PLS regression against a class-indicator matrix.

    The labels are one-hot encoded into an ``N x G`` indicator ``Y``, a PLS model is fitted
    on it, and a new sample's class comes from the ``G`` predicted indicator values. Two
    decision rules are offered, and they answer different questions:

    ``"max"``
        Take the largest predicted indicator. Every sample is assigned to exactly one
        class. This is the usual default and is what most PLS-DA software does.

    ``"bayes"``
        Take the largest class posterior, built from Gaussians fitted to the training
        indicator values of each class (in-class and out-of-class) and weighted by the
        class priors.

        This is the rule to prefer when the classes are badly unbalanced, and the reason
        is not the one that first suggests itself. A rare class's indicator column is
        pulled toward zero, because nine rows in ten want it there, so ``"max"`` hands
        almost everything to the common class: on a 1:9 fixture in the test suite it
        finds two of eight rare samples while reporting 92.5% accuracy, which is the
        classic accuracy trap. ``"bayes"`` reads each column against its own in-class and
        out-of-class spread instead of against the other columns, so a value that is high
        *for the rare column* still counts. On the same fixture it finds seven of eight,
        and comes out ahead on accuracy too.

    Both rules return exactly one label per sample. The per-class Bayesian *thresholds*,
    which do allow "in no class" and "in more than one class" readings, are exposed
    separately as :attr:`thresholds_`.

    Parameters
    ----------
    n_components : int
        Number of latent variables. As for :class:`PLS`, more is not better: PLS-DA on
        wide data will separate anything given enough components, which is what
        :meth:`permutation_test` exists to check.
    decision_rule : {"max", "bayes"}, optional
        Which rule :meth:`predict` applies. Default is ``"max"``.
    priors : "empirical", "uniform", or array-like of shape (n_classes,), optional
        Class priors used by the ``"bayes"`` rule and by :attr:`thresholds_`.
        ``"empirical"`` (the default) takes them from the training class frequencies;
        ``"uniform"`` gives every class ``1 / G``, which is what you want when the
        training set was deliberately balanced but the population is not.
    scale : bool, optional
        Passed to :class:`PLS`. Default True, which mean-centres and unit-variance-scales
        both blocks. Scaling the indicator block is standard for PLS-DA: without it a
        rare class contributes less variance and is fitted less well.
    max_iter, tol, copy, missing_data_settings, warn_on_uncentred
        Passed through to :class:`PLS` unchanged.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Sorted unique labels seen in ``fit``, in the column order of everything below.
    n_classes_ : int
        ``len(classes_)``.
    priors_ : np.ndarray of shape (n_classes,)
        The resolved priors, summing to 1.
    thresholds_ : pd.Series indexed by ``classes_``
        The per-class Bayesian decision threshold on the predicted indicator; see
        :func:`_bayes_threshold`.
    class_statistics_ : pd.DataFrame
        One row per class, columns ``mean_in``, ``sd_in``, ``mean_out``, ``sd_out``,
        ``prior``, ``threshold``: the fitted Gaussians the ``"bayes"`` rule uses.
    confusion_matrix_ : pd.DataFrame
        Training-set confusion matrix, rows the true class and columns the predicted one.
    accuracy_ : float
        Training-set accuracy. Optimistic by construction; use :meth:`score` on held-out
        data, or :meth:`permutation_test`, to learn anything about generalisation.
    sensitivity_, specificity_ : pd.Series indexed by ``classes_``
        Per-class true-positive and true-negative rates on the training set.

    Every fitted attribute of :class:`PLS` is also present (``scores_``, ``x_loadings_``,
    ``x_weights_``, ``r2_cumulative_``, ``hotellings_t2_``, ...), along with its
    convenience methods (``score_plot``, ``loading_plot``, ``vip``, ``spe_limit``,
    ``t2_contributions``, ...), because this class is a :class:`PLS`. Its cross-validation
    (:meth:`select_n_components`, :meth:`compare_cv_criteria`, :meth:`pseudo_validation_set`,
    :meth:`nested_cv` and :meth:`cross_validate`) takes the class labels and runs PLS's on
    their class-indicator matrix.

    Examples
    --------
    >>> model = PLSDA(n_components=2).fit(X, labels)        # doctest: +SKIP
    >>> model.predict(X_new)                                 # doctest: +SKIP
    array(['good', 'bad', 'good'], dtype=object)
    >>> model.confusion_matrix_                              # doctest: +SKIP
    >>> model.permutation_test(X, labels, n_permutations=99).p_value   # doctest: +SKIP

    References
    ----------
    Barker, M. & Rayens, W. (2003). Partial least squares for discrimination.
    *Journal of Chemometrics* 17:166-173.

    Brereton, R.G. & Lloyd, G.R. (2014). Partial least squares discriminant analysis:
    taking the magic away. *Journal of Chemometrics* 28:213-225.

    Westerhuis, J.A. et al. (2008). Assessment of PLSDA cross validation.
    *Metabolomics* 4:81-89.
    """

    _DECISION_RULES: typing.ClassVar[tuple[str, ...]] = ("max", "bayes")

    # Bound like PLS's own plot methods (ENG-05): a real method on the class, so `help`
    # and `inspect.signature` stay accurate and a subclass can override it.
    confusion_matrix_plot = _model_method(_confusion_matrix_plot)

    def __init__(  # noqa: PLR0913 - mirrors PLS's constructor plus two classifier options
        self,
        n_components: int,
        *,
        decision_rule: str = "max",
        priors: str | np.ndarray | list[float] = "empirical",
        scale: bool = True,
        max_iter: int = 500,
        tol: float = epsqrt,
        copy: bool = True,
        missing_data_settings: dict | None = None,
        warn_on_uncentred: bool = True,
    ):
        # `PLS.__init__` rather than `super().__init__`, and the difference is only
        # apparent. The MRO is PLSDA -> ClassifierMixin -> PLS, and `ClassifierMixin`
        # defines no `__init__`, so `super().__init__` resolved here anyway. Naming the
        # base says which constructor runs instead of leaving it to be read off the MRO,
        # and it does not depend on that mixin staying `__init__`-free. It is also what
        # CodeQL's py/missing-call-to-init can see: the query cannot follow the skip past
        # a mixin, so the `super()` form read as "PLS is never initialised", which would
        # have left every `n_components` unset had it been true.
        PLS.__init__(
            self,
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            missing_data_settings=missing_data_settings,
            warn_on_uncentred=warn_on_uncentred,
        )
        # Stored verbatim and read only in fit(), per the sklearn __init__ convention.
        self.decision_rule = decision_rule
        self.priors = priors

    def __sklearn_tags__(self):
        """Declare this a classifier, overriding the regressor tags inherited from PLS.

        :class:`PLS` carries ``RegressorMixin``, so the tag chain would otherwise leave
        ``estimator_type`` as ``"regressor"`` and a stale ``regressor_tags`` behind.
        ``ClassifierMixin`` sits ahead of ``PLS`` in the MRO and sets the type; this
        clears the leftover so ``is_regressor(PLSDA(...))`` is False rather than
        accidentally true through an attribute nobody reset.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.regressor_tags = None
        tags.target_tags.multi_output = False
        return tags

    def fit(self, X: DataMatrix, y: DataMatrix, sample_weight: np.ndarray | None = None) -> PLSDA:
        """Fit the PLS model on a one-hot encoding of ``y``, then the decision layer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain missing values, as for :class:`PLS`.
        y : array-like of shape (n_samples,)
            Class labels: strings, integers, or anything :func:`numpy.unique` can sort.
        sample_weight : array-like of shape (n_samples,), optional
            Passed through to :meth:`PLS.fit`.

        Returns
        -------
        self : PLSDA

        Raises
        ------
        ValueError
            If ``y`` holds fewer than two distinct labels (there is nothing to
            discriminate), if its length does not match ``X``, if a row has no label, or
            if ``decision_rule`` / ``priors`` is not a recognised value.
        """
        if self.decision_rule not in self._DECISION_RULES:
            msg = f"decision_rule must be one of {self._DECISION_RULES}; got {self.decision_rule!r}."
            raise ValueError(msg)

        labels, dummy = _indicator_matrix(X, y)
        classes = np.unique(labels)
        min_classes = 2
        if classes.size < min_classes:
            msg = f"PLSDA needs at least {min_classes} classes to discriminate between; y holds {classes.tolist()}."
            raise ValueError(msg)

        self.classes_ = classes
        self.n_classes_ = int(classes.size)
        self.priors_ = self._resolve_priors(labels)
        super().fit(X, dummy, sample_weight=sample_weight)
        self._fit_decision_layer(labels)
        return self

    def _resolve_priors(self, labels: np.ndarray) -> np.ndarray:
        """Turn the ``priors`` constructor argument into an array summing to 1."""
        if isinstance(self.priors, str):
            if self.priors == "uniform":
                return np.full(self.n_classes_, 1.0 / self.n_classes_)
            if self.priors == "empirical":
                counts = np.array([(labels == c).sum() for c in self.classes_])
                return counts / counts.sum()
            msg = f"priors must be 'empirical', 'uniform', or an array; got {self.priors!r}."
            raise ValueError(msg)

        priors = np.asarray(self.priors, dtype=float).ravel()
        if priors.shape[0] != self.n_classes_:
            msg = f"priors has {priors.shape[0]} entries but there are {self.n_classes_} classes."
            raise ValueError(msg)
        if np.any(priors <= 0) or not np.isclose(priors.sum(), 1.0):
            msg = f"priors must be positive and sum to 1; got {priors.tolist()}."
            raise ValueError(msg)
        return priors

    def _fit_decision_layer(self, labels: np.ndarray) -> None:
        """Fit the per-class Gaussians and record the training-set diagnostics.

        Uses ``predictions_``, the in-sample indicator predictions PLS already stores on
        the original (un-scaled) Y scale, rather than re-predicting: the numbers are the
        same and the training X does not have to be kept alive on the model.
        """
        scores = self.predictions_
        rows = []
        for position, klass in enumerate(self.classes_):
            column = scores.iloc[:, position].to_numpy()
            inside = labels == klass
            mean_in, sd_in = _mean_and_sd(column[inside])
            mean_out, sd_out = _mean_and_sd(column[~inside])
            prior = float(self.priors_[position])
            rows.append(
                {
                    "mean_in": mean_in,
                    "sd_in": max(sd_in, _SD_FLOOR),
                    "mean_out": mean_out,
                    "sd_out": max(sd_out, _SD_FLOOR),
                    "prior": prior,
                    "threshold": _bayes_threshold(mean_in, sd_in, mean_out, sd_out, prior),
                }
            )
        self.class_statistics_ = pd.DataFrame(rows, index=pd.Index(self.classes_, name="class"))
        self.thresholds_ = self.class_statistics_["threshold"]

        predicted = self._assign(scores)
        self.confusion_matrix_ = self._confusion(labels, predicted)
        self.accuracy_ = float((predicted == labels).mean())
        self.sensitivity_, self.specificity_ = self._rates(self.confusion_matrix_)

    def decision_function(self, X: DataMatrix) -> pd.DataFrame:
        """Predicted class-indicator values, one column per class.

        These are the raw PLS predictions of the one-hot ``Y``, so they are centred near 1
        for the class a sample belongs to and near 0 for the others, but they are not
        bounded to ``[0, 1]`` and do not sum to 1. Use :meth:`predict_proba` for a
        normalised quantity.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : pd.DataFrame of shape (n_samples, n_classes)
            Indexed by ``X``'s rows, columns in :attr:`classes_` order.
        """
        predicted = super().predict(X)  # which refuses an unfitted model
        predicted.columns = pd.Index(self.classes_, name="class")
        return predicted

    def predict_proba(self, X: DataMatrix) -> np.ndarray:
        """Class posteriors from the fitted per-class Gaussians.

        In column ``g`` the training indicator values of class ``g``'s members are modelled
        as ``N(mean_in, sd_in)`` and everyone else's as ``N(mean_out, sd_out)``. The
        evidence for class ``g`` is ``prior_g`` times the ratio of those two densities,
        and the posteriors are those normalised across the classes. This is a genuine
        probability model rather than a rescaling of the indicators: it accounts for how
        tightly each class scores, for how well it separates from the rest, and for how
        common it is.

        It is still only as good as the Gaussian assumption, which a bimodal class or one
        with three training members will not satisfy. The raw quantity the model actually
        computes is :meth:`decision_function`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Rows sum to 1; columns in :attr:`classes_` order, as sklearn requires.
        """
        evidence = self._log_evidence(self.decision_function(X))
        # Subtract the row maximum before exponentiating: the terms can underflow to zero
        # outright on a well-separated model, which would make the row sum 0/0.
        shifted = np.exp(evidence - evidence.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def _log_evidence(self, scores: pd.DataFrame) -> np.ndarray:
        """``log(prior_g) + log LR_g`` for each class, from the indicator values.

        ``LR_g`` is the likelihood ratio in column ``g``: how much more consistent the
        value is with belonging to class ``g`` than with not belonging to it. Using the
        *ratio* rather than the in-class density alone is what makes the columns
        comparable. The in-class densities are not: a class with eight members can have a
        very tight ``sd_in``, and its density then spikes high enough to outvote a class
        nine times as common, which is the opposite of what accounting for the prior is
        supposed to achieve. Dividing by the out-of-class density in the same column
        cancels that scale.

        It is also the quantity :attr:`thresholds_` is defined by: the threshold for class
        ``g`` is exactly where ``prior_g * LR_g`` crosses ``1 - prior_g``, so the rule and
        the reported thresholds cannot disagree.
        """
        values = scores.to_numpy()
        stats = self.class_statistics_
        return np.column_stack(
            [
                np.log(stats["prior"].iloc[position])
                + _gaussian_log_density(
                    values[:, position], stats["mean_in"].iloc[position], stats["sd_in"].iloc[position]
                )
                - _gaussian_log_density(
                    values[:, position], stats["mean_out"].iloc[position], stats["sd_out"].iloc[position]
                )
                for position in range(self.n_classes_)
            ]
        )

    def _assign(self, scores: pd.DataFrame) -> np.ndarray:
        """Apply the configured decision rule to a table of indicator values."""
        if self.decision_rule == "max":
            return self.classes_[np.argmax(scores.to_numpy(), axis=1)]
        return self.classes_[np.argmax(self._log_evidence(scores), axis=1)]

    def predict(self, X: DataMatrix) -> np.ndarray:  # type: ignore[override]
        """Predict a class label for every row of ``X``.

        The return type deliberately narrows :meth:`PLS.predict`, which hands back a
        DataFrame of predicted Y values: a classifier's ``predict`` returns labels, and
        sklearn's classifier contract requires exactly that. The DataFrame that
        :meth:`PLS.predict` would have returned is still available, as
        :meth:`decision_function`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Values drawn from :attr:`classes_`, chosen by :attr:`decision_rule`.

        See Also
        --------
        decision_function : the indicator values the rule is applied to.
        predict_proba : class posteriors.
        """
        return self._assign(self.decision_function(X))

    def score(self, X: DataMatrix, y: DataMatrix, sample_weight: np.ndarray | None = None) -> float:
        """Return the mean accuracy on ``X`` against the true labels ``y``.

        Overrides :meth:`PLS.score`, which returns R2 of the indicator predictions: a
        number that goes up when the indicators are fitted more tightly, not when more
        samples land in the right class. sklearn's convention (higher is better) holds
        either way, but only accuracy answers the question a classifier is asked.
        """
        correct = self.predict(X) == np.ravel(y)
        return float(np.average(correct, weights=None if sample_weight is None else np.ravel(sample_weight)))

    def _confusion(self, truth: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
        """Confusion matrix as a labelled frame: rows true, columns predicted."""
        counts = np.zeros((self.n_classes_, self.n_classes_), dtype=int)
        position = {klass: index for index, klass in enumerate(self.classes_)}
        for actual, guess in zip(truth, predicted, strict=True):
            counts[position[actual], position[guess]] += 1
        return pd.DataFrame(
            counts,
            index=pd.Index(self.classes_, name="true"),
            columns=pd.Index(self.classes_, name="predicted"),
        )

    @staticmethod
    def _rates(confusion: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Per-class sensitivity and specificity from a confusion matrix."""
        counts = confusion.to_numpy()
        true_positive = np.diag(counts)
        actual = counts.sum(axis=1)
        negatives = counts.sum() - actual
        true_negative = negatives - (counts.sum(axis=0) - true_positive)
        # A class absent from the rows has no positives and gives 0 / 0. So does a class that
        # is the only one in them: every row it is predicted for is then its own, so it has
        # no false positive and no true negative either. Both rates are NaN there.
        with np.errstate(invalid="ignore"):
            sensitivity = true_positive / actual
            specificity = true_negative / negatives
        index = confusion.index.rename("class")
        return pd.Series(sensitivity, index=index, name="sensitivity"), pd.Series(
            specificity, index=index, name="specificity"
        )

    def confusion(self, X: DataMatrix, y: DataMatrix) -> Bunch:
        """Confusion matrix and per-class rates on data the caller supplies.

        The ``confusion_matrix_`` / ``sensitivity_`` / ``specificity_`` attributes are the
        training-set versions and are optimistic; this is the same calculation on a
        held-out split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels. Labels outside :attr:`classes_` raise.

        Returns
        -------
        result : sklearn.utils.Bunch
            ``matrix`` (DataFrame), ``sensitivity`` and ``specificity`` (Series indexed by
            class), and ``accuracy`` (float).

        Raises
        ------
        ValueError
            If ``y`` holds a label the model was not fitted on.
        """
        check_is_fitted(self)
        truth = np.asarray(y).ravel()
        unseen = set(np.unique(truth)) - set(self.classes_.tolist())
        if unseen:
            msg = (
                f"y holds labels the model was not fitted on: {sorted(unseen)}. "
                f"Known classes: {self.classes_.tolist()}."
            )
            raise ValueError(msg)
        predicted = self.predict(X)
        matrix = self._confusion(truth, predicted)
        sensitivity, specificity = self._rates(matrix)
        return Bunch(
            matrix=matrix,
            sensitivity=sensitivity,
            specificity=specificity,
            accuracy=float((predicted == truth).mean()),
        )

    def roc_auc(self, X: DataMatrix, y: DataMatrix, *, positive_class: object = None) -> float:
        """Area under the ROC curve, from the continuous indicator rather than the label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels.
        positive_class : object, optional
            Which class counts as positive. Required when there are more than two classes,
            where the result is that class against all the others; for two classes it
            defaults to ``classes_[1]``.

        Returns
        -------
        auc : float
            Computed on :meth:`decision_function`, not on the hard labels, so it measures
            the ranking the model produces and is unaffected by the decision rule.

        Raises
        ------
        ValueError
            If ``positive_class`` is omitted with more than two classes, or names a class
            the model was not fitted on.
        """
        check_is_fitted(self)
        binary = 2
        if positive_class is None:
            if self.n_classes_ != binary:
                msg = (
                    f"roc_auc needs `positive_class` when there are {self.n_classes_} classes: "
                    "the result is one class against all the others."
                )
                raise ValueError(msg)
            positive_class = self.classes_[1]
        matches = np.flatnonzero(self.classes_ == positive_class)
        if matches.size == 0:
            msg = f"positive_class {positive_class!r} is not one of {self.classes_.tolist()}."
            raise ValueError(msg)
        column = int(matches[0])
        scores = self.decision_function(X).to_numpy()[:, column]
        return float(roc_auc_score(np.ravel(y) == positive_class, scores))

    def permutation_test(
        self,
        X: DataMatrix,
        y: DataMatrix,
        *,
        n_permutations: int = 99,
        cv: int | object | None = 5,
        random_state: int | np.random.Generator | None = None,
    ) -> Bunch:
        """Test whether the model separates the classes better than shuffled labels do.

        PLS-DA on wide data will separate almost anything: with more variables than
        samples there is always a direction that happens to line up with the labels. The
        permutation test is what tells a real effect from that, by refitting the same model
        on shuffled labels and asking how often chance does as well.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True labels.
        n_permutations : int, optional
            Number of label shuffles. Default 99, which puts the smallest attainable
            p-value at ``1 / 100``.
        cv : int, splitter, or None, optional
            Cross-validation for the accuracy that is compared. Default 5, stratified.
            ``None`` uses training accuracy, which is far quicker and actively misleading.
            Measured on 40 samples of 30 pure-noise variables with two random labels, 49
            permutations, four folds:

            ===================  =========  ============  =========
            Statistic            Separable  Pure noise
            ===================  =========  ============  =========
            Training accuracy    1.000      **1.000**
            Cross-validated      1.000      0.575
            p, ``cv=4``          0.020      **0.220**
            p, ``cv=None``       0.020      **0.020**
            ===================  =========  ============  =========

            The noise column is the whole argument. With more variables than samples the
            model fits random labels perfectly, so training accuracy says 1.000 either
            way and the cross-validated test correctly declines to call it significant,
            while the training-accuracy test reports p = 0.02 on data that has no signal
            in it at all. Westerhuis et al. (2008) is explicit that the comparison has to
            be made on cross-validated performance; ``cv=None`` is offered for a quick
            look and for well-conditioned data, not for a result to report.
        random_state : int, np.random.Generator, or None, optional
            Seeds the shuffles, per the reproducibility contract.

        Returns
        -------
        result : sklearn.utils.Bunch
            ``observed`` (float), ``null`` (np.ndarray of accuracies), ``p_value`` (float),
            ``n_permutations`` (int, the number of draws the null actually holds) and
            ``n_failed`` (int).

            A shuffled label set can leave a fold with nothing to fit at this component
            count, and NIPALS then goes singular. Those draws are dropped rather than
            scored as zero, which would push the null down and the p-value with it. When
            any are dropped a :class:`SpecificationWarning` says how many, because a null
            that keeps collapsing means the component count is too high for the data
            rather than that the model is good.

            The p-value is ``(1 + #{null >= observed}) / (1 + n_permutations)``. The
            observed statistic counts itself among the permutations, because no finite
            set of shuffles licenses a claim of exactly zero; the same convention is used
            by the Van der Voet and multiblock randomization tests in this package.
        """
        check_is_fitted(self)
        rng = check_random_state(random_state)
        labels = np.asarray(y).ravel()
        rows = pd.DataFrame(X)  # selected by position; PLS reads an array's columns by position too

        def accuracy(target: np.ndarray) -> float:
            if cv is None:
                model = self._clone_unfitted().fit(X, target)
                return float((model.predict(X) == target).mean())
            splitter = StratifiedKFold(cv) if isinstance(cv, int) else check_cv(cv)
            correct = 0
            for train_idx, test_idx in splitter.split(np.zeros(len(target)), target):
                x_train, x_test = rows.iloc[train_idx], rows.iloc[test_idx]
                model = self._clone_unfitted().fit(x_train, target[train_idx])
                correct += int((model.predict(x_test) == target[test_idx]).sum())
            return correct / len(target)

        observed = accuracy(labels)

        def attempt(target: np.ndarray) -> float | None:
            """One null draw, or None when this shuffle cannot be fitted at all.

            A shuffled label set can leave a fold with nothing left to fit at this
            component count: NIPALS deflates the block to noise and the inversion goes
            singular. Dropping that draw keeps the null honest; counting it as a zero
            would push the null down and the p-value with it, which is exactly the wrong
            direction for a test of chance separation.
            """
            try:
                return accuracy(target)
            except (np.linalg.LinAlgError, NotEnoughVarianceError):
                return None

        outcomes = [attempt(rng.permutation(labels)) for _ in range(int(n_permutations))]
        draws = [value for value in outcomes if value is not None]
        failed = len(outcomes) - len(draws)
        if not draws:
            msg = (
                f"Every one of the {n_permutations} permutations failed to fit at "
                f"n_components={self.n_components}. There is no null distribution to compare "
                "against; refit with fewer components, or with more samples per fold."
            )
            raise RuntimeError(msg)
        if failed:
            warnings.warn(
                f"{failed} of {n_permutations} permutations could not be fitted at "
                f"n_components={self.n_components} and were dropped; the p-value is over the "
                f"remaining {len(draws)}. Fewer components would make the null complete.",
                SpecificationWarning,
                stacklevel=2,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin the caller
            )
        null = np.asarray(draws)
        p_value = float((1 + int(np.sum(null >= observed))) / (1 + null.size))
        return Bunch(
            observed=observed,
            null=null,
            p_value=p_value,
            n_permutations=int(null.size),
            n_failed=failed,
        )

    def _clone_unfitted(self) -> PLSDA:
        """Build a fresh, unfitted copy carrying this model's constructor parameters."""
        return type(self)(**self.get_params())

    # PLS's cross-validation reads Y as a numeric response, so for class labels each of the
    # five below runs PLS's own on the class-indicator matrix: the regression this class
    # fits. A splitter is given the labels, so StratifiedKFold keeps each class's share in
    # every fold; an integer gives PLS's shuffled K-fold. The other settings are PLS's.

    @classmethod
    def select_n_components(
        cls, X: DataMatrix, Y: DataMatrix | pd.Series, *, cv: int | BaseCrossValidator = 5, **pls_kwargs
    ) -> Bunch:
        """Run :meth:`PLS.select_n_components` on the class-indicator matrix of the labels ``Y``."""
        labels, indicators = _indicator_matrix(X, Y)
        return PLS.select_n_components(X, indicators, cv=_label_splits(cv, X, labels), **pls_kwargs)

    @classmethod
    def compare_cv_criteria(
        cls, X: DataMatrix, Y: DataMatrix | pd.Series, *, cv: int | BaseCrossValidator = 7, **pls_kwargs
    ) -> Bunch:
        """Run :meth:`PLS.compare_cv_criteria` on the class-indicator matrix of the labels ``Y``.

        ``global_model`` is the :class:`PLS` fitted to that matrix.
        """
        labels, indicators = _indicator_matrix(X, Y)
        return PLS.compare_cv_criteria(X, indicators, cv=_label_splits(cv, X, labels), **pls_kwargs)

    @classmethod
    def pseudo_validation_set(
        cls, X: DataMatrix, Y: DataMatrix | pd.Series, *, cv: int | BaseCrossValidator = 7, **pls_kwargs
    ) -> Bunch:
        """Run :meth:`PLS.pseudo_validation_set` on the class-indicator matrix of the labels ``Y``.

        ``Y_pv`` is that matrix, and ``global_model`` the :class:`PLS` fitted to it.
        """
        labels, indicators = _indicator_matrix(X, Y)
        return PLS.pseudo_validation_set(X, indicators, cv=_label_splits(cv, X, labels), **pls_kwargs)

    @classmethod
    def nested_cv(
        cls, X: DataMatrix, Y: DataMatrix | pd.Series, *, outer_cv: int | BaseCrossValidator = 5, **pls_kwargs
    ) -> Bunch:
        """Run :meth:`PLS.nested_cv` on the class-indicator matrix of the labels ``Y``."""
        labels, indicators = _indicator_matrix(X, Y)
        return PLS.nested_cv(X, indicators, outer_cv=_label_splits(outer_cv, X, labels), **pls_kwargs)

    def cross_validate(self, X: DataMatrix, Y: DataMatrix | pd.Series, **settings) -> Bunch:
        """Run :meth:`PLS.cross_validate` for a :class:`PLS` with this model's settings and its labels ``Y``.

        That :class:`PLS` is fitted to the class-indicator matrix of ``Y``, and its
        coefficients are the ones resampled.
        """
        _, indicators = _indicator_matrix(X, Y)
        pls_settings = {
            name: value for name, value in self.get_params().items() if name not in ("decision_rule", "priors")
        }
        # cross_validate reads from the fitted model only the labels of its coefficients.
        return PLS(**pls_settings).fit(X, indicators).cross_validate(X, indicators, **settings)


def _indicator_matrix(X: DataMatrix, y: DataMatrix | pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the labels as one flat array, and the class-indicator matrix PLS-DA regresses ``X`` on.

    The matrix has one column per class, sorted and named ``str(class)``, holding 1.0 where
    a row is in that class and 0.0 elsewhere, and ``X``'s row labels.
    """
    labels = np.asarray(y).ravel()
    if labels.shape[0] != X.shape[0]:
        msg = f"X has {X.shape[0]} rows but y has {labels.shape[0]} labels."
        raise ValueError(msg)
    missing = int(pd.isna(labels).sum())
    if missing:
        msg = f"y has no label in {missing} of its rows; every row needs a class."
        raise ValueError(msg)
    classes = np.unique(labels)
    index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(labels.shape[0])
    indicators = np.where(labels[:, None] == classes[None, :], 1.0, 0.0)
    return labels, pd.DataFrame(indicators, index=index, columns=[str(klass) for klass in classes])


def _label_splits(cv: int | BaseCrossValidator, X: DataMatrix, labels: np.ndarray) -> int | BaseCrossValidator:
    """Return a splitter's folds of ``X``, split by the class labels; any other ``cv`` as given.

    A stratified splitter reads its target to keep each class's share in every fold, and
    refuses the indicator matrix as a multilabel target, so it is given the labels.
    """
    return check_cv(list(cv.split(X, labels))) if hasattr(cv, "split") else cv
