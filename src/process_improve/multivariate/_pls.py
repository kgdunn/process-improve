# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Projection to Latent Structures (PLS) regression estimator (ENG-01).

The sklearn-compatible :class:`PLS` regressor (NIPALS, missing-data aware), with
cross-validation, prediction intervals, diagnostics, confidence limits and
plotting bound as convenience methods after ``fit()``.
"""

from __future__ import annotations

import itertools
import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.metrics import r2_score
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedKFold, check_cv
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data
from tqdm import tqdm

from .._linalg import safe_inverse
from ..univariate.metrics import detect_outliers_esd
from ._base import _LatentVariableModel, _LazyFrame
from ._common import (
    Q2_MIN_INCREMENT,
    DataMatrix,
    NotEnoughVarianceError,
    SelectionRule,
    SpecificationWarning,
    UncentredDataWarning,
    _align_to_fit_features,
    _equal_weight_r2_total,
    _model_method,
    _nz,
    _reject_sparse,
    _select_n_components,
    _vandervoet_randomization,
    epsqrt,
    largest_ssq_column,
    resolve_loop_settings,
    y_is_exhausted,
    y_reach,
)
from ._cv_criteria import (
    _compare_cv_criteria,
    _CompareSettings,
    _pseudo_validation_set,
    _PseudoValidationSettings,
)
from ._diagnostics import _t2_score_sd
from ._diagnostics import (
    selectivity_ratio as _selectivity_ratio,
)
from ._diagnostics import (
    target_projection as _target_projection,
)
from ._impute import impute_low_rank
from ._nipals import quick_regress, ssq, terminate_check
from ._preprocessing import MCUVScaler, _uncentred_columns, _warn_scaling_traps
from ._projection import coerce_observed_mask, operator_for_pattern, project_rows
from .plots import (
    coefficient_plot as _coefficient_plot,
)
from .plots import (
    predictions_vs_observed_plot as _predictions_vs_observed_plot,
)

logger = logging.getLogger(__name__)

# The ``md_method`` values ``fit`` recognises. Deliberately smaller than
# ``_projection.PROJECTION_METHODS``: that tuple is the projection-time set, which also
# offers ``"scp"``. ``"scp"`` has no fit-time meaning of its own, because it is how NIPALS
# already scores an incomplete row during the fit, so asking for it is refused rather than
# quietly running NIPALS under another name.
_FIT_TIME_MD_METHODS = frozenset({"nipals", "tsr", "pmp"})


def _check_md_method(settings: dict) -> None:
    """Refuse a ``md_method`` the fit cannot carry out, before any work is done.

    An unrecognised value used to fall through to NIPALS silently, so a typo, or the
    projection-time name ``"scp"``, ran a different algorithm than the caller asked for.
    """
    md_method = settings["md_method"].lower()
    if md_method not in _FIT_TIME_MD_METHODS:
        raise ValueError(
            f"md_method must be one of {sorted(_FIT_TIME_MD_METHODS)}; got '{md_method}'. "
            "This is the fit-time setting, a smaller set than the method= accepted by "
            "project() and the contribution helpers."
        )


def _format_labels(labels: list, limit: int = 4) -> str:
    """Render column labels for a warning message, truncating a long list."""
    shown = ", ".join(repr(str(label)) for label in labels[:limit])
    if len(labels) > limit:
        shown += f", ... ({len(labels)} columns in total)"
    return shown


def _warn_if_uncentred(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    """Warn when ``PLS(scale=False)`` is handed a block that was never centred.

    With ``scale=False`` nothing is centred and no intercept is fitted, so a
    block carrying a non-zero mean displaces every prediction. The response is
    the damaging case: predictions come out offset by roughly the response mean,
    which drives R² and Q² large and negative on data that does contain a
    relationship. That reads as "there is nothing here", so the failure has to be
    announced rather than left to the caller to notice.

    Deliberately a warning, not an automatic centring: ``scale=False`` currently
    means "touch nothing", and quietly centring would change the numbers for
    every caller who already centres correctly.

    The category is :class:`UncentredDataWarning`, a :class:`SpecificationWarning`
    subclass, so a caller who fits un-centred on purpose can permit this one
    diagnostic without going blind to the rest. Narrower again, and touching no
    global filter: ``PLS(..., warn_on_uncentred=False)`` skips the call to this
    helper entirely.
    """
    for block, name in ((Y, "Y"), (X, "X")):
        offenders = _uncentred_columns(block)
        if not offenders:
            continue
        symptom = (
            "every prediction is offset by roughly the response mean, so R2 and Q2 "
            "go large and negative and the fit reads as 'no relationship in this "
            "data' even when the relationship is strong"
            if name == "Y"
            else "the un-centred columns act as an uncontrolled constant term, so the "
            "loadings and R2 describe the offset rather than the variation around it"
        )
        warnings.warn(
            f"PLS(scale=False) fits no intercept and centres nothing, but column(s) "
            f"{_format_labels(offenders)} of the {name} block have a mean that is large "
            f"relative to their own spread. The consequence is not an error: "
            f"{symptom}. Centre both blocks first (MCUVScaler().fit_transform(...), or "
            f"X - X.mean()), or pass scale=True and let the model do it. Fitting "
            f"un-centred on purpose? Pass warn_on_uncentred=False to silence this for "
            f"this model alone.",
            UncentredDataWarning,
            stacklevel=3,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin the caller
        )


def _held_out_scores(model: PLS, x_train: np.ndarray, x_test: np.ndarray, n_components: int) -> np.ndarray:
    """Return held-out rows' scores under a fold model's first ``n_components`` components (#676).

    Both blocks are in the space the model was fitted in. The smaller model's weights W,
    loadings P and scores are the first columns of the fold model's, since NIPALS extracts its
    components one at a time; its direct weights R = W (P'W)^-1 are the first columns of the
    fold model's only on complete data, where P'W is triangular, so with gaps in the training
    block they are worked out again. A complete row's score is ``x @ R``. A row with a missing
    cell would be NaN that way, so it is projected as :meth:`PLS.project` projects a new row with
    gaps: by trimmed score regression over its observed columns, with the smaller model's
    loadings, score variances and training residuals. A row with fewer observed columns than
    components has no such score and stays NaN, as does one whose regression is singular.
    """
    a = n_components
    loadings = model.x_loadings_.to_numpy()[:, :a]
    weights = model.x_weights_.to_numpy()[:, :a]
    if np.isnan(x_train).any():
        guide = weights @ safe_inverse(loadings.T @ weights, what="(P'W of the smaller model)")
        scores = x_test @ guide
    else:  # the product with every column, then cut, as before #676, so complete data scores the same
        guide = model.direct_weights_.to_numpy()
        scores = (x_test @ guide)[:, :a]
        guide = guide[:, :a]
    gaps = np.isnan(x_test).any(axis=1)
    if not gaps.any():
        return scores
    residuals = np.nan_to_num(x_train - model.scores_.to_numpy()[:, :a] @ loadings.T)
    variances = model.explained_variance_[:a]
    patterns, which = np.unique(~np.isnan(x_test[gaps]), axis=0, return_inverse=True)
    rows = np.flatnonzero(gaps)
    for k, observed in enumerate(patterns):
        if observed.sum() < a:
            continue
        try:
            # TSR, the default estimator, reads no weights (only SCP does).
            op = operator_for_pattern(loadings, guide, variances, observed, x_residuals=residuals)
        except np.linalg.LinAlgError:
            continue
        mine = rows[which.ravel() == k]
        scores[mine] = x_test[np.ix_(mine, np.flatnonzero(observed))] @ op.matrix.T
    return scores


def _direct_weights(x_weights: np.ndarray, x_loadings: np.ndarray) -> np.ndarray:
    """Return the direct weights ``R = W (P'W)^-1`` [K x A], which score X directly: ``T = X R``."""
    return x_weights @ safe_inverse(x_loadings.T @ x_weights, what="(x_loadings' @ x_weights)")


def _standard_error(per_fold: np.ndarray) -> np.ndarray:
    """Return the standard error of each row's mean over its folds (columns), ``ddof=1``.

    A row is NaN in every fold or in none (a count that left a held-out row unpredicted
    is NaN throughout), so its standard error is NaN or ``std / sqrt(folds)``; with a
    single fold there is no spread to measure.
    """
    folds = per_fold.shape[1]
    if folds < 2:
        return np.full(per_fold.shape[0], np.nan)
    return np.std(per_fold, axis=1, ddof=1) / np.sqrt(folds)


def _back_to_units(*scalers: MCUVScaler | None) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return ``(centre, scale)`` with ``units = scaled * scale + centre``, undoing ``scalers`` in the order given.

    Each scaler maps ``u`` to ``(u - center_) / scale_``, so the one applied last is undone
    first; a missing scaler is the identity.
    """
    centre: np.ndarray | float = 0.0
    scale: np.ndarray | float = 1.0
    for scaler in scalers:
        if scaler is not None:
            centre = centre * scaler.scale_.to_numpy() + scaler.center_.to_numpy()
            scale = scale * scaler.scale_.to_numpy()
    return centre, scale


def _fit_fold(
    model: PLS, X_train: pd.DataFrame, Y_train: pd.DataFrame, *, scale_inside_folds: bool
) -> tuple[tuple[np.ndarray | float, np.ndarray | float], tuple[np.ndarray | float, np.ndarray | float]]:
    """Fit a cross-validation fold's ``model``; return the centre and scale of X and of Y in its space.

    With ``scale_inside_folds`` the training rows are autoscaled first. The model may centre
    and scale them again (``scale=True``, or PRM's robust centre), so a block in the data's
    units is ``scaled * scale + centre`` of the same block in the model's space through both
    (#1027): the held-out rows are scored, and their predictions read, in the model's space.
    """
    fold_x = MCUVScaler().fit(X_train) if scale_inside_folds else None
    fold_y = MCUVScaler().fit(Y_train) if scale_inside_folds else None
    model.fit(
        X_train if fold_x is None else fold_x.transform(X_train),
        Y_train if fold_y is None else fold_y.transform(Y_train),
    )
    return _back_to_units(model._x_scaler, fold_x), _back_to_units(model._y_scaler, fold_y)


class PLS(_LatentVariableModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """Projection to Latent Structures (PLS) regression with diagnostics.

    Implements PLS via the NIPALS algorithm with production diagnostics: SPE,
    Hotelling's T², score contributions, and outlier detection. The API mirrors
    :class:`PCA` so that ``model.scores_``, ``model.spe_``, and
    ``model.detect_outliers()`` work identically for both model types.

    Parameters
    ----------
    n_components : int or None
        Number of latent components to extract. ``None`` asks for as many
        components as the data supports: it is resolved at fit time to
        ``min(n_samples, n_features)``, which is also the ceiling an explicit
        request is clamped to (with a ``SpecificationWarning``). The resolved
        count is available on the fitted attribute ``n_components_``.
    scale : bool, default=True
        Mean-center and unit-variance-scale both the X and Y blocks internally
        before fitting (``ddof=1``, done with :class:`MCUVScaler`). This mirrors
        :class:`sklearn.cross_decomposition.PLSRegression`, whose ``scale=True``
        default also scales X and Y; the parameter exists so ``PLS`` is a
        drop-in for the sklearn estimator. Predictions, ``predictions_`` and
        ``beta_coefficients_`` are returned on the original (un-scaled) data
        scale. When you scale externally (e.g. with ``MCUVScaler``), set
        ``scale=False`` to avoid the (idempotent) double scaling. Note: the
        cross-validation helpers (:meth:`select_n_components`) always re-fit an
        :class:`MCUVScaler` inside each training fold regardless of this flag.

        ``scale=False`` fits **no intercept**, so both blocks must already be
        centred. A response left on its natural scale is the trap: predictions
        come out offset by the response mean, and R² / Q² go large and negative
        on data that does contain a relationship. ``fit`` emits an
        :class:`UncentredDataWarning` when either block's column means are large
        relative to their spread; it does not centre for you, because
        ``scale=False`` means "touch nothing". Set ``warn_on_uncentred=False``
        when that fit is deliberate.
    max_iter : int, default=500
        Maximum number of iterations per component for the NIPALS algorithm.
        The same default as every other iterative estimator in the package; it
        was 1000 before 1.98.0 (#588).
    tol : float, default=sqrt(machine epsilon)
        Relative convergence tolerance for the NIPALS algorithm: the change
        between two successive score-vector iterations, relative to the norm
        of the current score vector (see :func:`terminate_check`).
    copy : bool, default=True
        Whether to copy X and Y before fitting.
    warn_on_uncentred : bool, default=True
        Emit the :class:`UncentredDataWarning` described under ``scale`` when
        ``scale=False`` and a block arrives un-centred. Set it to ``False`` for
        a fit that is un-centred on purpose (a demonstration of the offset, or a
        test that some other centring check fires), where the diagnostic is the
        expected outcome rather than a problem.

        This is the narrowest of the three opt-outs, and the one to reach for
        first. It silences the check for this model only, so an unrelated
        :class:`SpecificationWarning` raised elsewhere in the same block, or by
        this same ``fit`` call, still arrives. Filtering
        ``UncentredDataWarning`` as a category is next narrowest; suppressing
        all of ``SpecificationWarning`` is the blunt instrument, and hides
        clamped component counts and NIPALS non-convergence along with it.

        Has no effect when ``scale=True``: the model centres both blocks itself,
        so the condition cannot arise. Like every constructor parameter it is
        stored verbatim and survives :func:`~sklearn.base.clone`, so a
        deliberately un-centred fit stays quiet inside a ``Pipeline`` or a
        grid search.
    missing_data_settings : dict or None, default=None
        How to fit when the data has missing cells. Keys:

        - ``md_method``, one of:

          - ``"nipals"`` (the default): NIPALS skips each missing cell in its
            sums, so an incomplete row simply contributes less. Cheapest, and
            no imputation happens.
          - ``"tsr"``: trimmed score regression. The missing cells are filled
            by EM on a principal-component model of X and Y *together*,
            refitted every round from the completed data, and PLS is then
            fitted once to the result. Measured against the model the complete
            data would have given: on synthetic data with clear low-rank
            structure it lands 2.0 to 2.9 times closer than ``"nipals"``
            (5-45% of cells missing at random, closer on 17-19 trials in 20);
            on the LDPE process data 1.1 to 1.5 times closer (9-12 in 12). The
            gain grows with how much of the data the components capture,
            because a missing cell can only be rebuilt from what they describe.
            Roughly five times the cost of a ``"nipals"`` fit.
          - ``"pmp"``: the same loop, estimating each row by least-squares
            projection onto the components instead of by regression. Close to
            ``"tsr"`` throughout, and slower to settle as more goes missing.

          Any other value is refused, including ``"scp"``: that is a
          projection-time method (see :meth:`project`), and at fit time it is
          simply what NIPALS already does.
        - ``md_tol`` and ``md_max_iter`` were removed in 2.0 and raise a
          :class:`ValueError`; set ``tol`` and ``max_iter`` instead. Those two
          bound the NIPALS loop and, for ``"tsr"`` / ``"pmp"``, the imputation
          as well: the tolerance on the largest change in any imputed cell (in
          standard deviations) and the cap on imputation rounds.

        With ``"tsr"`` / ``"pmp"``, R², SPE and the other diagnostics are still
        computed on the *observed* cells only: an imputed cell is fitted by
        construction, so counting it would flatter the model.

    Attributes (after fitting)
    --------------------------
    n_components_ : int
        The resolved number of components actually fitted (the constructor
        parameter clamped to ``min(n_samples, n_features)``; the parameter
        itself is left as the user set it, including ``None``).
    scores_ : pd.DataFrame of shape (n_samples, n_components)
        X-block score matrix (T). This is the primary score matrix; equivalent
        to ``x_scores`` in older versions.
    y_scores_ : pd.DataFrame of shape (n_samples, n_components)
        Y-block score matrix (U).
    x_loadings_ : pd.DataFrame of shape (n_features, n_components)
        X-block loading matrix (P).
    y_loadings_ : pd.DataFrame of shape (n_targets, n_components)
        Y-block loading matrix (C).
    x_weights_ : pd.DataFrame of shape (n_features, n_components)
        X-block weight matrix (W).
    y_weights_ : pd.DataFrame of shape (n_targets, n_components)
        Y-block weight matrix.
    direct_weights_ : pd.DataFrame of shape (n_features, n_components)
        Direct (W*) weights: ``W (P'W)^{-1}``. Used for direct projection
        ``T = X @ W*``.
    beta_coefficients_ : pd.DataFrame of shape (n_features, n_targets)
        Regression coefficients linking X directly to Y.
    predictions_ : pd.DataFrame of shape (n_samples, n_targets)
        Y predictions from the training data.
    spe_ : pd.DataFrame of shape (n_samples, n_components)
        Per-row SPE diagnostic; stored as the square root of the row
        sum-of-squared X-residuals (so it is on the residual scale, not the
        squared scale). **One column per component, not one value per row**:
        column ``a`` is the SPE of the model truncated at ``a`` components, and
        the last column is the value at the full fitted model. Reach for a
        single number per observation with ``model.spe_.iloc[:, -1]``, not with
        ``np.asarray(model.spe_).ravel()``: ravel happens to give the right
        answer at one component and silently gives ``n_samples * n_components``
        values above it.
    hotellings_t2_ : pd.DataFrame of shape (n_samples, n_components)
        Cumulative Hotelling's T² statistic. Per-component, exactly as ``spe_``
        above: column ``a`` uses the first ``a`` components and the last column
        is the value at the full fitted model.
    r2_per_component_ : pd.Series of length n_components
        Fractional R² (on Y) explained by each component.
    r2_cumulative_ : pd.Series of length n_components
        Cumulative R² (on Y) after each component.
    r2_per_variable_ : pd.DataFrame of shape (n_features, n_components)
        Per-variable cumulative R² for X after each component.
    r2y_per_variable_ : pd.DataFrame of shape (n_targets, n_components)
        Per-variable R² for Y after each component.
    rmse_ : pd.DataFrame of shape (n_targets, n_components)
        Root mean squared error of Y predictions per component, on the original
        (un-scaled) Y scale, consistent with ``predictions_`` and
        ``prediction_interval``.
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component in X.
    scaling_factor_for_scores_ : pd.Series of length n_components
        Standard deviation per score (sqrt of explained variance).
    has_missing_data_ : bool
        Whether the training data contained missing values.
    fitting_info_ : dict
        Timing and iteration info from the fitting algorithm.

    See Also
    --------
    PCA : Principal Component Analysis.
    MCUVScaler : Mean-center unit-variance scaler.

    References
    ----------
    Abdi, "Partial least squares regression and projection on latent structure
    regression (PLS Regression)", 2010, DOI: 10.1002/wics.51

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.multivariate.methods import PLS, MCUVScaler
    >>> X = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]})
    >>> Y = pd.DataFrame({"y": [2.1, 3.9, 6.2, 7.8]})
    >>> pls = PLS(n_components=1)
    >>> pls = pls.fit(MCUVScaler().fit_transform(X), MCUVScaler().fit_transform(Y))
    >>> pls.scores_.shape
    (4, 1)
    """

    def __init__(  # noqa: PLR0913
        self,
        n_components: int,
        *,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = epsqrt,
        copy: bool = True,
        warn_on_uncentred: bool = True,
        # Own extra inputs, for the case when there is missing data
        missing_data_settings: dict | None = None,
    ):
        self.n_components: int = n_components
        # ``scale`` (like ``copy``) mirrors sklearn PLSRegression's constructor
        # signature for drop-in API compatibility. Unlike ``copy`` it is wired
        # up: when True, fit() centers and unit-variance-scales X and Y via
        # MCUVScaler (see fit()). It must be stored verbatim here and read only
        # in fit(), per the sklearn __init__ convention.
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        # Read only in fit(), and only on the scale=False branch: with scale=True
        # the model centres both blocks itself, so there is nothing to warn about.
        self.warn_on_uncentred = warn_on_uncentred
        self.missing_data_settings = missing_data_settings

    def __sklearn_tags__(self):
        """Declare sklearn capability tags (sklearn 1.6+).

        - ``input_tags.allow_nan=True`` because the NIPALS fit threads
          missing data through; ``fit`` and ``predict`` both pass
          ``ensure_all_finite="allow-nan"`` to ``validate_data``.
        - ``target_tags.multi_output=True`` because the X / Y blocks
          can both be multi-column (multi-target Y is the default
          chemometric PLS case).
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.target_tags.multi_output = True
        return tags

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ANN001, ARG002
        """Return the output column names of :meth:`transform`.

        :class:`PLS`'s ``transform`` returns the X scores (T matrix),
        labelled ``["T1", "T2", ..., "T{n_components}"]``. The
        ``input_features`` argument is accepted (Pipeline introspection
        passes it through) but unused: the output column count is the
        fitted ``n_components``, not the input feature count.

        Used by :meth:`set_output` (sklearn 1.2+) to label the
        :class:`~pandas.DataFrame` view of the scores when
        ``set_output(transform="pandas")`` is on, and by Pipeline
        introspection.
        """
        check_is_fitted(self, "direct_weights_")
        return np.asarray([f"T{a}" for a in self._component_names])

    # ENG-17: the 13 shared convenience methods, hotellings_t2_limit,
    # ellipse_coordinates and the rename __getattr__ are inherited from
    # _LatentVariableModel. PLS keeps only its two PLS-specific plot methods and
    # supplies its own rename map.
    predictions_vs_observed_plot = _model_method(_predictions_vs_observed_plot)
    coefficient_plot = _model_method(_coefficient_plot)
    # PLS-specific predictive-importance diagnostics (need beta_coefficients_).
    target_projection = _model_method(_target_projection)
    selectivity_ratio = _model_method(_selectivity_ratio)

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {
        "x_scores": "scores_",
        "y_scores": "y_scores_",
        "x_weights": "x_weights_",
        "y_weights": "y_weights_",
        "x_loadings": "x_loadings_",
        "y_loadings": "y_loadings_",
        "direct_weights": "direct_weights_",
        "beta_coefficients": "beta_coefficients_",
        "predictions": "predictions_",
        "squared_prediction_error": "spe_",
        "hotellings_t2": "hotellings_t2_",
        "R2": "r2_per_component_",
        "R2cum": "r2_cumulative_",
        "R2X_cum": "r2_per_variable_",
        "R2Y_cum": "r2y_per_variable_",
        "RMSE": "rmse_",
        "explained_variance": "explained_variance_",
        "scaling_factor_for_scores": "scaling_factor_for_scores_",
        "extra_info": "fitting_info_",
        "has_missing_data": "has_missing_data_",
        "N": "n_samples_",
        "K": "n_features_in_",
        "M": "n_targets_",
        "A": "n_components",
    }
    _RENAME_CONTEXT: typing.ClassVar[str] = "PLS"

    #: Which block ``r2_per_component_`` measures the variance of. PLS explains the
    #: Y block, PCA the X block, and subclasses inherit the right answer: PLSDA's
    #: r2_per_component_ is the variance of its class indicators, which is Y. Read by
    #: :func:`~process_improve.multivariate.plots.explained_variance_plot`, which
    #: defaults to "X" for any model that does not declare one. A class attribute
    #: rather than an ``isinstance`` check because ``_pls`` imports ``plots``, so the
    #: reverse import would close a cycle (#375).
    _variance_block: typing.ClassVar[str] = "Y"

    # Y-side fitted attributes: ndarrays while NIPALS fills them in, then wrapped
    # into the documented public DataFrames at the end of fit().
    y_scores_: np.ndarray | pd.DataFrame
    y_weights_: np.ndarray | pd.DataFrame
    y_loadings_: np.ndarray | pd.DataFrame
    # Fitted diagnostics: per-component arrays or scalar totals, plus, for an imputed
    # fit, which estimator ran (a string) and whether it converged.
    fitting_info_: dict[str, np.ndarray | int | float | str]
    # Declared here rather than at their first assignment, because an imputed fit
    # refits them from helper methods defined above ``fit``.
    _x_scaler: MCUVScaler | None
    _y_scaler: MCUVScaler | None

    # ENG-18: public DataFrame views built lazily from the private ndarrays.
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")
    spe_ = _LazyFrame("_spe", index="_sample_index", columns="_component_names")
    x_loadings_ = _LazyFrame("_x_loadings", index="_feature_names", columns="_component_names")
    x_weights_ = _LazyFrame("_x_weights", index="_feature_names", columns="_component_names")

    def _fit_nipals(  # noqa: PLR0915, C901
        self,
        X: DataMatrix,
        Y: DataMatrix,
        A: int,
        settings: dict,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit PLS via the NIPALS algorithm, handling missing data transparently.

        Parameters
        ----------
        X : DataMatrix
            Training X data (N x K).
        Y : DataMatrix
            Training Y data (N x M).
        A : int
            Number of components to extract.
        settings : dict
            Loop settings with keys ``md_tol`` and ``md_max_iter``.
        sample_weight : np.ndarray of shape (N,), optional
            Row weights, non-negative finite floats. When provided, X and Y
            are ``sqrt(w)``-rescaled at NIPALS entry so the cross-products
            ``X' u`` and ``Y' t`` become weighted (see #394). Loadings,
            weights, and beta are invariant under the rescale; scores are
            restored to the original sample scale at the end via
            ``X @ direct_weights`` (cleaner than dividing by ``sqrt(w)``,
            which would NaN at zero-weight rows).
        """
        N = self.n_samples_
        K = self.n_features_in_
        M = self.n_targets_

        # The blocks as given, copied in C order whatever the layout pandas kept them in, so
        # that every route to the same numbers sums them in the same order. Y is floats
        # already; an integer X, unscaled when scale=False, gives the products floats would.
        X_given, Y_given = np.asarray(X).copy(), np.asarray(Y).copy()
        Xd, Yd = X_given, Y_given

        # Weighted fit (#394): sqrt(w)-rescale X and Y up front. The NIPALS
        # cross-products X' u, Y' t, X' X etc. all become weighted sums; the
        # inner loop is otherwise unchanged. The blocks as given are kept to
        # rebuild original-scale scores after the loop (the alternative --
        # dividing the NIPALS-recovered scores by sqrt(w) -- NaNs out at
        # zero-weight rows, which the test plan explicitly requires to behave
        # like "fit on the remaining half").
        if sample_weight is not None:
            root = np.sqrt(sample_weight)[:, np.newaxis]
            Xd, Yd = X_given * root, Y_given * root

        self._scores = np.zeros((N, A))
        self.y_scores_ = np.zeros((N, A))
        self._x_weights = np.zeros((K, A))
        self.y_weights_ = np.zeros((M, A))
        self._x_loadings = np.zeros((K, A))
        self.y_loadings_ = np.zeros((M, A))

        # Every component fills its entry; a fit that stops early raises.
        timing, iterations = np.zeros(A), np.zeros(A)
        self.fitting_info_ = {"timing": timing, "iterations": iterations}

        reach = y_reach([Xd], Yd)
        exhausted = False  # pragma: no mutate - None reads as False in the or below
        for a in range(A):
            start_time = time.time()

            if ssq(Xd) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for X: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)
            if ssq(Yd) < epsqrt:
                emsg = (
                    "There is no variance left in the data array for Y: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)
            # Y has variance left, but none that X can still reach (#913): this component, and every later one,
            # takes the largest direction left in X (X as its own target) rather than iterating to max_iter.
            exhausted = exhausted or y_is_exhausted("PLS", a, [Xd], Yd, reach)
            target = Xd if exhausted else Yd

            # Seed u_a from the column with the greatest sum of squares (#195); ties go to the
            # first (#954). The deterministic sign convention applied below makes the fitted
            # sign independent of this seed. A missing cell of the seed starts at 0.
            u_a = largest_ssq_column(target) + 1.0

            # The first pass always runs: the ``+ 1.0`` offset that primes the loop
            # can be negligible relative to a large-magnitude seed column (e.g.
            # entries ~1e154), and the relative criterion (#504) would then report
            # convergence before ``t_a`` / ``w_a`` / ``c_a`` were ever assigned.
            # terminate_check ends the loop by md_max_iter at the latest.
            for itern in itertools.count(1):
                u_a_guess = u_a

                # 1: w_a = X'u_a / (u_a'u_a)
                w_a = quick_regress(Xd, u_a, per="column")

                # 2: Normalize w_a to unit length. Floor the norm: a collapsed ``w_a``
                # would make this 0/0 -> NaN and poison every later component. `_mbpls`
                # already guards the identical expression this way; see `_nz` (#513).
                w_a = w_a / _nz(float(np.sqrt(ssq(w_a))))

                # 3: t_a = X w_a / (w_a'w_a)
                t_a = quick_regress(Xd, w_a, per="row")

                # 4: c_a = Y't_a / (t_a't_a)
                c_a = quick_regress(target, t_a, per="column")

                # 5: u_a = Y c_a / (c_a'c_a)
                u_a = quick_regress(target, c_a, per="row")

                if terminate_check(u_a_guess, u_a, iterations=itern, settings=settings):
                    break

            if exhausted:  # the Y side of a component along X: a Y-loading of about zero, and no Y-score
                c_a = quick_regress(Yd, t_a, per="column")
                u_a = np.zeros_like(t_a)

            timing[a] = time.time() - start_time
            iterations[a] = itern
            logger.debug(
                "PLS NIPALS: component %d converged in %d iterations (tol=%g)",
                a + 1,
                itern,
                settings["md_tol"],
            )

            # ``terminate_check`` stops the loop when ``iterations >=
            # md_max_iter``, so ``itern`` is capped AT the maximum and the
            # previous strict ``>`` comparison could never fire: non-convergent
            # fits returned silently.
            if itern >= settings["md_max_iter"]:
                warnings.warn(
                    f"PLS NIPALS: component {a + 1} reached the maximum number of "
                    f"iterations ({settings['md_max_iter']}) without converging.",
                    SpecificationWarning,
                    stacklevel=2,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin the caller
                )

            # 6: Compute loadings and deflate
            p_a = quick_regress(Xd, t_a, per="column")
            Xd = Xd - np.dot(t_a, p_a.T)
            Yd = Yd - np.dot(t_a, c_a.T)

            # Flip signs so largest-magnitude loading element is positive
            sign = -1.0 if np.sign(p_a[np.argmax(np.abs(p_a))]) < 1 else 1.0
            t_a, u_a, w_a, p_a, c_a = sign * t_a, sign * u_a, sign * w_a, sign * p_a, sign * c_a

            self._scores[:, a] = t_a.flatten()
            self.y_scores_[:, a] = u_a.flatten()
            self._x_weights[:, a] = w_a.flatten()
            self._x_loadings[:, a] = p_a.flatten()
            self.y_loadings_[:, a] = c_a.flatten()
            # In PLS mode A (PLSRegression), y_weights == y_loadings
            self.y_weights_[:, a] = c_a.flatten()

        # Weighted fit post-fix (#394): the NIPALS scores stored above are on
        # the sqrt(w)-rescaled rows (T_w = sqrt(W) @ T_orig). Recover
        # original-scale scores by re-projecting the unweighted X / Y via the
        # converged loadings: T = X @ W (P'W)^{-1}, the same identity that
        # makes direct_weights_ meaningful. This rebuild correctly handles
        # zero-weight rows (which NIPALS leaves as identically zero, but
        # whose natural projected score is X[i] @ direct_weights).
        if sample_weight is not None:
            self._scores = X_given @ _direct_weights(self._x_weights, self._x_loadings)
            # y_scores are diagnostic only; the deflation identity
            # u_a = Y_a c_a / (c_a' c_a) gives the natural y-side scores
            # using the unweighted (deflated) Y.
            Y_deflated = Y_given
            for a in range(A):
                c_a = self.y_loadings_[:, [a]]
                denom = float((c_a.T @ c_a).item())
                # A component along X alone (#913) has a Y-loading of about zero.
                if denom > epsqrt:  # pragma: no mutate - >= differs only at a squared length of exactly epsqrt
                    self.y_scores_[:, a] = (Y_deflated @ c_a / denom).flatten()
                Y_deflated = Y_deflated - (self._scores[:, [a]] @ c_a.T)

    def _fit_imputed(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        A: int,
        settings: dict,
        sample_weight: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill the missing cells from a model of ``[X Y]`` jointly, then fit once (#189).

        The missing cells are estimated by
        :func:`~process_improve.multivariate._impute.impute_low_rank`: EM on a
        principal-component model of the joint matrix, the model-building algorithm
        ``PCA(algorithm="tsr")`` also runs. PLS is then fitted to the completed data,
        scaled by the completed data's own statistics.

        The first version of this rebuilt the missing cells from the PLS model itself,
        alternating fit and rebuild. That is the wrong tool twice over. PLS picks its
        components for their covariance with Y, not to reproduce X, so it rebuilds X
        only as well as those components happen to span it: on the LDPE data at three
        components it was a coin flip against plain NIPALS. And the alternation is not
        an EM, so nothing made it converge: on a synthetic fixture about one fit in
        seven cycled indefinitely. The joint model converged on all of those, and
        beat NIPALS on 9 to 12 of 12 LDPE trials where the PLS-based loop managed 5 to 8.

        Parameters
        ----------
        X, Y : pd.DataFrame
            The training blocks in their original units, NaN where missing.
        A : int
            Number of components, used for the imputation model as well as for PLS.
        settings : dict
            Resolved missing-data settings: ``md_method`` (``"tsr"`` or ``"pmp"``),
            and ``md_tol`` and ``md_max_iter`` for the imputation rounds.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Applied to the PLS fit. The imputation models the data as given.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``X`` and ``Y`` in the fitted model's scaled space, still NaN where they
            were missing, so the diagnostics are computed on observed cells only.
        """
        method = settings["md_method"].lower()
        joint = pd.concat([X, Y], axis=1).to_numpy(dtype=float)
        # A column with nothing observed has nothing to estimate it from. NIPALS gives
        # such a column no weight, so it is held out of the imputation and set to a
        # constant, which scaling then reduces to zero: the same outcome, without
        # making the choice of md_method decide whether the data is fittable.
        empty = np.all(np.isnan(joint), axis=0)
        imputation = impute_low_rank(
            joint[:, ~empty], A, method=method, tol=settings["md_tol"], max_iter=settings["md_max_iter"]
        )
        completed = np.zeros_like(joint)
        completed[:, ~empty] = imputation.completed

        if not imputation.converged:
            warnings.warn(
                f"PLS {method.upper()}: the imputed cells were still moving by {imputation.shift:.3g} "
                f"standard deviations after {imputation.rounds} rounds (tol={settings['md_tol']:g}). "
                "Raise max_iter, or loosen tol if that change is already negligible for your data.",
                SpecificationWarning,
                stacklevel=3,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin the caller
            )

        # The scalers read only the values, and the rows of a weight above 0, so the completed
        # blocks need their column labels (the scalers' feature names) and not their row labels.
        n_x = X.shape[1]
        completed_x = pd.DataFrame(completed[:, :n_x], columns=X.columns)
        completed_y = pd.DataFrame(completed[:, n_x:], columns=Y.columns)
        x_scaled, y_scaled = self._rescaled(completed_x, completed_y, sample_weight)
        self._fit_nipals(x_scaled, y_scaled, A, settings, sample_weight=sample_weight)
        self.fitting_info_.update(
            {
                "md_method": method,
                "md_rounds": imputation.rounds,
                "md_converged": imputation.converged,
                "md_shift": imputation.shift,
            }
        )
        return self._scaled_as_fitted(X, Y)

    def _rescaled(
        self, X: pd.DataFrame, Y: pd.DataFrame, sample_weight: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refit the scalers on complete blocks and return them scaled; identity when ``scale=False``."""
        if self._centres_and_scales():
            self._x_scaler, self._y_scaler = self._make_scalers(X, Y, sample_weight)
        x_scaled, y_scaled = self._scaled_as_fitted(X, Y)
        return x_scaled.to_numpy(), y_scaled.to_numpy()

    def _scaled_as_fitted(self, X: pd.DataFrame, Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Put X and Y into the current scaled space, with the scalers as they stand."""
        # The two scalers are set and cleared together; the second test narrows the type for mypy.
        if self._x_scaler is None or self._y_scaler is None:  # pragma: no mutate - "and" is the same
            return X, Y
        return self._x_scaler.transform(X), self._y_scaler.transform(Y)

    def _centres_and_scales(self) -> bool:
        """Whether ``fit`` passes X and Y through :meth:`_make_scalers`: when ``scale``, and always for PRM."""
        return bool(self.scale)

    def _make_scalers(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        sample_weight: np.ndarray | None,
    ) -> tuple[MCUVScaler, MCUVScaler]:
        """Build the (X, Y) centring / scaling pair that ``fit`` will use.

        Split out as its own method so a resistant subclass can substitute a
        different notion of "middle" and "spread" without reimplementing
        ``fit``: :class:`~process_improve.multivariate.methods.PRM` centres at
        the L1 median, because the ordinary mean and standard deviation are
        exactly as fragile as the fit they are preparing (#191).

        The scalers are fitted on the rows that actually enter the fit. With
        ``sample_weight``, zero-weight rows are excluded from NIPALS (via the
        ``sqrt(w)`` rescale), so they must not influence the center / scale
        either; otherwise a zero-weight row would stop being equivalent to a
        dropped row. All rows are then transformed with those statistics.

        Parameters
        ----------
        X, Y : pd.DataFrame
            The training blocks, before any scaling.
        sample_weight : np.ndarray of shape (n_samples,) or None
            Row weights, as passed to :meth:`fit`.

        Returns
        -------
        tuple[MCUVScaler, MCUVScaler]
            Fitted scalers for X and Y respectively.
        """
        x_fit_rows = X if sample_weight is None else X[sample_weight > 0]
        y_fit_rows = Y if sample_weight is None else Y[sample_weight > 0]
        return MCUVScaler().fit(x_fit_rows), MCUVScaler().fit(y_fit_rows)

    def fit(  # noqa: PLR0912, PLR0915, C901
        self,
        X: DataMatrix,
        Y: DataMatrix,
        sample_weight: np.ndarray | None = None,
    ) -> PLS:
        """
        Fit a projection to latent structures (PLS) model to the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where ``n_samples`` is the number of samples (rows)
            and ``n_features`` is the number of features (columns).
        Y : array-like, shape (n_samples, n_targets)
            Training data, where ``n_samples`` is the number of samples (rows)
            and ``n_targets`` is the number of target outputs (columns).
        sample_weight : array-like of shape (n_samples,), optional
            Non-negative row weights for a weighted PLS fit (#394). NIPALS
            is run on ``sqrt(w)``-rescaled X and Y, which is equivalent to
            weighting the cross-products ``X' W u`` and ``Y' W t``. Loadings,
            weights and beta are computed correctly; scores are returned on
            the original sample scale. Zero weights effectively exclude the
            corresponding rows (``sample_weight=[1,1,0,0,1]`` reproduces the
            unweighted fit on rows ``[0,1,4]``). Forwarded to ``score()`` /
            ``r2_score`` for any caller that also threads it through.

        Returns
        -------
        PLS
            Model object.

        References
        ----------
        Abdi, "Partial least squares regression and projection on latent structure
        regression (PLS Regression)", 2010, DOI: 10.1002/wics.51
        """
        # Accept a Series for Y by promoting it to one column with its own labels; a 1-D array
        # becomes one column below, labelled like X's rows as a 2-D array is (#1027).
        if isinstance(Y, pd.Series):
            Y = Y.to_frame()

        # Validate sample_weight up front so the error path is the same
        # regardless of whether the caller passes ndarray or list / scalar.
        # We defer the row-count check until after validate_data has set
        # n_samples_.
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).ravel()
            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight must be finite (no NaN / inf).")
            if np.any(sample_weight < 0):
                raise ValueError("sample_weight must be non-negative.")

        # Reject sparse before validate_data does, so the message names the
        # ColumnTransformer knob rather than `.toarray()` (#399).
        _reject_sparse(X, "PLS")
        # Capture DataFrame metadata before validate_data converts X to ndarray
        # so the downstream DataFrame view keeps its row/column labels; pandas
        # numbers an array's rows and columns from 0.
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(self, X, ensure_min_samples=2, ensure_all_finite="allow-nan")
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=sample_index)
        # The responses as floats, a pandas missing value (pd.NA) as NaN, as validate_data reads X:
        # without scaling nothing else converts them, and pd.NA raised TypeError (#1027). Columns
        # of floats are left as they are: a copy can change the memory layout, and with it the
        # order the scaler sums them in, so the fit would differ in its last digits from one
        # scaled outside the model.
        if not (Y.dtypes == np.float64).all():
            Y = Y.astype(float)  # pragma: no mutate - pandas reads astype(None) as float64 too

        self.n_samples_: int = X.shape[0]
        Ny: int = Y.shape[0]
        self.n_targets_: int = Y.shape[1]
        if Ny != self.n_samples_:
            raise ValueError(
                f"The X and Y arrays must have the same number of rows: X has {self.n_samples_} and Y has {Ny}."
            )
        if sample_weight is not None and sample_weight.shape[0] != self.n_samples_:
            raise ValueError(
                f"sample_weight has {sample_weight.shape[0]} entries; expected {self.n_samples_} to match X / Y."
            )

        N = self.n_samples_
        K = self.n_features_in_

        # The remainder of fit() uses the pandas DataFrame API (.isna / .index /
        # .columns); the NIPALS core in _fit_nipals already np.asarray-copies the
        # numeric values. Narrow the static type accordingly (the DataFrame path
        # is unchanged at runtime).
        assert isinstance(X, pd.DataFrame)
        assert isinstance(Y, pd.DataFrame)

        # Honour ``scale`` (see __init__): center + unit-variance-scale X and Y
        # with the library's MCUVScaler, matching sklearn PLSRegression's
        # ``scale=True`` default. The NIPALS core and every fitted attribute run
        # in this scaled space; the user-facing predictions_ / beta_coefficients_
        # / predict() outputs below are mapped back to the original data scale.
        # On already-scaled input (center ~ 0, scale ~ 1) this is a no-op, so
        # callers who pre-scale (scale=False) or pass MCUVScaler output are
        # unaffected. MCUVScaler is NaN-aware and leaves constant columns at 1.
        self._x_scaler = None
        self._y_scaler = None
        # An imputed fit (md_method "tsr" / "pmp") re-estimates the scaling from the
        # completed data every round, so it needs the blocks before this first scaling.
        unscaled_x, unscaled_y = X, Y
        if self._centres_and_scales():
            self._x_scaler, self._y_scaler = self._make_scalers(X, Y, sample_weight)
            X = self._x_scaler.transform(X)
            Y = self._y_scaler.transform(Y)
        elif self.warn_on_uncentred:
            _warn_if_uncentred(X, Y)

        # Check if number of components is supported against maximum requested
        min_dim = min(N, K)
        if self.n_components is not None and int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1; got {self.n_components}.")
        A = min_dim if self.n_components is None else int(self.n_components)
        if min_dim < A:
            warn = (
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({N}) or "
                f"the number of columns ({K})."
            )
            warnings.warn(
                warn,
                SpecificationWarning,
                stacklevel=2,  # pragma: no mutate - mutmut's trampoline adds frames, so no test can pin the caller
            )
            A = min_dim
        # The resolved (possibly clamped) component count is a fitted attribute;
        # the constructor parameter n_components is left exactly as the user set
        # it, per the sklearn clone/get_params contract (#505).
        self.n_components_ = A

        self.has_missing_data_ = bool(Y.isna().to_numpy().any() or X.isna().to_numpy().any())

        # One resolution, whether or not the data has gaps, through the resolver every
        # iterative estimator shares (#588). ``tol`` and ``max_iter`` supply the values.
        #
        # Seeding from ``self`` is what makes ``PLS(tol=...)`` reach the NIPALS loop in
        # both cases: the missing-data branch used to hard-code its tolerance at epsqrt
        # while taking the iteration cap from the constructor, so a caller's ``tol``
        # applied to complete data and was dropped as soon as a cell went missing.
        #
        # ``md_method`` is set before the resolver so a caller's dict can still override
        # it: it is the one axis ``missing_data_settings`` still owns. It defaults to
        # NIPALS, the cheapest path: it handles per-cell NaN directly via skipna sums
        # inside its iterations, with no imputation. The resolved settings stay local:
        # mutating the constructor parameter would leak into clone() (#505).
        settings: dict[str, typing.Any] = {"md_method": "nipals"}  # pragma: no mutate - read lowercased below
        settings.update(
            resolve_loop_settings(
                tol=self.tol,
                max_iter=self.max_iter,
                missing_data_settings=self.missing_data_settings,
                estimator_name="PLS",
            )
        )
        _check_md_method(settings)
        if settings["md_method"].lower() == "nipals":
            self._fit_nipals(X, Y, A, settings, sample_weight=sample_weight)
        else:
            X, Y = self._fit_imputed(unscaled_x, unscaled_y, A, settings, sample_weight=sample_weight)

        # --- Common post-fit path: wrap numpy arrays into pandas ---

        direct_weights = _direct_weights(self._x_weights, self._x_loadings)
        # beta = RC' [KxM]: direct link from k-th X variable to m-th Y variable.
        # NIPALS ran in the (optionally) scaled space, so this beta maps scaled X
        # to scaled Y. Rescale to the original data units so it stays the
        # documented raw-X -> raw-Y relationship (and matches sklearn's coef_):
        # beta_orig[k, m] = beta_scaled[k, m] * y_scale[m] / x_scale[k].
        _, x_scale = _back_to_units(self._x_scaler)
        y_centre, y_scale = _back_to_units(self._y_scaler)
        T, P, C = self._scores, self._x_loadings, self.y_loadings_
        beta_coefficients = direct_weights / np.atleast_1d(x_scale)[:, np.newaxis] @ C.T * y_scale

        component_names = list(range(1, A + 1))
        # ENG-18: scores_ / x_weights_ / x_loadings_ / spe_ are stored as private
        # ndarrays (the source of truth); their public DataFrame views are built
        # lazily by the _LazyFrame descriptors from the metadata attrs below.
        self._sample_index = X.index
        self._feature_names = X.columns
        self._component_names = component_names
        self.y_scores_ = pd.DataFrame(self.y_scores_, index=Y.index, columns=component_names)
        self.y_weights_ = pd.DataFrame(self.y_weights_, index=Y.columns, columns=component_names)
        self.y_loadings_ = pd.DataFrame(C, index=Y.columns, columns=component_names)
        # NIPALS predictions are in scaled-Y space; report on the original scale.
        self.predictions_ = pd.DataFrame(T @ C.T * y_scale + y_centre, index=Y.index, columns=Y.columns)
        self.direct_weights_ = pd.DataFrame(direct_weights, index=X.columns, columns=component_names)
        self.beta_coefficients_ = pd.DataFrame(beta_coefficients, index=X.columns, columns=Y.columns)
        # fit refuses fewer than two rows (ensure_min_samples=2), so N - 1 is at least 1.
        self.explained_variance_ = np.diag(T.T @ T) / (N - 1)
        # The training residual block, kept for the trimmed-score-regression
        # estimator in ``project`` / ``projection_matrix``: TSR regresses on
        # the covariance of the observed columns, which is the model plane
        # plus this spread around it. Kept as E (N x K) rather than its K x K
        # Gram, because the estimator contracts it to A columns straight away.
        # A missing training cell has no residual, so it enters as zero.
        # Y is floats; an integer X, unscaled, subtracts and sums as floats would.
        X_values, Y_values = X.to_numpy(), Y.to_numpy()
        self._x_residuals = np.nan_to_num(X_values - T @ P.T)
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )
        # A component with ~0 score standard deviation (rank-deficient fit)
        # contributes 0/0 to T²; _t2_score_sd leaves it out, as PCA does, rather
        # than dividing by ~zero and poisoning T² with inf/NaN.
        score_sd = _t2_score_sd(self.scaling_factor_for_scores_.to_numpy())
        self.hotellings_t2_ = pd.DataFrame(
            np.cumsum((T / score_sd) ** 2, axis=1), index=X.index, columns=component_names
        )

        # What is left of X and of Y after the first a components, for a = 1..A. Every
        # sum skips a missing cell. R2Y is 1 - SSE/SST, the convention of the X side:
        # the SS(Yhat)/SS(Y) form is identical when the scores are orthogonal, but with
        # missing data they are not, and SS(Yhat) can DECREASE as a component is added,
        # which made r2_per_component_ negative and vip() NaN. Per-variable R^2 is
        # undefined for a column with no variance to explain: NaN there, SEC-21 (#270).
        x_total, y_total = ssq(X_values, axis=0), ssq(Y_values, axis=0)
        spe, r2x, r2y, r2_cumulative, rmse = [], [], [], [], []
        for a in range(1, A + 1):
            x_left = X_values - T[:, :a] @ P[:, :a].T
            y_left = Y_values - T[:, :a] @ C[:, :a].T
            spe.append(np.sqrt(ssq(x_left, axis=1)))
            r2x.append(1 - ssq(x_left, axis=0) / np.where(x_total > 0, x_total, np.nan))
            r2y.append(1 - ssq(y_left, axis=0) / np.where(y_total > 0, y_total, np.nan))
            r2_cumulative.append(1 - ssq(y_left) / np.sum(y_total))
            # Over the observed cells, as the R2 is: one missing Y cell made its column NaN (#1027).
            rmse.append(np.sqrt(np.nanmean(y_left**2, axis=0)))
        self._spe = np.column_stack(spe)
        self.r2_cumulative_ = pd.Series(r2_cumulative, index=component_names, name="Output cumulative R²")
        self.r2_per_component_ = pd.Series(
            np.diff(r2_cumulative, prepend=0.0), index=component_names, name="Output R² per component"
        )
        self.r2_per_variable_ = pd.DataFrame(np.column_stack(r2x), index=X.columns, columns=component_names)
        self.r2y_per_variable_ = pd.DataFrame(np.column_stack(r2y), index=Y.columns, columns=component_names)
        # rmse_ is reported on the ORIGINAL Y scale, consistent with predictions_,
        # diagnose().y_hat and prediction_interval(), which uses it as the residual error.
        self.rmse_ = pd.DataFrame(
            np.column_stack(rmse) * np.atleast_1d(y_scale)[:, np.newaxis], index=Y.columns, columns=component_names
        )

        return self

    def transform(self, X: DataMatrix, Y: DataMatrix | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Project X (and optionally Y) into the latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        Y : array-like of shape (n_samples, n_targets), optional
            Ignored. Present for API compatibility with sklearn pipelines.

        Returns
        -------
        X_scores : pd.DataFrame of shape (n_samples, n_components)
            Projected X data (scores).
        """
        return self._scaled_rows(X) @ self.direct_weights_

    def _scaled_rows(self, X: DataMatrix) -> pd.DataFrame:
        """Return new rows checked against the fit, in the space the model was fitted in.

        Every method that takes new rows reads them through here. A DataFrame's string column
        labels are matched by name, as in :func:`_align_to_fit_features`; any other columns are
        read by position, as ``validate_data`` and the scaler read them, so the rows are the same
        values to every method whatever their labels and whether or not the model scales. A
        DataFrame keeps its row labels; pandas numbers an array's rows from 0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New rows; NaN marks a missing cell.

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            The rows, scaled as the fit scaled its own, columns labelled as fitted.
        """
        check_is_fitted(self, "direct_weights_")
        # sklearn's validate_data(reset=False) checks feature-name *order*; _align_to_fit_features
        # only needs set equality, so a reordered DataFrame is put in the fitted order first.
        if isinstance(X, pd.DataFrame):
            X = _align_to_fit_features(X, self._feature_names)
        X_arr = validate_data(
            self,
            X,
            reset=False,  # pragma: no mutate - validate_data reads reset=None as False
            ensure_all_finite="allow-nan",
        )
        rows = pd.DataFrame(X_arr, index=X.index if isinstance(X, pd.DataFrame) else None, columns=self._feature_names)
        return rows if self._x_scaler is None else self._x_scaler.transform(rows)

    def fit_transform(self, X: DataMatrix, Y: DataMatrix | None = None) -> pd.DataFrame:
        """Fit the model and return X scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_targets)
            Required despite the ``None`` default: the default is present only
            for signature-compatibility with the sklearn ``fit_transform``
            protocol, and PLS cannot be fitted without responses. Omitting
            ``Y`` raises :class:`ValueError`, under ``python -O`` as well.

        Returns
        -------
        X_scores : pd.DataFrame of shape (n_samples, n_components)
        """
        if Y is None:
            raise ValueError(
                "Y is required by PLS.fit_transform; got None. The None default exists only to "
                "match the sklearn fit_transform(X, y=None) signature, and PLS cannot be fitted "
                "without responses. Pass the response block, e.g. model.fit_transform(X, Y)."
            )
        self.fit(X, Y)
        return self.scores_

    def score(self, X: DataMatrix, Y: DataMatrix, sample_weight: np.ndarray | None = None) -> float:
        """Return the R² score for the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Y : array-like of shape (n_samples, n_targets)
            True target values.
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        score : float
            R² of ``self.predict(X)`` w.r.t. *Y*.
        """
        y_pred = self.predict(X)
        return float(r2_score(Y, y_pred, sample_weight=sample_weight))

    def predict(self, X: DataMatrix) -> pd.DataFrame:
        """Predict Y for new observations.

        Returns just the predicted ``y_hat`` so the call satisfies the
        scikit-learn :class:`~sklearn.base.RegressorMixin` contract (and
        therefore composes inside :class:`~sklearn.pipeline.Pipeline`,
        :func:`~sklearn.model_selection.cross_val_score`, and
        :class:`~sklearn.model_selection.GridSearchCV`). For the rich
        diagnostic view (scores, Hotelling's T², SPE, plus ``y_hat``),
        see :meth:`diagnose`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_hat : pd.DataFrame of shape (n_samples, n_targets)
            Predicted target values, indexed by ``X``'s rows and labelled
            with the target column names captured during ``fit``.

        See Also
        --------
        diagnose : richer per-prediction diagnostics.

        Examples
        --------
        >>> y_pred = pls.predict(X_new)
        >>> diag = pls.diagnose(X_new)   # for scores / T² / SPE
        """
        return self.diagnose(X).y_hat

    def diagnose(self, X: DataMatrix) -> Bunch:
        """Project new data and compute predictions plus diagnostics.

        This is the rich view that :meth:`predict` used to return before
        1.35.0: alongside ``y_hat`` it reports the X scores, cumulative
        Hotelling's T², and SPE for every row of ``X`` so the user can
        flag out-of-model observations *and* read their predicted Y from
        one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``scores`` (pd.DataFrame, ``(n_samples, n_components)``).
            - ``hotellings_t2`` (pd.Series, ``(n_samples,)``): T² of the full
              model, the last column of the fitted ``hotellings_t2_``.
            - ``spe`` (pd.Series, ``(n_samples,)``): SPE of the full model.
            - ``y_hat`` (pd.DataFrame, ``(n_samples, n_targets)``): the
              predictions, on the original Y scale.

        Notes
        -----
        :meth:`PCA.diagnose` returns ``hotellings_t2`` as a cumulative
        ``pd.DataFrame`` with one column per component, of which the last is
        this Series. The two shapes are kept as they are because each is public
        API; compare them through that last column.

        See Also
        --------
        predict : sklearn-compatible call returning just ``y_hat``.

        Examples
        --------
        >>> result = pls.diagnose(scaler_x.transform(X_new))
        >>> result.y_hat           # Predicted Y values
        >>> result.spe             # SPE for each new observation
        >>> result.hotellings_t2   # T² for each new observation
        """
        # Scores, T², SPE and y_hat are all computed in the space the model was fitted in.
        X = self._scaled_rows(X)
        scores = X @ self.direct_weights_

        # Hotelling's T² (cumulative over all components), leaving out a component fit leaves out (#1026)
        # A Series of the scores' row labels: the DataFrame's row sums carry them.
        t2 = pd.Series(((scores / _t2_score_sd(self.scaling_factor_for_scores_.to_numpy())) ** 2).sum(axis=1))
        t2.name = "Hotelling's T²"

        # SPE: residual after reconstructing X from its scores. Reconstruct in NumPy so the
        # feature columns stay aligned with X: a DataFrame ``scores @ self._x_loadings.T`` product
        # relabels the columns 0..K-1, which then misaligns the ``X - X_hat`` subtraction to all-NaN
        # and collapses every SPE to 0.
        x_hat = scores.to_numpy() @ self._x_loadings.T
        residuals = X.to_numpy() - x_hat
        spe_values = pd.Series(np.sqrt(np.power(residuals, 2).sum(axis=1)), index=X.index, name="SPE")

        # Y predictions (computed in scaled-Y space; mapped back to original units)
        y_hat = scores @ self.y_loadings_.T
        if self._y_scaler is not None:
            y_hat = self._y_scaler.inverse_transform(y_hat)

        return Bunch(scores=scores, hotellings_t2=t2, spe=spe_values, y_hat=y_hat)

    def project(
        self,
        X: DataMatrix,
        *,
        method: str = "tsr",  # pragma: no mutate - _validate_method lowercases it, so "TSR" is the same default
        ridge: float = 0.0,
    ) -> Bunch:
        """Estimate scores, prediction and diagnostics for rows with missing values.

        Whereas :meth:`transform` and :meth:`diagnose` propagate NaN into the
        scores, this method estimates the scores of partially-observed rows
        from the observed columns only, using the missing-data estimators of
        Arteaga and Ferrer (2002): trimmed score regression (``"tsr"``, the
        default and statistically the strongest), single-component projection
        (``"scp"``, the score step of NIPALS itself: project onto the observed
        part of each weight vector, deflate with the loadings), or projection
        to the model plane (``"pmp"``). Rows with no missing values take the
        standard complete-data path, so their scores are bitwise identical to
        :meth:`transform`. As the observed part grows to the whole row, TSR
        and SCP tend to the model's own scores; PMP, the least-squares fit of
        the observed columns onto the loadings, does not for a PLS model,
        whose scores come from the weights rather than the loadings, so
        prefer the other two here.

        This is the "batch so far" primitive for predicting the final quality
        of a running batch: the future part of the unfolded row is missing by
        construction, and the prediction at each decision point is this
        projection followed by the ordinary Y regression (Garcia-Munoz,
        Kourti and MacGregor, 2004; Flores-Cerrillo and MacGregor, 2004).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New observations on the original (unscaled) X units when the
            model was fitted with ``scale=True``, or in the model's scaled
            space otherwise, exactly as :meth:`transform` expects. NaN marks
            a missing entry; rows that are entirely NaN are rejected.
        method : {"tsr", "scp", "pmp"}, default="tsr"
            The score estimator; see
            :mod:`process_improve.multivariate._projection`.
        ridge : float, default=0.0
            Non-negative regularisation added to the matrix inverted by the
            ``"tsr"`` and ``"pmp"`` estimators. Raise it above zero when
            ``condition_number`` reports near-singularity (typically very
            early in a batch, when few columns are observed).

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``scores`` (DataFrame), ``y_hat`` (DataFrame, on the
            original Y units), ``hotellings_t2`` (Series; total over all
            components), ``spe`` (Series; square root of the residual sum of
            squares over the *observed* columns only), ``condition_number``
            (Series; 1.0 when nothing is missing) and ``n_observed``
            (Series). SPE and T2 of a partially-observed row must be compared
            against limits built from the same missingness pattern, not the
            full-observation limits.
        """
        # Missing entries survive the scaling as NaN, so the pattern of observed columns is unchanged.
        X_df = self._scaled_rows(X)
        sample_index = X_df.index
        raw = project_rows(
            self._x_loadings,
            self.direct_weights_.to_numpy(),
            self.explained_variance_,
            X_df.to_numpy(),
            method=method,
            ridge=ridge,
            x_weights=self._x_weights,
            x_residuals=getattr(self, "_x_residuals", None),
        )
        scores = pd.DataFrame(raw.scores, index=sample_index, columns=self._component_names)
        # A component fit leaves out of T² is left out here too (#1026).
        s = _t2_score_sd(self.scaling_factor_for_scores_.to_numpy())
        t2 = pd.Series(np.sum((raw.scores / s) ** 2, axis=1), index=sample_index, name="Hotelling's T2")
        y_hat = scores @ self.y_loadings_.T
        if self._y_scaler is not None:
            y_hat = self._y_scaler.inverse_transform(y_hat)
        return Bunch(
            scores=scores,
            y_hat=y_hat,
            hotellings_t2=t2,
            spe=pd.Series(raw.spe, index=sample_index, name="SPE"),
            condition_number=pd.Series(raw.condition_number, index=sample_index, name="condition_number"),
            n_observed=pd.Series(raw.n_observed, index=sample_index, name="n_observed"),
        )

    def projection_matrix(
        self,
        observed: object,
        *,
        method: str = "tsr",  # pragma: no mutate - _validate_method lowercases it, so "TSR" is the same default
        ridge: float = 0.0,
    ) -> Bunch:
        """Build the fixed linear operator mapping observed columns to score estimates.

        For a fixed missingness pattern, every estimator in :meth:`project`
        is a fixed linear map ``t_hat = M @ z_observed`` on the model's
        scaled X space. This method exposes that matrix so callers that reuse
        one pattern many times (an online monitor at time sample ``k``, or a
        mid-course optimiser treating the candidate future columns as
        observed) can precompute it once. Note the matrix acts on *scaled*
        values: when the model was fitted with ``scale=True``, apply the
        internal centring and scaling first (as :meth:`project` does).

        Parameters
        ----------
        observed : array-like
            Either a boolean mask of length ``n_features_in_`` (True =
            observed), or a list of feature labels to treat as observed.
        method : {"tsr", "scp", "pmp"}, default="tsr"
        ridge : float, default=0.0

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``matrix`` (DataFrame, n_components x n_observed,
            columns labelled by the observed features), ``condition_number``
            (float) and ``method``.
        """
        check_is_fitted(self, "direct_weights_")
        mask = coerce_observed_mask(observed, self._feature_names)
        op = operator_for_pattern(
            self._x_loadings,
            self.direct_weights_.to_numpy(),
            self.explained_variance_,
            mask,
            method=method,
            ridge=ridge,
            x_weights=self._x_weights,
            x_residuals=getattr(self, "_x_residuals", None),
        )
        matrix = pd.DataFrame(
            op.matrix,
            index=self._component_names,
            columns=pd.Index(self._feature_names)[mask],
        )
        return Bunch(matrix=matrix, condition_number=op.condition_number, method=op.method)

    def invert(
        self,
        y_desired: float | np.ndarray | pd.Series | pd.DataFrame | dict,
        *,
        null_space_coordinates: np.ndarray | None = None,
    ) -> Bunch:
        r"""Invert the PLS model: find inputs that yield a desired response.

        PLS is normally used in the forward direction (:meth:`predict`): given
        inputs ``X``, predict the response ``Y``. Model *inversion* runs the
        model backwards: fix the response you want (``y_desired``) and solve for
        an input vector that the model predicts will achieve it. This is the
        basis of latent-variable product and process design (Jaeckle and
        MacGregor, 2000).

        Because a PLS model usually retains more components ``A`` than the rank
        ``r`` of the response, the target pins down only ``r`` of the ``A`` score
        directions and the inversion is underdetermined: a whole
        ``(A - r)``-dimensional family of input vectors yields the same
        prediction. That family is the *null space*. This method returns the
        minimum-norm (direct-inversion) solution together with an orthonormal
        basis for the null space, so callers can move along it to satisfy
        secondary criteria (cost, safety, operability) without changing the
        predicted response.

        For a single response (``r = 1``), García-Carrión et al. (2025) proved
        that this null space is the same linear space as the *orthogonal space*
        isolated by an O-PLS model with the same total number of components.

        Parameters
        ----------
        y_desired : float, array-like, pandas Series/DataFrame, or dict
            The desired response, on the original (un-scaled) Y scale. A scalar
            is accepted for a single-target model; otherwise supply one value per
            target. A Series/DataFrame/dict is aligned to the fitted target
            names; a plain array must follow the fitted target order.
        null_space_coordinates : np.ndarray, optional
            Coordinates along the null-space basis, of length ``A - r`` (the
            null-space dimension). When given, the returned solution is
            ``tau_direct_inversion + null_space_basis @ null_space_coordinates``
            reconstructed into the input space. All such solutions yield the same
            predicted response. When omitted, the minimum-norm (direct-inversion)
            solution is returned.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            ``x_new`` : pd.Series of shape (n_features,)
                The estimated input vector, on the original (un-scaled) X scale.
            ``scores`` : pd.Series of length A
                The score vector (tau) of the solution.
            ``y_hat`` : pd.Series of length n_targets
                The model's prediction at ``x_new``; equals ``y_desired`` up to
                numerical error, a check that the inversion is consistent.
            ``null_space_basis`` : pd.DataFrame of shape (A, A - r)
                Orthonormal basis of the null space, in score coordinates. Empty
                (zero columns) when ``A == r`` and the solution is unique.
            ``null_space_dimension`` : int
                ``A - r``, the number of free directions.
            ``hotellings_t2`` : float
                Hotelling's T² of the solution, to flag extrapolation beyond the
                calibration data. Compare against :meth:`hotellings_t2_limit`.

        See Also
        --------
        predict : the forward direction, X -> Y.
        hotellings_t2_limit : confidence limit to judge ``hotellings_t2``.

        References
        ----------
        C. M. Jaeckle and J. F. MacGregor, "Industrial applications of product
        design through the inversion of latent variable models", Chemometrics
        and Intelligent Laboratory Systems, 50 (2000): 199-210,
        DOI: 10.1016/S0169-7439(99)00058-1.

        S. García-Carrión et al., "On the equivalence between null space and
        orthogonal space in latent variable regression modeling", Journal of
        Chemometrics, 39 (2025): e70057, DOI: 10.1002/cem.70057.

        Examples
        --------
        >>> result = pls.invert(y_desired=25.0)
        >>> result.x_new              # input vector giving the target response
        >>> result.null_space_basis   # directions that leave the response fixed
        >>> pls.predict(result.x_new.to_frame().T)   # ~= 25.0
        """
        check_is_fitted(self, "y_loadings_")

        # ``beta_coefficients_`` is only ever a DataFrame (unlike ``y_loadings_``,
        # which is an ndarray inside ``_fit_nipals`` before being wrapped), so its
        # columns are the reliable source of the fitted target names.
        target_names = self.beta_coefficients_.columns
        feature_names = self._feature_names
        component_names = self._component_names
        n_targets = len(target_names)

        # --- Coerce y_desired to a 1-row DataFrame on the original Y scale ---
        if isinstance(y_desired, pd.DataFrame):
            y_des_df = y_desired.reindex(columns=target_names)
        elif isinstance(y_desired, (pd.Series, dict)):
            y_des_df = pd.Series(y_desired).reindex(target_names).to_frame().T
        else:
            y_arr = np.atleast_2d(np.asarray(y_desired, dtype=float).ravel())
            if y_arr.shape[1] != n_targets:
                raise ValueError(
                    f"y_desired has {y_arr.shape[1]} value(s); expected {n_targets} to match the fitted "
                    f"target(s): {list(target_names)}."
                )
            y_des_df = pd.DataFrame(y_arr, columns=target_names)
        if y_des_df.isna().any(axis=None):
            raise ValueError(f"y_desired is missing value(s) for target(s): {list(target_names)}.")

        # Move the target into the scaled Y space the model was fit in.
        y_des_scaled = (
            self._y_scaler.transform(y_des_df).to_numpy() if self._y_scaler is not None else y_des_df.to_numpy()
        )  # shape (1, M)

        # Score -> scaled-Y map: y_scaled = tau @ q_map, with q_map [A x M].
        q_map = np.asarray(self.y_loadings_).T
        # Minimum-norm (direct-inversion) score, tau_DI [1 x A] (paper Eq. 9).
        tau = y_des_scaled @ np.linalg.pinv(q_map)

        # Null-space basis: score directions g with g @ q_map = 0 (paper Eq. 14).
        # The left singular vectors of q_map for its zero singular values span it.
        # numpy's matrix_rank tolerance; q_map has at least one singular value, as A and M are at least 1.
        u_mat, sing, _ = np.linalg.svd(q_map)
        tol = max(q_map.shape) * np.finfo(float).eps * sing[0]
        rank = int((sing > tol).sum())  # pragma: no mutate - > and >= differ only at a value of exactly tol
        null_space_basis = u_mat[:, rank:]  # [A x (A - rank)]
        null_space_dimension = null_space_basis.shape[1]

        # Optionally move along the null space; every such point predicts y_desired.
        if null_space_coordinates is not None:
            coords = np.atleast_1d(np.asarray(null_space_coordinates, dtype=float))
            if coords.shape[0] != null_space_dimension:
                raise ValueError(
                    f"null_space_coordinates has {coords.shape[0]} value(s); expected "
                    f"{null_space_dimension} to match the null-space dimension (A - r)."
                )
            tau = tau + null_space_basis @ coords

        # Reconstruct the input vector from its scores, then un-scale to raw units.
        x_new = tau @ self._x_loadings.T  # [1 x K]
        if self._x_scaler is not None:
            x_new = self._x_scaler.inverse_transform(x_new).to_numpy()

        # Prediction at the solution (should equal y_desired up to rounding).
        y_hat = tau @ q_map  # [1 x M]
        if self._y_scaler is not None:
            y_hat = self._y_scaler.inverse_transform(y_hat).to_numpy()

        # The T² diagnose would give x_new, by fit's rule for a component with no spread (#1026).
        t2 = float(np.sum((tau.ravel() / _t2_score_sd(self.scaling_factor_for_scores_.to_numpy())) ** 2))
        ns_columns = [f"NS{i}" for i in range(1, null_space_dimension + 1)]

        return Bunch(
            x_new=pd.Series(x_new.ravel(), index=feature_names, name="x_new"),
            scores=pd.Series(tau.ravel(), index=component_names, name="scores"),
            y_hat=pd.Series(y_hat.ravel(), index=target_names, name="y_hat"),
            null_space_basis=pd.DataFrame(null_space_basis, index=component_names, columns=ns_columns),
            null_space_dimension=null_space_dimension,
            hotellings_t2=t2,
        )

    @classmethod
    def select_n_components(  # noqa: C901, PLR0912, PLR0913, PLR0915
        cls,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 5,
        n_repeats: int | None = None,
        random_state: int | None = None,
        selection_rule: SelectionRule = "1se",
        scale_inside_folds: bool = True,
        min_q2_increase: float = Q2_MIN_INCREMENT,
        n_permutations: int = 999,
        alpha: float = 0.01,
        stability_threshold: float = 0.6,
        **pls_kwargs,
    ) -> Bunch:
        """Select the number of PLS components via cross-validation.

        Fits PLS models on cross-validation training folds and evaluates the
        out-of-fold prediction error for every component count
        ``1, 2, ..., max_components``. Reports per-fold and pooled RMSECV plus
        the validated cumulative R² curves, and recommends a component count
        from one of four rules (see ``selection_rule`` below).

        The defaults are the research-backed combination: the
        one-standard-error rule on top of repeated, shuffled K-fold CV, with
        :class:`MCUVScaler` re-fit inside every training fold so test data
        never leaks into the centring/scaling estimates.

        Unlike the calibration statistics stored on a fitted model
        (``rmse_``, ``r2_cumulative_``), the metrics returned here estimate
        performance on unseen data and are therefore suitable for choosing
        ``n_components``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training X. With the default ``scale_inside_folds=True`` the raw,
            unscaled X may be passed; scaling is fit inside every training
            fold.
        Y : array-like of shape (n_samples, n_targets)
            Training Y. Same treatment as ``X`` under ``scale_inside_folds``.
        max_components : int, optional
            Maximum number of components to evaluate. Default is the largest
            value supported by every cross-validation training fold,
            ``min(min_fold_size, n_features)``.
        cv : int or sklearn CV splitter, default 5
            If an integer, used as the ``n_splits`` of a shuffled
            :class:`~sklearn.model_selection.KFold` (or
            :class:`~sklearn.model_selection.RepeatedKFold` when
            ``n_repeats > 1``). Any sklearn splitter object (e.g.
            ``KFold(10, shuffle=True)`` or ``LeaveOneOut()``) is also accepted
            and is used as-is (``n_repeats`` is then ignored).
        n_repeats : int, optional
            Number of times the K-fold split is repeated with a fresh shuffle,
            used only when ``cv`` is an integer. The signature default is
            ``None``, which is resolved to ``10`` inside the function
            (giving a ``cv * 10`` per-fold sample for the 1-SE rule); pass
            ``1`` to disable repeats. Repeated K-fold's standard errors are
            slightly optimistic because test folds overlap across repeats;
            that is fine for the 1-SE *selection* rule but should not be
            reported as an unbiased generalisation variance.
        random_state : int, optional
            Seed forwarded to ``KFold`` / ``RepeatedKFold`` for reproducible
            shuffling. Ignored when ``cv`` is a pre-built splitter.
        selection_rule : {"1se", "min", "q2_increment", "randomization"}, default "1se"
            How the recommended component count is chosen.
            See :data:`~process_improve.multivariate._common.SelectionRule`
            for the rule semantics. ``"1se"`` is the default; ``"min"`` is the
            argmin RMSECV (the pre-1.28 default, prone to running to the
            maximum component count); ``"q2_increment"`` is the Wold's-R-style
            cumulative-Q² threshold; ``"randomization"`` is Van der Voet's
            (1994) permutation test (uses ``n_permutations`` and ``alpha``)
            that picks the smallest model whose predictive ability is
            statistically indistinguishable from the reference (argmin
            RMSECV) one.
        scale_inside_folds : bool, default True
            When True (the default), fit a fresh :class:`MCUVScaler` on each
            training fold's X and Y, apply it to the held-out rows, fit PLS in
            scaled space, then inverse-transform the predictions so RMSECV is
            reported on the original Y scale. This removes the centring /
            scaling leakage of the prior default. Set to False to keep the
            pre-1.28 behaviour, in which case ``X`` and ``Y`` should already
            be scaled; a :class:`SpecificationWarning` is emitted. A fold
            model that scales (``scale=True``, the default) still autoscales
            its own rows, so pass ``scale=False`` as well to keep a scaling
            of your own. Each held-out row is predicted as its fold model
            predicts it, whichever scalings apply.

            Pass the **raw, unscaled** blocks under the default. In-fold
            re-standardisation overwrites whatever scaling the caller applied,
            so two deliberately different strategies (autoscale versus Pareto,
            say) become the same model and report RMSECV identical to several
            decimal places: a comparison between them shows no difference for
            reasons that have nothing to do with the data. A
            :class:`SpecificationWarning` is emitted when ``X`` arrives already
            centred and unit-variance scaled, which is the detectable half of
            that case; a block scaled some other way cannot be recognised, so
            the rule is the caller's to keep.
        min_q2_increase : float, default 0.01
            Threshold used only when ``selection_rule="q2_increment"``: the
            smallest increase in cumulative validated :math:`Q^2_Y` that
            justifies keeping an extra component.
        n_permutations : int, default 999
            Used only when ``selection_rule="randomization"``: number of
            sign-flip permutations driving the Van der Voet test.
        alpha : float, default 0.01
            Used only when ``selection_rule="randomization"``: significance
            level. The smallest component count whose Van der Voet *p*-value
            exceeds ``alpha`` is recommended. R's ``pls::selectNcomp`` uses
            the same default; smaller values pick more parsimonious models.
        stability_threshold : float, default 0.6
            For the per-repeat stability-selection diagnostic (``"1se"`` /
            ``"min"`` rules with ``n_repeats > 1`` only): the recommendation
            is judged ``selection_is_stable=True`` iff the modal vote share
            in ``selection_distribution`` is at least this fraction.
            Meinshausen & Bühlmann (2010, *JRSS-B*) suggest 0.6-0.9 for
            their variable-selection analogue; we default to the
            permissive end.
        **pls_kwargs
            Additional keyword arguments passed to the ``PLS()`` constructor
            (e.g. ``missing_data_settings``).

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - recommended number of components (int).
            - ``rmsecv`` - pooled RMSECV per component count
              (pd.DataFrame, indexed ``1..A``; columns are the Y-variable
              names plus ``"total"``).
            - ``per_fold_rmsecv`` - per-fold total RMSECV (pd.DataFrame,
              indexed ``1..A``; one column per fold across all repeats).
              Drives the 1-SE rule.
            - ``se_rmsecv`` - standard error of the per-fold RMSECV per
              component count (pd.Series, indexed ``1..A``).
            - ``q2_se`` - standard error on the Q2 scale (the per-fold total
              PRESS standard error divided by the total Y sum-of-squares),
              i.e. the half-width of a +/-1 SE band around
              ``r2y_validated["total"]`` (pd.Series, indexed ``1..A``).
            - ``r2y_validated`` - validated cumulative :math:`R^2_Y`
              (pd.DataFrame, indexed ``1..A``; one column per Y-variable, then
              ``"total"`` and ``"scaled_total"``). ``"total"`` pools the
              targets on the original Y scale, so a wide-ranging target
              dominates it; ``"scaled_total"`` weights every target equally,
              which is the pooling a fitted model's ``r2_y_cumulative_`` uses,
              so those two are the columns to compare fitted against held-out.
            - ``r2x_validated`` - validated cumulative :math:`R^2_X`
              (pd.DataFrame, indexed ``1..A``; columns are the X-variable
              names plus ``"total"``).
            - ``press`` - pooled Y prediction error sum of squares per
              component count (pd.Series, indexed ``1..A``).
            - ``cv_predictions`` - out-of-fold predictions of Y at the
              recommended component count, on the original Y scale
              (pd.DataFrame). For repeated K-fold, the *first* repeat's
              held-out predictions are reported so each row appears exactly
              once.
            - ``selection_rule`` - the rule used to pick ``n_components``.
            - ``randomization_pvalues`` - per-component Van der Voet
              right-tail *p*-values when ``selection_rule="randomization"``;
              ``None`` otherwise.
            - ``selection_distribution`` - per-repeat *vote share* over
              candidate component counts (pd.Series indexed ``1..A``).
              Populated only for ``selection_rule in {"1se", "min"}`` and
              ``n_repeats > 1``; ``None`` otherwise. A concentrated
              distribution signals a confident recommendation; a flat or
              multi-modal one flags it for review.
            - ``selection_mode`` - the most-voted component count, or
              ``None`` when ``selection_distribution`` is ``None``.
            - ``selection_is_stable`` - ``True`` iff the modal vote share
              meets ``stability_threshold``; ``None`` when no distribution
              was computed.

        Notes
        -----
        The pooled RMSECV in ``rmsecv["total"]`` is the square root of the
        total PRESS over all fold-test rows divided by ``(N_eff * M)`` where
        ``N_eff = N * n_repeats`` under repeated CV; the ``per_fold_rmsecv``
        column for fold *f* is the square root of fold-*f*'s sum-of-squared
        residuals over its own test rows.

        References
        ----------
        Breiman, Friedman, Olshen & Stone (1984), *CART*, sec.3.4.3 (1-SE rule).
        Hastie, Tibshirani & Friedman, *ESL*, sec.7.10. Kohavi (1995, IJCAI)
        recommends 10-fold stratified CV for model selection.

        Examples
        --------
        >>> from sklearn.model_selection import KFold
        >>> # Default: 1-SE on 10 x 5-fold repeated CV with in-fold scaling.
        >>> result = PLS.select_n_components(X, Y, max_components=6, random_state=0)
        >>> result.n_components, result.selection_rule
        >>> # Opt-in to the older argmin-RMSECV rule:
        >>> PLS.select_n_components(X, Y, max_components=6, selection_rule="min")
        >>> # Caller-supplied splitter (n_repeats is ignored here):
        >>> PLS.select_n_components(X, Y, cv=KFold(10, shuffle=True, random_state=0))
        """
        # A DataFrame stays as it is; an array is numbered from 0, a Series keeps its name.
        X, Y = pd.DataFrame(X), pd.DataFrame(Y)

        _warn_scaling_traps(X, scale_inside_folds=scale_inside_folds, fold="CV fold", metric="RMSECV")

        N, K = X.shape
        M = Y.shape[1]

        if isinstance(cv, int):
            if cv < 2:
                raise ValueError(f"cv must be >= 2 when given as an int; got {cv}.")
            repeats = 10 if n_repeats is None else int(n_repeats)
            if repeats < 1:
                raise ValueError(f"n_repeats must be >= 1; got {repeats}.")
            splitter: BaseCrossValidator
            if repeats == 1:
                splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            else:
                splitter = RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
            splits = list(splitter.split(X))  # (Repeated)KFold reads only the number of rows
            first_repeat_fold_count = cv
        else:
            splits = list(check_cv(cv).split(X, Y))
            first_repeat_fold_count = len(splits)
        if not splits:
            raise ValueError("The cross-validation splitter produced no folds.")
        n_folds_total = len(splits)
        min_train_size = min(len(train_idx) for train_idx, _ in splits)

        # Centring inside a fold removes one DoF, so a training fold of n rows, re-centred,
        # has rank at most n - 1. Cap A there when the fold is scaled, or when its model
        # centres the fold itself (PLS(scale=True) does, and PRM always): without in-fold
        # scaling the cap was n, and a fold model on as few rows as columns ran out of X (#1027).
        probe: dict[str, typing.Any] = {"n_components": 1}  # pragma: no mutate - only whether it centres is read
        centred = scale_inside_folds or cls(**probe, **pls_kwargs)._centres_and_scales()
        upper = min(min_train_size - centred, K)
        if max_components is None:
            max_components = upper
        A = min(int(max_components), upper)
        if A < 1:
            raise ValueError("No components can be evaluated; the data or folds are too small.")

        component_index = pd.Index(range(1, A + 1), name="n_components")

        press_y = np.zeros((A, M))
        press_x = np.zeros((A, K))
        # Per-fold total-RMSECV across every fold-fit (n_folds_total columns).
        # Drives the 1-SE rule's standard error.
        per_fold_rmse = np.full((A, n_folds_total), np.nan)
        # Per-fold total PRESS (sum of squared Y residuals over the fold), on the
        # same linear scale as the Q2 normalisation; used to put a +/-1 SE band
        # on the Q2 curve, mirroring PCA's se_press -> q2_se rescaling.
        per_fold_press_total = np.full((A, n_folds_total), np.nan)
        # Out-of-fold predictions: with repeated K-fold each row appears in
        # multiple test folds, so populate only from the *first* repeat (which
        # covers every row exactly once) to preserve the existing semantic that
        # cv_predictions has one row per observation.
        oof = np.full((A, N, M), np.nan)
        # A count that leaves a held-out row with an observed response unpredicted (a row with
        # fewer observed columns than components) is not comparable with the counts that predict
        # every row, so it gets no PRESS (#676).
        unscored = np.zeros(A, dtype=bool)

        x_columns = list(X.columns)
        y_columns = list(Y.columns)
        x_values = X.to_numpy()
        y_values = Y.to_numpy()
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            model = cls(n_components=A, **pls_kwargs)
            (x_centre, x_scale), (y_centre, y_scale) = _fit_fold(
                model, X.iloc[train_idx], Y.iloc[train_idx], scale_inside_folds=scale_inside_folds
            )
            # ``y_values`` and ``x_values`` are on the ORIGINAL (un-scaled) input data: RMSECV /
            # r2y_validated / r2x_validated are all reported on the input scale. The model scores
            # rows in its own space.
            x_train = (x_values[train_idx] - x_centre) / x_scale
            x_test = x_values[test_idx]
            x_test_model = (x_test - x_centre) / x_scale
            y_test = y_values[test_idx]
            n_test = len(test_idx)
            y_loadings = np.asarray(model.y_loadings_)  # shape (M, A)
            x_loadings = model.x_loadings_.to_numpy()  # shape (K, A)
            for a in range(1, A + 1):
                scores_test = _held_out_scores(model, x_train, x_test_model, a)
                y_hat = scores_test @ y_loadings[:, :a].T * y_scale + y_centre
                x_hat = scores_test @ x_loadings[:, :a].T * x_scale + x_centre
                unscored[a - 1] |= bool(np.isnan(y_hat[~np.isnan(y_test)]).any())
                residuals_y = y_test - y_hat
                fold_press = float(np.nansum(residuals_y**2))
                press_y[a - 1] += np.nansum(residuals_y**2, axis=0)
                press_x[a - 1] += np.nansum((x_test - x_hat) ** 2, axis=0)
                per_fold_rmse[a - 1, fold_idx] = np.sqrt(fold_press / max(1, n_test * M))
                per_fold_press_total[a - 1, fold_idx] = fold_press
                if fold_idx < first_repeat_fold_count:
                    oof[a - 1, test_idx, :] = y_hat

        press_y, press_x, per_fold_rmse, per_fold_press_total = (
            np.where(unscored[:, np.newaxis], np.nan, values)
            for values in (press_y, press_x, per_fold_rmse, per_fold_press_total)
        )

        # PRESS accumulates one residual per test-row evaluation, so the
        # divisor is the actual number of those evaluations. Counting them
        # directly, rather than assuming the splitter partitions the rows once
        # per repeat, is what makes a non-partition splitter come out right:
        # the docstring accepts any sklearn splitter, and a ShuffleSplit
        # holding out half the rows ten times produces 5N evaluations while
        # the old formula still divided by N, inflating RMSECV by sqrt(5).
        # For KFold and RepeatedKFold the counts are 1 and n_repeats per row,
        # so this reproduces the previous N and N * n_repeats exactly.
        test_counts = np.zeros(N)
        for _, test_idx in splits:
            test_counts[test_idx] += 1.0
        n_eff = float(test_counts.sum())

        # The TSS denominators are weighted by the same per-row coverage, so
        # each row contributes to the "predict the mean" reference exactly as
        # often as it contributes to PRESS. A splitter that tests some rows
        # more than others (or not at all) is then handled exactly, rather
        # than through a whole-dataset repeat multiplier.
        y_centred_sq = (y_values - np.nanmean(y_values, axis=0)) ** 2
        x_centred_sq = (x_values - np.nanmean(x_values, axis=0)) ** 2
        tss_y = np.nansum(test_counts[:, None] * y_centred_sq, axis=0)
        tss_x = np.nansum(test_counts[:, None] * x_centred_sq, axis=0)

        rmsecv = pd.DataFrame(
            np.column_stack([np.sqrt(press_y / n_eff), np.sqrt(press_y.sum(axis=1) / (n_eff * M))]),
            index=component_index,
            columns=[*y_columns, "total"],
        )

        def _validated_r2(press: np.ndarray, tss: np.ndarray) -> np.ndarray:
            # `tss` already carries the per-row coverage weighting, so no repeat multiplier
            # is needed. A column, or a block, with no spread over the rows tested has no R2:
            # dividing by NaN gives NaN without a warning.
            per_var = 1.0 - press / np.where(tss > 0, tss, np.nan)
            total = 1.0 - press.sum(axis=1) / np.where(tss.sum() > 0, tss.sum(), np.nan)
            return np.column_stack([per_var, total])

        r2y_columns = _validated_r2(press_y, tss_y)
        r2y_validated = pd.DataFrame(
            np.column_stack([r2y_columns, _equal_weight_r2_total(r2y_columns[:, :M])]),
            index=component_index,
            columns=[*y_columns, "total", "scaled_total"],
        )
        r2x_validated = pd.DataFrame(
            _validated_r2(press_x, tss_x),
            index=component_index,
            columns=[*x_columns, "total"],
        )
        press = pd.Series(press_y.sum(axis=1), index=component_index, name="PRESS")

        per_fold_rmsecv = pd.DataFrame(
            per_fold_rmse,
            index=component_index,
            columns=[f"fold_{i + 1}" for i in range(n_folds_total)],
        )
        # Standard error across folds (and repeats), ``ddof=1``; with a single fold the SE
        # is NaN and the 1-SE rule degenerates to argmin gracefully.
        se_values = _standard_error(per_fold_rmse)
        se_rmsecv = pd.Series(se_values, index=component_index, name="SE(RMSECV)")

        # Q2 standard error: the standard error of the per-fold total PRESS, rescaled by the
        # Y sum-of-squares that normalises the validated Q2 (Q2_Y_total = 1 - PRESS /
        # tss_y.sum()). This is the half-width of a +/-1 SE band around the
        # ``r2y_validated["total"]`` curve, computed the same way as PCA's ``q2_se``.
        #
        # Both sides of the ratio have to be on the scale of one pass over the data. The
        # numerator is put there by multiplying the mean per-fold PRESS by the folds in a
        # pass. ``tss_y`` is not: its per-row coverage weighting counts every row once per
        # repeat that tested it, so it carries ``repeats`` passes and has to be divided down
        # to match. Without that the band came out a factor of ``n_repeats`` too narrow,
        # which quietly turned selection_rule="1se" into "min" whenever a repeated splitter
        # was used. Every repeat has ``first_repeat_fold_count`` folds, at least one.
        repeats = n_folds_total // first_repeat_fold_count
        ss_y_pass = np.where(tss_y.sum() > 0, tss_y.sum(), np.nan) / repeats
        q2_se = pd.Series(
            _standard_error(per_fold_press_total) * first_repeat_fold_count / ss_y_pass,
            index=component_index,
            name="SE(Q2)",
        )

        # If every CV fold produced NaN (e.g. zero-variance Y per fold) the
        # cross-validation never converged to anything we can judge. Raise so
        # the caller knows, rather than silently returning ``1``.
        # SEC-21 (#270) sub-item 8.
        total_rmsecv = rmsecv["total"].to_numpy()
        if np.all(np.isnan(total_rmsecv)):
            raise RuntimeError(
                "Cross-validation produced NaN total-RMSECV for every component count; "
                "no recommendation can be made. Likely cause: a per-fold zero-variance Y "
                "column, or every fold trivially degenerate."
            )
        # Per-observation squared total residual at every component count.
        # Drives Van der Voet's randomization test; cheap to compute and
        # otherwise diagnostic (rows of oof[a] that are NaN - typically
        # because some custom splitter never held them out - are skipped via
        # nansum).
        per_obs_sse = np.nansum((y_values[None, :, :] - oof) ** 2, axis=2)  # (A, N)

        randomization_pvalues: pd.Series | None = None
        if selection_rule == "randomization":
            # Only the counts that predicted every held-out row are compared (#676); a count
            # left out has no p-value. With every count scored this is the whole table.
            scored = np.flatnonzero(~unscored)
            pick, scored_p = _vandervoet_randomization(
                per_obs_sse[scored],
                total_rmsecv=total_rmsecv[scored],
                n_permutations=n_permutations,
                alpha=alpha,
                random_state=random_state,
            )
            recommended = int(scored[pick - 1]) + 1
            p_values = np.full(A, np.nan)
            p_values[scored] = scored_p
            randomization_pvalues = pd.Series(p_values, index=component_index, name="p-value (Van der Voet)")
        else:
            recommended = _select_n_components(
                selection_rule,
                mean_error=total_rmsecv,
                se_error=se_values,
                q2_cumulative=r2y_validated["total"].to_numpy(),
                min_q2_increase=min_q2_increase,
            )
        cv_predictions = pd.DataFrame(oof[recommended - 1], index=Y.index, columns=Y.columns)

        # Stability selection: re-apply the chosen rule per repeat and
        # tabulate how often each component count wins. A multi-modal or
        # flat distribution flags the recommendation as low-confidence.
        # Only meaningful when the rule operates on the per-fold RMSE
        # curve (1se/min) and we ran more than one repeat; q2_increment
        # and randomization don't decompose per repeat without more
        # bookkeeping than they're worth at this stage.
        selection_distribution: pd.Series | None = None
        selection_mode: int | None = None
        selection_is_stable: bool | None = None
        # The fold loop gives every (component count, fold) cell of per_fold_rmse a finite
        # value, the root of a nansum, so every repeat has folds to judge and casts a vote. A
        # count that left a held-out row unpredicted is NaN in every fold, and the rules pass
        # over it (#676). The 1se and min rules read no Q2 curve.
        if selection_rule in ("1se", "min") and repeats > 1:
            votes: list[int] = []
            for r in range(repeats):
                fold_subset = per_fold_rmse[:, r * first_repeat_fold_count : (r + 1) * first_repeat_fold_count]
                pick = _select_n_components(
                    selection_rule, mean_error=fold_subset.mean(axis=1), se_error=_standard_error(fold_subset)
                )
                votes.append(int(pick))
            counts = pd.Series(votes).value_counts()
            dist = (counts / len(votes)).reindex(component_index, fill_value=0.0)
            dist.name = "vote_share"
            selection_distribution = dist
            selection_mode = int(np.argmax(dist.to_numpy())) + 1
            selection_is_stable = bool(dist.max() >= stability_threshold)

        return Bunch(
            n_components=recommended,
            rmsecv=rmsecv,
            per_fold_rmsecv=per_fold_rmsecv,
            se_rmsecv=se_rmsecv,
            q2_se=q2_se,
            r2y_validated=r2y_validated,
            r2x_validated=r2x_validated,
            press=press,
            cv_predictions=cv_predictions,
            selection_rule=selection_rule,
            randomization_pvalues=randomization_pvalues,
            selection_distribution=selection_distribution,
            selection_mode=selection_mode,
            selection_is_stable=selection_is_stable,
        )

    @classmethod
    def compare_cv_criteria(  # noqa: PLR0913 - each setting controls a named criterion (#605)
        cls,
        X: DataMatrix,
        Y: DataMatrix | pd.Series,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 7,
        random_state: int | np.random.Generator | None = None,
        n_permutations: int = 999,
        n_cv_permutations: int = 199,
        alpha: float = 0.05,
        angle_threshold: float = 30.0,
        conf_level: float = 0.95,
        **pls_kwargs,
    ) -> Bunch:
        r"""Compare predictive and latent-structure validation criteria for a PLS model.

        Runs one K-fold cross-validation (one fit of this class per fold at
        ``max_components``, truncated for smaller counts) and reports, for every number of
        components, criteria that each answer a different question, together with the
        number of components each one recommends. Disagreement between them is the point:
        a component can predict Y without being stable, or be stable and real without
        improving the prediction.

        Missing values (NaN) are handled as the NIPALS fit handles them. Every held-out
        row is scored the way NIPALS scores a training row, from its observed cells, and
        a missing Y cell is left out of PRESS, of the slope ratio and of the score
        correlation. The PRESS identity and ``s_a = 1`` in sample stay exact.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor block, on its raw scale, NaN where missing. Each training fold is
            autoscaled with its own :class:`MCUVScaler`, as in
            :meth:`PLS.select_n_components`.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response block, NaN where missing. A row with no observed response still helps
            fit X.
        max_components : int, optional
            Largest number of components to evaluate. Capped at one fewer than the
            smallest training fold, and at the number of features.
        cv : int or sklearn splitter, default=7
            Number of shuffled K-fold segments, or a splitter that holds out every row
            exactly once (``KFold``, ``LeaveOneOut``, ``GroupKFold``, ...). With an integer
            ``random_state`` the folds are those of
            ``PLS.select_n_components(..., cv=cv, n_repeats=1, random_state=random_state)``.
        random_state : int, numpy.random.Generator or None, default=None
            Seed for the fold assignment and for the permutation tests.
        n_permutations : int, default=999
            Permutations for the van der Voet and the covariance tests (cheap: no refits).
        n_cv_permutations : int, default=199
            Permutations of the rows of Y used to calibrate the held-out score correlation
            of each component. Each one repeats the cross-validation with kernel PLS refits from the
            folds' ``X'X`` and ``X'Y`` (fast for tens to hundreds of features). With ``0``
            the normal approximation
            :math:`r \sim N(0, 1/N)` is used instead; it is quick but too permissive,
            because fold models share rows and a strong low-rank X widens the null
            distribution of a pooled cross-validated correlation.
        alpha : float, default=0.05
            Significance level for the tests and for the score-correlation threshold.
            :meth:`PLS.select_n_components` uses 0.01 for van der Voet.
        angle_threshold : float, default=30.0
            Jackknife-scaled subspace angle, in degrees, above which the leading weights
            are judged unstable.
        conf_level : float, default=0.95
            Confidence level of the SPE and Hotelling's :math:`T^2` limits whose
            out-of-sample alarm rates are checked.
        **pls_kwargs
            Passed to every fit of this class (for example ``max_iter``, ``tol``).
            ``scale=False`` is rejected: every fold is autoscaled.

        Returns
        -------
        result : sklearn.utils.Bunch
            With these fields:

            - ``table`` (pandas.DataFrame, indexed by ``n_components``), with columns:

              - ``r2y``, ``q2y``, ``q2y_se``: in-sample :math:`R^2_Y` (pooled in scaled
                units), cross-validated :math:`Q^2_Y` (every target weighted equally, which
                for one target is the ordinary :math:`Q^2`), and its standard error across
                folds.
              - ``vdv_p``: van der Voet *p*-value against the minimum-PRESS model.
              - ``cv_anova_p``: CV-ANOVA *p*-value for the whole model, one response only
                (NaN otherwise). It is a monotone function of :math:`Q^2`.
              - ``r_train``, ``r_cv``, ``r_cv_threshold``, ``r_cv_p``: correlation of
                :math:`t_a` and :math:`u_a` on the training rows; the same correlation
                pooled over the held-out rows (about the training-fold centre, so it is
                unaffected by sign flips between folds); the :math:`1-\alpha` quantile of
                its null distribution under permuted Y; and its one-sided permutation
                *p*-value.
              - ``slope_ratio``: :math:`s_a`, the held-out slope of the deflated response
                on :math:`t_a` relative to the training slope. PRESS falls exactly when
                :math:`s_a > 1/2`; see the module notes.
              - ``cov_perm_p``: sequential permutation *p*-value for the covariance of
                component ``a``.
              - ``angle_component_deg``, ``angle_subspace_deg``: angle between each fold's
                weight vector :math:`w_a` and the full-data one, and the largest principal
                angle between the spans of the first ``a`` weights, each scaled by the
                delete-d jackknife (:math:`\tan\theta \to \sqrt{G-1}\,\text{rms}(\tan\theta_k)`)
                so that it estimates how far the full-data direction may lie from the
                population one. Raw fold angles shrink as the number of folds grows (fold
                models share most of their rows); the scaled ones do not. A large
                component angle with a small subspace angle means the components swap or
                mix inside a stable space: loadings are then not interpretable one by one,
                but SPE, :math:`T^2` and inversion, which depend on the span, are.
              - ``d_ratio_median``, ``d_ratio_min``: Procrustes D ratios
                :math:`c_{ka}^\top c_a / c_a^\top c_a` across folds; a negative minimum
                means some fold reversed the component's inner relation.
              - ``pv_spe_alarm_rate``, ``pv_t2_alarm_rate``: fraction of the Procrustes
                pseudo-validation rows above the full-data model's SPE and :math:`T^2`
                limits at ``conf_level``; nominally ``1 - conf_level``.
              - ``pv_spe_limit_ratio``: the SPE limit fitted to the pseudo-validation
                (held-out) SPE, divided by the full-data limit. The full-data limit is
                fitted to training residuals, which are smaller than those of new rows,
                so the ratio is above 1, and more so the fewer rows per variable.

            - ``recommendations`` (pandas.DataFrame): one row per selection rule
              (``q2_max``, ``q2_1se``, ``van_der_voet``, ``score_correlation``,
              ``covariance_permutation``, ``subspace_stability``) with the recommended
              ``n_components``, the ``rule`` and the ``question`` it answers. The
              structural rules count leading components that pass, stopping at the first
              failure, and can return 0. For ``subspace_stability``, a pair of components
              that swap inside a stable span (the span of the first fails, the span of
              both passes) counts as two.
            - ``press`` (pandas.Series): cross-validated PRESS in original Y units.
            - ``press_baseline`` (float): PRESS of predicting each held-out row by its
              training fold's mean.
            - ``d_ratios`` (pandas.DataFrame): Procrustes D ratio per fold and component.
            - ``spe_limits`` (pandas.DataFrame, indexed by ``n_components``): the SPE limit
              at ``conf_level`` fitted to the training residuals (``full_data``, what
              ``global_model`` reports) and to the pseudo-validation SPE
              (``pseudo_validation``). Use the second to monitor new rows. It is a limit
              for complete rows: a held-out row with missing cells has its squared SPE
              scaled by ``K / n_observed`` before the fit, and the SPE of a new row with
              gaps should be scaled the same way before the comparison.
            - ``cv_splits`` (list of (train, test) index arrays): the folds used.
            - ``global_model`` (PLS): the model fitted to all rows with
              ``max_components`` components.
            - ``alpha``, ``conf_level``, ``angle_threshold``, ``alarm_rate_upper``
              (float): the settings used, and the binomial upper bound for the alarm
              rates.

        Raises
        ------
        ValueError
            For a row whose X is entirely missing, a column that is entirely missing, a
            training fold with fewer than two observed values in a column, a splitter that
            does not hold out every row exactly once, or an out-of-range setting.

        See Also
        --------
        select_n_components : Choose the number of components from prediction alone.
        pseudo_validation_set : Build the Procrustes pseudo-validation set explicitly.
        process_improve.multivariate.plots.cv_criteria_plot : Plot the ``table``.

        Examples
        --------
        >>> result = PLS.compare_cv_criteria(X, y, max_components=6, random_state=0)
        >>> result.recommendations["n_components"]
        >>> result.table[["q2y", "r_cv", "slope_ratio", "angle_subspace_deg"]]
        """
        settings = _CompareSettings(
            max_components=max_components,
            cv=cv,
            random_state=random_state,
            n_permutations=n_permutations,
            n_cv_permutations=n_cv_permutations,
            alpha=alpha,
            angle_threshold=angle_threshold,
            conf_level=conf_level,
        )
        return _compare_cv_criteria(cls, X, Y, settings, pls_kwargs)

    @classmethod
    def pseudo_validation_set(  # noqa: PLR0913 - mirrors the published pcvpls settings (#605)
        cls,
        X: DataMatrix,
        Y: DataMatrix | pd.Series,
        *,
        n_components: int,
        cv: int | BaseCrossValidator = 7,
        scope: typing.Literal["global", "local"] = "global",
        random_state: int | np.random.Generator | None = None,
        **pls_kwargs,
    ) -> Bunch:
        r"""Build a Procrustes pseudo-validation set for a PLS model.

        Procrustes cross-validation (Kucheryavskiy, Rodionova and Pomerantsev, 2023) turns
        the variation between cross-validation fold models into a data set, ``X_pv``, of
        the same size as ``X``, which the model fitted to all rows can be applied to like an
        independent test set. For every held-out row, the full-data model returns:

        * scores equal to the row's scores in its fold model, multiplied per component by
          the D ratio :math:`c_{ka}^\top c_a / c_a^\top c_a`;
        * an SPE equal to the row's SPE in its fold model.

        With one response and ``scope="global"``, the full-data model's predictions for
        ``X_pv`` also equal the fold models' predictions for the held-out rows. ``Y_pv``
        is ``Y`` unchanged.

        With missing values in X, the held-out rows are scored from their observed cells,
        ``X_pv`` has the same missing cells as ``X``, and the properties above are exact
        for the complete rows.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor block, raw scale, NaN where missing.
        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response block, NaN where missing.
        n_components : int
            Number of components of the model being validated.
        cv : int or sklearn splitter, default=7
            Number of shuffled K-fold segments, or a splitter that holds out every row
            exactly once.
        scope : {"global", "local"}, default="global"
            ``"global"`` (the published default) fits the fold models on rows of the fully
            autoscaled data without re-centring them; the prediction property above is
            then exact. ``"local"`` autoscales each training fold separately, as
            :meth:`PLS.select_n_components` does; scores and SPE are still reproduced
            exactly, predictions approximately.
        random_state : int, numpy.random.Generator or None, default=None
            Seed for the fold assignment and the random directions of the residual part.
        **pls_kwargs
            Passed to every fit of this class. ``scale=False`` is rejected.

        Returns
        -------
        result : sklearn.utils.Bunch
            With these fields:

            - ``X_pv`` (pandas.DataFrame): pseudo-validation predictors, on the raw scale
              of ``X``.
            - ``Y_pv`` (pandas.DataFrame): ``Y``, unchanged.
            - ``scores`` (pandas.DataFrame): the scores the full-data model gives ``X_pv``.
            - ``local_spe`` (pandas.Series): SPE of each held-out row in its fold model;
              the full-data model gives ``X_pv`` the same values. A limit fitted to these,
              ``spe_calculation(pv.local_spe, conf_level=0.95)``, is calibrated for new
              rows; the full-data model's own limit is too tight for them.
            - ``d_ratios`` (pandas.DataFrame): D ratio per fold and component.
            - ``cv_splits`` (list of (train, test) index arrays): the folds used.
            - ``global_model`` (PLS): the model fitted to all rows; apply it to ``X_pv``
              with ``diagnose``.
            - ``scope`` (str): the scope used.

        See Also
        --------
        compare_cv_criteria : Uses the same construction to report alarm rates per component.

        Examples
        --------
        >>> pv = PLS.pseudo_validation_set(X, y, n_components=2, random_state=0)
        >>> diagnostics = pv.global_model.diagnose(pv.X_pv)
        >>> (diagnostics.spe > pv.global_model.spe_limit()).mean()   # out-of-sample SPE alarm rate
        """
        settings = _PseudoValidationSettings(n_components=n_components, cv=cv, scope=scope, random_state=random_state)
        return _pseudo_validation_set(cls, X, Y, settings, pls_kwargs)

    @classmethod
    def nested_cv(  # noqa: PLR0913
        cls,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        max_components: int | None = None,
        outer_cv: int | BaseCrossValidator = 5,
        inner_cv: int = 5,
        n_inner_repeats: int = 10,
        selection_rule: SelectionRule = "1se",
        scale_inside_folds: bool = True,
        min_q2_increase: float = Q2_MIN_INCREMENT,
        n_permutations: int = 999,
        alpha: float = 0.01,
        random_state: int | None = None,
        **pls_kwargs,
    ) -> Bunch:
        """Nested cross-validation for an honest PLS performance estimate.

        Outer loop splits the data into outer-train / outer-test; the inner
        loop runs :meth:`select_n_components` on the outer-train (with the
        configured ``selection_rule`` over ``inner_cv * n_inner_repeats``
        folds) to pick the component count; a final PLS is fit on the
        outer-train at that count and used to predict the outer-test. The
        accumulated out-of-fold predictions give RMSEP that is *not*
        optimism-biased by the selection decision - the headline number to
        report when a clean test set is not available.

        Parameters
        ----------
        X, Y : array-like
            Training data. Treated as in :meth:`select_n_components` (raw
            if ``scale_inside_folds=True``, pre-scaled otherwise).
        max_components : int, optional
            Forwarded to the inner :meth:`select_n_components`.
        outer_cv : int or sklearn splitter, default 5
            Number of outer folds (or a custom splitter).
        inner_cv : int, default 5
            Number of inner folds passed to :meth:`select_n_components`.
        n_inner_repeats : int, default 10
            Number of inner-CV repeats per outer fold; the inner
            ``random_state`` is offset by the outer-fold index so each
            outer fold sees a fresh inner shuffle.
        selection_rule : str, default "1se"
            Selection rule applied inside the inner loop. See
            :data:`~process_improve.multivariate._common.SelectionRule`.
        scale_inside_folds : bool, default True
            Mirrors :meth:`select_n_components`. Also applied to the final
            outer-train fit, with the test-fold predictions inverse-
            transformed to the original Y scale before RMSEP accumulates.
        min_q2_increase, n_permutations, alpha
            Forwarded to the inner :meth:`select_n_components` per rule.
        random_state : int, optional
            Seed for the outer-fold shuffle and the inner CV. The inner
            seed is offset per outer fold so each outer split sees a fresh
            shuffled inner CV.
        **pls_kwargs
            Forwarded to :class:`PLS` for both the inner CV and the final
            outer-train fits.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``rmsep`` - honest held-out RMSEP per Y column plus a
              ``"total"`` entry (pd.Series).
            - ``q2y`` - validated :math:`Q^2_Y` per Y column plus
              ``"total"`` (pd.Series).
            - ``cv_predictions`` - out-of-fold predictions of Y at the
              per-outer-fold selected component counts (pd.DataFrame on
              the original Y scale).
            - ``selected_components_per_fold`` - list of inner
              recommendations, one per outer fold.
            - ``selected_components_distribution`` - vote share over
              candidate counts (pd.Series).

        Notes
        -----
        Runtime is roughly ``outer_cv * inner_cv * n_inner_repeats *
        max_components`` PLS fits. With the defaults that is 5 * 5 * 10 *
        max_components fits per call; for ``max_components=10`` and a
        moderate dataset that completes in seconds. Drop ``n_inner_repeats``
        if you need to bring it down further.

        Examples
        --------
        >>> from process_improve.multivariate import PLS
        >>> result = PLS.nested_cv(X, Y, max_components=8, random_state=0)
        >>> result.rmsep["total"]
        """
        # A DataFrame stays as it is; an array is numbered from 0, a Series keeps its name.
        X, Y = pd.DataFrame(X), pd.DataFrame(Y)

        N = X.shape[0]
        M = Y.shape[1]

        if isinstance(outer_cv, int):
            if outer_cv < 2:
                raise ValueError(f"outer_cv must be >= 2 when given as an int; got {outer_cv}.")
            outer_splitter: BaseCrossValidator = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
        else:
            outer_splitter = outer_cv
        outer_splits = list(outer_splitter.split(X, Y))
        if not outer_splits:
            raise ValueError("The outer cross-validation splitter produced no folds.")

        y_columns = list(Y.columns)
        y_values = Y.to_numpy()
        oof_predictions = np.full((N, M), np.nan)
        selected_components_per_fold: list[int] = []

        for outer_idx, (train_idx, test_idx) in enumerate(outer_splits):
            X_outer_train = X.iloc[train_idx]
            Y_outer_train = Y.iloc[train_idx]
            X_outer_test = X.iloc[test_idx]

            inner_seed = None if random_state is None else int(random_state) + outer_idx
            inner_result = cls.select_n_components(
                X_outer_train,
                Y_outer_train,
                max_components=max_components,
                cv=inner_cv,
                n_repeats=n_inner_repeats,
                selection_rule=selection_rule,
                scale_inside_folds=scale_inside_folds,
                min_q2_increase=min_q2_increase,
                n_permutations=n_permutations,
                alpha=alpha,
                random_state=inner_seed,
                **pls_kwargs,
            )
            n_comp_inner = int(inner_result.n_components)
            selected_components_per_fold.append(n_comp_inner)

            # Final outer-train fit at the inner-selected component count.
            model = cls(n_components=n_comp_inner, **pls_kwargs)
            (x_centre, x_scale), (y_centre, y_scale) = _fit_fold(
                model, X_outer_train, Y_outer_train, scale_inside_folds=scale_inside_folds
            )
            x_train = (X_outer_train.to_numpy() - x_centre) / x_scale
            x_test = (X_outer_test.to_numpy() - x_centre) / x_scale
            scores_test = _held_out_scores(model, x_train, x_test, n_comp_inner)
            oof_predictions[test_idx, :] = scores_test @ np.asarray(model.y_loadings_).T * y_scale + y_centre

        # RMSEP from the out-of-fold predictions (each row is predicted by
        # exactly one outer fold).
        covered = ~np.isnan(oof_predictions).any(axis=1)
        if not covered.any():
            raise RuntimeError("Nested CV produced no covered observations; check the outer splitter.")
        # Restrict every sum below to the rows the outer splitter actually held
        # out, and skip cells whose observed Y is missing. Mixing the two (a
        # PRESS over covered rows against a TSS over all rows) inflates Q2
        # whenever the splitter does not cover everything: an outer
        # ShuffleSplit touching 21 of 60 rows reported Q2 = 0.81 on pure noise,
        # where the honest value on the covered rows is 0.10. A covered row is
        # predicted in every column, so a residual is missing exactly where Y is.
        covered_y = y_values[covered]
        per_y_valid = (~np.isnan(covered_y)).sum(axis=0)
        total_valid = per_y_valid.sum()
        per_y_press = np.nansum((covered_y - oof_predictions[covered]) ** 2, axis=0)
        # A column with no observed cell among those rows has no RMSEP, and one with no spread there no Q2.
        rmsep_per_y = np.sqrt(np.divide(per_y_press, per_y_valid, out=np.full(M, np.nan), where=per_y_valid > 0))
        rmsep_total = np.sqrt(per_y_press.sum() / total_valid) if total_valid > 0 else np.nan
        rmsep = pd.Series(
            np.concatenate([rmsep_per_y, [rmsep_total]]),
            index=[*y_columns, "total"],
            name="RMSEP",
        )

        # Validated Q^2_Y per column and total, against the column mean; the TSS
        # spans exactly the cells the PRESS does.
        tss_per_y = np.nansum((covered_y - np.nanmean(y_values, axis=0)) ** 2, axis=0)
        q2y_per_y = 1.0 - np.divide(per_y_press, tss_per_y, out=np.full(M, np.nan), where=tss_per_y > 0)
        tss_total = tss_per_y.sum()
        q2y_total = 1.0 - per_y_press.sum() / tss_total if tss_total > 0 else np.nan
        q2y = pd.Series(
            np.concatenate([q2y_per_y, [q2y_total]]),
            index=[*y_columns, "total"],
            name="Q2Y",
        )

        cv_predictions = pd.DataFrame(oof_predictions, index=Y.index, columns=Y.columns)
        counts = pd.Series(selected_components_per_fold).value_counts().sort_index()
        distribution = counts / counts.sum()
        distribution.name = "vote_share"
        distribution.index.name = "n_components"

        return Bunch(
            rmsep=rmsep,
            q2y=q2y,
            cv_predictions=cv_predictions,
            selected_components_per_fold=selected_components_per_fold,
            selected_components_distribution=distribution,
        )

    def detect_outliers(self, conf_level: float = 0.95) -> list[dict]:
        """Detect outlier observations using SPE and Hotelling's T² diagnostics.

        Same approach as ``PCA.detect_outliers``: combines statistical limits
        with the robust generalized ESD test.

        Parameters
        ----------
        conf_level : float, default 0.95
            Confidence level in [0.8, 0.999].

        Returns
        -------
        outliers : list of dict
            Sorted from most severe to least. Each dict contains
            ``observation``, ``outlier_types``, ``spe``, ``hotellings_t2``,
            ``spe_limit``, ``hotellings_t2_limit``, ``severity``.

        Examples
        --------
        >>> pls = PLS(n_components=3).fit(X_scaled, Y_scaled)
        >>> outliers = pls.detect_outliers(conf_level=0.95)
        >>> for o in outliers:
        ...     print(f"{o['observation']}: {o['outlier_types']}")
        """
        check_is_fitted(self, "spe_")
        if not (0.8 <= conf_level <= 0.999):
            raise ValueError(f"conf_level must be between 0.8 and 0.999, got {conf_level}.")

        N = self.n_samples_

        spe_values = self.spe_.iloc[:, -1]
        t2_values = self.hotellings_t2_.iloc[:, -1]

        spe_lim = self.spe_limit(conf_level=conf_level)
        t2_lim = self.hotellings_t2_limit(conf_level=conf_level)

        max_outliers = max(1, N // 5)
        alpha = 1 - conf_level

        spe_outlier_idx, _ = detect_outliers_esd(spe_values.to_numpy(), max_outliers_detected=max_outliers, alpha=alpha)
        t2_outlier_idx, _ = detect_outliers_esd(t2_values.to_numpy(), max_outliers_detected=max_outliers, alpha=alpha)

        spe_flagged = set(spe_outlier_idx) | {int(i) for i in np.flatnonzero(spe_values.to_numpy() > spe_lim)}
        above_t2_limit = t2_values.to_numpy() > t2_lim  # pragma: no mutate - > and >= tie only at the limit itself
        t2_flagged = set(t2_outlier_idx) | {int(i) for i in np.flatnonzero(above_t2_limit)}

        all_flagged = spe_flagged | t2_flagged
        results = []
        for i in all_flagged:
            types = []
            if i in spe_flagged:
                types.append("spe")
            if i in t2_flagged:
                types.append("hotellings_t2")

            spe_val = float(spe_values.iloc[i])
            t2_val = float(t2_values.iloc[i])
            # A degenerate limit carries no information about how severe an
            # observation is, so it must not contribute to the ranking. A
            # perfect fit gives spe_lim == 0, which used to raise
            # ZeroDivisionError outright, and a limit at machine-epsilon scale
            # produced impressive-looking severities that were ratios of
            # floating-point noise. The ratio itself is scale-invariant (value
            # and limit share units), so the guard is only against a limit of
            # zero. Deliberately NOT an absolute epsilon, which would make
            # severity depend on the units of the data. The T2 limit is
            # positive, and infinite when A == N, where its ratio is 0 by itself.
            severity = max(spe_val / spe_lim if spe_lim > 0.0 else 0.0, t2_val / t2_lim)

            results.append(
                {
                    "observation": spe_values.index[i],
                    "outlier_types": types,
                    "spe": spe_val,
                    "hotellings_t2": t2_val,
                    "spe_limit": spe_lim,
                    "hotellings_t2_limit": t2_lim,
                    "severity": round(severity, 4),
                }
            )

        results.sort(key=lambda d: d["severity"], reverse=True)
        return results

    def cross_validate(  # noqa: PLR0913, PLR0915, C901
        self,
        X: DataMatrix,
        Y: DataMatrix,
        *,
        cv: int | str = "loo",
        n_bootstrap: int = 0,
        conf_level: float = 0.95,
        random_state: int | None = None,
        show_progress: bool = True,
        sample_weight: np.ndarray | None = None,
    ) -> Bunch:
        """Cross-validate the PLS model and compute error bars for beta coefficients.

        Refits the model on data subsets (jackknife, K-fold, or bootstrap),
        collects ``beta_coefficients_`` from each refit, and computes
        confidence intervals. Also returns cross-validated predictions
        and prediction-error metrics (RMSE, Q²).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix (same data used for ``fit``).
        Y : array-like of shape (n_samples, n_targets)
            Response matrix (same data used for ``fit``).
        cv : int or ``"loo"``, default ``"loo"``
            Cross-validation strategy:

            * ``"loo"`` - leave-one-out (jackknife). Produces N resamples.
            * ``int`` - number of folds for K-fold CV.
        n_bootstrap : int, default 0
            If > 0, use bootstrap resampling instead of CV folds.
            The value specifies the number of bootstrap rounds.
            Overrides the ``cv`` parameter when set.
        conf_level : float, default 0.95
            Confidence level for the beta-coefficient intervals, in (0, 1).
        random_state : int or None, default None
            Random seed for reproducibility (K-fold shuffle and bootstrap).
        show_progress : bool, default True
            Whether to display a ``tqdm`` progress bar.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Per-sample non-negative weights. Threaded into every sub-fit so
            each resample's PLS uses the same weighting scheme as the parent
            model. Default ``None`` (all samples weighted equally).

        Returns
        -------
        result : :class:`~sklearn.utils.Bunch`
            Dictionary-like object with the following keys:

            **Beta-coefficient uncertainty**

            beta_samples : np.ndarray of shape (n_resamples, n_features, n_targets)
                Raw beta coefficients from every resample.
            beta_mean : pd.DataFrame of shape (n_features, n_targets)
                Mean beta across resamples.
            beta_std : pd.DataFrame of shape (n_features, n_targets)
                Standard error of the beta coefficients.
            beta_ci_lower : pd.DataFrame of shape (n_features, n_targets)
                Lower bound of the confidence interval.
            beta_ci_upper : pd.DataFrame of shape (n_features, n_targets)
                Upper bound of the confidence interval.
            significant : pd.DataFrame of shape (n_features, n_targets)
                ``True`` where the confidence interval excludes zero.

            **Prediction metrics**

            y_hat_cv : pd.DataFrame of shape (n_samples, n_targets)
                Cross-validated predictions (out-of-fold). Only available for
                jackknife and K-fold; ``None`` for bootstrap.
            press : float
                Prediction Error Sum of Squares (sum over all Y elements).
                Only for jackknife / K-fold.
            rmse_cv : pd.Series of length n_targets
                Root-mean-square error per Y variable (cross-validated).
                Only for jackknife / K-fold.
            q_squared : pd.Series of length n_targets
                Cross-validated R² (Q²) per Y variable.
                Only for jackknife / K-fold.

            **Metadata**

            n_resamples : int
                Number of resamples performed.
            method : str
                ``"jackknife"``, ``"kfold"``, or ``"bootstrap"``.
            conf_level : float
                The confidence level used.

        Examples
        --------
        >>> from process_improve.multivariate import PLS, MCUVScaler
        >>> scaler_x = MCUVScaler().fit(X)
        >>> scaler_y = MCUVScaler().fit(Y)
        >>> X_s, Y_s = scaler_x.transform(X), scaler_y.transform(Y)
        >>> pls = PLS(n_components=2).fit(X_s, Y_s)
        >>> cv_results = pls.cross_validate(X_s, Y_s, cv="loo")
        >>> cv_results.beta_mean          # mean beta across LOO resamples
        >>> cv_results.significant        # which betas are significantly != 0
        >>> cv_results.q_squared          # cross-validated R²
        """
        check_is_fitted(self, "beta_coefficients_")

        X, Y = pd.DataFrame(X), pd.DataFrame(Y)
        N = X.shape[0]
        M = Y.shape[1]

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same number of rows, got {X.shape[0]} and {Y.shape[0]}.")
        if not (0.5 < conf_level < 1.0):
            raise ValueError(f"conf_level must be between 0.5 and 1.0, got {conf_level}.")

        # Validate sample_weight up front (#394). The per-fold fits below
        # subset it by training index, so the user's weights stay
        # row-aligned to X / Y throughout.
        if sample_weight is not None:
            sample_weight = np.ravel(sample_weight)
            if sample_weight.shape[0] != N:
                raise ValueError(f"sample_weight has {sample_weight.shape[0]} entries; expected {N} to match X / Y.")

        # --- The resamples: (training rows, held-out rows); a bootstrap holds out none ---
        resamples: list[tuple[np.ndarray, np.ndarray | None]]
        if n_bootstrap > 0:
            method = "bootstrap"
            rng = np.random.default_rng(random_state)
            resamples = [(rng.choice(N, size=N), None) for _ in range(n_bootstrap)]  # with replacement
            desc = "Bootstrap"
        elif cv == "loo":
            method = "jackknife"
            resamples = [(np.delete(np.arange(N), i), np.array([i])) for i in range(N)]
            desc = "Jackknife (LOO)"
        else:
            method = "kfold"
            resamples = list(KFold(n_splits=int(cv), shuffle=True, random_state=random_state).split(X))
            desc = f"{int(cv)}-Fold CV"

        # --- Collect beta coefficients (and out-of-fold predictions for CV) ---
        beta_collection: list[np.ndarray] = []
        y_hat_cv = np.full((N, M), np.nan)
        for train_idx, test_idx in tqdm(resamples, desc=desc, disable=not show_progress):
            sub_model = clone(self).fit(
                X.iloc[train_idx],
                Y.iloc[train_idx],
                sample_weight=None if sample_weight is None else sample_weight[train_idx],
            )
            beta_collection.append(sub_model.beta_coefficients_.to_numpy())
            if test_idx is not None:
                y_hat_cv[test_idx, :] = sub_model.predict(X.iloc[test_idx]).to_numpy()

        beta_samples = np.array(beta_collection)  # (n_resamples, K, M)
        actual_n_resamples = beta_samples.shape[0]

        # --- Beta-coefficient statistics ---
        beta_mean_arr = beta_samples.mean(axis=0)
        alpha = 1 - conf_level
        if method == "bootstrap":  # percentile CI
            beta_std_arr = beta_samples.std(axis=0, ddof=1)
            beta_ci_lower_arr = np.percentile(beta_samples, 100 * alpha / 2, axis=0)
            beta_ci_upper_arr = np.percentile(beta_samples, 100 * (1 - alpha / 2), axis=0)
        else:
            # The G sub-models are delete-one (jackknife, G = N) or delete-a-block (K-fold,
            # G = K) jackknife estimates, not independent replicates: their training sets
            # overlap. The jackknife standard error is sqrt((G-1)/G * sum(dev^2)) (Westad &
            # Martens, 2000, modified jackknife for PLS); the plain sample SD used for K-fold
            # before was (G-1)/sqrt(G) times too small (1.79x for K=5), over-declaring
            # significance. The CI is on the t-distribution with G - 1 degrees of freedom.
            deviations_sq = np.sum((beta_samples - beta_mean_arr) ** 2, axis=0)
            beta_std_arr = np.sqrt((actual_n_resamples - 1) / actual_n_resamples * deviations_sq)
            t_crit = t_dist.ppf(1 - alpha / 2, df=actual_n_resamples - 1)
            beta_ci_lower_arr = beta_mean_arr - t_crit * beta_std_arr
            beta_ci_upper_arr = beta_mean_arr + t_crit * beta_std_arr

        # Significance: CI does not contain zero
        significant_arr = (beta_ci_lower_arr > 0) | (beta_ci_upper_arr < 0)

        x_cols = self.beta_coefficients_.index
        y_cols = self.beta_coefficients_.columns

        beta_mean_df = pd.DataFrame(beta_mean_arr, index=x_cols, columns=y_cols)
        beta_std_df = pd.DataFrame(beta_std_arr, index=x_cols, columns=y_cols)
        beta_ci_lower_df = pd.DataFrame(beta_ci_lower_arr, index=x_cols, columns=y_cols)
        beta_ci_upper_df = pd.DataFrame(beta_ci_upper_arr, index=x_cols, columns=y_cols)
        significant_df = pd.DataFrame(significant_arr, index=x_cols, columns=y_cols)

        # --- Cross-validated prediction metrics ---
        press_val = None
        rmse_cv_series = None
        q_squared_series = None
        y_hat_cv_df = None

        if method != "bootstrap":
            y_hat_cv_df = pd.DataFrame(y_hat_cv, index=Y.index, columns=Y.columns)
            residuals = Y.values - y_hat_cv
            press_val = float(np.nansum(residuals**2))
            # nanmean, not mean: a single NaN in Y otherwise propagates through
            # the column mean and turns the whole ss_total (and Q2) into NaN.
            ss_total = np.nansum((Y.values - np.nanmean(Y.values, axis=0)) ** 2, axis=0)
            ss_res = np.nansum(residuals**2, axis=0)

            rmse_vals = np.sqrt(np.nanmean(residuals**2, axis=0))
            rmse_cv_series = pd.Series(rmse_vals, index=Y.columns, name="RMSE_CV")

            q2_vals = 1.0 - ss_res / ss_total
            q_squared_series = pd.Series(q2_vals, index=Y.columns, name="Q_squared")

        return Bunch(
            beta_samples=beta_samples,
            beta_mean=beta_mean_df,
            beta_std=beta_std_df,
            beta_ci_lower=beta_ci_lower_df,
            beta_ci_upper=beta_ci_upper_df,
            significant=significant_df,
            y_hat_cv=y_hat_cv_df,
            press=press_val,
            rmse_cv=rmse_cv_series,
            q_squared=q_squared_series,
            n_resamples=actual_n_resamples,
            method=method,
            conf_level=conf_level,
        )

    def prediction_interval(
        self,
        X: DataMatrix,
        *,
        conf_level: float = 0.95,
        cv_result: Bunch | None = None,
    ) -> Bunch:
        """Prediction interval for the Y predictions of new observations.

        The interval combines the residual error variance with the leverage of
        each new observation in the latent-variable space. For a new
        observation the prediction-interval half-width on target ``m`` is

        ``t * s_E[m] * sqrt(1 + 1/N + T2_new / (N - 1))``

        where ``s_E`` is the residual error standard deviation, ``T2_new`` is
        the Hotelling's T² of the new observation, ``N`` is the number of
        calibration samples, and ``t`` is the Student-t quantile.

        Parameters
        ----------
        X : array-like of shape (n_new, n_features)
            New observations, pre-processed the same way as the training data.
        conf_level : float, default=0.95
            Confidence level for the interval, in (0.5, 1.0).
        cv_result : sklearn.utils.Bunch or None, default=None
            The result of :meth:`cross_validate`. When supplied, its
            cross-validated RMSE (``rmse_cv``) is used for the error variance,
            which is preferable to the optimistic calibration RMSE used
            otherwise.

        Returns
        -------
        sklearn.utils.Bunch
            With keys ``y_hat`` (point predictions), ``lower`` and ``upper``
            (prediction-interval bounds) - each a DataFrame of shape
            (n_new, n_targets) - and ``conf_level``.
        """
        check_is_fitted(self, "beta_coefficients_")
        if not (0.5 < conf_level < 1.0):
            raise ValueError(f"conf_level must be between 0.5 and 1.0, got {conf_level}.")

        diagnostics = self.diagnose(X)
        y_hat = diagnostics.y_hat
        t2_new = diagnostics.hotellings_t2.to_numpy()

        n_samples = self.n_samples_
        n_components = int(self.n_components_)

        # Residual error std per Y variable: prefer the cross-validated RMSE
        # when a cross_validate() result is supplied (calibration RMSE is
        # optimistic for genuinely new observations).
        df = max(n_samples - n_components - 1, 1)
        if cv_result is not None:
            # Already an out-of-sample error: it needs no dof correction.
            error_std = np.asarray(cv_result.rmse_cv)
        else:
            # `rmse_` divides the residual sum of squares by N, but a
            # prediction interval needs the residual variance on the same
            # N - A - 1 degrees of freedom as the t quantile below. Without
            # this rescaling the interval is too narrow by sqrt(N/(N-A-1)),
            # which is 1.12 at N=20 with A=3, and a nominal 95% interval
            # delivered roughly 73 percent coverage at N=15.
            error_std = self.rmse_.iloc[:, -1].to_numpy() * np.sqrt(n_samples / df)

        # Leverage of a new observation in the latent space.
        leverage = 1.0 / n_samples + t2_new / (n_samples - 1)

        t_crit = t_dist.ppf(1 - (1 - conf_level) / 2, df)
        half_width = t_crit * np.sqrt(1.0 + leverage)[:, None] * error_std[None, :]

        lower = pd.DataFrame(y_hat.values - half_width, index=y_hat.index, columns=y_hat.columns)
        upper = pd.DataFrame(y_hat.values + half_width, index=y_hat.index, columns=y_hat.columns)
        return Bunch(y_hat=y_hat, lower=lower, upper=upper, conf_level=conf_level)


def compare_cv_criteria(X: DataMatrix, Y: DataMatrix | pd.Series, **kwargs) -> Bunch:
    """Compare predictive and latent-structure validation criteria for a PLS model.

    Function form of :meth:`PLS.compare_cv_criteria`, which documents the settings and
    the returned fields: Q2, the 1-SE rule, van der Voet and CV-ANOVA beside the
    held-out score correlation, the out-of-sample slope ratio, a covariance permutation
    test, weight-subspace angles and Procrustes cross-validation, per component, with
    the number of components each rule recommends.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor block, on its raw scale.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response block.
    **kwargs
        The keyword arguments of :meth:`PLS.compare_cv_criteria`.

    Returns
    -------
    sklearn.utils.Bunch
        See :meth:`PLS.compare_cv_criteria`.

    Examples
    --------
    >>> from process_improve.multivariate import compare_cv_criteria
    >>> result = compare_cv_criteria(X, y, max_components=6, random_state=0)
    >>> result.recommendations["n_components"]
    """
    return PLS.compare_cv_criteria(X, Y, **kwargs)


def pseudo_validation_set(X: DataMatrix, Y: DataMatrix | pd.Series, **kwargs) -> Bunch:
    """Build a Procrustes pseudo-validation set for a PLS model.

    Function form of :meth:`PLS.pseudo_validation_set`, which documents the settings
    and the returned fields.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor block, raw scale.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response block.
    **kwargs
        The keyword arguments of :meth:`PLS.pseudo_validation_set`; ``n_components`` is
        required.

    Returns
    -------
    sklearn.utils.Bunch
        See :meth:`PLS.pseudo_validation_set`.

    Examples
    --------
    >>> pv = pseudo_validation_set(X, y, n_components=2, random_state=0)
    >>> pv.global_model.diagnose(pv.X_pv).spe
    """
    return PLS.pseudo_validation_set(X, Y, **kwargs)
