r"""(c) Kevin Dunn, 2010-2026. MIT License.

Compare predictive and latent-structure validation of a PLS model, component by component.

Cross-validated :math:`Q^2` answers one question: does the model predict
:math:`\mathbf{Y}` on rows it has not seen? A PLS model used for process monitoring
(SPE and :math:`T^2`) or for model inversion also needs its latent directions to be
a reproducible property of the process, and :math:`Q^2` does not test that. The
two questions can have different answers, so :func:`compare_cv_criteria` runs one
K-fold loop and reports, for every number of components, a set of criteria that each
answer a stated question, with the component count each one recommends.

* Prediction: cumulative :math:`Q^2`, the one-standard-error rule, the van der Voet
  randomization test, and the CV-ANOVA *p*-value.
* Inner relation on held-out rows: the correlation of the held-out
  :math:`t_a` and :math:`u_a` scores, and the out-of-sample slope ratio
  :math:`s_a`.
* Covariance significance: a sequential permutation test on the deflated
  :math:`\mathbf{X}_a^\top\mathbf{Y}_a` cross-product.
* Stability of the weights: per-component and subspace angles between each fold
  model and the model fitted to all rows.
* Monitoring: Procrustes cross-validation (Kucheryavskiy, Rodionova and
  Pomerantsev, 2023), which places every held-out row in the coordinates of the
  model fitted to all rows, so that the SPE and :math:`T^2` alarm rates of that
  model can be checked on rows it did not see.

The slope ratio links the predictive and the structural views exactly. For held-out
rows, with :math:`\mathbf{R}_{a-1}` the residuals after :math:`a-1` components and
:math:`\tilde{\mathbf{c}}_a` the fold's Y loading in original units,

.. math::

    s_a = \frac{\sum_k \mathbf{t}_{ak}^\top \mathbf{R}_{a-1,k}\,\tilde{\mathbf{c}}_{ak}}
               {\sum_k \|\tilde{\mathbf{c}}_{ak}\|^2\,\mathbf{t}_{ak}^\top\mathbf{t}_{ak}},
    \qquad
    \text{PRESS}_{a-1} - \text{PRESS}_a
    = (2 s_a - 1)\sum_k \|\tilde{\mathbf{c}}_{ak}\|^2\,\mathbf{t}_{ak}^\top\mathbf{t}_{ak}.

On the training rows :math:`s_a = 1`. On new rows, :math:`Q^2` rises only when
:math:`s_a > 1/2`, while the held-out score correlation is positive as soon as
:math:`s_a > 0`. A component with :math:`0 < s_a < 1/2` points the right way on new
data, but its training slope is more than twice too steep. Positive is not
significant, though: in simulations with a real but weak component, 4 of 83
components in this band beat the permutation null at the 5% level, no more than
chance would give. Tested against its null, the score correlation is not more
sensitive than :math:`Q^2`; the band marks a direction that is real but cannot yet be
told from noise with this many rows.

Missing values are handled as the NIPALS fit handles them. Every held-out row is
scored the way NIPALS scores an incomplete training row (regression on the observed
part of each weight vector, then deflation), and a missing Y cell is left out of every
sum. The identity above and :math:`s_a = 1` in sample hold exactly with gaps.

References
----------
Van der Voet, H. (1994). Comparing the predictive accuracy of models using a simple
randomization test. *Chemom. Intell. Lab. Syst.*, 25(2), 313-323.

Eriksson, L., Trygg, J. and Wold, S. (2008). CV-ANOVA for significance testing of PLS
and OPLS models. *J. Chemometrics*, 22(11-12), 594-600.

Kucheryavskiy, S., Rodionova, O. and Pomerantsev, A. (2023). Procrustes
cross-validation of multivariate regression models. *Anal. Chim. Acta*, 1255, 341096.
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import binom, norm
from scipy.stats import f as f_dist
from sklearn.model_selection import BaseCrossValidator, KFold, check_cv
from sklearn.utils import Bunch

from .._random import check_random_state
from ._common import _equal_weight_r2_total, _select_n_components, _vandervoet_randomization
from ._limits import hotellings_t2_limit, spe_calculation
from ._nipals import _kernel_pls, _sign_align
from ._preprocessing import MCUVScaler, _warn_scaling_traps

if typing.TYPE_CHECKING:
    from ._common import DataMatrix

# The public entry points are ``PLS.compare_cv_criteria`` / ``PLS.pseudo_validation_set``
# and the module-level functions of the same names in ``_pls``. This module takes the
# estimator class as an argument instead of importing ``PLS``, so ``_pls`` can import it
# without an import cycle.

#: A zero-argument callable returning an unfitted PLS-family estimator.
ModelFactory = Callable[[], Any]

#: Degrees of freedom charged per PLS component by CV-ANOVA (Eriksson, Trygg and Wold, 2008).
CV_ANOVA_DF_PER_COMPONENT = 2

#: Permutations evaluated per vectorised batch in the covariance permutation test.
_PERMUTATION_BATCH = 256


@dataclass
class _Fold:
    """One cross-validation fold: its rows, held-out data, and fitted model matrices."""

    train: np.ndarray
    test: np.ndarray
    x_test: np.ndarray  # held-out X in the fold model's scaled space, NaN where missing, (n_test, K)
    scores: np.ndarray  # held-out scores, scored as NIPALS scores its training rows, (n_test, A)
    y_centre: np.ndarray  # maps the fold's scaled Y back to original units, (M,)
    y_scale: np.ndarray  # (M,)
    weights: np.ndarray  # W, (K, A)
    direct_weights: np.ndarray  # W* = W (P'W)^-1, (K, A)
    x_loadings: np.ndarray  # P, (K, A)
    y_loadings: np.ndarray  # C, (M, A)


def _as_frames(X: DataMatrix, Y: DataMatrix | pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce X and Y to float DataFrames with one row per observation.

    Missing cells (NaN) are allowed, with two exceptions that leave nothing to work
    with: a row whose X is entirely missing cannot be scored, and a column that is
    entirely missing cannot be scaled. A row whose Y is entirely missing is kept; it
    helps fit the X side and is left out of every Y-based criterion.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
    if isinstance(Y, pd.Series):
        Y_df = Y.to_frame()
    elif isinstance(Y, pd.DataFrame):
        Y_df = Y
    else:
        y = np.asarray(Y, dtype=float)
        Y_df = pd.DataFrame(y.reshape(-1, 1) if y.ndim == 1 else y, index=X_df.index)
    if X_df.shape[0] != Y_df.shape[0]:
        raise ValueError(f"X and Y must have the same number of rows; got {X_df.shape[0]} and {Y_df.shape[0]}.")
    X_df = X_df.astype(float)
    Y_df = Y_df.astype(float)
    empty_rows = X_df.index[X_df.isna().all(axis=1)]
    if len(empty_rows):
        raise ValueError(f"Rows with every X value missing cannot be scored; remove them: {list(empty_rows)[:5]}.")
    for name, frame in (("X", X_df), ("Y", Y_df)):
        empty_columns = frame.columns[frame.isna().all(axis=0)]
        if len(empty_columns):
            raise ValueError(f"Columns of {name} with every value missing: {list(empty_columns)[:5]}.")
    return X_df, Y_df


def _check_fold_coverage(X: pd.DataFrame, Y: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Every training fold needs two observed values per column to centre and scale it."""
    for k, (train, _) in enumerate(splits, start=1):
        for name, frame in (("X", X), ("Y", Y)):
            counts = frame.iloc[train].notna().sum(axis=0)
            sparse = counts.index[counts < 2]
            if len(sparse):
                raise ValueError(
                    f"Training fold {k} has fewer than two observed values in {name} column(s) "
                    f"{list(sparse)[:5]}; use fewer folds or impute."
                )


def _partition_splits(
    cv: int | BaseCrossValidator,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    rng: np.random.Generator,
    random_state: int | np.random.Generator | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test splits in which every row is held out exactly once.

    An integer ``random_state`` goes to :class:`~sklearn.model_selection.KFold`
    unchanged, so the splits are the ones :meth:`PLS.select_n_components` makes with
    the same seed. A Generator, or ``None``, supplies a derived integer seed.
    """
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError(f"cv must be >= 2 when given as an int; got {cv}.")
        if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
            seed = int(random_state)
        else:
            seed = int(rng.integers(2**31))
        splits = list(KFold(n_splits=int(cv), shuffle=True, random_state=seed).split(X, Y))
    else:
        splits = list(check_cv(cv).split(X, Y))
    counts = np.zeros(X.shape[0], dtype=int)
    for _, test in splits:
        counts[test] += 1
    if not splits or np.any(counts != 1):
        raise ValueError(
            "The cross-validation splitter must hold out every row exactly once (a partition, such as "
            "KFold or LeaveOneOut). Repeated or random splitters (RepeatedKFold, ShuffleSplit) are not "
            "supported, because the pooled held-out scores and the pseudo-validation set need one "
            "held-out copy of each row."
        )
    return splits


def _component_cap(splits: list[tuple[np.ndarray, np.ndarray]], K: int, requested: int | None) -> int:
    """Largest component count every fold can fit, capped by ``requested``.

    Centring inside a fold removes one degree of freedom, so a fold with ``n`` training
    rows supports at most ``n - 1`` components (the cap ``select_n_components`` uses).
    """
    min_train = min(len(train) for train, _ in splits)
    upper = min(min_train - 1, K)
    if requested is not None and int(requested) < 1:
        raise ValueError(f"max_components must be >= 1; got {requested}.")
    A = upper if requested is None else min(int(requested), upper)
    if A < 1:
        raise ValueError("No components can be evaluated; the data or the folds are too small.")
    return A


def _nipals_scores(x: np.ndarray, weights: np.ndarray, x_loadings: np.ndarray) -> np.ndarray:
    """Score rows the way NIPALS scores its training rows, using only the observed cells.

    For each component in turn, the score is the regression of the row's deflated
    observed cells on the matching cells of ``w_a``, and the row is then deflated by
    ``t_a p_a'``. A missing cell stays out of every step.

    With complete training data, ``P'W`` is unit upper triangular, and on a complete row
    this equals ``x @ W*``, the projection :meth:`PLS.select_n_components` uses. With
    gaps in the training data, ``P'W`` is not triangular, so ``x @ W*`` is not how the
    model scored its own rows, even for a complete row. Scoring held-out rows this way
    keeps them on the same footing as the training scores the model was fitted to, so
    scoring the training rows gives ``s_a = 1``. It is sequential, so the first ``a``
    scores are those of the ``a``-component model, and it inverts no matrix. Trimmed
    score regression has neither property: its estimate of the first ``a`` scores
    changes with ``A``, and its ``A x A`` matrix is singular once ``A`` exceeds the
    number of observed cells in a row.
    """
    observed = ~np.isnan(x)
    residual = np.where(observed, x, 0.0)
    scores = np.zeros((x.shape[0], weights.shape[1]))
    for a in range(weights.shape[1]):
        w = weights[:, a]
        denominator = observed @ w**2
        with np.errstate(divide="ignore", invalid="ignore"):
            scores[:, a] = np.where(denominator > 0, residual @ w / denominator, 0.0)
        residual = np.where(observed, residual - np.outer(scores[:, a], x_loadings[:, a]), 0.0)
    return scores


def _fold_from_model(
    model: Any,  # noqa: ANN401 - any fitted PLS-family estimator
    split: tuple[np.ndarray, np.ndarray],
    x_test: pd.DataFrame,
    y_map: tuple[np.ndarray, np.ndarray],
) -> _Fold:
    """Collect the fitted matrices a fold contributes; ``y_map`` is the (centre, scale) of Y."""
    return _Fold(
        train=split[0],
        test=split[1],
        x_test=x_test.to_numpy(dtype=float),
        scores=_nipals_scores(x_test.to_numpy(dtype=float), model.x_weights_.to_numpy(), model.x_loadings_.to_numpy()),
        y_centre=np.asarray(y_map[0], dtype=float),
        y_scale=np.asarray(y_map[1], dtype=float),
        weights=model.x_weights_.to_numpy(),
        direct_weights=model.direct_weights_.to_numpy(),
        x_loadings=model.x_loadings_.to_numpy(),
        y_loadings=model.y_loadings_.to_numpy(),
    )


def _global_scaling(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[MCUVScaler, MCUVScaler]:
    """Scalers mapping raw data into the space of the model fitted to all rows."""
    return MCUVScaler().fit(X), MCUVScaler().fit(Y)


def _fit_folds(
    make_model: ModelFactory,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    scope: Literal["local", "global"],
) -> list[_Fold]:
    """Fit one model per fold, each made by ``make_model()``.

    ``"local"`` scope autoscales each training fold with its own :class:`MCUVScaler`,
    exactly as :meth:`PLS.select_n_components` does. ``"global"`` scope fits the fold
    models on rows of the fully autoscaled data without re-centring them, which is the
    default of the published Procrustes cross-validation; ``make_model`` must then build
    an estimator with ``scale=False``.
    """
    _check_fold_coverage(X, Y, splits)
    folds: list[_Fold] = []
    if scope == "global":
        x_scaler, y_scaler = _global_scaling(X, Y)
        Xs, Ys = x_scaler.transform(X), y_scaler.transform(Y)
        y_map = (y_scaler.center_.to_numpy(), y_scaler.scale_.to_numpy())
        for train, test in splits:
            model = make_model().fit(Xs.iloc[train], Ys.iloc[train])
            folds.append(_fold_from_model(model, (train, test), Xs.iloc[test], y_map))
        return folds

    for train, test in splits:
        scaler_x = MCUVScaler().fit(X.iloc[train])
        scaler_y = MCUVScaler().fit(Y.iloc[train])
        model = make_model().fit(scaler_x.transform(X.iloc[train]), scaler_y.transform(Y.iloc[train]))
        y_map = (scaler_y.center_.to_numpy(), scaler_y.scale_.to_numpy())
        folds.append(_fold_from_model(model, (train, test), scaler_x.transform(X.iloc[test]), y_map))
    return folds


def _correlation_about_origin(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation about zero (uncentred), NaN when either vector is zero.

    Held-out rows are centred on their training fold's means, the reference the model
    predicts from; re-centring them on their own means would hide an offset. About the
    origin, the correlation is also unchanged when a fold flips the sign of a component
    (t and u flip together). On training scores, which have zero mean, it equals the
    Pearson correlation.
    """
    denominator = float(np.sqrt((x @ x) * (y @ y)))
    return float(x @ y) / denominator if denominator > 0 else float("nan")


def _heldout_pass(folds: list[_Fold], y_values: np.ndarray, n_components: int) -> Bunch:
    """Accumulate PRESS, the slope ratio and the pooled scores over every held-out fold.

    PRESS is computed in original Y units, the same way :meth:`PLS.select_n_components`
    computes it. The slope ratio is accumulated in the same units, so that
    ``PRESS[a-1] - PRESS[a] == (2 * slope_ratio[a] - 1) * slope_weight[a]`` exactly.

    A missing Y cell has no residual. It is left out of PRESS, and the identity stays
    exact because the slope ratio uses the same observed cells: each row's weight is
    ``t_i^2`` times the sum of ``c~_m^2`` over the responses observed in that row. A
    row with no observed response is left out of the pooled t-u scores.
    """
    N, M = y_values.shape
    A = n_components
    n_folds = len(folds)
    press_y = np.zeros((A, M))
    per_obs_sse = np.full((A, N), np.nan)
    per_fold_press = np.full((A, n_folds), np.nan)
    per_fold_rmse = np.full((A, n_folds), np.nan)
    slope_numerator = np.zeros(A)
    slope_weight = np.zeros(A)
    press_baseline = 0.0
    t_pool: list[list[np.ndarray]] = [[] for _ in range(A)]
    u_pool: list[list[np.ndarray]] = [[] for _ in range(A)]

    for k, fold in enumerate(folds):
        y_test = y_values[fold.test]
        observed = ~np.isnan(y_test)
        has_y = observed.any(axis=1)
        n_cells = max(1, int(observed.sum()))
        # a = 0 predicts the training-fold mean; a missing cell's residual is zero.
        residual_prev = np.where(observed, y_test - fold.y_centre, 0.0)
        press_baseline += float(np.sum(residual_prev**2))
        for a in range(A):
            t = fold.scores[:, a]
            c = fold.y_loadings[:, a]
            c_original = c * fold.y_scale
            slope_numerator[a] += float(t @ residual_prev @ c_original)
            slope_weight[a] += float(t**2 @ (observed @ c_original**2))
            # u_a = Y_res c_a / (c_a'c_a) over the observed responses, with Y_res the
            # held-out Y deflated by the earlier held-out components, in scaled units.
            cc = observed @ c**2
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.where(cc > 0, (residual_prev / fold.y_scale) @ c / cc, 0.0)
            t_pool[a].append(t[has_y])
            u_pool[a].append(u[has_y])

            residual = np.where(observed, residual_prev - np.outer(t, c_original), 0.0)
            squared = residual**2
            press_y[a] += squared.sum(axis=0)
            per_obs_sse[a, fold.test] = np.where(has_y, squared.sum(axis=1), np.nan)
            per_fold_press[a, k] = float(squared.sum())
            per_fold_rmse[a, k] = float(np.sqrt(squared.sum() / n_cells))
            residual_prev = residual

    with np.errstate(divide="ignore", invalid="ignore"):
        slope_ratio = np.where(slope_weight > 0, slope_numerator / slope_weight, np.nan)
    r_cv = np.array([_correlation_about_origin(np.concatenate(t_pool[a]), np.concatenate(u_pool[a])) for a in range(A)])
    return Bunch(
        press_y=press_y,
        per_obs_sse=per_obs_sse,
        per_fold_press=per_fold_press,
        per_fold_rmse=per_fold_rmse,
        press_baseline=press_baseline,
        slope_ratio=slope_ratio,
        slope_weight=slope_weight,
        r_cv=r_cv,
    )


def _autoscale_parameters(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and ``ddof=1`` standard deviations over observed cells, with the :class:`MCUVScaler` guard."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0, ddof=1)
    tiny = float(np.finfo(float).tiny) ** 0.5
    return centre, np.where(~np.isfinite(scale) | (scale <= tiny), 1.0, scale)


def _score_correlation_null(
    x_values: np.ndarray,
    y_values: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out t-u correlation of every component, on Y as given and with its rows permuted.

    ``shape`` is ``(n_permutations, n_components)``, the shape of the null array.

    Each permutation shuffles the rows of Y and repeats the whole cross-validation,
    with the folds refitted by kernel PLS (Dayal and MacGregor, 1997) from the fold's
    ``X'X``, which a permutation of Y leaves unchanged, and the new ``X'Y``; for
    complete data the kernel algorithm gives the NIPALS model. The observed statistic
    is computed by the same pipeline on the unpermuted Y, so that the two are compared
    like with like. For complete data it equals the NIPALS ``r_cv``.

    A missing training cell is set to zero after autoscaling (the fold mean), in X and in
    Y, so it adds nothing to a cross-product. Held-out rows are scored from their
    observed cells, as the NIPALS fit scores its rows, rather than from mean-filled ones,
    which would shrink the score of every incomplete row. A Y row carries its missing
    cells with it when the rows are permuted. ``u`` is regressed on the observed
    responses only, and a row with no observed response is left out.

    The whole of Y is permuted, not the residual of the first ``a - 1`` components.
    A residual permutation is not a valid null here: PLS components form a Krylov
    sequence, so "the first ``a - 1`` components plus noise" still holds structure that
    the refitted fold models need further components to reproduce.

    The normal approximation ``r ~ N(0, 1/N)`` is not used either. Each held-out score
    is a weighted sum of the other folds' responses, so the pooled correlation is a
    quadratic form in Y whose null spread depends on the structure of X, and is
    typically wider.

    Returns
    -------
    observed : np.ndarray of shape (n_components,)
        The held-out correlation of every component on Y as given.
    null : np.ndarray of shape (n_permutations, n_components)
        For each permutation, the held-out correlation of every component.
    """
    n_permutations, A = shape
    folds = []
    for train, test in splits:
        centre, scale = _autoscale_parameters(x_values[train])
        x_train = np.nan_to_num((x_values[train] - centre) / scale)
        folds.append((train, test, x_train, (x_values[test] - centre) / scale, x_train.T @ x_train))

    def _heldout_correlations(y: np.ndarray) -> np.ndarray:
        tu = np.zeros(A)
        tt = np.zeros(A)
        uu = np.zeros(A)
        for train, test, x_train, x_test, xtx in folds:
            centre, scale = _autoscale_parameters(y[train])
            y_train = np.nan_to_num((y[train] - centre) / scale)
            weights, x_loadings, _, y_loadings, _ = _kernel_pls(xtx, x_train.T @ y_train, A)
            t_test = _nipals_scores(x_test, weights, x_loadings)
            observed = ~np.isnan(y[test])
            has_y = observed.any(axis=1)
            residual = np.nan_to_num((y[test] - centre) / scale)
            for a in range(A):
                c = y_loadings[:, a]
                cc = observed @ c**2
                t = np.where(has_y, t_test[:, a], 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = np.where(cc > 0, residual @ c / cc, 0.0)
                tu[a] += float(t @ u)
                tt[a] += float(t @ t)
                uu[a] += float(u @ u)
                residual = np.where(observed, residual - np.outer(t, c), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tt * uu > 0, tu / np.sqrt(tt * uu), np.nan)

    null = np.array([_heldout_correlations(y_values[rng.permutation(len(y_values))]) for _ in range(n_permutations)])
    return _heldout_correlations(y_values), null


def _permutation_summary(null: np.ndarray, observed: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided permutation *p*-value and the ``1 - alpha`` null quantile, per column."""
    p_values = np.full(null.shape[1], np.nan)
    thresholds = np.full(null.shape[1], np.nan)
    for a in range(null.shape[1]):
        finite = null[np.isfinite(null[:, a]), a]
        if np.isfinite(observed[a]) and finite.size:
            p_values[a] = (np.sum(finite >= observed[a]) + 1) / (null.shape[0] + 1)
            thresholds[a] = np.quantile(finite, 1.0 - alpha, method="higher")
    return p_values, thresholds


def _covariance_permutation_pvalues(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    model: Any,  # noqa: ANN401 - any fitted PLS-family estimator
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Sequential permutation test of the covariance each component captures.

    Component ``a`` is tested on the blocks deflated by the ``a - 1`` components before
    it. The statistic is the largest singular value of :math:`\mathbf{X}_a^\top
    \mathbf{Y}_a`, the covariance that NIPALS maximises; the null permutes the rows of
    the deflated :math:`\mathbf{Y}_a` while :math:`\mathbf{X}_a` stays fixed. Because
    :math:`\mathbf{X}_a` is orthogonal to the earlier scores, this is the Freedman-Lane
    scheme. A test on the undeflated :math:`\mathbf{X}^\top\mathbf{Y}` could not reach
    past the first component for a single response, since that matrix has rank one.

    A missing cell is held at zero through every deflation, so it adds nothing to the
    cross-product; a Y row carries its missing cells with it when the rows are permuted.
    """
    N = x_scaled.shape[0]
    M = y_scaled.shape[1]
    scores = model.scores_.to_numpy()
    x_loadings = model.x_loadings_.to_numpy()
    y_loadings = model.y_loadings_.to_numpy()
    x_observed = ~np.isnan(x_scaled)
    y_observed = ~np.isnan(y_scaled)
    x_a = np.where(x_observed, x_scaled, 0.0)
    y_a = np.where(y_observed, y_scaled, 0.0)
    p_values = np.empty(scores.shape[1])
    for a in range(scores.shape[1]):
        observed = float(np.linalg.norm(x_a.T @ y_a, ord=2))
        exceed = 0
        done = 0
        while done < n_permutations:
            batch = min(_PERMUTATION_BATCH, n_permutations - done)
            order = rng.permuted(np.tile(np.arange(N), (batch, 1)), axis=1)
            cross = np.einsum("nk,bnm->bkm", x_a, y_a[order])  # (batch, K, M)
            null = np.linalg.norm(cross[:, :, 0], axis=1) if M == 1 else np.linalg.norm(cross, ord=2, axis=(1, 2))
            exceed += int(np.sum(null >= observed))
            done += batch
        p_values[a] = (exceed + 1) / (n_permutations + 1)
        t = scores[:, [a]]
        x_a = np.where(x_observed, x_a - t @ x_loadings[:, [a]].T, 0.0)
        y_a = np.where(y_observed, y_a - t @ y_loadings[:, [a]].T, 0.0)
    return p_values


def _jackknife_angle(fold_angles_deg: np.ndarray, n_folds: int) -> np.ndarray:
    r"""Scale fold-to-full-data angles to the sampling angle of the full-data estimate.

    A fold model shares :math:`(G-1)/G` of its rows with the full-data model, so the
    raw angle between them understates how far the full-data direction could be from
    the population one, and shrinks towards zero as the number of folds :math:`G`
    grows. The delete-d jackknife corrects this: the sampling spread of the full-data
    estimate is :math:`\sqrt{G-1}` times the root-mean-square spread of the fold
    estimates around it. Applied to :math:`\tan\theta`, the result is comparable
    between 7-fold and leave-one-out cross-validation.
    """
    tangent = np.tan(np.radians(np.clip(fold_angles_deg, 0.0, 90.0 - 1e-9)))
    rms = np.sqrt(np.mean(tangent**2, axis=0))
    return np.degrees(np.arctan(np.sqrt(max(n_folds - 1, 1)) * rms))


def _weight_angles(global_weights: np.ndarray, folds: list[_Fold]) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife-scaled per-component and subspace angles (degrees) of the weights.

    The per-component angle is between each fold's :math:`w_a` and the full-data one
    (sign-free); the subspace angle is the largest principal angle between the spans
    of the first ``a`` weights. ``span(W[:, :a]) == span(W*[:, :a])`` because ``P'W``
    is unit upper triangular in NIPALS, so either can be used.
    """
    A = global_weights.shape[1]
    reference = global_weights / np.linalg.norm(global_weights, axis=0)
    component = np.full((len(folds), A), np.nan)
    subspace = np.full((len(folds), A), np.nan)
    for k, fold in enumerate(folds):
        weights = fold.weights / np.linalg.norm(fold.weights, axis=0)
        cosines = np.clip(np.abs(np.sum(weights * reference, axis=0)), 0.0, 1.0)
        component[k] = np.degrees(np.arccos(cosines))
        for a in range(A):
            # subspace_angles returns the principal angles in descending order.
            subspace[k, a] = np.degrees(subspace_angles(reference[:, : a + 1], weights[:, : a + 1])[0])
    return _jackknife_angle(component, len(folds)), _jackknife_angle(subspace, len(folds))


def _procrustes_scores(model: Any, folds: list[_Fold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN401
    """Pseudo-validation scores, local squared SPE, and D ratios (Kucheryavskiy et al., 2023).

    Each fold model is sign-aligned to the full-data model. The held-out scores in the
    fold model (``T_k = X_k W*_k``, or trimmed score regression for a row with missing
    cells) are scaled per component by ``d_ka = c_ka'c_a / c_a'c_a``, the ratio of the
    fold's Y loading to the full-data one, giving the pseudo-validation scores. The
    local squared SPE of each held-out row, summed over its observed cells, is what the
    pseudo-validation row's residual is built to reproduce.

    Returns
    -------
    scores : np.ndarray of shape (N, A)
    squared_spe : np.ndarray of shape (N, A)
    d_ratios : np.ndarray of shape (n_folds, A)
    """
    global_direct = model.direct_weights_.to_numpy()
    global_y_loadings = model.y_loadings_.to_numpy()
    N = sum(len(fold.test) for fold in folds)
    A = global_direct.shape[1]
    cc = np.sum(global_y_loadings**2, axis=0)
    pv_scores = np.zeros((N, A))
    squared_spe = np.zeros((N, A))
    d_ratios = np.zeros((len(folds), A))
    for k, fold in enumerate(folds):
        # _sign_align only reports the signs; apply them to the scores, P and C together.
        signs = _sign_align(fold.direct_weights, global_direct)
        x_loadings = fold.x_loadings * signs
        y_loadings = fold.y_loadings * signs
        local_scores = fold.scores * signs
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(cc > 0, np.sum(y_loadings * global_y_loadings, axis=0) / cc, np.nan)
        d_ratios[k] = d
        pv_scores[fold.test] = local_scores * d
        for a in range(A):
            residual = fold.x_test - local_scores[:, : a + 1] @ x_loadings[:, : a + 1].T
            squared_spe[fold.test, a] = np.nansum(residual**2, axis=1)  # observed cells only
    return pv_scores, squared_spe, d_ratios


def _leading_count(passes: np.ndarray, swaps: np.ndarray | None = None) -> int:
    """Count the leading components that pass, stopping at the first failure (or NaN).

    ``swaps[a]`` marks components ``a`` and ``a + 1`` (0-based) as a pair that mixes
    between folds inside a stable two-dimensional span. A failure at ``a`` does not stop
    the count when ``swaps[a]`` holds and ``a + 1`` passes: the pair is counted together.
    """
    count = 0
    a = 0
    while a < len(passes):
        if passes[a]:
            count, a = a + 1, a + 1
        elif swaps is not None and swaps[a] and a + 1 < len(passes) and passes[a + 1]:
            count, a = a + 2, a + 2
        else:
            break
    return count


def _swapped_pairs(angle_subspace_deg: np.ndarray, threshold: float) -> np.ndarray:
    """Where the span of the first ``a`` weights is unstable but the span of ``a + 1`` is stable.

    Two components of nearly equal strength can swap or rotate into each other from
    fold to fold. The span of the first of them is then unstable, while the span of the
    pair is not; the per-component angles of both are large.
    """
    unstable = ~(angle_subspace_deg < threshold)
    stable_next = np.r_[angle_subspace_deg[1:] < threshold, False]
    return unstable & stable_next


def _cv_anova_pvalues(q2: np.ndarray, N: int, M: int) -> np.ndarray:
    r"""CV-ANOVA *p*-value per component count (single response only).

    With :math:`SS_{reg} = SS_{tot} - \text{PRESS}`, the F statistic reduces to
    :math:`F = Q^2/(1-Q^2)\cdot df_{res}/df_{reg}`, a monotone function of :math:`Q^2`
    for fixed degrees of freedom: CV-ANOVA calibrates :math:`Q^2` as a *p*-value, it
    does not rank models differently.
    """
    A = len(q2)
    p_values = np.full(A, np.nan)
    if M != 1:
        return p_values
    for a in range(1, A + 1):
        df_reg = CV_ANOVA_DF_PER_COMPONENT * a
        df_res = N - 1 - df_reg
        if df_res <= 0 or not np.isfinite(q2[a - 1]):
            continue
        if q2[a - 1] <= 0:
            p_values[a - 1] = 1.0
        elif q2[a - 1] >= 1:
            p_values[a - 1] = 0.0
        else:
            f_stat = q2[a - 1] / (1 - q2[a - 1]) * df_res / df_reg
            p_values[a - 1] = float(f_dist.sf(f_stat, df_reg, df_res))
    return p_values


def _check_scale(pls_kwargs: dict) -> None:
    """Reject ``scale=False``: every fold is autoscaled, so the full-data model must be too."""
    if pls_kwargs.get("scale", True) is False:
        raise ValueError(
            "scale=False is not supported: each training fold is autoscaled, and the model fitted to all "
            "rows must live in the same space for the angles and the Procrustes step to compare them. "
            "Pass X and Y on their raw scale."
        )


def _check_probability(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1); got {value}.")


@dataclass(frozen=True)
class _CompareSettings:
    """Settings of :meth:`PLS.compare_cv_criteria`; see that method for their meaning."""

    max_components: int | None = None
    cv: int | BaseCrossValidator = 7
    random_state: int | np.random.Generator | None = None
    n_permutations: int = 999
    n_cv_permutations: int = 199
    alpha: float = 0.05
    angle_threshold: float = 30.0
    conf_level: float = 0.95

    def validate(self) -> None:
        """Raise ``ValueError`` for an out-of-range setting."""
        if int(self.n_permutations) < 1:
            raise ValueError(f"n_permutations must be >= 1; got {self.n_permutations}.")
        if int(self.n_cv_permutations) < 0:
            raise ValueError(f"n_cv_permutations must be >= 0; got {self.n_cv_permutations}.")
        _check_probability("alpha", self.alpha)
        _check_probability("conf_level", self.conf_level)
        if not 0.0 < self.angle_threshold <= 90.0:
            raise ValueError(f"angle_threshold must lie in (0, 90] degrees; got {self.angle_threshold}.")


@dataclass(frozen=True)
class _PseudoValidationSettings:
    """Settings of :meth:`PLS.pseudo_validation_set`; see that method for their meaning."""

    n_components: int
    cv: int | BaseCrossValidator = 7
    scope: Literal["global", "local"] = "global"
    random_state: int | np.random.Generator | None = None


def _predictive_criteria(
    held: Bunch, y_values: np.ndarray, settings: _CompareSettings, rng: np.random.Generator
) -> Bunch:
    """Q2 and its SE, RMSECV and its SE, van der Voet and CV-ANOVA, from the held-out pass.

    Every sum runs over the observed Y cells, the ones PRESS is summed over.
    """
    observed = ~np.isnan(y_values)
    M = y_values.shape[1]
    A, n_folds = held.per_fold_press.shape
    tss_y = np.nansum((y_values - np.nanmean(y_values, axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q2_per_target = np.where(tss_y > 0, 1.0 - held.press_y / np.where(tss_y > 0, tss_y, 1.0), np.nan)
    q2_total = 1.0 - held.press_y.sum(axis=1) / tss_y.sum() if tss_y.sum() > 0 else np.full(A, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se_press = np.nanstd(held.per_fold_press, axis=1, ddof=1) / np.sqrt(n_folds)
        se_rmsecv = np.nanstd(held.per_fold_rmse, axis=1, ddof=1) / np.sqrt(n_folds)
    rmsecv = np.sqrt(held.press_y.sum(axis=1) / max(1, int(observed.sum())))
    vdv_recommended, vdv_p = _vandervoet_randomization(
        held.per_obs_sse,
        total_rmsecv=rmsecv,
        n_permutations=int(settings.n_permutations),
        alpha=settings.alpha,
        random_state=rng,
    )
    return Bunch(
        q2y=_equal_weight_r2_total(q2_per_target),
        q2y_se=se_press * n_folds / tss_y.sum() if tss_y.sum() > 0 else np.full(A, np.nan),
        cv_anova_p=_cv_anova_pvalues(q2_total, int(observed[:, 0].sum()), M),
        vdv_p=vdv_p,
        vdv_recommended=int(vdv_recommended),
        rmsecv=rmsecv,
        se_rmsecv=se_rmsecv,
    )


def _monitoring_criteria(model: Any, folds: list[_Fold], conf_level: float, alpha: float) -> Bunch:  # noqa: ANN401
    """Procrustes D ratios, pseudo-validation SPE / T2 alarm rates, and the SPE limit refitted to held-out rows.

    The full-data SPE limit is fitted to training residuals, which are smaller than the
    residuals of rows the model has not seen, so it is too tight for new rows, and more
    so the fewer rows there are per variable. The pseudo-validation SPE is the
    held-out residual, so a limit fitted to it is calibrated for new rows.
    """
    N = sum(len(fold.test) for fold in folds)
    pv_scores, squared_spe, d_ratios = _procrustes_scores(model, folds)
    score_sd = model.scaling_factor_for_scores_.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_t2 = np.cumsum((pv_scores / np.where(score_sd > 0, score_sd, np.nan)) ** 2, axis=1)
    pv_spe = np.sqrt(squared_spe)
    A = pv_scores.shape[1]
    spe_alarm = np.full(A, np.nan)
    t2_alarm = np.full(A, np.nan)
    spe_limits = np.full((A, 2), np.nan)
    for a in range(A):
        spe_lim = float(spe_calculation(model.spe_.iloc[:, a].to_numpy(), conf_level=conf_level))
        spe_limits[a] = spe_lim, float(spe_calculation(pv_spe[:, a], conf_level=conf_level))
        t2_lim = hotellings_t2_limit(conf_level=conf_level, n_components=a + 1, n_rows=N)
        if np.isfinite(spe_lim):
            spe_alarm[a] = float(np.mean(pv_spe[:, a] > spe_lim))
        if np.isfinite(t2_lim) and np.all(np.isfinite(pv_t2[:, a])):
            t2_alarm[a] = float(np.mean(pv_t2[:, a] > t2_lim))
    return Bunch(
        d_ratios=d_ratios,
        spe_alarm=spe_alarm,
        t2_alarm=t2_alarm,
        spe_limits=spe_limits,
        alarm_rate_upper=float(binom.ppf(1.0 - alpha, N, 1.0 - conf_level) / N),
    )


def _recommendations(table: pd.DataFrame, predictive: Bunch, settings: _CompareSettings) -> pd.DataFrame:
    """One row per selection rule: the recommended component count, the rule, and its question."""
    alpha = settings.alpha
    q2y = table["q2y"].to_numpy()
    swaps = _swapped_pairs(table["angle_subspace_deg"].to_numpy(), settings.angle_threshold)
    rules = {
        "q2_max": (
            int(np.nanargmax(q2y)) + 1 if np.isfinite(q2y).any() else 1,
            "Largest cumulative Q2.",
            "Does the model predict Y?",
        ),
        "q2_1se": (
            _select_n_components("1se", mean_error=predictive.rmsecv, se_error=predictive.se_rmsecv),
            "Fewest components whose RMSECV is within one standard error of the minimum.",
            "Does the model predict Y?",
        ),
        "van_der_voet": (
            predictive.vdv_recommended,
            f"Fewest components whose predictions are not worse than the minimum-PRESS model (p > {alpha}).",
            "Is the gain over fewer components real?",
        ),
        "score_correlation": (
            _leading_count(table["r_cv_p"].to_numpy() < alpha),
            f"Leading components whose held-out t-u correlation beats the Y-permutation null (p < {alpha}).",
            "Does each component's inner relation hold on new rows?",
        ),
        "covariance_permutation": (
            _leading_count(table["cov_perm_p"].to_numpy() < alpha),
            f"Leading components whose covariance beats {int(settings.n_permutations)} row permutations (p < {alpha}).",
            "Is each component's covariance larger than chance?",
        ),
        "subspace_stability": (
            _leading_count(table["angle_subspace_deg"].to_numpy() < settings.angle_threshold, swaps),
            (
                f"Leading components whose jackknife-scaled subspace angle is below {settings.angle_threshold} "
                "degrees; a pair of components that swap inside a stable span counts as two."
            ),
            "Does the model's latent space survive a change of rows?",
        ),
    }
    return pd.DataFrame(
        [(name, n, rule, question) for name, (n, rule, question) in rules.items()],
        columns=["criterion", "n_components", "rule", "question"],
    ).set_index("criterion")


def _compare_cv_criteria(
    estimator: Callable[..., Any],
    X: DataMatrix,
    Y: DataMatrix | pd.Series,
    settings: _CompareSettings,
    pls_kwargs: dict,
) -> Bunch:
    """Run :meth:`PLS.compare_cv_criteria` for the estimator class ``estimator``."""
    X_df, Y_df = _as_frames(X, Y)
    _check_scale(pls_kwargs)
    settings.validate()
    _warn_scaling_traps(X_df, scale_inside_folds=True, fold="CV fold", metric="Q2")

    rng = check_random_state(settings.random_state)
    splits = _partition_splits(settings.cv, X_df, Y_df, rng=rng, random_state=settings.random_state)
    rng_vdv, rng_perm, rng_null = rng.spawn(3)
    A = _component_cap(splits, X_df.shape[1], settings.max_components)
    folds = _fit_folds(lambda: estimator(n_components=A, **pls_kwargs), X_df, Y_df, splits, scope="local")
    model = estimator(n_components=A, **pls_kwargs).fit(X_df, Y_df)
    y_values = Y_df.to_numpy()
    held = _heldout_pass(folds, y_values, A)
    predictive = _predictive_criteria(held, y_values, settings, rng_vdv)

    if int(settings.n_cv_permutations) > 0:
        observed, null = _score_correlation_null(
            X_df.to_numpy(), y_values, splits, (int(settings.n_cv_permutations), A), rng_null
        )
        r_cv_p, r_cv_threshold = _permutation_summary(null, observed, settings.alpha)
    else:
        n_with_y = int((~np.isnan(y_values).all(axis=1)).sum())
        r_cv_p = norm.sf(held.r_cv * np.sqrt(n_with_y))
        r_cv_threshold = np.full(A, norm.ppf(1.0 - settings.alpha) / np.sqrt(n_with_y))
    x_scaler, y_scaler = _global_scaling(X_df, Y_df)
    cov_perm_p = _covariance_permutation_pvalues(
        x_scaler.transform(X_df).to_numpy(),
        y_scaler.transform(Y_df).to_numpy(),
        model,
        int(settings.n_permutations),
        rng_perm,
    )
    angle_component, angle_subspace = _weight_angles(model.x_weights_.to_numpy(), folds)
    monitoring = _monitoring_criteria(model, folds, settings.conf_level, settings.alpha)
    rows_with_y = ~np.isnan(y_values).all(axis=1)  # a row with no response has u = 0 by construction
    t_scores, u_scores = np.asarray(model.scores_)[rows_with_y], np.asarray(model.y_scores_)[rows_with_y]

    component_index = pd.Index(range(1, A + 1), name="n_components")
    table = pd.DataFrame(
        {
            "r2y": model.r2_cumulative_.to_numpy(),
            "q2y": predictive.q2y,
            "q2y_se": predictive.q2y_se,
            "vdv_p": predictive.vdv_p,
            "cv_anova_p": predictive.cv_anova_p,
            "r_train": [_correlation_about_origin(t_scores[:, a], u_scores[:, a]) for a in range(A)],
            "r_cv": held.r_cv,
            "r_cv_threshold": r_cv_threshold,
            "r_cv_p": r_cv_p,
            "slope_ratio": held.slope_ratio,
            "cov_perm_p": cov_perm_p,
            "angle_component_deg": angle_component,
            "angle_subspace_deg": angle_subspace,
            "d_ratio_median": np.median(monitoring.d_ratios, axis=0),
            "d_ratio_min": np.min(monitoring.d_ratios, axis=0),
            "pv_spe_alarm_rate": monitoring.spe_alarm,
            "pv_t2_alarm_rate": monitoring.t2_alarm,
            "pv_spe_limit_ratio": monitoring.spe_limits[:, 1] / monitoring.spe_limits[:, 0],
        },
        index=component_index,
    )
    fold_index = [f"fold_{k + 1}" for k in range(len(folds))]
    return Bunch(
        table=table,
        recommendations=_recommendations(table, predictive, settings),
        press=pd.Series(held.press_y.sum(axis=1), index=component_index, name="PRESS"),
        press_baseline=held.press_baseline,
        d_ratios=pd.DataFrame(monitoring.d_ratios, index=fold_index, columns=component_index),
        spe_limits=pd.DataFrame(
            monitoring.spe_limits, index=component_index, columns=["full_data", "pseudo_validation"]
        ),
        cv_splits=splits,
        global_model=model,
        alpha=settings.alpha,
        conf_level=settings.conf_level,
        angle_threshold=settings.angle_threshold,
        alarm_rate_upper=monitoring.alarm_rate_upper,
    )


def _column_normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0)
    return values / np.where(norms > 0, norms, 1.0)


def _pseudo_validation_set(
    estimator: Callable[..., Any],
    X: DataMatrix,
    Y: DataMatrix | pd.Series,
    settings: _PseudoValidationSettings,
    pls_kwargs: dict,
) -> Bunch:
    """Run :meth:`PLS.pseudo_validation_set` for the estimator class ``estimator``."""
    if settings.scope not in ("global", "local"):
        raise ValueError(f"scope must be 'global' or 'local'; got {settings.scope!r}.")
    X_df, Y_df = _as_frames(X, Y)
    _check_scale(pls_kwargs)
    N, K = X_df.shape
    rng = check_random_state(settings.random_state)
    splits = _partition_splits(settings.cv, X_df, Y_df, rng=rng, random_state=settings.random_state)
    (rng_pcv,) = rng.spawn(1)
    cap = _component_cap(splits, K, None)
    A = int(settings.n_components)
    if not 1 <= A <= cap:
        raise ValueError(f"n_components must lie in [1, {cap}] for these folds; got {settings.n_components}.")

    fold_kwargs = (
        {**pls_kwargs, "scale": False, "warn_on_uncentred": False} if settings.scope == "global" else pls_kwargs
    )
    folds = _fit_folds(lambda: estimator(n_components=A, **fold_kwargs), X_df, Y_df, splits, scope=settings.scope)
    model = estimator(n_components=A, **pls_kwargs).fit(X_df, Y_df)
    pv_scores, squared_spe, d_ratios = _procrustes_scores(model, folds)

    # X_pv = T_pv P' + E, with E orthogonal to the model (E W* = 0) and each row of E
    # carrying the fold model's residual norm, so the full-data model reports exactly
    # the held-out scores (rescaled by D) and the held-out SPE for every complete row.
    # E mixes the held-out rows at random (missing cells at the mean), and the missing
    # cells of X are missing again in X_pv.
    direct = model.direct_weights_.to_numpy()
    x_loadings = model.x_loadings_.to_numpy()
    x_pv = np.zeros((N, K))
    for fold in folds:
        n_test = len(fold.test)
        mixed = _column_normalise(rng_pcv.standard_normal((n_test, n_test)) @ np.nan_to_num(fold.x_test))
        residual = mixed - (mixed @ direct) @ x_loadings.T
        norms = np.linalg.norm(residual, axis=1)
        target = np.sqrt(squared_spe[fold.test, A - 1])
        factor = np.where(norms > 0, target / np.where(norms > 0, norms, 1.0), 0.0)
        x_pv[fold.test] = pv_scores[fold.test] @ x_loadings.T + residual * factor[:, None]

    x_pv[X_df.isna().to_numpy()] = np.nan
    x_scaler, _ = _global_scaling(X_df, Y_df)
    component_index = pd.Index(range(1, A + 1), name="n_components")
    return Bunch(
        X_pv=x_scaler.inverse_transform(pd.DataFrame(x_pv, index=X_df.index, columns=X_df.columns)),
        Y_pv=Y_df.copy(),
        scores=pd.DataFrame(pv_scores, index=X_df.index, columns=component_index),
        local_spe=pd.Series(np.sqrt(squared_spe[:, A - 1]), index=X_df.index, name="SPE"),
        d_ratios=pd.DataFrame(d_ratios, index=[f"fold_{k + 1}" for k in range(len(folds))], columns=component_index),
        cv_splits=splits,
        global_model=model,
        scope=settings.scope,
    )
