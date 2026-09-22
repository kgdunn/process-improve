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
data but its training slope is more than twice too steep: the score correlation
keeps it and :math:`Q^2` drops it.

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
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import binom, norm
from scipy.stats import f as f_dist
from sklearn.model_selection import BaseCrossValidator, KFold, check_cv
from sklearn.utils import Bunch

from .._random import check_random_state
from ._adaptive import _kernel_pls, _sign_align
from ._common import _equal_weight_r2_total, _select_n_components
from ._limits import hotellings_t2_limit, spe_calculation
from ._pls import PLS, _vandervoet_randomization
from ._preprocessing import MCUVScaler, _warn_scaling_traps

if typing.TYPE_CHECKING:
    from ._common import DataMatrix

__all__ = ["compare_cv_criteria", "pseudo_validation_set"]

#: Degrees of freedom charged per PLS component by CV-ANOVA (Eriksson, Trygg and Wold, 2008).
CV_ANOVA_DF_PER_COMPONENT = 2

#: Permutations evaluated per vectorised batch in the covariance permutation test.
_PERMUTATION_BATCH = 256


@dataclass
class _Fold:
    """One cross-validation fold: its rows, held-out data, and fitted model matrices."""

    train: np.ndarray
    test: np.ndarray
    x_test: np.ndarray  # held-out X in the fold model's scaled space, (n_test, K)
    y_centre: np.ndarray  # maps the fold's scaled Y back to original units, (M,)
    y_scale: np.ndarray  # (M,)
    weights: np.ndarray  # W, (K, A)
    direct_weights: np.ndarray  # W* = W (P'W)^-1, (K, A)
    x_loadings: np.ndarray  # P, (K, A)
    y_loadings: np.ndarray  # C, (M, A)


def _as_frames(X: DataMatrix, Y: DataMatrix | pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce X and Y to float DataFrames with one row per observation, and reject gaps."""
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
    # Held-out scores are X_test @ W*, which is undefined for a missing cell, and the
    # NIPALS missing-data path breaks the triangular P'W that the angle and Procrustes
    # calculations rely on.
    if X_df.isna().to_numpy().any() or Y_df.isna().to_numpy().any():
        raise ValueError("X and Y must not contain missing values; impute or remove them first.")
    return X_df, Y_df


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


def _fold_from_model(  # noqa: PLR0913
    model: PLS,
    train: np.ndarray,
    test: np.ndarray,
    x_test: np.ndarray,
    y_centre: np.ndarray,
    y_scale: np.ndarray,
) -> _Fold:
    """Collect the fitted matrices a fold contributes to the criteria."""
    return _Fold(
        train=train,
        test=test,
        x_test=x_test,
        y_centre=np.asarray(y_centre, dtype=float),
        y_scale=np.asarray(y_scale, dtype=float),
        weights=model.x_weights_.to_numpy(),
        direct_weights=model.direct_weights_.to_numpy(),
        x_loadings=model.x_loadings_.to_numpy(),
        y_loadings=typing.cast("pd.DataFrame", model.y_loadings_).to_numpy(),
    )


def _global_scaling(X: pd.DataFrame, Y: pd.DataFrame, pls_kwargs: dict) -> tuple[MCUVScaler | None, MCUVScaler | None]:
    """Return the scalers that map raw data into the full-data model's space (None if unscaled)."""
    if not pls_kwargs.get("scale", True):
        return None, None
    return MCUVScaler().fit(X), MCUVScaler().fit(Y)


def _fit_folds(  # noqa: PLR0913
    estimator: type[PLS],
    X: pd.DataFrame,
    Y: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_components: int,
    pls_kwargs: dict,
    *,
    scope: Literal["local", "global"],
) -> list[_Fold]:
    """Fit one model per fold.

    ``"local"`` scope autoscales each training fold with its own :class:`MCUVScaler`,
    exactly as :meth:`PLS.select_n_components` does. ``"global"`` scope fits the fold
    models on rows of the fully autoscaled data without re-centring them, which is the
    default of the published Procrustes cross-validation.
    """
    folds: list[_Fold] = []
    if scope == "global":
        x_scaler, y_scaler = _global_scaling(X, Y, pls_kwargs)
        Xs = x_scaler.transform(X) if x_scaler is not None else X
        Ys = y_scaler.transform(Y) if y_scaler is not None else Y
        y_centre = y_scaler.center_.to_numpy() if y_scaler is not None else np.zeros(Y.shape[1])
        y_scale = y_scaler.scale_.to_numpy() if y_scaler is not None else np.ones(Y.shape[1])
        fold_kwargs = {**pls_kwargs, "scale": False, "warn_on_uncentred": False}
        for train, test in splits:
            model = estimator(n_components=n_components, **fold_kwargs).fit(Xs.iloc[train], Ys.iloc[train])
            folds.append(_fold_from_model(model, train, test, Xs.iloc[test].to_numpy(), y_centre, y_scale))
        return folds

    for train, test in splits:
        scaler_x = MCUVScaler().fit(X.iloc[train])
        scaler_y = MCUVScaler().fit(Y.iloc[train])
        model = estimator(n_components=n_components, **pls_kwargs).fit(
            scaler_x.transform(X.iloc[train]), scaler_y.transform(Y.iloc[train])
        )
        x_test = scaler_x.transform(X.iloc[test]).to_numpy()
        folds.append(
            _fold_from_model(model, train, test, x_test, scaler_y.center_.to_numpy(), scaler_y.scale_.to_numpy())
        )
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
    """Project every held-out fold and accumulate PRESS, the slope ratio and the pooled scores.

    PRESS is computed in original Y units, the same way :meth:`PLS.select_n_components`
    computes it. The slope ratio is accumulated in the same units, so that
    ``PRESS[a-1] - PRESS[a] == (2 * slope_ratio[a] - 1) * slope_weight[a]`` exactly.
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
        scores = fold.x_test @ fold.direct_weights
        y_loadings = fold.y_loadings
        residual_prev = y_test - fold.y_centre  # a = 0: predict the training-fold mean
        press_baseline += float(np.sum(residual_prev**2))
        for a in range(A):
            t = scores[:, a]
            c = y_loadings[:, a]
            c_original = c * fold.y_scale
            slope_numerator[a] += float(t @ residual_prev @ c_original)
            slope_weight[a] += float(c_original @ c_original) * float(t @ t)
            # u_a = Y_res c_a / (c_a'c_a), with Y_res the held-out Y deflated by the
            # earlier held-out components, in the fold model's scaled units.
            cc = float(c @ c)
            u = (residual_prev / fold.y_scale) @ c / cc if cc > 0 else np.zeros_like(t)
            t_pool[a].append(t)
            u_pool[a].append(u)

            y_hat = scores[:, : a + 1] @ y_loadings[:, : a + 1].T * fold.y_scale + fold.y_centre
            residual = y_test - y_hat
            squared = residual**2
            press_y[a] += squared.sum(axis=0)
            per_obs_sse[a, fold.test] = squared.sum(axis=1)
            per_fold_press[a, k] = float(squared.sum())
            per_fold_rmse[a, k] = float(np.sqrt(squared.sum() / max(1, len(fold.test) * M)))
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
    """Column means and ``ddof=1`` standard deviations, with the :class:`MCUVScaler` guard."""
    centre = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    tiny = float(np.finfo(float).tiny) ** 0.5
    return centre, np.where(~np.isfinite(scale) | (scale <= tiny), 1.0, scale)


def _score_correlation_pvalues(  # noqa: PLR0913
    x_values: np.ndarray,
    y_values: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    observed: np.ndarray,
    n_permutations: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation test of the held-out t-u correlation of every component.

    Each permutation shuffles the rows of Y and repeats the whole cross-validation,
    with the folds refitted by kernel PLS (Dayal and MacGregor, 1997) from the fold's
    ``X'X``, which a permutation of Y leaves unchanged, and the new ``X'Y``; for
    complete data the kernel algorithm gives the NIPALS model.

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
    p_values : np.ndarray of shape (A,)
    thresholds : np.ndarray of shape (A,)
        The ``1 - alpha`` quantile of each component's null distribution.
    """
    A = len(observed)
    folds = []
    for train, test in splits:
        centre, scale = _autoscale_parameters(x_values[train])
        x_train = (x_values[train] - centre) / scale
        folds.append((train, test, x_train, (x_values[test] - centre) / scale, x_train.T @ x_train))

    def _heldout_correlations(y: np.ndarray) -> np.ndarray:
        tu = np.zeros(A)
        tt = np.zeros(A)
        uu = np.zeros(A)
        for train, test, x_train, x_test, xtx in folds:
            centre, scale = _autoscale_parameters(y[train])
            _, _, direct, y_loadings, _ = _kernel_pls(xtx, x_train.T @ ((y[train] - centre) / scale), A)
            t_test = x_test @ direct
            residual = (y[test] - centre) / scale
            for a in range(A):
                c = y_loadings[:, a]
                cc = float(c @ c)
                t = t_test[:, a]
                u = residual @ c / cc if cc > 0 else np.zeros_like(t)
                tu[a] += float(t @ u)
                tt[a] += float(t @ t)
                uu[a] += float(u @ u)
                residual = residual - np.outer(t, c)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tt * uu > 0, tu / np.sqrt(tt * uu), np.nan)

    null = np.array([_heldout_correlations(y_values[rng.permutation(len(y_values))]) for _ in range(n_permutations)])
    p_values = np.full(A, np.nan)
    thresholds = np.full(A, np.nan)
    for a in range(A):
        finite = null[np.isfinite(null[:, a]), a]
        if np.isfinite(observed[a]) and finite.size:
            p_values[a] = (np.sum(finite >= observed[a]) + 1) / (n_permutations + 1)
            thresholds[a] = np.quantile(finite, 1.0 - alpha, method="higher")
    return p_values, thresholds


def _covariance_permutation_pvalues(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    model: PLS,
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
    """
    N = x_scaled.shape[0]
    M = y_scaled.shape[1]
    scores = model.scores_.to_numpy()
    x_loadings = model.x_loadings_.to_numpy()
    y_loadings = typing.cast("pd.DataFrame", model.y_loadings_).to_numpy()
    x_a = x_scaled.copy()
    y_a = y_scaled.copy()
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
        x_a = x_a - t @ x_loadings[:, [a]].T
        y_a = y_a - t @ y_loadings[:, [a]].T
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


def _procrustes_scores(model: PLS, folds: list[_Fold]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pseudo-validation scores, local squared SPE, and D ratios (Kucheryavskiy et al., 2023).

    Each fold model is sign-aligned to the full-data model. The held-out scores in the
    fold model, ``T_k = X_k W*_k``, are scaled per component by
    ``d_ka = c_ka'c_a / c_a'c_a``, the ratio of the fold's Y loading to the full-data one,
    giving the pseudo-validation scores. The local squared SPE of each held-out row is
    what the pseudo-validation row's residual is built to reproduce.

    Returns
    -------
    scores : np.ndarray of shape (N, A)
    squared_spe : np.ndarray of shape (N, A)
    d_ratios : np.ndarray of shape (n_folds, A)
    """
    global_direct = model.direct_weights_.to_numpy()
    global_y_loadings = typing.cast("pd.DataFrame", model.y_loadings_).to_numpy()
    N = sum(len(fold.test) for fold in folds)
    A = global_direct.shape[1]
    cc = np.sum(global_y_loadings**2, axis=0)
    pv_scores = np.zeros((N, A))
    squared_spe = np.zeros((N, A))
    d_ratios = np.zeros((len(folds), A))
    for k, fold in enumerate(folds):
        # _sign_align only reports the signs; apply them to W*, P and C together.
        signs = _sign_align(fold.direct_weights, global_direct)
        direct = fold.direct_weights * signs
        x_loadings = fold.x_loadings * signs
        y_loadings = fold.y_loadings * signs
        local_scores = fold.x_test @ direct
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(cc > 0, np.sum(y_loadings * global_y_loadings, axis=0) / cc, np.nan)
        d_ratios[k] = d
        pv_scores[fold.test] = local_scores * d
        for a in range(A):
            residual = fold.x_test - local_scores[:, : a + 1] @ x_loadings[:, : a + 1].T
            squared_spe[fold.test, a] = np.sum(residual**2, axis=1)
    return pv_scores, squared_spe, d_ratios


def _leading_count(passes: np.ndarray) -> int:
    """Count the leading components that pass, stopping at the first failure (or NaN)."""
    failures = np.flatnonzero(~passes)
    return int(failures[0]) if failures.size else len(passes)


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


def _check_probability(name: str, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1); got {value}.")


def _compare_cv_criteria(  # noqa: PLR0913, PLR0915
    estimator: type[PLS],
    X: DataMatrix,
    Y: DataMatrix | pd.Series,
    *,
    max_components: int | None,
    cv: int | BaseCrossValidator,
    random_state: int | np.random.Generator | None,
    n_permutations: int,
    n_cv_permutations: int,
    alpha: float,
    angle_threshold: float,
    conf_level: float,
    pls_kwargs: dict,
) -> Bunch:
    """Shared implementation of :func:`compare_cv_criteria` and :meth:`PLS.compare_cv_criteria`."""
    X_df, Y_df = _as_frames(X, Y)
    if int(n_permutations) < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}.")
    if int(n_cv_permutations) < 0:
        raise ValueError(f"n_cv_permutations must be >= 0; got {n_cv_permutations}.")
    _check_probability("alpha", alpha)
    _check_probability("conf_level", conf_level)
    if not 0.0 < angle_threshold <= 90.0:
        raise ValueError(f"angle_threshold must lie in (0, 90] degrees; got {angle_threshold}.")
    _warn_scaling_traps(X_df, scale_inside_folds=True, fold="CV fold", metric="Q2")

    N, K = X_df.shape
    M = Y_df.shape[1]
    rng = check_random_state(random_state)
    splits = _partition_splits(cv, X_df, Y_df, rng=rng, random_state=random_state)
    rng_vdv, rng_perm, rng_null = rng.spawn(3)
    A = _component_cap(splits, K, max_components)
    folds = _fit_folds(estimator, X_df, Y_df, splits, A, pls_kwargs, scope="local")
    model = estimator(n_components=A, **pls_kwargs).fit(X_df, Y_df)
    component_index = pd.Index(range(1, A + 1), name="n_components")

    # Prediction: PRESS, Q2, the 1-SE rule, van der Voet and CV-ANOVA.
    y_values = Y_df.to_numpy()
    held = _heldout_pass(folds, y_values, A)
    tss_y = np.sum((y_values - y_values.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q2_per_target = np.where(tss_y > 0, 1.0 - held.press_y / np.where(tss_y > 0, tss_y, 1.0), np.nan)
    q2_total = 1.0 - held.press_y.sum(axis=1) / tss_y.sum() if tss_y.sum() > 0 else np.full(A, np.nan)
    q2y = _equal_weight_r2_total(q2_per_target)
    n_folds = len(folds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se_press = np.nanstd(held.per_fold_press, axis=1, ddof=1) / np.sqrt(n_folds)
        se_rmsecv = np.nanstd(held.per_fold_rmse, axis=1, ddof=1) / np.sqrt(n_folds)
    q2y_se = se_press * n_folds / tss_y.sum() if tss_y.sum() > 0 else np.full(A, np.nan)
    rmsecv = np.sqrt(held.press_y.sum(axis=1) / (N * M))
    vdv_recommended, vdv_p = _vandervoet_randomization(
        held.per_obs_sse, total_rmsecv=rmsecv, n_permutations=int(n_permutations), alpha=alpha, random_state=rng_vdv
    )

    # Inner relation and covariance.
    t_scores = np.asarray(model.scores_)
    u_scores = np.asarray(model.y_scores_)
    r_train = np.array([_correlation_about_origin(t_scores[:, a], u_scores[:, a]) for a in range(A)])
    x_scaler, y_scaler = _global_scaling(X_df, Y_df, pls_kwargs)
    x_scaled = (x_scaler.transform(X_df) if x_scaler is not None else X_df).to_numpy()
    y_scaled = (y_scaler.transform(Y_df) if y_scaler is not None else Y_df).to_numpy()
    if int(n_cv_permutations) > 0:
        r_cv_p, r_cv_threshold = _score_correlation_pvalues(
            X_df.to_numpy(), y_values, splits, held.r_cv, int(n_cv_permutations), alpha, rng_null
        )
    else:
        r_cv_p = norm.sf(held.r_cv * np.sqrt(N))
        r_cv_threshold = np.full(A, norm.ppf(1.0 - alpha) / np.sqrt(N))
    cov_perm_p = _covariance_permutation_pvalues(x_scaled, y_scaled, model, int(n_permutations), rng_perm)

    # Stability of the weights, and Procrustes cross-validation.
    angle_component, angle_subspace = _weight_angles(model.x_weights_.to_numpy(), folds)
    pv_scores, squared_spe, d_ratios = _procrustes_scores(model, folds)
    score_sd = model.scaling_factor_for_scores_.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pv_t2 = np.cumsum((pv_scores / np.where(score_sd > 0, score_sd, np.nan)) ** 2, axis=1)
    pv_spe = np.sqrt(squared_spe)
    pv_spe_alarm = np.full(A, np.nan)
    pv_t2_alarm = np.full(A, np.nan)
    for a in range(A):
        spe_lim = float(spe_calculation(model.spe_.iloc[:, a].to_numpy(), conf_level=conf_level))
        t2_lim = hotellings_t2_limit(conf_level=conf_level, n_components=a + 1, n_rows=N)
        if np.isfinite(spe_lim):
            pv_spe_alarm[a] = float(np.mean(pv_spe[:, a] > spe_lim))
        if np.isfinite(t2_lim) and np.all(np.isfinite(pv_t2[:, a])):
            pv_t2_alarm[a] = float(np.mean(pv_t2[:, a] > t2_lim))
    alarm_rate_upper = float(binom.ppf(1.0 - alpha, N, 1.0 - conf_level) / N)

    table = pd.DataFrame(
        {
            "r2y": model.r2_cumulative_.to_numpy(),
            "q2y": q2y,
            "q2y_se": q2y_se,
            "vdv_p": vdv_p,
            "cv_anova_p": _cv_anova_pvalues(q2_total, N, M),
            "r_train": r_train,
            "r_cv": held.r_cv,
            "r_cv_threshold": r_cv_threshold,
            "r_cv_p": r_cv_p,
            "slope_ratio": held.slope_ratio,
            "cov_perm_p": cov_perm_p,
            "angle_component_deg": angle_component,
            "angle_subspace_deg": angle_subspace,
            "d_ratio_median": np.median(d_ratios, axis=0),
            "d_ratio_min": np.min(d_ratios, axis=0),
            "pv_spe_alarm_rate": pv_spe_alarm,
            "pv_t2_alarm_rate": pv_t2_alarm,
        },
        index=component_index,
    )

    finite_q2 = np.isfinite(q2y)
    rules = {
        "q2_max": (
            int(np.nanargmax(q2y)) + 1 if finite_q2.any() else 1,
            "Largest cumulative Q2.",
            "Does the model predict Y?",
        ),
        "q2_1se": (
            _select_n_components("1se", mean_error=rmsecv, se_error=se_rmsecv),
            "Fewest components whose RMSECV is within one standard error of the minimum.",
            "Does the model predict Y?",
        ),
        "van_der_voet": (
            int(vdv_recommended),
            f"Fewest components whose predictions are not worse than the minimum-PRESS model (p > {alpha}).",
            "Is the gain over fewer components real?",
        ),
        "score_correlation": (
            _leading_count(r_cv_p < alpha),
            f"Leading components whose held-out t-u correlation beats the Y-permutation null (p < {alpha}).",
            "Does each component's inner relation hold on new rows?",
        ),
        "covariance_permutation": (
            _leading_count(cov_perm_p < alpha),
            f"Leading components whose covariance beats {int(n_permutations)} row permutations (p < {alpha}).",
            "Is each component's covariance larger than chance?",
        ),
        "subspace_stability": (
            _leading_count(angle_subspace < angle_threshold),
            f"Leading components whose jackknife-scaled weight subspace angle is below {angle_threshold} degrees.",
            "Does the model's latent space survive a change of rows?",
        ),
        "pv_spe_alarm": (
            _leading_count(pv_spe_alarm <= alarm_rate_upper),
            (
                f"Leading components whose out-of-sample SPE alarm rate is at most {alarm_rate_upper:.3g}, "
                f"the upper {1 - alpha:.3g} binomial quantile of the nominal rate {1 - conf_level:.3g}."
            ),
            "Are the monitoring limits right on new rows?",
        ),
    }
    recommendations = pd.DataFrame(
        [(name, n, rule, question) for name, (n, rule, question) in rules.items()],
        columns=["criterion", "n_components", "rule", "question"],
    ).set_index("criterion")

    return Bunch(
        table=table,
        recommendations=recommendations,
        press=pd.Series(held.press_y.sum(axis=1), index=component_index, name="PRESS"),
        press_baseline=held.press_baseline,
        d_ratios=pd.DataFrame(d_ratios, index=[f"fold_{k + 1}" for k in range(n_folds)], columns=component_index),
        cv_splits=splits,
        global_model=model,
        alpha=alpha,
        conf_level=conf_level,
        angle_threshold=angle_threshold,
        alarm_rate_upper=alarm_rate_upper,
    )


def compare_cv_criteria(  # noqa: PLR0913
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

    Runs one K-fold cross-validation (one :class:`PLS` fit per fold at
    ``max_components``, truncated for smaller counts) and reports, for every number of
    components, criteria that each answer a different question, together with the
    number of components each one recommends. Disagreement between them is the point:
    a component can predict Y without being stable, or be stable and real without
    improving the prediction.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor block, on its raw scale. Each training fold is autoscaled with its
        own :class:`MCUVScaler`, as in :meth:`PLS.select_n_components`.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response block.
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
        Passed to :class:`PLS` for every fit (for example ``max_iter``, ``tol``).

    Returns
    -------
    result : sklearn.utils.Bunch
        ``table`` : pandas.DataFrame, indexed by ``n_components``
            ``r2y``, ``q2y``, ``q2y_se``
                In-sample :math:`R^2_Y` (pooled in scaled units), cross-validated
                :math:`Q^2_Y` (every target weighted equally, which for one target is
                the ordinary :math:`Q^2`), and its standard error across folds.
            ``vdv_p``
                Van der Voet *p*-value against the minimum-PRESS model.
            ``cv_anova_p``
                CV-ANOVA *p*-value for the whole model, one response only (NaN otherwise).
                It is a monotone function of :math:`Q^2`.
            ``r_train``, ``r_cv``, ``r_cv_threshold``, ``r_cv_p``
                Correlation of :math:`t_a` and :math:`u_a` on the training rows, the same
                correlation pooled over the held-out rows (about the training-fold
                centre, so it is unaffected by sign flips between folds), the
                :math:`1-\alpha` quantile of its null distribution under permuted Y, and
                its one-sided permutation *p*-value.
            ``slope_ratio``
                :math:`s_a`, the held-out slope of the deflated response on
                :math:`t_a` relative to the training slope. PRESS falls exactly when
                :math:`s_a > 1/2`; see the module notes.
            ``cov_perm_p``
                Sequential permutation *p*-value for the covariance of component ``a``.
            ``angle_component_deg``, ``angle_subspace_deg``
                Angle between each fold's weight vector :math:`w_a` and the full-data
                one, and the largest principal angle between the spans of the first
                ``a`` weights, each scaled by the delete-d jackknife
                (:math:`\tan\theta \to \sqrt{G-1}\,\text{rms}(\tan\theta_k)`) so it
                estimates how far the full-data direction may lie from the population
                one. Raw fold angles shrink as the number of folds grows (fold models
                share most of their rows); the scaled ones do not. A large component
                angle with a small subspace angle means the components swap or mix
                inside a stable space: loadings are then not interpretable one by one,
                but SPE, :math:`T^2` and inversion, which depend on the span, are.
            ``d_ratio_median``, ``d_ratio_min``
                Procrustes D ratios :math:`c_{ka}^\top c_a / c_a^\top c_a` across folds;
                a negative minimum means some fold reversed the component's inner relation.
            ``pv_spe_alarm_rate``, ``pv_t2_alarm_rate``
                Fraction of the Procrustes pseudo-validation rows above the full-data
                model's SPE and :math:`T^2` limits at ``conf_level``; nominally
                ``1 - conf_level``.
        ``recommendations`` : pandas.DataFrame
            One row per selection rule (``q2_max``, ``q2_1se``, ``van_der_voet``,
            ``score_correlation``, ``covariance_permutation``, ``subspace_stability``,
            ``pv_spe_alarm``) with the recommended ``n_components``, the ``rule`` and
            the ``question`` it answers. The structural rules count leading components
            that pass, stopping at the first failure, and can return 0.
        ``press`` : pandas.Series
            Cross-validated PRESS in original Y units.
        ``press_baseline`` : float
            PRESS of predicting each held-out row by its training fold's mean.
        ``d_ratios`` : pandas.DataFrame
            Procrustes D ratio per fold and component.
        ``cv_splits`` : list of (train, test) index arrays
        ``global_model`` : PLS
            Model fitted to all rows with ``max_components`` components.
        ``alpha``, ``conf_level``, ``angle_threshold``, ``alarm_rate_upper`` : float
            The settings used, and the binomial upper bound for the alarm rates.

    Raises
    ------
    ValueError
        For missing values, a splitter that does not hold out every row exactly once,
        or an out-of-range setting.

    See Also
    --------
    PLS.select_n_components : Choose the number of components from prediction alone.
    pseudo_validation_set : Build the Procrustes pseudo-validation set explicitly.
    process_improve.multivariate.plots.cv_criteria_plot : Plot the ``table``.

    Examples
    --------
    >>> from process_improve.multivariate import compare_cv_criteria
    >>> result = compare_cv_criteria(X, y, max_components=6, random_state=0)
    >>> result.recommendations["n_components"]
    >>> result.table[["q2y", "r_cv", "slope_ratio", "angle_subspace_deg"]]
    """
    return _compare_cv_criteria(
        PLS,
        X,
        Y,
        max_components=max_components,
        cv=cv,
        random_state=random_state,
        n_permutations=n_permutations,
        n_cv_permutations=n_cv_permutations,
        alpha=alpha,
        angle_threshold=angle_threshold,
        conf_level=conf_level,
        pls_kwargs=pls_kwargs,
    )


def _column_normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=0)
    return values / np.where(norms > 0, norms, 1.0)


def _pseudo_validation_set(  # noqa: PLR0913
    estimator: type[PLS],
    X: DataMatrix,
    Y: DataMatrix | pd.Series,
    *,
    n_components: int,
    cv: int | BaseCrossValidator,
    scope: Literal["global", "local"],
    random_state: int | np.random.Generator | None,
    pls_kwargs: dict,
) -> Bunch:
    """Shared implementation of :func:`pseudo_validation_set` and :meth:`PLS.pseudo_validation_set`."""
    if scope not in ("global", "local"):
        raise ValueError(f"scope must be 'global' or 'local'; got {scope!r}.")
    X_df, Y_df = _as_frames(X, Y)
    N, K = X_df.shape
    rng = check_random_state(random_state)
    splits = _partition_splits(cv, X_df, Y_df, rng=rng, random_state=random_state)
    (rng_pcv,) = rng.spawn(1)
    cap = _component_cap(splits, K, None)
    if int(n_components) < 1 or int(n_components) > cap:
        raise ValueError(f"n_components must lie in [1, {cap}] for these folds; got {n_components}.")
    A = int(n_components)

    folds = _fit_folds(estimator, X_df, Y_df, splits, A, pls_kwargs, scope=scope)
    model = estimator(n_components=A, **pls_kwargs).fit(X_df, Y_df)
    pv_scores, squared_spe, d_ratios = _procrustes_scores(model, folds)

    # X_pv = T_pv P' + E, with E orthogonal to the model (E W* = 0) and each row of E
    # carrying the fold model's residual norm, so the full-data model reports exactly
    # the held-out scores (rescaled by D) and the held-out SPE for every row.
    direct = model.direct_weights_.to_numpy()
    x_loadings = model.x_loadings_.to_numpy()
    x_pv = np.zeros((N, K))
    for fold in folds:
        n_test = len(fold.test)
        mixed = _column_normalise(rng_pcv.standard_normal((n_test, n_test)) @ fold.x_test)
        residual = mixed - (mixed @ direct) @ x_loadings.T
        norms = np.linalg.norm(residual, axis=1)
        target = np.sqrt(squared_spe[fold.test, A - 1])
        factor = np.where(norms > 0, target / np.where(norms > 0, norms, 1.0), 0.0)
        x_pv[fold.test] = pv_scores[fold.test] @ x_loadings.T + residual * factor[:, None]

    x_scaler, _ = _global_scaling(X_df, Y_df, pls_kwargs)
    X_pv = pd.DataFrame(x_pv, index=X_df.index, columns=X_df.columns)
    if x_scaler is not None:
        X_pv = x_scaler.inverse_transform(X_pv)
    component_index = pd.Index(range(1, A + 1), name="n_components")
    return Bunch(
        X_pv=X_pv,
        Y_pv=Y_df.copy(),
        scores=pd.DataFrame(pv_scores, index=X_df.index, columns=component_index),
        local_spe=pd.Series(np.sqrt(squared_spe[:, A - 1]), index=X_df.index, name="SPE"),
        d_ratios=pd.DataFrame(d_ratios, index=[f"fold_{k + 1}" for k in range(len(folds))], columns=component_index),
        cv_splits=splits,
        global_model=model,
        scope=scope,
    )


def pseudo_validation_set(  # noqa: PLR0913
    X: DataMatrix,
    Y: DataMatrix | pd.Series,
    *,
    n_components: int,
    cv: int | BaseCrossValidator = 7,
    scope: Literal["global", "local"] = "global",
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

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor block, raw scale.
    Y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Response block.
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
        Passed to :class:`PLS` for every fit.

    Returns
    -------
    result : sklearn.utils.Bunch
        ``X_pv`` : pandas.DataFrame
            Pseudo-validation predictors, on the raw scale of ``X``.
        ``Y_pv`` : pandas.DataFrame
            ``Y``, unchanged.
        ``scores`` : pandas.DataFrame
            Scores the full-data model gives ``X_pv``.
        ``local_spe`` : pandas.Series
            SPE of each held-out row in its fold model; the full-data model gives
            ``X_pv`` the same values.
        ``d_ratios`` : pandas.DataFrame
            D ratio per fold and component.
        ``cv_splits`` : list of (train, test) index arrays
        ``global_model`` : PLS
            Model fitted to all rows; apply it to ``X_pv`` with ``diagnose``.
        ``scope`` : str

    See Also
    --------
    compare_cv_criteria : Uses the same construction to report alarm rates per component.

    Examples
    --------
    >>> pv = pseudo_validation_set(X, y, n_components=2, random_state=0)
    >>> diagnostics = pv.global_model.diagnose(pv.X_pv)
    >>> (diagnostics.spe > pv.global_model.spe_limit()).mean()   # out-of-sample SPE alarm rate
    """
    return _pseudo_validation_set(
        PLS,
        X,
        Y,
        n_components=n_components,
        cv=cv,
        scope=scope,
        random_state=random_state,
        pls_kwargs=pls_kwargs,
    )
