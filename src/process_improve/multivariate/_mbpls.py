# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Multi-block PLS (MBPLS) estimator and its randomization test (ENG-01).

Holds :class:`MBPLS`, the hierarchical / superblock multi-block PLS regressor,
and :func:`randomization_test_mbpls`, the permutation test for the significance
of each fitted component.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import typing
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedKFold
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..visualization.themes import REFERENCE_LINE_COLOR
from ._base import _HotellingsT2LimitMixin
from ._common import (
    SelectionRule,
    SpecificationWarning,
    _equal_weight_r2_total,
    _nz,
    _scale_block_contributions,
    _select_n_components,
    epsqrt,
)
from ._diagnostics import _select_rows
from ._limits import spe_calculation
from ._nipals import quick_regress, ssq
from ._preprocessing import MCUVScaler

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _stacked_super_weights(
    block_weights: dict[str, np.ndarray],
    super_weight: np.ndarray,
    sqrt_kb: dict[str, float],
    block_names: Sequence[str],
) -> np.ndarray:
    r"""Give the super score's weight on every variable, stacked over the blocks in ``block_names`` order.

    Written out over the variables rather than over the blocks,

    .. math::
        t_i = \frac{1}{\mathbf{w}_s' \mathbf{w}_s} \sum_b w_{s,b}
              \frac{\mathbf{x}_{ib}' \mathbf{w}_b}{\sqrt{K_b}}
            = \frac{\mathbf{x}_i' \mathbf{g}}{\mathbf{w}_s' \mathbf{w}_s},
        \qquad
        \mathbf{g} = \big[\, w_{s,b}\, \mathbf{w}_b / \sqrt{K_b} \,\big]_b

    the super score is one linear combination of all the variables of all the blocks.
    ``g`` is that combination.
    """
    return np.concatenate([super_weight[b] * block_weights[name] / sqrt_kb[name] for b, name in enumerate(block_names)])


def _pooled_super_score(
    x_def: dict[str, np.ndarray],
    stacked_weights: np.ndarray,
    super_weight: np.ndarray,
    block_names: Sequence[str],
) -> np.ndarray:
    r"""Score every row by one masked regression of the whole row onto the stacked weights.

    The alternative is to score each block on its own and add the per-block scores up. On a
    complete row the two give the same number, but they part company as soon as a row has missing
    cells: a block observed in one variable out of twenty still hands a score built on that one
    variable into the sum, weighted as though it were as well determined as the others. Regressing
    the whole row at once instead lets every cell the row does have carry its share, wherever in
    the blocks that cell sits.

    The ratio is rescaled by :math:`\mathbf{g}'\mathbf{g}` so a complete row reproduces the
    weighted sum of block scores exactly, which is what leaves every complete-data model
    unchanged. A row with nothing observed under a non-zero weight has no score: it comes back
    as NaN.
    """
    stacked_x = np.hstack([x_def[name] for name in block_names])
    observed = ~np.isnan(stacked_x)
    numerator = np.nan_to_num(stacked_x) @ stacked_weights
    denominator = observed @ (stacked_weights**2)
    scale = float(stacked_weights @ stacked_weights) / _nz(float(super_weight @ super_weight))
    usable = denominator > 0
    return np.where(usable, numerator / np.where(usable, denominator, 1.0) * scale, np.nan)


def _rows_with_data(values: np.ndarray) -> np.ndarray:
    """Report, for each row, whether it has at least one observed cell."""
    return np.any(~np.isnan(values), axis=1)


def _validate_fit_arguments(X: dict[str, pd.DataFrame], y: pd.DataFrame) -> pd.DataFrame:
    """Check the shapes and types ``fit`` was handed, and return ``y`` as a DataFrame."""
    if not isinstance(X, dict) or len(X) == 0:
        raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)
    for name, block in X.items():
        if not isinstance(block, pd.DataFrame):
            raise TypeError(f"X['{name}'] must be a pandas DataFrame; got {type(block).__name__}.")

    n_samples = X[next(iter(X))].shape[0]
    for name in X:
        if X[name].shape[0] != n_samples:
            raise ValueError(
                f"All X-blocks must have the same row count. Block '{name}' has "
                f"{X[name].shape[0]} rows; expected {n_samples}."
            )
    if y.shape[0] != n_samples:
        raise ValueError(f"y has {y.shape[0]} rows; expected {n_samples} to match X-blocks.")
    return y


def _resolve_missing_data_settings(missing_data_settings: dict | None, algo: str) -> dict:
    """Resolve, and for the ``"nipals"`` path validate, the iterative-algorithm settings."""
    settings = {"md_tol": epsqrt, "md_max_iter": 1000}
    if isinstance(missing_data_settings, dict):
        settings.update(missing_data_settings)
    settings["md_max_iter"] = int(settings["md_max_iter"])
    if algo == "nipals":
        if not settings["md_tol"] < 10:
            raise ValueError("Tolerance should not be too large.")
        if not settings["md_tol"] > epsqrt**1.95:
            raise ValueError("Tolerance must exceed machine precision.")
    return settings


def _reject_degenerate_missingness(X: dict[str, pd.DataFrame], y: pd.DataFrame, block_names: Sequence[str]) -> None:
    """Refuse the missing-data patterns the masked NIPALS path cannot estimate from.

    A column with nothing in it leaves the masked NIPALS denominator at zero for that
    variable's loading, so it is refused. A *row* with nothing in one block is allowed: the
    super score is estimated from the cells that row has in the other blocks, which is the
    multiblock case of an observation missing one whole analysis. Only a row with nothing
    observed in any block carries no information at all, and that one is refused.
    """
    for name in block_names:
        values = X[name].values
        col_all_nan = np.all(np.isnan(values), axis=0)
        if np.any(col_all_nan):
            bad = X[name].columns[col_all_nan].tolist()
            raise ValueError(
                f"Block '{name}' has columns with all values missing: {bad}. Drop these columns before fitting."
            )
    observed_anywhere = np.zeros(X[block_names[0]].shape[0], dtype=bool)
    for name in block_names:
        observed_anywhere |= _rows_with_data(X[name].values)
    if not np.all(observed_anywhere):
        bad_rows = np.where(~observed_anywhere)[0].tolist()
        raise ValueError(
            f"Rows at positions {bad_rows} have all values missing in every X-block. "
            "Drop these observations or impute them before fitting."
        )
    y_values = y.values
    y_col_all_nan = np.all(np.isnan(y_values), axis=0)
    if np.any(y_col_all_nan):
        bad = y.columns[y_col_all_nan].tolist()
        raise ValueError(f"Y has columns with all values missing: {bad}. Drop these targets before fitting.")
    y_row_all_nan = np.all(np.isnan(y_values), axis=1)
    if np.any(y_row_all_nan):
        bad_rows = np.where(y_row_all_nan)[0].tolist()
        raise ValueError(
            f"Y has rows with all values missing at positions {bad_rows}. "
            "Drop these observations or impute them before fitting."
        )


class _MBPLSLoopContext(typing.NamedTuple):
    """The quantities one component's NIPALS iteration needs that do not change between components."""

    algo: str
    block_names: Sequence[str]
    sqrt_kb: dict[str, float]
    block_has_data: dict[str, np.ndarray]
    tol: float
    max_iter: int


class _MBPLSComponent(typing.NamedTuple):
    """What one component's NIPALS iteration converged to, after the sign convention."""

    block_weights: dict[str, np.ndarray]
    block_scores: dict[str, np.ndarray]
    super_weight: np.ndarray
    super_score: np.ndarray
    super_y_score: np.ndarray
    super_y_loading: np.ndarray
    iterations: int


def _block_projections(
    x_def: dict[str, np.ndarray], u_a: np.ndarray, context: _MBPLSLoopContext
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Regress every block on ``u``, giving each block's weight, its score, and the score summary.

    The ``"nipals"`` branch is mask-aware: each projection is a per-column (or per-row)
    regression that uses only the entries that are not NaN, and divides by the masked sum of
    squares. It reuses the same primitives as single-block PCA NIPALS.
    """
    local_w: dict[str, np.ndarray] = {}
    local_t: dict[str, np.ndarray] = {}
    t_b_summary = np.zeros((u_a.shape[0], len(context.block_names)))
    if context.algo == "nipals":
        u_a_col = u_a.reshape(-1, 1)
        for b_idx, name in enumerate(context.block_names):
            w_b = quick_regress(x_def[name], u_a_col).flatten()
            w_b = w_b / _nz(float(np.sqrt(ssq(w_b.reshape(-1, 1)))))
            t_b = quick_regress(x_def[name], w_b.reshape(-1, 1)).flatten() / context.sqrt_kb[name]
            local_w[name] = w_b
            local_t[name] = t_b
            t_b_summary[:, b_idx] = np.where(context.block_has_data[name], t_b, np.nan)
    else:
        for b_idx, name in enumerate(context.block_names):
            w_b = x_def[name].T @ u_a / _nz(float(u_a @ u_a))
            w_b = w_b / _nz(float(np.linalg.norm(w_b)))
            t_b = x_def[name] @ w_b / _nz(float(w_b @ w_b)) / context.sqrt_kb[name]
            local_w[name] = w_b
            local_t[name] = t_b
            t_b_summary[:, b_idx] = t_b
    return local_w, local_t, t_b_summary


def _apply_sign_convention(
    component: _MBPLSComponent,
    block_names: Sequence[str],
) -> _MBPLSComponent:
    """Flip the component so the largest element of the super weight is positive."""
    w_s = component.super_weight
    flip_idx = int(np.argmax(np.abs(w_s)))
    if w_s[flip_idx] >= 0:
        return component
    return component._replace(
        block_weights={name: -component.block_weights[name] for name in block_names},
        block_scores={name: -component.block_scores[name] for name in block_names},
        super_weight=-w_s,
        super_score=-component.super_score,
        super_y_score=-component.super_y_score,
        super_y_loading=-component.super_y_loading,
    )


def _fit_one_component(
    x_def: dict[str, np.ndarray], y_def: np.ndarray, n_targets: int, context: _MBPLSLoopContext
) -> _MBPLSComponent:
    """Iterate one hierarchical NIPALS component to convergence on the deflated blocks."""
    n_samples = y_def.shape[0]
    # Deterministic start (#503): seed ``u`` from the column of the (deflated) Y block with the
    # largest sum of squares, exactly as single-block PLS does (#195). No RNG is involved, so the
    # fit is reproducible without a random_state parameter, and the highest-variance Y column is
    # closest to the leading component. The sign convention applied after convergence makes the
    # fitted signs independent of this seed. (NaN is replaced by 0 for the missing-data path.)
    start_col = int(np.argmax(np.nansum(y_def**2, axis=0)))
    u_a = np.nan_to_num(y_def[:, start_col].astype(float).copy())
    prev = u_a + 1.0
    local_w: dict[str, np.ndarray] = {}
    local_t: dict[str, np.ndarray] = {}
    t_super = np.zeros(n_samples)
    w_s = np.zeros(len(context.block_names))
    c_a = np.zeros(n_targets)
    itern = 0
    # Relative convergence criterion (#504): the change between two successive ``u`` iterations is
    # judged against the size of the current ``u`` vector, so the decision is invariant to a global
    # rescaling of the data. The denominator is floored via ``_nz`` so an all-zero ``u`` vector
    # cannot divide by zero.
    while np.linalg.norm(prev - u_a) / _nz(float(np.linalg.norm(u_a))) > context.tol and itern < context.max_iter:
        prev = u_a
        local_w, local_t, t_b_summary = _block_projections(x_def, u_a, context)

        if context.algo == "nipals":
            # Masked, so a block this row has nothing in does not vote on the super weight.
            w_s = quick_regress(t_b_summary, u_a.reshape(-1, 1)).flatten()
        else:
            w_s = t_b_summary.T @ u_a / _nz(float(u_a @ u_a))
        w_s = w_s / _nz(float(np.linalg.norm(w_s)))
        # One masked regression of the whole row onto the stacked weights, rather than a weighted
        # sum of per-block scores: identical on complete rows, and on a row with missing cells it
        # uses every cell the row has instead of one score per block.
        t_super = _pooled_super_score(
            x_def,
            _stacked_super_weights(local_w, w_s, context.sqrt_kb, context.block_names),
            w_s,
            context.block_names,
        )
        # A row observed only where the weights are zero has no score for this component: place it
        # at the centre so the iteration stays finite. The guard in ``_reject_degenerate_missingness``
        # has already refused a row with nothing observed anywhere.
        t_super = np.nan_to_num(t_super)
        if context.algo == "nipals":
            t_super_col = t_super.reshape(-1, 1)
            c_a = quick_regress(y_def, t_super_col).flatten()
            u_a = quick_regress(y_def, c_a.reshape(-1, 1)).flatten()
        else:
            c_a = y_def.T @ t_super / _nz(float(t_super @ t_super))
            u_a = y_def @ c_a / _nz(float(c_a @ c_a))
        itern += 1

    component = _MBPLSComponent(
        block_weights=local_w,
        block_scores=local_t,
        super_weight=w_s,
        super_score=t_super,
        super_y_score=u_a,
        super_y_loading=c_a,
        iterations=itern,
    )
    return _apply_sign_convention(component, context.block_names)


def _deflate(
    x_def: dict[str, np.ndarray], y_def: np.ndarray, component: _MBPLSComponent, context: _MBPLSLoopContext
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    """Remove the component from every X-block and from Y, using the super score.

    Returns the deflated blocks, the deflated Y, and the per-block loadings that did the
    deflating.
    """
    t_super = component.super_score
    t_super_col = t_super.reshape(-1, 1)
    loadings: dict[str, np.ndarray] = {}
    for name in context.block_names:
        if context.algo == "nipals":
            p_b = quick_regress(x_def[name], t_super_col).flatten()
        else:
            p_b = x_def[name].T @ t_super / _nz(float(t_super @ t_super))
        x_def[name] = x_def[name] - np.outer(t_super, p_b)
        loadings[name] = p_b
    return x_def, y_def - np.outer(t_super, component.super_y_loading), loadings


class _InitialSumsOfSquares(typing.NamedTuple):
    """The sums of squares of the preprocessed data, the denominators of every R² below."""

    x_block: dict[str, float]
    x_variable: dict[str, np.ndarray]
    y_total: float
    y_variable: np.ndarray

    @classmethod
    def of(cls, x_blocks_pp: dict[str, np.ndarray], y_pp: np.ndarray) -> _InitialSumsOfSquares:
        """Take the sums of squares of the preprocessed blocks and Y."""
        return cls(
            x_block={name: float(np.nansum(values**2)) for name, values in x_blocks_pp.items()},
            x_variable={name: np.nansum(values**2, axis=0) for name, values in x_blocks_pp.items()},
            y_total=float(np.nansum(y_pp**2)),
            y_variable=np.nansum(y_pp**2, axis=0),
        )


@dataclasses.dataclass
class _MBPLSArrays:
    """The numpy workspace a fit fills in, one column per component, wrapped in pandas at the end."""

    super_scores: np.ndarray
    super_y_scores: np.ndarray
    super_weights: np.ndarray
    super_y_loadings: np.ndarray
    block_scores: dict[str, np.ndarray]
    block_weights: dict[str, np.ndarray]
    block_loadings: dict[str, np.ndarray]
    block_spe: dict[str, np.ndarray]
    r2_x_block_cum: np.ndarray
    r2_x_var_cum: dict[str, np.ndarray]
    r2_y_cum: np.ndarray
    r2_y_var_cum: np.ndarray
    timing: np.ndarray
    iterations: np.ndarray

    @classmethod
    def allocate(cls, block_widths: dict[str, int], n_samples: int, n_targets: int, n_components: int) -> _MBPLSArrays:
        """Allocate the workspace for a fit of the given shape."""
        n_blocks = len(block_widths)
        per_block = {name: np.zeros((n_samples, n_components)) for name in block_widths}
        per_variable = {name: np.zeros((width, n_components)) for name, width in block_widths.items()}
        return cls(
            super_scores=np.zeros((n_samples, n_components)),
            super_y_scores=np.zeros((n_samples, n_components)),
            super_weights=np.zeros((n_blocks, n_components)),
            super_y_loadings=np.zeros((n_targets, n_components)),
            block_scores=per_block,
            block_weights={name: values.copy() for name, values in per_variable.items()},
            block_loadings={name: values.copy() for name, values in per_variable.items()},
            block_spe={name: values.copy() for name, values in per_block.items()},
            r2_x_block_cum=np.zeros((n_blocks, n_components)),
            r2_x_var_cum={name: values.copy() for name, values in per_variable.items()},
            r2_y_cum=np.zeros(n_components),
            r2_y_var_cum=np.zeros((n_targets, n_components)),
            timing=np.zeros(n_components),
            iterations=np.zeros(n_components, dtype=int),
        )

    def record_component(
        self, a: int, component: _MBPLSComponent, loadings: dict[str, np.ndarray], context: _MBPLSLoopContext
    ) -> None:
        """Store component ``a``'s scores, weights and loadings.

        A block a row has nothing observed in gets NaN for its block score, not a zero, which
        would place the row at that block's average.
        """
        for name in context.block_names:
            self.block_loadings[name][:, a] = loadings[name]
            self.block_weights[name][:, a] = component.block_weights[name]
            self.block_scores[name][:, a] = np.where(context.block_has_data[name], component.block_scores[name], np.nan)
        self.super_scores[:, a] = component.super_score
        self.super_y_scores[:, a] = component.super_y_score
        self.super_weights[:, a] = component.super_weight
        self.super_y_loadings[:, a] = component.super_y_loading
        self.iterations[a] = component.iterations

    def record_explained_variation(
        self,
        a: int,
        x_def: dict[str, np.ndarray],
        y_def: np.ndarray,
        initial: _InitialSumsOfSquares,
        context: _MBPLSLoopContext,
    ) -> None:
        """Store the cumulative R²X and R²Y through component ``a``, and the per-block SPE.

        R² is undefined for a zero-variance block or column; it is reported as NaN rather than
        dividing by zero (inf/nan plus a warning) or returning a misleading 1.0.
        """
        for b_idx, name in enumerate(context.block_names):
            ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
            self.r2_x_block_cum[b_idx, a] = (
                1 - np.sum(ssq_remain_per_var) / initial.x_block[name] if initial.x_block[name] > 0 else np.nan
            )
            per_var = initial.x_variable[name]
            self.r2_x_var_cum[name][:, a] = np.where(
                per_var > 0, 1 - ssq_remain_per_var / np.where(per_var > 0, per_var, 1.0), np.nan
            )
            self.block_spe[name][:, a] = np.where(
                context.block_has_data[name], np.sqrt(np.nansum(x_def[name] ** 2, axis=1)), np.nan
            )
        ssq_y_remain_per_var = np.nansum(y_def**2, axis=0)
        self.r2_y_cum[a] = 1 - np.sum(ssq_y_remain_per_var) / initial.y_total if initial.y_total > 0 else np.nan
        self.r2_y_var_cum[:, a] = np.where(
            initial.y_variable > 0,
            1 - ssq_y_remain_per_var / np.where(initial.y_variable > 0, initial.y_variable, 1.0),
            np.nan,
        )


def _cumulative_to_per_component(cumulative: np.ndarray) -> np.ndarray:
    """Difference a cumulative R² along its last (component) axis, keeping the first value as it is."""
    per_component = np.empty_like(cumulative)
    per_component[..., 0] = cumulative[..., 0]
    if cumulative.shape[-1] > 1:
        per_component[..., 1:] = np.diff(cumulative, axis=-1)
    return per_component


class MBPLS(_HotellingsT2LimitMixin, RegressorMixin, BaseEstimator):
    r"""Multi-block PLS (hierarchical / superblock formulation).

    Generic multi-block PLS as described by Westerhuis, Kourti & MacGregor
    (1998) and Westerhuis & Smilde (2001). Each X-block is preprocessed
    independently (mean-centred and unit-variance scaled), then divided by
    ``sqrt(K_b)`` so that blocks of unequal width contribute fairly to the
    super-score.

    Parameters
    ----------
    n_components : int
        Number of latent variables to extract.
    max_iter : int, default=500
        Maximum NIPALS iterations per latent variable.
    tol : float or None, default=None
        Relative convergence tolerance on the change in the Y-block score
        ``u``: the norm of the change between two successive iterations,
        divided by the norm of the current ``u`` vector (#504). If ``None``,
        ``epsqrt`` (about 1.49e-8) is used, the same default as PCA / PLS /
        TPLS. The legacy absolute tolerance ``np.finfo(float).eps ** (6/7)``
        (about 3.8e-14) sits below the floating-point oscillation floor of a
        relative criterion, so it would never be reached in practice.
    algorithm : str, default="auto"
        Algorithm to use for fitting the model.

        - ``"auto"``: dense vectorised hierarchical NIPALS when every
          block (X and Y) is complete; mask-aware NIPALS when any block
          contains missing values.
        - ``"dense"``: dense vectorised hierarchical NIPALS. Raises if
          any block contains missing values.
        - ``"nipals"``: mask-aware hierarchical NIPALS. Always uses the
          NaN-tolerant inner-loop primitives, even when the data is
          complete (slower than ``"dense"`` but produces equivalent
          results).

        With missing data the super score of each row is estimated by one masked regression of
        the whole row onto the stacked block weights, rather than by adding up a score per block.
        The two are the same number on a complete row, so no fitted model changes; on an
        incomplete row the pooled form lets every observed cell carry its share, wherever in the
        blocks it sits, instead of letting a block seen in one variable speak as loudly as a block
        seen in twenty. A row with nothing observed in one block is therefore still scored, from
        the blocks it does have; only a row observed in no block at all is refused.

    missing_data_settings : dict or None, default=None
        Settings for the iterative ``"nipals"`` path. Keys: ``md_tol``
        (convergence tolerance on the score-vector change between
        iterations), ``md_max_iter`` (maximum NIPALS iterations per
        component). Defaults to ``{"md_tol": epsqrt, "md_max_iter": 1000}``.

    Attributes (after fitting)
    --------------------------
    block_names_ : list[str]
        Ordered list of X-block names (the keys of the input dict).
    block_widths_ : dict[str, int]
        Number of variables in each X-block.
    n_samples_ : int
        Number of rows fitted.
    n_targets_ : int
        Number of Y columns.
    n_features_in_ : int
        Total number of X variables summed across blocks.
    feature_names_in_ : np.ndarray
        Concatenated column names, one per feature, in block order.
    preproc_ : dict[str, MCUVScaler]
        Per-block preprocessors used to mean-centre and unit-variance
        scale each X-block.
    y_preproc_ : MCUVScaler
        Preprocessor used on Y.
    super_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block (consensus) X-scores ``T``. Finite for every row that has at least one
        observed cell, in any block.
    super_y_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block Y-scores ``U``.
    super_weights_ : pd.DataFrame, shape (n_blocks, n_components)
        Super-block weights ``w_super``; rows indexed by block name.
    super_y_loadings_ : pd.DataFrame, shape (n_targets, n_components)
        Y-block loadings ``c``.
    super_hotellings_t2_ : pd.DataFrame, shape (n_samples, n_components)
        Cumulative Hotelling's T^2 on the super-scores per component.
    super_vip_ : pd.Series
        Variable-importance in projection for each X-block, indexed by
        block name.
    block_scores_ : dict[str, pd.DataFrame]
        Per-block X-scores ``t_b``, each shape ``(n_samples, n_components)``. NaN for a row with
        nothing observed in that block: the block has no score of its own there, and reporting
        zero would place the row at the block's average instead.
    block_weights_ : dict[str, pd.DataFrame]
        Per-block X-weights ``w_b``, each shape ``(K_b, n_components)``.
        Each column has unit norm.
    block_loadings_ : dict[str, pd.DataFrame]
        Per-block X-loadings ``p_b`` (used for deflation), each shape
        ``(K_b, n_components)``.
    block_spe_ : dict[str, pd.DataFrame]
        Per-block squared prediction error per sample and component. NaN where the row has
        nothing observed in that block, for the same reason as ``block_scores_``.
    block_hotellings_t2_ : dict[str, pd.DataFrame]
        Per-block cumulative Hotelling's T^2 per sample and component.
    block_vip_ : dict[str, pd.Series]
        Per-block variable-importance in projection, indexed by variable
        name inside each block.
    predictions_ : pd.DataFrame, shape (n_samples, n_targets)
        In-sample Y predictions on the *original* scale.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variance of the super-score per component (ddof=1).
    scaling_factor_for_super_scores_ : pd.Series
        ``sqrt(explained_variance_)`` per component.
    r2_x_per_block_cumulative_ : pd.DataFrame, shape (n_blocks, n_components)
        Cumulative R^2X per block and component.
    r2_x_per_block_per_component_ : pd.DataFrame, shape (n_blocks, n_components)
        Incremental R^2X per block and component.
    r2_x_per_variable_ : dict[str, pd.DataFrame]
        Cumulative R^2X per variable within each block.
    r2_y_cumulative_ : pd.Series, shape (n_components,)
        Cumulative R^2Y per component.
    r2_y_per_component_ : pd.Series, shape (n_components,)
        Incremental R^2Y per component.
    r2_y_per_variable_ : pd.DataFrame, shape (n_targets, n_components)
        Cumulative R^2Y per Y-variable and component.
    fitting_info_ : dict
        Per-component iteration count and timing.
    has_missing_data_ : bool
        Whether any X-block or Y had NaN values.
    algorithm_ : str
        The resolved algorithm actually used for the fit. With
        ``algorithm="auto"``, this is ``"dense"`` for complete data
        and ``"nipals"`` for NaN-containing data.

    Notes
    -----
    Block weighting uses the convention :math:`X_b / \sqrt{K_b}` so that
    every block contributes the same total sum of squares to the
    super-score, regardless of how many variables it has.

    Missing data
    ------------
    When any X-block or Y contains NaN entries, the ``"auto"``
    algorithm routes to a mask-aware NIPALS variant. The X-block
    weights, block scores, block loadings used for deflation, Y-block
    loadings and Y-block scores are each computed as a regression that
    uses only the observed entries; the masked sum-of-squares is used
    as the denominator so missing values neither bias the latent
    direction nor contribute to the score. The mask is preserved
    across components automatically because deflation propagates NaN
    through subtraction. This is the standard skip-NaN NIPALS update;
    see Walczak & Massart (2001) and Arteaga & Ferrer (2002).

    The fit refuses to run if any X-block or Y has a column with all
    entries missing, or a row with all entries missing for that
    block; either case leaves the masked denominator at zero. Drop or
    impute such rows or columns before fitting. Predict-time score
    estimation for new observations with NaN (Trimmed Score Regression
    / Projection to the Model Plane) is a separate follow-up.

    References
    ----------
    Westerhuis, J. A., Kourti, T. & MacGregor, J. F. *Analysis of
    multiblock and hierarchical PCA and PLS models.* Journal of
    Chemometrics, 12 (1998), 301-321.

    Westerhuis, J. A. & Smilde, A. K. *Deflation in multiblock PLS.*
    Journal of Chemometrics, 15 (2001), 485-493.

    Walczak, B. & Massart, D. L. *Dealing with missing data: Part I.*
    Chemom. Intell. Lab. Syst., 58 (2001), 15-27.

    Arteaga, F. & Ferrer, A. *Dealing with missing data in MSPC: several
    methods, different interpretations, some examples.* J. Chemometrics,
    16 (2002), 408-418.
    """

    _valid_algorithms: typing.ClassVar[list[str]] = ["auto", "dense", "nipals"]

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int],
        "max_iter": [int],
        "tol": [float, None],
        "algorithm": [str],
        "missing_data_settings": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        *,
        max_iter: int = 500,
        tol: float | None = None,
        algorithm: str = "auto",
        missing_data_settings: dict | None = None,
    ):
        super().__init__()
        if n_components <= 0:
            raise ValueError(f"n_components must be positive; got {n_components}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive; got {max_iter}.")
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.missing_data_settings = missing_data_settings

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> MBPLS:
        """Fit the multi-block PLS model.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            X-blocks. Keys are block names; values are DataFrames sharing the
            same row index (and row count). Each block is preprocessed
            independently.
        y : pd.DataFrame
            Y-block. Same row index / row count as the X-blocks.
        """
        y = _validate_fit_arguments(X, y)
        self._record_data_shape(X, y)
        algo = self._resolve_algorithm(X, y)
        # Resolve the iterative-algorithm settings. Only the validation inside is load-bearing
        # today: the resolved ``md_tol`` and ``md_max_iter`` do not yet reach the NIPALS path.
        _resolve_missing_data_settings(self.missing_data_settings, algo)
        if algo == "nipals":
            _reject_degenerate_missingness(X, y, self.block_names_)

        x_blocks_pp, y_pp = self._preprocess(X, y)
        context = _MBPLSLoopContext(
            algo=algo,
            block_names=self.block_names_,
            # Algorithmic block weighting: X_b / sqrt(K_b)
            sqrt_kb={name: float(np.sqrt(width)) for name, width in self.block_widths_.items()},
            # A row with nothing observed in a block has no score *for that block*; it still has a
            # super score, estimated from the blocks it does have.
            block_has_data={name: _rows_with_data(values) for name, values in x_blocks_pp.items()},
            tol=epsqrt if self.tol is None else float(self.tol),
            max_iter=self.max_iter,
        )

        work = _MBPLSArrays.allocate(self.block_widths_, self.n_samples_, self.n_targets_, self.n_components_)
        initial = _InitialSumsOfSquares.of(x_blocks_pp, y_pp)
        x_def = {name: values.copy() for name, values in x_blocks_pp.items()}
        y_def = y_pp.copy()

        for a in range(self.n_components_):
            start = time.time()
            component = _fit_one_component(x_def, y_def, self.n_targets_, context)
            x_def, y_def, loadings = _deflate(x_def, y_def, component, context)
            work.record_component(a, component, loadings, context)
            work.record_explained_variation(a, x_def, y_def, initial, context)
            work.timing[a] = time.time() - start

        self._store_latent_frames(work)
        self._report_convergence(work)
        self._store_explained_variation(work)
        self._store_diagnostics(work)
        return self

    def _record_data_shape(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> None:
        """Record the block names, widths, row and column labels, and counts this fit is over."""
        self.block_names_: list[str] = list(X.keys())
        first = X[self.block_names_[0]]
        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._y_columns = y.columns
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}

        self.n_samples_ = int(first.shape[0])
        self.n_targets_ = int(y.shape[1])
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        # feature_names_in_: sklearn convention (#392). Flat concatenation of
        # all blocks' column names in block-iteration order. Lets
        # ``Pipeline.get_feature_names_out`` and SHAP / eli5 / model-card
        # tooling introspect a multiblock fit through the same surface as a
        # single-block estimator.
        self.feature_names_in_ = np.concatenate([self._block_columns[name].to_numpy() for name in self.block_names_])
        # Fitted mirror of the constructor parameter, so shared helpers (the
        # T2 limit mixin, spe_limit, the plot pre-checks) read one resolved
        # attribute across PCA / PLS / MBPCA / MBPLS (#505).
        self.n_components_ = int(self.n_components)

    def _resolve_algorithm(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> str:
        """Decide which algorithm this fit runs, recording whether the data has gaps."""
        self.has_missing_data_ = any(np.any(X[name].isna().values) for name in self.block_names_) or bool(
            np.any(y.isna().values)
        )
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognised. Must be one of {self._valid_algorithms}."
            )
        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "dense"
        if algo == "dense" and self.has_missing_data_:
            raise ValueError("Algorithm 'dense' cannot handle missing data. Use 'nipals' or 'auto' instead.")
        self.algorithm_ = algo
        return algo

    def _preprocess(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Centre and scale each X-block and Y independently, and return the numpy values."""
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        self.y_preproc_ = MCUVScaler().fit(y)
        x_blocks_pp: dict[str, np.ndarray] = {
            name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_
        }
        return x_blocks_pp, self.y_preproc_.transform(y).values.astype(float)

    def _store_latent_frames(self, work: _MBPLSArrays) -> None:
        """Wrap the fitted scores, weights and loadings in pandas, and predict the fitted rows."""
        component_names = list(range(1, self.n_components_ + 1))
        self.super_scores_ = pd.DataFrame(work.super_scores, index=self._sample_index, columns=component_names)
        self.super_y_scores_ = pd.DataFrame(work.super_y_scores, index=self._sample_index, columns=component_names)
        self.super_weights_ = pd.DataFrame(work.super_weights, index=self.block_names_, columns=component_names)
        self.super_y_loadings_ = pd.DataFrame(work.super_y_loadings, index=self._y_columns, columns=component_names)

        self.block_scores_ = {
            name: pd.DataFrame(work.block_scores[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_weights_ = {
            name: pd.DataFrame(work.block_weights[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(work.block_loadings[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }

        # In-sample predictions on the original Y scale
        y_hat_pp = work.super_scores @ work.super_y_loadings.T
        y_hat = self.y_preproc_.inverse_transform(pd.DataFrame(y_hat_pp, columns=self._y_columns))
        y_hat.index = self._sample_index
        self.predictions_ = y_hat

        self.explained_variance_ = np.diag(work.super_scores.T @ work.super_scores) / max(1, self.n_samples_ - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )

    def _report_convergence(self, work: _MBPLSArrays) -> None:
        """Record how each component's iteration went, and warn about any that ran out of steps."""
        converged = work.iterations < self.max_iter
        self.fitting_info_ = {"timing": work.timing, "iterations": work.iterations, "converged": converged}
        logger.debug("MBPLS (%s): iterations per component = %s", self.algorithm_, list(work.iterations))
        if not np.all(converged):
            failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
            warnings.warn(
                f"MBPLS NIPALS did not converge within max_iter={self.max_iter} for "
                f"component(s) {failed}; results for those components may be unreliable.",
                SpecificationWarning,
                # Three frames up: this helper, ``fit``, and the sklearn ``_fit_context`` wrapper.
                stacklevel=3,
            )

    def _store_explained_variation(self, work: _MBPLSArrays) -> None:
        """Wrap the R² bookkeeping in pandas, and derive the per-block and super VIP from it."""
        component_names = list(range(1, self.n_components_ + 1))
        r2_x_block_per_a = _cumulative_to_per_component(work.r2_x_block_cum)
        r2_y_per_a = _cumulative_to_per_component(work.r2_y_cum)

        self.r2_x_per_block_cumulative_ = pd.DataFrame(
            work.r2_x_block_cum, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_block_per_component_ = pd.DataFrame(
            r2_x_block_per_a, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_variable_ = {
            name: pd.DataFrame(work.r2_x_var_cum[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }
        self.r2_y_cumulative_ = pd.Series(work.r2_y_cum, index=component_names, name="Cumulative R²Y")
        self.r2_y_per_component_ = pd.Series(r2_y_per_a, index=component_names, name="R²Y per component")
        self.r2_y_per_variable_ = pd.DataFrame(work.r2_y_var_cum, index=self._y_columns, columns=component_names)

        # Per-block VIP_jb = sqrt(K_b * sum_a(r2_x_block_a * w_b[j,a]^2) / sum_a r2_x_block_a)
        self.block_vip_: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            r2 = r2_x_block_per_a[b_idx, :]
            if np.sum(r2) > 0:
                w = self.block_weights_[name].values  # (K_b, A)
                vip_b = np.sqrt(self.block_widths_[name] * np.sum(r2 * w**2, axis=1) / np.sum(r2))
            else:
                vip_b = np.zeros(self.block_widths_[name])
            self.block_vip_[name] = pd.Series(vip_b, index=self._block_columns[name], name=f"VIP[{name}]")

        # Super VIP_b = sqrt(B * sum_a(r2_y_a * w_super[b,a]^2) / sum_a r2_y_a)
        n_blocks = len(self.block_names_)
        if np.sum(r2_y_per_a) > 0:
            ws = self.super_weights_.values  # (B, A)
            super_vip = np.sqrt(n_blocks * np.sum(r2_y_per_a * ws**2, axis=1) / np.sum(r2_y_per_a))
        else:
            super_vip = np.zeros(n_blocks)
        self.super_vip_ = pd.Series(super_vip, index=self.block_names_, name="Super VIP")

    def _store_diagnostics(self, work: _MBPLSArrays) -> None:
        """Wrap the per-block SPE, and accumulate the per-block and super Hotelling's T²."""
        component_names = list(range(1, self.n_components_ + 1))
        self.block_spe_ = {
            name: pd.DataFrame(work.block_spe[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }

        # Cumulative T^2 from block scores and super scores (using per-component score variance)
        block_t2: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            scores_np = self.block_scores_[name].values  # (N, A)
            score_var = np.var(scores_np, axis=0, ddof=1)
            score_var = np.where(score_var > 0, score_var, 1.0)
            block_t2[name] = np.cumsum((scores_np**2) / score_var, axis=1)
        self.block_hotellings_t2_ = {
            name: pd.DataFrame(block_t2[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        super_t2 = np.cumsum((work.super_scores**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

    def block_spe_limit(self, block: str, conf_level: float = 0.95) -> float:
        """SPE limit for one X-block using the Nomikos & MacGregor chi-square approximation.

        Operates on the same scale as ``block_spe_[block]`` (sqrt of row sum
        of squares), so the value can be drawn directly on a SPE plot.
        """
        check_is_fitted(self, "block_spe_")
        if block not in self.block_spe_:
            raise KeyError(f"Unknown block '{block}'. Known blocks: {list(self.block_spe_)}.")
        return spe_calculation(self.block_spe_[block].iloc[:, -1].to_numpy(), conf_level=conf_level)

    def super_spe_limit(self, conf_level: float = 0.95) -> float:
        """SPE limit for the merged super-block (sum of per-block SPE squared)."""
        check_is_fitted(self, "block_spe_")
        merged_spe_squared = np.zeros(self.n_samples_)
        for name in self.block_names_:
            merged_spe_squared += self.block_spe_[name].iloc[:, -1].to_numpy() ** 2
        return spe_calculation(np.sqrt(merged_spe_squared), conf_level=conf_level)

    def spe_contributions(self, X: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Per-variable squared residuals for each X-block (SPE contributions).

        For each new observation and each X-block, reconstruct the block as
        ``T_super @ P_b^T`` (matching the deflation step used during fit) and
        return the squared per-variable residuals. Useful for fault diagnosis:
        the variable with the largest contribution is the most likely culprit
        for a high SPE.

        Returns
        -------
        dict[str, pd.DataFrame]
            One DataFrame per block, shape ``(n_samples, K_b)``. Values are
            preprocessed-scale squared residuals (centred and scaled inside
            the model). Sum across columns equals ``block_spe_[b].iloc[:, -1] ** 2``.
        """
        check_is_fitted(self, "block_loadings_")
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks: {sorted(missing)}.")

        result = self._project(X)
        super_scores = result.super_scores.values  # (N, A)
        out: dict[str, pd.DataFrame] = {}
        sample_index = next(iter(result.block_scores.values())).index
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            x_pp = self.preproc_[name].transform(block).values.astype(float)
            x_hat = super_scores @ self.block_loadings_[name].values.T
            residuals_sq = (x_pp - x_hat) ** 2
            out[name] = pd.DataFrame(residuals_sq, index=sample_index, columns=self._block_columns[name])
        return out

    def _deflated_blocks(self, X: dict[str, pd.DataFrame], component: int) -> tuple[dict[str, np.ndarray], pd.Index]:
        """Preprocessed blocks, deflated through the first ``component - 1`` components.

        The super score at component *a* is formed from the data that remain
        after the earlier components have been removed, so a decomposition of
        that score has to start from the same deflated data.
        """
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks: {sorted(missing)}.")
        a_max = int(self.n_components)
        if not (1 <= int(component) <= a_max):
            msg = f"component must be a 1-based index within 1..{a_max}, got {component}."
            raise ValueError(msg)

        sample_index: pd.Index | None = None
        x_def: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            if sample_index is None:
                sample_index = block.index
            x_def[name] = self.preproc_[name].transform(block).values.astype(float)

        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}
        for a in range(int(component) - 1):
            w_s = self.super_weights_.values[:, a]
            weights = {name: self.block_weights_[name].values[:, a] for name in self.block_names_}
            t_super = np.nan_to_num(
                _pooled_super_score(
                    x_def, _stacked_super_weights(weights, w_s, sqrt_kb, self.block_names_), w_s, self.block_names_
                )
            )
            for name in self.block_names_:
                p_b = self.block_loadings_[name].values[:, a]
                x_def[name] = x_def[name] - np.outer(t_super, p_b)

        assert sample_index is not None
        return x_def, sample_index

    def score_contributions(
        self,
        X: dict[str, pd.DataFrame],
        component: int = 1,
        scaling: str = "none",
    ) -> dict[str, pd.DataFrame]:
        r"""Per-block per-variable contributions to a super-score.

        The multi-block analogue of :meth:`PLS.score_contributions`. A super
        score is a weighted sum of the (deflated, preprocessed) variables across
        every block, so it splits exactly into one term per variable:

        .. math::

            c_{b,ij}^{(a)} = \tilde{x}_{b,ij}^{(a)}\,
                \frac{w_b[j, a]\, w_\mathrm{super}[b, a]}{\sqrt{K_b}},
            \qquad
            \sum_b \sum_j c_{b,ij}^{(a)} = t_{\mathrm{super},ia},

        where :math:`\tilde{x}^{(a)}` is the block data deflated through the
        first :math:`a-1` components, which is what the super score at component
        :math:`a` is actually formed from.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            Raw (un-preprocessed) X-blocks, keyed by block name, exactly as
            passed to :meth:`fit`. The stored per-block preprocessing is applied
            internally.
        component : int, default=1
            **1-based** component index whose super score is decomposed.
        scaling : {"none", "maximum", "within"}, default="none"
            Presentation scaling, as for :meth:`PLS.score_contributions`. Under
            ``"none"`` the contributions sum across all blocks to the super
            score. ``"maximum"`` divides by the largest absolute contribution
            over every block; ``"within"`` divides each observation by the total
            absolute contribution it accumulates across every block, so both
            scalings are taken over the blocks jointly rather than one block at
            a time.

        Returns
        -------
        dict[str, pd.DataFrame]
            One frame per X-block, of shape (n_samples, K_b).

        Examples
        --------
        >>> mbpls = MBPLS(n_components=2).fit(blocks, Y)
        >>> contrib = mbpls.score_contributions(blocks, component=1)
        >>> sum(frame.sum(axis=1) for frame in contrib.values())  # super score 1
        """
        check_is_fitted(self, "block_weights_")
        deflated, sample_index = self._deflated_blocks(X, component)
        a = int(component) - 1

        raw: dict[str, np.ndarray] = {}
        for b_idx, name in enumerate(self.block_names_):
            sqrt_kb = float(np.sqrt(self.block_widths_[name]))
            w_b = self.block_weights_[name].values[:, a]
            w_s = float(self.super_weights_.values[b_idx, a])
            raw[name] = deflated[name] * (w_b * w_s / sqrt_kb)

        raw = _scale_block_contributions(raw, scaling)
        return {
            name: pd.DataFrame(values, index=sample_index, columns=self._block_columns[name])
            for name, values in raw.items()
        }

    def group_contributions(
        self,
        X: dict[str, pd.DataFrame],
        group: Sequence,
        reference: Sequence | None = None,
        component: int = 1,
    ) -> dict[str, pd.Series]:
        """Per-block per-variable contributions to a group's average super score.

        The multi-block analogue of :meth:`PLS.group_contributions`. See that
        method for the definition; the only difference is that the result is
        returned one Series per X-block, and the sum over every block equals the
        group's average super score (or the difference between the two groups'
        average super scores when ``reference`` is given).
        """
        per_block = self.score_contributions(X, component=component)
        out: dict[str, pd.Series] = {}
        for name, frame in per_block.items():
            deviation = _select_rows(frame, group, "group").mean(axis=0)
            if reference is not None:
                deviation = deviation - _select_rows(frame, reference, "reference").mean(axis=0)
            out[name] = pd.Series(deviation, name=f"group_contributions[{name}]")
        return out

    def super_score_plot(self, pc_horiz: int = 1, pc_vert: int = 2) -> go.Figure:
        """Scatter plot of super-scores for two components."""
        check_is_fitted(self, "super_scores_")
        a_max = int(self.n_components)
        if not (1 <= pc_horiz <= a_max and 1 <= pc_vert <= a_max):
            raise ValueError(f"pc_horiz and pc_vert must be in 1..{a_max}.")
        x = self.super_scores_[pc_horiz].to_numpy()
        y = self.super_scores_[pc_vert].to_numpy()
        labels = [str(i) for i in self.super_scores_.index]
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    name="Super-scores",
                )
            ]
        )
        fig.update_layout(
            xaxis_title=f"t_super[{pc_horiz}]",
            yaxis_title=f"t_super[{pc_vert}]",
            title=f"MBPLS super-score plot: PC{pc_horiz} vs PC{pc_vert}",
        )
        return fig

    def super_weights_bar_plot(self, component: int = 1) -> go.Figure:
        """Bar plot of super-weights ``w_super`` for a single component."""
        check_is_fitted(self, "super_weights_")
        a_max = int(self.n_components)
        if not (1 <= component <= a_max):
            raise ValueError(f"component must be in 1..{a_max}.")
        weights = self.super_weights_[component]
        fig = go.Figure(data=[go.Bar(x=list(weights.index), y=weights.to_numpy(), name=f"w_super[{component}]")])
        fig.update_layout(
            xaxis_title="Block",
            yaxis_title=f"w_super[{component}]",
            title=f"MBPLS super-weights, component {component}",
        )
        return fig

    def predictions_vs_observed_plot(self, y_observed: pd.DataFrame, variable: str | None = None) -> go.Figure:
        """Scatter plot of predicted vs observed Y, with y=x reference and RMSEE annotation.

        Parameters
        ----------
        y_observed : pd.DataFrame
            The observed Y on the original scale, same columns as the training Y.
        variable : str or None, default=None
            If given, plot only that Y-variable. If ``None``, plot the first one.
        """
        check_is_fitted(self, "predictions_")
        if variable is None:
            variable = str(self.predictions_.columns[0])
        if variable not in self.predictions_.columns:
            raise ValueError(f"Unknown Y-variable '{variable}'. Known: {list(self.predictions_.columns)}.")
        observed = pd.Series(y_observed[variable].values, name="observed").reset_index(drop=True)
        predicted = pd.Series(self.predictions_[variable].values, name="predicted").reset_index(drop=True)
        rmsee = float(np.sqrt(np.mean((observed.to_numpy() - predicted.to_numpy()) ** 2)))
        lo = float(min(observed.min(), predicted.min()))
        hi = float(max(observed.max(), predicted.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        fig = go.Figure(
            data=[
                go.Scatter(x=observed, y=predicted, mode="markers", name="Predicted vs observed"),
                go.Scatter(
                    x=[lo - pad, hi + pad],
                    y=[lo - pad, hi + pad],
                    mode="lines",
                    line={"color": REFERENCE_LINE_COLOR, "dash": "dash"},
                    name="y = x",
                ),
            ]
        )
        fig.add_annotation(
            x=lo + 0.05 * (hi - lo),
            y=hi - 0.05 * (hi - lo),
            text=f"RMSEE = {rmsee:.4g}",
            showarrow=False,
        )
        fig.update_layout(
            xaxis_title=f"Observed: {variable}",
            yaxis_title=f"Predicted: {variable}",
            title=f"Predicted vs observed for {variable}",
        )
        return fig

    def display_results(self, show_cumulative: bool = True) -> str:
        """Format a short text summary of per-block R²X, overall R²Y, iterations and timing."""
        check_is_fitted(self, "super_scores_")
        rows: list[str] = []
        rows.append(f"MBPLS model: {self.n_components} component(s), {len(self.block_names_)} X-block(s)")
        header = "  PC | " + " | ".join(f"R²X[{name}]" for name in self.block_names_) + " | R²Y"
        rows.append(header)
        rows.append("-" * len(header))
        for a in range(self.n_components):
            cells = [f"{i:>3d}" for i in [a + 1]]
            for name in self.block_names_:
                src = self.r2_x_per_block_cumulative_ if show_cumulative else self.r2_x_per_block_per_component_
                cells.append(f"{src.loc[name].iloc[a]:>9.4f}")
            r2y_src = self.r2_y_cumulative_ if show_cumulative else self.r2_y_per_component_
            cells.append(f"{r2y_src.iloc[a]:>9.4f}")
            rows.append(" | ".join(cells))
        rows.append("")
        rows.append(f"  Iterations per PC: {list(self.fitting_info_['iterations'])}")
        rows.append(f"  Time per PC (ms):  {[round(float(t * 1000), 1) for t in self.fitting_info_['timing']]}")
        return "\n".join(rows)

    def transform(self, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Project new data to super-scores using the fitted model."""
        check_is_fitted(self, "super_weights_")
        return self._project(X).super_scores

    def diagnose(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data and return the full diagnostics Bunch.

        Returns a :class:`sklearn.utils.Bunch` with fields ``super_scores``
        (DataFrame, n_samples x n_components), ``block_scores`` (dict[str,
        DataFrame]), ``predictions`` (DataFrame on original Y scale),
        ``block_spe`` (dict[str, Series], per-block SPE of the new
        observations) and ``hotellings_t2`` (Series of cumulative
        Hotelling's T² over all components, per new observation).

        The rename (since 1.38.4, #395) matches :meth:`PLS.diagnose`
        and PCA.diagnose; :meth:`predict` is kept as a deprecation shim.
        """
        check_is_fitted(self, "super_weights_")
        return self._project(X)

    @classmethod
    def select_n_components(  # noqa: PLR0913, PLR0915
        cls,
        X: dict[str, pd.DataFrame],
        y: pd.DataFrame,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 5,
        n_repeats: int | None = None,
        random_state: int | None = None,
        selection_rule: SelectionRule = "1se",
        **mbpls_kwargs,
    ) -> Bunch:
        """Select the number of multi-block PLS components by cross-validation.

        Whole rows are held out. The super score of a held-out row is computed
        from its X-blocks alone and its Y is what the model predicts, so the
        value being predicted never enters its own prediction. That is the same
        argument that makes row-wise cross-validation sound for a single-block
        :meth:`PLS.select_n_components`, and it is unaffected by there being
        several X-blocks.

        Each block is centred and scaled inside :meth:`fit`, on the training
        rows only, so the fold statistics never see the held-out rows.

        One model is fitted per fold **and** per component count, because the
        hierarchical NIPALS deflation means an ``a``-component model is not
        recoverable from an ``A``-component one. The cost is
        ``cv * n_repeats * max_components`` fits.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            X-blocks, keyed by block name, all sharing ``y``'s row index.
        y : pd.DataFrame
            Y-block, one row per observation.
        max_components : int, optional
            Largest component count to evaluate. Defaults to the largest the
            smallest training fold supports, capped at the total width of the
            X-blocks.
        cv : int or sklearn CV splitter, default 5
            An integer is used as the ``n_splits`` of a shuffled
            :class:`~sklearn.model_selection.KFold`, or of a
            :class:`~sklearn.model_selection.RepeatedKFold` when
            ``n_repeats > 1``. A splitter object is used as given, and
            ``n_repeats`` is then ignored.
        n_repeats : int, optional
            How many times to repeat the split with a fresh shuffle. Resolved
            to 10 when ``cv`` is an integer; pass 1 to disable repeats.
        random_state : int, optional
            Seed for the shuffling. Ignored when ``cv`` is a splitter.
        selection_rule : {"1se", "min", "q2_increment"}, default "1se"
            How ``n_components`` is chosen from the curve. See
            :data:`~process_improve.multivariate._common.SelectionRule`.
            ``"randomization"`` is not offered here.
        **mbpls_kwargs
            Passed to every :class:`MBPLS` fitted, for instance ``tol`` or
            ``algorithm``.

        Returns
        -------
        sklearn.utils.Bunch
            With ``n_components`` (int), ``rmsecv`` and ``se_rmsecv`` (Series
            indexed ``1..A``), ``per_fold_rmsecv`` (DataFrame, components by
            fold), ``press`` (Series), ``r2y_validated`` (DataFrame with one
            column per target, plus ``"total"`` on the original Y scale and
            ``"scaled_total"`` with every target weighted equally),
            ``cv_predictions`` (DataFrame
            of the held-out predictions of the recommended model, averaged
            over repeats) and ``selection_rule``.

        Raises
        ------
        ValueError
            If ``X`` is not a non-empty dict of frames sharing ``y``'s index,
            or if no component count could be evaluated.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> from process_improve.multivariate.methods import MBPLS
        >>> rng = np.random.default_rng(0)
        >>> t = rng.standard_normal((40, 2))
        >>> blocks = {
        ...     "a": pd.DataFrame(t @ rng.standard_normal((2, 5)) + rng.standard_normal((40, 5)) * 0.3),
        ...     "b": pd.DataFrame(t @ rng.standard_normal((2, 4)) + rng.standard_normal((40, 4)) * 0.3),
        ... }
        >>> Y = pd.DataFrame(t @ rng.standard_normal((2, 2)) + rng.standard_normal((40, 2)) * 0.3)
        >>> out = MBPLS.select_n_components(blocks, Y, max_components=3, cv=5, n_repeats=2, random_state=0)
        >>> 1 <= out.n_components <= 3
        True
        """
        if not isinstance(X, dict):
            raise TypeError(f"X must be a dict of DataFrames, one per block; got {type(X).__name__}.")
        if not X:
            raise ValueError("X must hold at least one block.")
        y = pd.DataFrame(y)
        for name, block in X.items():
            if not isinstance(block, pd.DataFrame):
                raise TypeError(f"Block {name!r} must be a pandas DataFrame; got {type(block).__name__}.")
            if len(block) != len(y):
                raise ValueError(f"Block {name!r} has {len(block)} rows; y has {len(y)}.")
        n_samples = len(y)
        total_width = sum(block.shape[1] for block in X.values())

        if isinstance(cv, int):
            repeats = 10 if n_repeats is None else int(n_repeats)
            splitter: BaseCrossValidator = (
                KFold(cv, shuffle=True, random_state=random_state)
                if repeats == 1
                else RepeatedKFold(n_splits=cv, n_repeats=repeats, random_state=random_state)
            )
            n_splits = cv
        else:
            splitter, repeats, n_splits = cv, 1, cv.get_n_splits(y)

        if max_components is not None and int(max_components) < 1:
            raise ValueError(f"max_components must be at least 1; got {max_components}.")

        splits = list(splitter.split(y))
        smallest_train = min(len(train) for train, _ in splits)
        # At least one component is always evaluated: a fold too small to support more
        # still supports one, and reporting nothing would hide that from the caller.
        ceiling = max(1, min(smallest_train - 1, total_width))
        A = ceiling if max_components is None else min(int(max_components), ceiling)

        component_index = pd.Index(range(1, A + 1), name="n_components")
        targets = list(y.columns)
        press = np.zeros((A, len(targets)))
        per_fold = np.full((A, len(splits)), np.nan)
        gathered: dict[int, list[pd.DataFrame]] = {a: [] for a in component_index}
        tested = np.zeros(n_samples)  # how often each row is held out, which need not be uniform

        for fold, (train, test) in enumerate(splits):
            tested[test] += 1.0
            tr, te = y.index[train], y.index[test]
            train_blocks = {name: block.iloc[train] for name, block in X.items()}
            test_blocks = {name: block.iloc[test] for name, block in X.items()}
            for a in component_index:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SpecificationWarning)
                    model = cls(n_components=a, **mbpls_kwargs).fit(train_blocks, y.loc[tr])
                    predicted = model.diagnose(test_blocks).predictions
                residual = y.loc[te].to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
                press[a - 1] += np.nansum(residual**2, axis=0)
                per_fold[a - 1, fold] = float(np.sqrt(np.nanmean(residual**2)))
                gathered[a].append(pd.DataFrame(predicted, index=te, columns=targets))

        # Weight the "predict the mean" reference by the same per-row coverage that
        # built PRESS, so each row counts in the denominator exactly as often as it
        # counted in the numerator. A splitter that tests some rows more than others,
        # or not at all, is then handled exactly rather than by a flat repeat count.
        # This is what PLS.select_n_components does, and both sides are on the
        # original Y scale, so the two are directly comparable.
        values = y.to_numpy(dtype=float)
        centred_sq = (values - np.nanmean(values, axis=0)) ** 2
        tss = np.nansum(tested[:, None] * centred_sq, axis=0)
        per_target = np.where(tss > 0, 1.0 - press / np.where(tss > 0, tss, 1.0), np.nan)
        total = np.where(tss.sum() > 0, 1.0 - press.sum(axis=1) / tss.sum(), np.nan)

        n_predicted = float(tested.sum())
        rmsecv = pd.Series(
            np.sqrt(press.sum(axis=1) / max(n_predicted * len(targets), 1.0)), index=component_index, name="RMSECV"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            se = np.nanstd(per_fold, axis=1, ddof=1) / np.sqrt(np.maximum(1, np.sum(~np.isnan(per_fold), axis=1)))
        se_rmsecv = pd.Series(se, index=component_index, name="SE of RMSECV")
        r2y_validated = pd.DataFrame(
            np.column_stack([per_target, total, _equal_weight_r2_total(per_target)]),
            index=component_index,
            columns=[*targets, "total", "scaled_total"],
        )

        recommended = _select_n_components(
            selection_rule,
            mean_error=rmsecv.to_numpy(),
            se_error=se_rmsecv.to_numpy(),
            q2_cumulative=r2y_validated["total"].to_numpy(),
        )
        held_out = pd.concat(gathered[recommended]).groupby(level=0).mean().reindex(y.index)

        return Bunch(
            n_components=int(recommended),
            rmsecv=rmsecv,
            se_rmsecv=se_rmsecv,
            per_fold_rmsecv=pd.DataFrame(
                per_fold, index=component_index, columns=[f"fold_{i + 1}" for i in range(len(splits))]
            ),
            press=pd.Series(press.sum(axis=1), index=component_index, name="PRESS"),
            r2y_validated=r2y_validated,
            cv_predictions=held_out,
            selection_rule=selection_rule,
            n_splits=n_splits,
        )

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Forward to :meth:`diagnose`; emits a :class:`DeprecationWarning`.

        .. deprecated:: 1.38.4
            Use :meth:`MBPLS.diagnose` instead. The sklearn-convention
            ``predict`` name suggests a regression-style ndarray return,
            but the historical return is the rich diagnostics Bunch. The
            rename aligns with :meth:`PLS.diagnose` and :meth:`PCA.diagnose`
            and frees the name for a future contract that returns just
            the ``predictions`` field. Will be removed in 2.0.0.
        """
        warnings.warn(
            "MBPLS.predict is deprecated and will be removed in 2.0.0; use MBPLS.diagnose instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.diagnose(X)

    def _project(self, X: dict[str, pd.DataFrame]) -> Bunch:
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks for prediction: {sorted(missing)}.")

        # Preprocess each block
        x_pp: dict[str, np.ndarray] = {}
        sample_index: pd.Index | None = None
        for name in self.block_names_:
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            if block.shape[1] != self.block_widths_[name]:
                raise ValueError(f"Block '{name}' must have {self.block_widths_[name]} columns; got {block.shape[1]}.")
            x_pp[name] = self.preproc_[name].transform(block).values.astype(float)
            if sample_index is None:
                sample_index = block.index

        n_components = int(self.n_components)
        n_new = next(iter(x_pp.values())).shape[0]
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        super_scores = np.zeros((n_new, n_components))
        block_scores: dict[str, np.ndarray] = {name: np.zeros((n_new, n_components)) for name in self.block_names_}

        x_def = {name: x_pp[name].copy() for name in self.block_names_}
        block_has_data = {name: _rows_with_data(x_pp[name]) for name in self.block_names_}
        for a in range(n_components):
            weights = {}
            for name in self.block_names_:
                w_b = self.block_weights_[name].values[:, a]
                weights[name] = w_b
                t_b = quick_regress(x_def[name], w_b.reshape(-1, 1)).flatten() / sqrt_kb[name]
                block_scores[name][:, a] = np.where(block_has_data[name], t_b, np.nan)
            w_s = self.super_weights_.values[:, a]
            # The same pooled regression the fit uses, so a projected score matches a fitted one.
            t_super = _pooled_super_score(
                x_def, _stacked_super_weights(weights, w_s, sqrt_kb, self.block_names_), w_s, self.block_names_
            )
            super_scores[:, a] = t_super
            t_super_finite = np.nan_to_num(t_super)
            for name in self.block_names_:
                p_b = self.block_loadings_[name].values[:, a]
                x_def[name] = x_def[name] - np.outer(t_super_finite, p_b)

        component_names = list(range(1, n_components + 1))
        super_scores_df = pd.DataFrame(super_scores, index=sample_index, columns=component_names)
        block_scores_df = {
            name: pd.DataFrame(block_scores[name], index=sample_index, columns=component_names)
            for name in self.block_names_
        }
        y_hat_pp = super_scores @ self.super_y_loadings_.values.T
        predictions = self.y_preproc_.inverse_transform(pd.DataFrame(y_hat_pp, columns=self._y_columns))
        assert sample_index is not None  # block_names_ is non-empty, so it was set in the loop
        predictions.index = sample_index

        # Per-block SPE for new observations (residual after final deflation)
        block_spe = {
            name: pd.Series(np.sqrt(np.nansum(x_def[name] ** 2, axis=1)), index=sample_index, name=f"SPE[{name}]")
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        hotellings_t2 = pd.Series(
            np.sum((super_scores**2) / super_score_var, axis=1), index=sample_index, name="Hotelling's T²"
        )

        return Bunch(
            super_scores=super_scores_df,
            block_scores=block_scores_df,
            predictions=predictions,
            block_spe=block_spe,
            hotellings_t2=hotellings_t2,
        )


def randomization_test_mbpls(
    model: MBPLS,
    X: dict[str, pd.DataFrame],
    y: pd.DataFrame,
    n_permutations: int = 200,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    r"""Randomization (permutation) test for component significance in MBPLS.

    For each component ``a``, the null hypothesis is "there is no real
    relationship between X and Y at this component"; the test permutes the
    rows of ``y``, refits a fresh MBPLS with the same number of components,
    and recomputes the test statistic. The risk is the fraction of
    permutations whose statistic equals or exceeds the original model's.

    Statistic: per-component absolute correlation between the super X-score
    and the super Y-score, ``|t_super(:,a)' u_super(:,a)| / (||t|| * ||u||)``.

    Parameters
    ----------
    model : MBPLS
        A fitted MBPLS model.
    X, y : dict[str, DataFrame], DataFrame
        The same training data used to fit ``model``.
    n_permutations : int, default=200
        Number of Y-row permutations to evaluate.
    seed : int or None, default=None
        Seed for the permutation RNG (``None`` uses non-reproducible
        randomness).

    Returns
    -------
    pd.DataFrame
        Indexed by component ``1..A`` with columns:

        - ``observed`` : the actual model's per-component statistic.
        - ``risk_pct`` : Monte-Carlo estimate (in %) of the right-tail
          probability, ``100 * (n_exceed + 1) / (n_permutations + 1)``. Low
          values (e.g. < 5%) suggest the component is significant; values near
          50% suggest the component is no better than chance.

          The ``+ 1`` on each side counts the observed statistic among the
          permutations, which is what keeps the estimate a valid p-value: the
          uncorrected ``n_exceed / n_permutations`` can report exactly 0, and no
          finite permutation set can license the claim that the true tail
          probability is zero. The floor is ``100 / (n_permutations + 1)``, so
          the default 999 permutations cannot resolve below 0.1%. This matches
          the convention already used by the Van der Voet test in
          :mod:`~process_improve.multivariate._pls` (#513).

    References
    ----------
    Wiklund, S., Nilsson, D., Eriksson, L., Sjöström, M., Wold, S. &
    Faber, K. *A randomization test for PLS component selection.* J.
    Chemometrics, 21 (2007), 427-439.
    """
    check_is_fitted(model, "super_scores_")
    rng = np.random.default_rng(seed)
    a_components = int(model.n_components)

    def _objective(mod: MBPLS) -> np.ndarray:
        t = mod.super_scores_.values
        u = mod.super_y_scores_.values
        out = np.zeros(t.shape[1])
        for a in range(t.shape[1]):
            num = float(np.abs(t[:, a] @ u[:, a]))
            denom = float(np.linalg.norm(t[:, a]) * np.linalg.norm(u[:, a]))
            # SEC-33 (#282): float ``==`` zero only catches the exact-zero
            # case; a sub-eps near-zero denom produced a meaningless ratio
            # that the permutation test treated as an observed statistic.
            out[a] = 0.0 if denom <= epsqrt else num / denom
        return out

    observed = _objective(model)
    n_exceed = np.zeros(a_components, dtype=int)
    n_samples = y.shape[0]
    for _ in range(int(n_permutations)):
        perm_idx = rng.permutation(n_samples)
        y_perm = y.iloc[perm_idx].reset_index(drop=True)
        # Reset X indices to align row positions (otherwise pandas will
        # join on index and silently misalign).
        x_reset = {name: X[name].reset_index(drop=True) for name in X}
        permuted_model = MBPLS(n_components=a_components).fit(x_reset, y_perm)
        stat = _objective(permuted_model)
        n_exceed += (stat >= observed).astype(int)

    component_names = list(range(1, a_components + 1))
    # See the `risk_pct` note in the docstring for why both sides carry the + 1.
    risk_pct = 100.0 * (n_exceed + 1) / (n_permutations + 1)
    return pd.DataFrame(
        {"observed": observed, "risk_pct": risk_pct},
        index=pd.Index(component_names, name="component"),
    )
