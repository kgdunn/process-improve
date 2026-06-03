# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Multi-block PLS (MBPLS) estimator and its randomization test (ENG-01).

Holds :class:`MBPLS`, the hierarchical / superblock multi-block PLS regressor,
and :func:`randomization_test_mbpls`, the permutation test for the significance
of each fitted component.
"""

from __future__ import annotations

import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ..visualization.themes import REFERENCE_LINE_COLOR
from ._base import _HotellingsT2LimitMixin
from ._common import SpecificationWarning, _nz, epsqrt
from ._limits import spe_calculation
from ._nipals import quick_regress, ssq
from ._preprocessing import MCUVScaler

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]


logger = logging.getLogger(__name__)


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
        Convergence tolerance on the change in the Y-block score. If
        ``None``, ``np.finfo(float).eps ** (6/7)`` is used (matching the
        legacy multi-block reference implementation).
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
    super_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block (consensus) X-scores ``T``.
    super_y_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block Y-scores ``U``.
    super_weights_ : pd.DataFrame, shape (n_blocks, n_components)
        Super-block weights ``w_super``; rows indexed by block name.
    super_y_loadings_ : pd.DataFrame, shape (n_targets, n_components)
        Y-block loadings ``c``.
    block_scores_ : dict[str, pd.DataFrame]
        Per-block X-scores ``t_b``, each shape ``(n_samples, n_components)``.
    block_weights_ : dict[str, pd.DataFrame]
        Per-block X-weights ``w_b``, each shape ``(K_b, n_components)``.
        Each column has unit norm.
    block_loadings_ : dict[str, pd.DataFrame]
        Per-block X-loadings ``p_b`` (used for deflation), each shape
        ``(K_b, n_components)``.
    predictions_ : pd.DataFrame, shape (n_samples, n_targets)
        In-sample Y predictions on the *original* scale.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variance of the super-score per component (ddof=1).
    scaling_factor_for_super_scores_ : pd.Series
        ``sqrt(explained_variance_)`` per component.
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
    def fit(self, X: dict[str, pd.DataFrame], y: pd.DataFrame) -> MBPLS:  # noqa: C901, PLR0912, PLR0915
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
        if not isinstance(X, dict) or len(X) == 0:
            raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
        for name, block in X.items():
            if not isinstance(block, pd.DataFrame):
                raise TypeError(f"X['{name}'] must be a pandas DataFrame; got {type(block).__name__}.")

        self.block_names_: list[str] = list(X.keys())
        first = X[self.block_names_[0]]
        n_samples = first.shape[0]
        for name in self.block_names_:
            if X[name].shape[0] != n_samples:
                raise ValueError(
                    f"All X-blocks must have the same row count. Block '{name}' has "
                    f"{X[name].shape[0]} rows; expected {n_samples}."
                )
        if y.shape[0] != n_samples:
            raise ValueError(f"y has {y.shape[0]} rows; expected {n_samples} to match X-blocks.")

        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._y_columns = y.columns
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}

        self.n_samples_ = int(n_samples)
        self.n_targets_ = int(y.shape[1])
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        n_components = int(self.n_components)
        n_blocks = len(self.block_names_)

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

        # Resolve iterative-algorithm settings (used by the 'nipals' path).
        settings = {"md_tol": epsqrt, "md_max_iter": 1000}
        if isinstance(self.missing_data_settings, dict):
            settings.update(self.missing_data_settings)
        settings["md_max_iter"] = int(settings["md_max_iter"])
        if algo == "nipals":
            if not settings["md_tol"] < 10:
                raise ValueError("Tolerance should not be too large.")
            if not settings["md_tol"] > epsqrt**1.95:
                raise ValueError("Tolerance must exceed machine precision.")
            # Degeneracy guards: any column or any (block, row) entirely NaN
            # leaves the masked NIPALS denominator at zero, which would
            # silently produce a spurious score or loading. Refuse the fit
            # rather than coerce the user into a misleading result.
            for name in self.block_names_:
                values = X[name].values
                col_all_nan = np.all(np.isnan(values), axis=0)
                if np.any(col_all_nan):
                    bad = X[name].columns[col_all_nan].tolist()
                    raise ValueError(
                        f"Block '{name}' has columns with all values missing: {bad}. "
                        "Drop these columns before fitting."
                    )
                row_all_nan = np.all(np.isnan(values), axis=1)
                if np.any(row_all_nan):
                    bad_rows = np.where(row_all_nan)[0].tolist()
                    raise ValueError(
                        f"Block '{name}' has rows with all values missing at positions {bad_rows}. "
                        "Drop these observations or impute them before fitting."
                    )
            y_values = y.values
            y_col_all_nan = np.all(np.isnan(y_values), axis=0)
            if np.any(y_col_all_nan):
                bad = y.columns[y_col_all_nan].tolist()
                raise ValueError(
                    f"Y has columns with all values missing: {bad}. Drop these targets before fitting."
                )
            y_row_all_nan = np.all(np.isnan(y_values), axis=1)
            if np.any(y_row_all_nan):
                bad_rows = np.where(y_row_all_nan)[0].tolist()
                raise ValueError(
                    f"Y has rows with all values missing at positions {bad_rows}. "
                    "Drop these observations or impute them before fitting."
                )

        # Preprocess each X-block and Y independently
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        self.y_preproc_ = MCUVScaler().fit(y)
        x_blocks_pp: dict[str, np.ndarray] = {
            name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_
        }
        y_pp = self.y_preproc_.transform(y).values.astype(float)

        # Algorithmic block weighting: X_b / sqrt(K_b)
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        # Storage (numpy arrays during fit; wrapped in pandas at the end)
        super_scores_np = np.zeros((n_samples, n_components))
        super_y_scores_np = np.zeros((n_samples, n_components))
        super_weights_np = np.zeros((n_blocks, n_components))
        super_y_loadings_np = np.zeros((self.n_targets_, n_components))
        block_scores_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }
        block_weights_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        block_loadings_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }

        x_def: dict[str, np.ndarray] = {name: x_blocks_pp[name].copy() for name in self.block_names_}
        y_def = y_pp.copy()

        # Initial sums of squares (for R^2 bookkeeping)
        ssq_x_init = {name: float(np.nansum(x_blocks_pp[name] ** 2)) for name in self.block_names_}
        ssq_y_init = float(np.nansum(y_pp ** 2))
        ssq_x_init_per_var = {
            name: np.nansum(x_blocks_pp[name] ** 2, axis=0) for name in self.block_names_
        }
        ssq_y_init_per_var = np.nansum(y_pp ** 2, axis=0)

        # Per-component cumulative R^2 storage (filled inside the loop)
        r2_x_block_cum = np.zeros((n_blocks, n_components))
        r2_x_var_cum: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        r2_y_cum = np.zeros(n_components)
        r2_y_var_cum = np.zeros((self.n_targets_, n_components))
        block_spe_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }

        tol = float(np.finfo(float).eps ** (6 / 7)) if self.tol is None else float(self.tol)
        timing = np.zeros(n_components)
        iterations = np.zeros(n_components, dtype=int)
        rng = np.random.default_rng(0)

        for a in range(n_components):
            start = time.time()
            u_a = rng.standard_normal(n_samples)
            prev = u_a * 2
            local_w: dict[str, np.ndarray] = {}
            local_t: dict[str, np.ndarray] = {}
            t_b_summary = np.zeros((n_samples, n_blocks))
            t_super = np.zeros(n_samples)
            w_s = np.zeros(n_blocks)
            c_a = np.zeros(self.n_targets_)
            itern = 0
            while np.linalg.norm(prev - u_a) > tol and itern < self.max_iter:
                prev = u_a
                if algo == "nipals":
                    # Mask-aware NIPALS: each projection is a per-column (or
                    # per-row) regression that uses only the entries that
                    # are not NaN, and divides by the masked sum of squares.
                    # Reuses the same primitives as single-block PCA NIPALS.
                    u_a_col = u_a.reshape(-1, 1)
                    for b_idx, name in enumerate(self.block_names_):
                        w_b = quick_regress(x_def[name], u_a_col).flatten()
                        w_b = w_b / _nz(float(np.sqrt(ssq(w_b.reshape(-1, 1)))))
                        t_b = quick_regress(x_def[name], w_b.reshape(-1, 1)).flatten() / sqrt_kb[name]
                        local_w[name] = w_b
                        local_t[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                else:
                    for b_idx, name in enumerate(self.block_names_):
                        w_b = x_def[name].T @ u_a / _nz(float(u_a @ u_a))
                        w_b = w_b / _nz(float(np.linalg.norm(w_b)))
                        t_b = x_def[name] @ w_b / _nz(float(w_b @ w_b)) / sqrt_kb[name]
                        local_w[name] = w_b
                        local_t[name] = t_b
                        t_b_summary[:, b_idx] = t_b

                w_s = t_b_summary.T @ u_a / _nz(float(u_a @ u_a))
                w_s = w_s / _nz(float(np.linalg.norm(w_s)))
                t_super = t_b_summary @ w_s / _nz(float(w_s @ w_s))
                if algo == "nipals":
                    t_super_col = t_super.reshape(-1, 1)
                    c_a = quick_regress(y_def, t_super_col).flatten()
                    u_a = quick_regress(y_def, c_a.reshape(-1, 1)).flatten()
                else:
                    c_a = y_def.T @ t_super / _nz(float(t_super @ t_super))
                    u_a = y_def @ c_a / _nz(float(c_a @ c_a))
                itern += 1

            # Sign convention: largest |w_super| element positive
            flip_idx = int(np.argmax(np.abs(w_s)))
            if w_s[flip_idx] < 0:
                w_s = -w_s
                t_super = -t_super
                u_a = -u_a
                c_a = -c_a
                for name in self.block_names_:
                    local_w[name] = -local_w[name]
                    local_t[name] = -local_t[name]

            # Deflate using the super-score
            t_super_col = t_super.reshape(-1, 1)
            for name in self.block_names_:
                if algo == "nipals":
                    p_b = quick_regress(x_def[name], t_super_col).flatten()
                else:
                    p_b = x_def[name].T @ t_super / _nz(float(t_super @ t_super))
                x_def[name] = x_def[name] - np.outer(t_super, p_b)
                block_loadings_np[name][:, a] = p_b
                block_weights_np[name][:, a] = local_w[name]
                block_scores_np[name][:, a] = local_t[name]
            y_def = y_def - np.outer(t_super, c_a)

            super_scores_np[:, a] = t_super
            super_y_scores_np[:, a] = u_a
            super_weights_np[:, a] = w_s
            super_y_loadings_np[:, a] = c_a

            # Track per-block cumulative R^2_X and per-Y-variable cumulative R^2_Y
            for b_idx, name in enumerate(self.block_names_):
                ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
                # R^2 is undefined for a zero-variance block/column; report NaN
                # rather than dividing by zero (inf/nan + warning) or returning a
                # misleading 1.0.
                r2_x_block_cum[b_idx, a] = (
                    1 - np.sum(ssq_remain_per_var) / ssq_x_init[name] if ssq_x_init[name] > 0 else np.nan
                )
                r2_x_var_cum[name][:, a] = np.where(
                    ssq_x_init_per_var[name] > 0,
                    1 - ssq_remain_per_var / np.where(ssq_x_init_per_var[name] > 0, ssq_x_init_per_var[name], 1.0),
                    np.nan,
                )
                block_spe_np[name][:, a] = np.sqrt(np.nansum(x_def[name] ** 2, axis=1))
            ssq_y_remain_per_var = np.nansum(y_def ** 2, axis=0)
            r2_y_cum[a] = 1 - np.sum(ssq_y_remain_per_var) / ssq_y_init if ssq_y_init > 0 else np.nan
            r2_y_var_cum[:, a] = np.where(
                ssq_y_init_per_var > 0,
                1 - ssq_y_remain_per_var / np.where(ssq_y_init_per_var > 0, ssq_y_init_per_var, 1.0),
                np.nan,
            )

            timing[a] = time.time() - start
            iterations[a] = itern

        component_names = list(range(1, n_components + 1))
        self.super_scores_ = pd.DataFrame(super_scores_np, index=self._sample_index, columns=component_names)
        self.super_y_scores_ = pd.DataFrame(super_y_scores_np, index=self._sample_index, columns=component_names)
        self.super_weights_ = pd.DataFrame(super_weights_np, index=self.block_names_, columns=component_names)
        self.super_y_loadings_ = pd.DataFrame(super_y_loadings_np, index=self._y_columns, columns=component_names)

        self.block_scores_ = {
            name: pd.DataFrame(block_scores_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_weights_ = {
            name: pd.DataFrame(
                block_weights_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(
                block_loadings_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }

        # In-sample predictions on the original Y scale
        y_hat_pp = super_scores_np @ super_y_loadings_np.T
        y_hat = self.y_preproc_.inverse_transform(pd.DataFrame(y_hat_pp, columns=self._y_columns))
        y_hat.index = self._sample_index
        self.predictions_ = y_hat

        self.explained_variance_ = np.diag(super_scores_np.T @ super_scores_np) / max(1, n_samples - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )
        converged = iterations < self.max_iter
        self.fitting_info_ = {"timing": timing, "iterations": iterations, "converged": converged}
        logger.debug("MBPLS (%s): iterations per component = %s", self.algorithm_, list(iterations))
        if not np.all(converged):
            failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
            warnings.warn(
                f"MBPLS NIPALS did not converge within max_iter={self.max_iter} for "
                f"component(s) {failed}; results for those components may be unreliable.",
                SpecificationWarning,
                stacklevel=2,
            )

        # --- Per-component (incremental) R^2 from cumulative ---
        r2_x_block_per_a = np.zeros_like(r2_x_block_cum)
        r2_x_block_per_a[:, 0] = r2_x_block_cum[:, 0]
        if n_components > 1:
            r2_x_block_per_a[:, 1:] = np.diff(r2_x_block_cum, axis=1)
        r2_y_per_a = np.empty_like(r2_y_cum)
        r2_y_per_a[0] = r2_y_cum[0]
        if n_components > 1:
            r2_y_per_a[1:] = np.diff(r2_y_cum)

        self.r2_x_per_block_cumulative_ = pd.DataFrame(
            r2_x_block_cum, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_block_per_component_ = pd.DataFrame(
            r2_x_block_per_a, index=self.block_names_, columns=component_names
        )
        self.r2_x_per_variable_ = {
            name: pd.DataFrame(r2_x_var_cum[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }
        self.r2_y_cumulative_ = pd.Series(r2_y_cum, index=component_names, name="Cumulative R²Y")
        self.r2_y_per_component_ = pd.Series(r2_y_per_a, index=component_names, name="R²Y per component")
        self.r2_y_per_variable_ = pd.DataFrame(r2_y_var_cum, index=self._y_columns, columns=component_names)

        # --- Per-block VIP and super VIP ---
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
        if np.sum(r2_y_per_a) > 0:
            ws = self.super_weights_.values  # (B, A)
            super_vip = np.sqrt(n_blocks * np.sum(r2_y_per_a * ws**2, axis=1) / np.sum(r2_y_per_a))
        else:
            super_vip = np.zeros(n_blocks)
        self.super_vip_ = pd.Series(super_vip, index=self.block_names_, name="Super VIP")

        # --- Per-block SPE (already accumulated) and per-block / super Hotelling's T^2 ---
        self.block_spe_ = {
            name: pd.DataFrame(block_spe_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }

        # Cumulative T^2 from block scores and super scores (using per-component score variance)
        block_t2: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            scores_np = self.block_scores_[name].values  # (N, A)
            score_var = np.var(scores_np, axis=0, ddof=1)
            score_var = np.where(score_var > 0, score_var, 1.0)
            t2 = np.cumsum((scores_np**2) / score_var, axis=1)
            block_t2[name] = t2
        self.block_hotellings_t2_ = {
            name: pd.DataFrame(block_t2[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        super_t2 = np.cumsum((super_scores_np**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

        return self

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

    def score_contributions(
        self,
        t_super_start: np.ndarray | pd.Series,
        t_super_end: np.ndarray | pd.Series | None = None,
        components: list[int] | None = None,
        *,
        weighted: bool = False,
    ) -> dict[str, pd.Series]:
        r"""Per-block per-variable contributions to a super-score movement.

        The multi-block analogue of :meth:`PCA.score_contributions`. Decomposes
        a super-score-space delta back into preprocessed-scale variable-space
        deltas, one per X-block.

        For MBPLS, the super-score at component *a* is built from the per-block
        scores (which themselves are weighted regressions of the block on
        ``w_b``), so the back-projection through ``w_super[b,a] * w_b[:,a] /
        sqrt(K_b)`` gives the variable contribution.

        Parameters
        ----------
        t_super_start : array-like, shape (n_components,)
            Super-score row of the observation of interest. Typically a row
            from ``self.super_scores_`` or from ``predict(X_new).super_scores``.
        t_super_end : array-like, optional
            Reference point in super-score space. Defaults to the model
            centre (zeros).
        components : list of int, optional
            **1-based** component indices to decompose over. ``None`` (default)
            uses all components - appropriate for Hotelling's T² contributions.
        weighted : bool, default=False
            If ``True``, divide the super-score delta by
            ``sqrt(explained_variance_)`` per component before back-projecting,
            giving contributions to the T² statistic instead of the Euclidean
            super-score distance.

        Returns
        -------
        dict[str, pd.Series]
            One Series per X-block (length ``K_b``), indexed by variable
            (column) name.
        """
        check_is_fitted(self, "block_weights_")
        t_start = np.asarray(t_super_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_super_end is None else np.asarray(t_super_end, dtype=float)
        idx = np.arange(self.n_components) if components is None else np.array(components) - 1
        dt = t_end[idx] - t_start[idx]  # (len(idx),)
        if weighted:
            # ``explained_variance_[a] == 0`` for a degenerate component
            # would silently produce inf/NaN weighted contributions.
            # Clamp the divisor so weighting is a no-op on such components.
            # SEC-21 (#270) sub-item 5.
            ev = np.asarray(self.explained_variance_)[idx]
            dt = dt / np.sqrt(np.where(ev > epsqrt, ev, 1.0))

        out: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            sqrt_kb = float(np.sqrt(self.block_widths_[name]))
            ws = self.super_weights_.values[b_idx, idx]  # (len(idx),)
            wb = self.block_weights_[name].values[:, idx]  # (K_b, len(idx))
            # Effective per-component contribution per variable: w_super[b] * w_b[:,a] / sqrt(K_b)
            effective = wb * (ws / sqrt_kb)  # (K_b, len(idx))
            contrib = effective @ dt  # (K_b,)
            out[name] = pd.Series(contrib, index=self._block_columns[name], name=f"score_contributions[{name}]")
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

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data and predict Y on the original scale.

        Returns a :class:`sklearn.utils.Bunch` with fields ``super_scores``,
        ``block_scores`` (dict[str, DataFrame]) and ``predictions`` (DataFrame
        on original Y scale).
        """
        check_is_fitted(self, "super_weights_")
        return self._project(X)

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
                raise ValueError(
                    f"Block '{name}' must have {self.block_widths_[name]} columns; got {block.shape[1]}."
                )
            x_pp[name] = self.preproc_[name].transform(block).values.astype(float)
            if sample_index is None:
                sample_index = block.index

        n_components = int(self.n_components)
        n_new = next(iter(x_pp.values())).shape[0]
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        super_scores = np.zeros((n_new, n_components))
        block_scores: dict[str, np.ndarray] = {
            name: np.zeros((n_new, n_components)) for name in self.block_names_
        }

        x_def = {name: x_pp[name].copy() for name in self.block_names_}
        for a in range(n_components):
            t_b_row = np.zeros((n_new, len(self.block_names_)))
            for b_idx, name in enumerate(self.block_names_):
                w_b = self.block_weights_[name].values[:, a]
                t_b = x_def[name] @ w_b / sqrt_kb[name]
                block_scores[name][:, a] = t_b
                t_b_row[:, b_idx] = t_b
            w_s = self.super_weights_.values[:, a]
            t_super = t_b_row @ w_s
            super_scores[:, a] = t_super
            for name in self.block_names_:
                p_b = self.block_loadings_[name].values[:, a]
                x_def[name] = x_def[name] - np.outer(t_super, p_b)

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
            name: pd.Series(
                np.sqrt(np.nansum(x_def[name] ** 2, axis=1)), index=sample_index, name=f"SPE[{name}]"
            )
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
    and the super Y-score, ``|t_super(:,a)' u_super(:,a)| / (||t|| * ||u||)``,
    matching the legacy ConnectMV randomization-objective for PLS.

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
        - ``risk_pct`` : fraction (in %) of permutations with statistic
          ``>= observed``. Low values (e.g. < 5%) suggest the component is
          significant; values near 50% suggest the component is no better
          than chance.

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
    risk_pct = 100.0 * n_exceed / n_permutations
    return pd.DataFrame(
        {"observed": observed, "risk_pct": risk_pct},
        index=pd.Index(component_names, name="component"),
    )
