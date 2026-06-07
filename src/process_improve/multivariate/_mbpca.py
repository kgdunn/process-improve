# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Multi-block PCA (MBPCA) estimator (ENG-01).

Holds :class:`MBPCA`, the hierarchical / superblock multi-block PCA transformer.
"""

from __future__ import annotations

import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

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


class MBPCA(_HotellingsT2LimitMixin, TransformerMixin, BaseEstimator):
    r"""Multi-block PCA (hierarchical / consensus PCA).

    Generic multi-block PCA following the consensus-PCA / hierarchical PCA
    formulation of Westerhuis, Kourti & MacGregor (1998). Each X-block is
    preprocessed independently (mean-centred and unit-variance scaled),
    then divided by ``sqrt(K_b)`` so blocks of unequal width contribute
    fairly to the consensus super-score.

    The hierarchical NIPALS loop alternates: (i) regress each block on the
    super-score to get block loadings and block scores, (ii) collect block
    scores into a super-block, (iii) regress the super-block to get a new
    super-score / super-loading, repeat to convergence. After convergence,
    deflate every block by the super-score and the corresponding block
    loading scaled by the super-loading element.

    Parameters
    ----------
    n_components : int
    max_iter : int, default=500
    tol : float or None, default=None
        Convergence tolerance on the super-score change. ``None`` uses
        ``np.finfo(float).eps ** (9/10)`` (matches the legacy reference).
    algorithm : str, default="auto"
        Algorithm to use for fitting the model.

        - ``"auto"``: dense vectorised hierarchical NIPALS when the data is
          complete; mask-aware NIPALS (NaN-tolerant) when any block contains
          missing values.
        - ``"dense"``: dense vectorised hierarchical NIPALS. Raises if any
          block contains missing values.
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
    block_names_, block_widths_                       (as MBPLS)
    super_scores_                                     DataFrame (N x A)
    super_loadings_                                   DataFrame (B x A)
    block_scores_, block_loadings_                    dict[str, DataFrame]
    r2_x_per_block_cumulative_, r2_x_per_block_per_component_
    r2_x_per_variable_                                dict[str, DataFrame]
    block_vip_, super_vip_
    block_spe_, block_hotellings_t2_, super_hotellings_t2_
    explained_variance_, scaling_factor_for_super_scores_
    fitting_info_, has_missing_data_, algorithm_

    Notes
    -----
    The deflation step is :math:`X_b \leftarrow X_b - t_{\rm super}\,
    (p_b\,p_s[b]\,\sqrt{K_b})^\top`, derived in Westerhuis et al. 1998.
    The legacy MATLAB ``mbpca.m`` had this step marked as broken by the
    original author; this implementation re-derives it directly from the
    paper and is independently validated against the pure-numpy reference
    oracles in the test suite.

    Missing data
    ------------
    When any block contains NaN entries, the ``"auto"`` algorithm
    routes to a mask-aware NIPALS variant. Each per-block projection
    in the inner loop is computed as a regression that uses only the
    observed entries; the masked sum-of-squares is used as the
    denominator so missing values neither bias the loading direction
    nor contribute to the score. The mask is preserved across
    components automatically because deflation propagates NaN through
    subtraction. This is the standard skip-NaN NIPALS update; see
    Walczak & Massart (2001) and Arteaga & Ferrer (2002).

    The fit refuses to run if any block has a column with all entries
    missing, or any block has a row with all entries missing for that
    block; either case leaves the masked denominator at zero. Drop or
    impute such rows or columns before fitting. Predict-time score
    estimation for new observations with NaN (Trimmed Score Regression
    / Projection to the Model Plane) is a separate follow-up.

    References
    ----------
    Westerhuis, J. A., Kourti, T. & MacGregor, J. F. *Analysis of
    multiblock and hierarchical PCA and PLS models.* J. Chemometrics, 12
    (1998), 301-321.

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
    def fit(self, X: dict[str, pd.DataFrame], y: None = None) -> MBPCA:  # noqa: ARG002, C901, PLR0912, PLR0915
        """Fit the multi-block PCA model."""
        if not isinstance(X, dict) or len(X) == 0:
            raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
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

        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}

        self.n_samples_ = int(n_samples)
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        # feature_names_in_: sklearn convention (#392). Flat concatenation of
        # all blocks' column names in block-iteration order. Lets
        # ``Pipeline.get_feature_names_out`` and SHAP / eli5 / model-card
        # tooling introspect a multiblock fit through the same surface as a
        # single-block estimator.
        self.feature_names_in_ = np.concatenate(
            [self._block_columns[name].to_numpy() for name in self.block_names_]
        )
        n_components = int(self.n_components)
        n_blocks = len(self.block_names_)

        self.has_missing_data_ = any(np.any(X[name].isna().values) for name in self.block_names_)
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognised. "
                f"Must be one of {self._valid_algorithms}."
            )
        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "dense"
        if algo == "dense" and self.has_missing_data_:
            raise ValueError(
                "Algorithm 'dense' cannot handle missing data. "
                "Use 'nipals' or 'auto' instead."
            )
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

        # Preprocess each block independently
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        x_blocks_pp: dict[str, np.ndarray] = {
            name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_
        }
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        # Working copies for deflation and stats accumulation
        x_def: dict[str, np.ndarray] = {name: x_blocks_pp[name].copy() for name in self.block_names_}
        ssq_x_init = {name: float(np.nansum(x_blocks_pp[name] ** 2)) for name in self.block_names_}
        ssq_x_init_per_var = {
            name: np.nansum(x_blocks_pp[name] ** 2, axis=0) for name in self.block_names_
        }

        super_scores_np = np.zeros((n_samples, n_components))
        super_loadings_np = np.zeros((n_blocks, n_components))
        block_scores_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }
        block_loadings_np: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        r2_x_block_cum = np.zeros((n_blocks, n_components))
        r2_x_var_cum: dict[str, np.ndarray] = {
            name: np.zeros((self.block_widths_[name], n_components)) for name in self.block_names_
        }
        block_spe_np: dict[str, np.ndarray] = {
            name: np.zeros((n_samples, n_components)) for name in self.block_names_
        }

        tol = float(np.finfo(float).eps ** (9 / 10)) if self.tol is None else float(self.tol)
        timing = np.zeros(n_components)
        iterations = np.zeros(n_components, dtype=int)
        rng = np.random.default_rng(0)

        for a in range(n_components):
            start = time.time()
            t_super = rng.standard_normal(n_samples)
            prev = t_super * 2
            t_b_summary = np.zeros((n_samples, n_blocks))
            local_loadings: dict[str, np.ndarray] = {}
            local_scores: dict[str, np.ndarray] = {}
            p_s = np.zeros(n_blocks)
            itern = 0
            while np.linalg.norm(prev - t_super) > tol and itern < self.max_iter:
                prev = t_super
                if algo == "nipals":
                    # Mask-aware NIPALS: each projection is a per-column (or
                    # per-row) regression that uses only the entries that
                    # are not NaN, and divides by the masked sum of squares.
                    # Reuses the same primitives as single-block PCA NIPALS.
                    t_super_col = t_super.reshape(-1, 1)
                    for b_idx, name in enumerate(self.block_names_):
                        p_b = quick_regress(x_def[name], t_super_col).flatten()
                        p_b = p_b / _nz(float(np.sqrt(ssq(p_b.reshape(-1, 1)))))
                        t_b = quick_regress(x_def[name], p_b.reshape(-1, 1)).flatten() / sqrt_kb[name]
                        local_loadings[name] = p_b
                        local_scores[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                else:
                    for b_idx, name in enumerate(self.block_names_):
                        p_b = x_def[name].T @ t_super / _nz(float(t_super @ t_super))
                        p_b = p_b / _nz(float(np.linalg.norm(p_b)))
                        t_b = x_def[name] @ p_b / _nz(float(p_b @ p_b)) / sqrt_kb[name]
                        local_loadings[name] = p_b
                        local_scores[name] = t_b
                        t_b_summary[:, b_idx] = t_b
                p_s = t_b_summary.T @ t_super / _nz(float(t_super @ t_super))
                p_s = p_s / _nz(float(np.linalg.norm(p_s)))
                t_super = t_b_summary @ p_s / _nz(float(p_s @ p_s))
                itern += 1

            # Sign convention: largest |super_loading| element positive
            flip_idx = int(np.argmax(np.abs(p_s)))
            if p_s[flip_idx] < 0:
                p_s = -p_s
                t_super = -t_super
                for name in self.block_names_:
                    local_loadings[name] = -local_loadings[name]
                    local_scores[name] = -local_scores[name]

            # Deflate each block using the super-score and block loading scaled by super-loading
            for b_idx, name in enumerate(self.block_names_):
                p_deflate = local_loadings[name] * p_s[b_idx] * sqrt_kb[name]
                x_def[name] = x_def[name] - np.outer(t_super, p_deflate)
                block_loadings_np[name][:, a] = local_loadings[name]
                block_scores_np[name][:, a] = local_scores[name]

            super_scores_np[:, a] = t_super
            super_loadings_np[:, a] = p_s

            # Per-block cumulative R²X and SPE
            for b_idx, name in enumerate(self.block_names_):
                ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
                # R^2 is undefined for a zero-variance block/column; report NaN
                # rather than dividing by zero (inf/nan + warning) or 1.0.
                r2_x_block_cum[b_idx, a] = (
                    1 - np.sum(ssq_remain_per_var) / ssq_x_init[name] if ssq_x_init[name] > 0 else np.nan
                )
                r2_x_var_cum[name][:, a] = np.where(
                    ssq_x_init_per_var[name] > 0,
                    1 - ssq_remain_per_var / np.where(ssq_x_init_per_var[name] > 0, ssq_x_init_per_var[name], 1.0),
                    np.nan,
                )
                block_spe_np[name][:, a] = np.sqrt(np.nansum(x_def[name] ** 2, axis=1))

            timing[a] = time.time() - start
            iterations[a] = itern

        # Wrap in pandas containers
        component_names = list(range(1, n_components + 1))
        self.super_scores_ = pd.DataFrame(super_scores_np, index=self._sample_index, columns=component_names)
        self.super_loadings_ = pd.DataFrame(
            super_loadings_np, index=self.block_names_, columns=component_names
        )
        self.block_scores_ = {
            name: pd.DataFrame(block_scores_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(
                block_loadings_np[name], index=self._block_columns[name], columns=component_names
            )
            for name in self.block_names_
        }

        self.explained_variance_ = np.diag(super_scores_np.T @ super_scores_np) / max(1, n_samples - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )
        converged = iterations < self.max_iter
        self.fitting_info_ = {"timing": timing, "iterations": iterations, "converged": converged}
        logger.debug("MBPCA (%s): iterations per component = %s", self.algorithm_, list(iterations))
        if not np.all(converged):
            failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
            warnings.warn(
                f"MBPCA NIPALS did not converge within max_iter={self.max_iter} for "
                f"component(s) {failed}; results for those components may be unreliable.",
                SpecificationWarning,
                stacklevel=2,
            )

        # Per-component (incremental) R²X
        r2_x_block_per_a = np.zeros_like(r2_x_block_cum)
        r2_x_block_per_a[:, 0] = r2_x_block_cum[:, 0]
        if n_components > 1:
            r2_x_block_per_a[:, 1:] = np.diff(r2_x_block_cum, axis=1)

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

        # VIPs (per-block: variance-of-X explanation; super: same on super-loadings)
        self.block_vip_: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            r2 = r2_x_block_per_a[b_idx, :]
            if np.sum(r2) > 0:
                p = self.block_loadings_[name].values
                vip_b = np.sqrt(self.block_widths_[name] * np.sum(r2 * p**2, axis=1) / np.sum(r2))
            else:
                vip_b = np.zeros(self.block_widths_[name])
            self.block_vip_[name] = pd.Series(vip_b, index=self._block_columns[name], name=f"VIP[{name}]")

        # Per-block SPE / T² and super T²
        self.block_spe_ = {
            name: pd.DataFrame(block_spe_np[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        block_t2: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            scores_np = self.block_scores_[name].values
            score_var = np.var(scores_np, axis=0, ddof=1)
            score_var = np.where(score_var > 0, score_var, 1.0)
            block_t2[name] = np.cumsum((scores_np**2) / score_var, axis=1)
        self.block_hotellings_t2_ = {
            name: pd.DataFrame(block_t2[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        super_score_var = np.where(self.explained_variance_ > 0, self.explained_variance_, 1.0)
        super_t2 = np.cumsum((super_scores_np**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

        return self

    def transform(self, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Project new data to super-scores using the fitted model."""
        check_is_fitted(self, "super_loadings_")
        return self._project(X).super_scores

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data; return super_scores, block_scores, block_spe, hotellings_t2."""
        check_is_fitted(self, "super_loadings_")
        return self._project(X)

    def _project(self, X: dict[str, pd.DataFrame]) -> Bunch:
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks for prediction: {sorted(missing)}.")

        x_pp: dict[str, np.ndarray] = {}
        sample_index = None
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
                p_b = self.block_loadings_[name].values[:, a]
                t_b = x_def[name] @ p_b / _nz(p_b @ p_b) / sqrt_kb[name]
                block_scores[name][:, a] = t_b
                t_b_row[:, b_idx] = t_b
            p_s = self.super_loadings_.values[:, a]
            t_super = t_b_row @ p_s / _nz(p_s @ p_s)
            super_scores[:, a] = t_super
            for b_idx, name in enumerate(self.block_names_):
                p_b = self.block_loadings_[name].values[:, a]
                p_deflate = p_b * p_s[b_idx] * sqrt_kb[name]
                x_def[name] = x_def[name] - np.outer(t_super, p_deflate)

        component_names = list(range(1, n_components + 1))
        super_scores_df = pd.DataFrame(super_scores, index=sample_index, columns=component_names)
        block_scores_df = {
            name: pd.DataFrame(block_scores[name], index=sample_index, columns=component_names)
            for name in self.block_names_
        }
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
            block_spe=block_spe,
            hotellings_t2=hotellings_t2,
        )

    def block_spe_limit(self, block: str, conf_level: float = 0.95) -> float:
        """SPE limit for one X-block (Nomikos & MacGregor chi-square approximation)."""
        check_is_fitted(self, "block_spe_")
        if block not in self.block_spe_:
            raise KeyError(f"Unknown block '{block}'. Known blocks: {list(self.block_spe_)}.")
        return spe_calculation(self.block_spe_[block].iloc[:, -1].to_numpy(), conf_level=conf_level)

    def super_spe_limit(self, conf_level: float = 0.95) -> float:
        """SPE limit for the merged super-block."""
        check_is_fitted(self, "block_spe_")
        merged_spe_squared = np.zeros(self.n_samples_)
        for name in self.block_names_:
            merged_spe_squared += self.block_spe_[name].iloc[:, -1].to_numpy() ** 2
        return spe_calculation(np.sqrt(merged_spe_squared), conf_level=conf_level)

    def spe_contributions(self, X: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Per-variable squared residuals for each X-block (SPE contributions).

        Reconstruction matches the MBPCA deflation step:
        ``X_b = T_super @ (P_b * p_super[b] * sqrt(K_b))^T`` summed over
        components. Returns squared residuals on the preprocessed scale; sum
        across columns equals ``block_spe_[b].iloc[:, -1] ** 2``.
        """
        check_is_fitted(self, "block_loadings_")
        if not isinstance(X, dict):
            raise TypeError("X must be a dict[str, pd.DataFrame].")
        missing = set(self.block_names_) - set(X)
        if missing:
            raise ValueError(f"Missing X-blocks: {sorted(missing)}.")

        result = self._project(X)
        super_scores = result.super_scores.values  # (N, A)
        sample_index = next(iter(result.block_scores.values())).index
        sqrt_kb = {name: float(np.sqrt(self.block_widths_[name])) for name in self.block_names_}

        out: dict[str, pd.DataFrame] = {}
        for b_idx, name in enumerate(self.block_names_):
            block = X[name]
            if not isinstance(block, pd.DataFrame):
                block = pd.DataFrame(block, columns=self._block_columns[name])
            x_pp = self.preproc_[name].transform(block).values.astype(float)
            # X_b reconstruction summed over components
            p_b = self.block_loadings_[name].values  # (K_b, A)
            p_s = self.super_loadings_.values[b_idx, :]  # (A,)
            p_eff = p_b * p_s * sqrt_kb[name]  # (K_b, A) effective loading
            x_hat = super_scores @ p_eff.T
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
        r"""Per-block per-variable contributions to a super-score movement (MBPCA).

        Decomposes a super-score-space delta into preprocessed-scale variable
        contributions per X-block. The MBPCA back-projection mirrors the
        deflation step used during fit:

        .. math::

            \text{contrib}_{b,j} = \sum_a (\Delta t_\mathrm{super}[a]) \cdot
            P_b[j, a] \cdot p_\mathrm{super}[b, a] \cdot \sqrt{K_b}

        See :meth:`MBPLS.score_contributions` for the parameter and return
        descriptions; the API is identical.
        """
        check_is_fitted(self, "block_loadings_")
        t_start = np.asarray(t_super_start, dtype=float)
        t_end = np.zeros(self.n_components) if t_super_end is None else np.asarray(t_super_end, dtype=float)
        idx = np.arange(self.n_components) if components is None else np.array(components) - 1
        dt = t_end[idx] - t_start[idx]
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
            ps = self.super_loadings_.values[b_idx, idx]  # (len(idx),)
            pb = self.block_loadings_[name].values[:, idx]  # (K_b, len(idx))
            effective = pb * (ps * sqrt_kb)  # (K_b, len(idx))
            contrib = effective @ dt  # (K_b,)
            out[name] = pd.Series(contrib, index=self._block_columns[name], name=f"score_contributions[{name}]")
        return out

    def super_score_plot(self, pc_horiz: int = 1, pc_vert: int = 2) -> go.Figure:
        """Scatter plot of MBPCA super-scores for two components."""
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
            title=f"MBPCA super-score plot: PC{pc_horiz} vs PC{pc_vert}",
        )
        return fig

    def super_loadings_bar_plot(self, component: int = 1) -> go.Figure:
        """Bar plot of MBPCA super-loadings for a single component."""
        check_is_fitted(self, "super_loadings_")
        a_max = int(self.n_components)
        if not (1 <= component <= a_max):
            raise ValueError(f"component must be in 1..{a_max}.")
        loadings = self.super_loadings_[component]
        fig = go.Figure(data=[go.Bar(x=list(loadings.index), y=loadings.to_numpy(), name=f"p_super[{component}]")])
        fig.update_layout(
            xaxis_title="Block",
            yaxis_title=f"p_super[{component}]",
            title=f"MBPCA super-loadings, component {component}",
        )
        return fig

    def display_results(self, show_cumulative: bool = True) -> str:
        """Format a short text summary of per-block R²X, iterations and timing."""
        check_is_fitted(self, "super_scores_")
        rows: list[str] = [f"MBPCA model: {self.n_components} component(s), {len(self.block_names_)} X-block(s)"]
        header = "  PC | " + " | ".join(f"R²X[{name}]" for name in self.block_names_)
        rows.append(header)
        rows.append("-" * len(header))
        src = self.r2_x_per_block_cumulative_ if show_cumulative else self.r2_x_per_block_per_component_
        for a in range(self.n_components):
            cells = [f"{a + 1:>3d}"]
            cells.extend(f"{src.loc[name].iloc[a]:>9.4f}" for name in self.block_names_)
            rows.append(" | ".join(cells))
        rows.append("")
        rows.append(f"  Iterations per PC: {list(self.fitting_info_['iterations'])}")
        rows.append(f"  Time per PC (ms):  {[round(float(t * 1000), 1) for t in self.fitting_info_['timing']]}")
        return "\n".join(rows)
