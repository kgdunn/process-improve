# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Multi-block PCA (MBPCA) estimator (ENG-01).

Holds :class:`MBPCA`, the hierarchical / superblock multi-block PCA transformer.
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
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from ._base import _HotellingsT2LimitMixin
from ._common import SpecificationWarning, _nz, _scale_block_contributions, epsqrt
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


def _validate_blocks(X: dict[str, pd.DataFrame]) -> None:
    """Reject anything that is not a non-empty dict of equally tall DataFrames."""
    if not isinstance(X, dict) or len(X) == 0:
        raise TypeError("X must be a non-empty dict[str, pd.DataFrame].")
    for name, block in X.items():
        if not isinstance(block, pd.DataFrame):
            raise TypeError(f"X['{name}'] must be a pandas DataFrame; got {type(block).__name__}.")
    names = list(X)
    n_samples = X[names[0]].shape[0]
    for name in names:
        if X[name].shape[0] != n_samples:
            raise ValueError(
                f"All X-blocks must have the same row count. Block '{name}' has "
                f"{X[name].shape[0]} rows; expected {n_samples}."
            )


def _resolve_missing_data_settings(missing_data_settings: dict | None, algo: str) -> dict:
    """Resolve and validate the iterative-algorithm settings for the NIPALS path."""
    settings = {"md_tol": epsqrt, "md_max_iter": 1000}
    if isinstance(missing_data_settings, dict):
        settings.update(missing_data_settings)
    settings["md_max_iter"] = int(settings["md_max_iter"])
    if algo == "nipals":
        if not settings["md_tol"] < 10:  # the historical ceiling, kept verbatim
            raise ValueError("Tolerance should not be too large.")
        if not settings["md_tol"] > epsqrt**1.95:
            raise ValueError("Tolerance must exceed machine precision.")
    return settings


def _reject_degenerate_missingness(X: dict[str, pd.DataFrame], block_names: Sequence[str]) -> None:
    """Refuse a block with an all-missing column or row.

    Either one leaves the masked NIPALS denominator at zero, which would silently produce
    a spurious score or loading. Refusing is better than coercing the caller into a
    misleading result.
    """
    for name in block_names:
        values = X[name].values
        col_all_nan = np.all(np.isnan(values), axis=0)
        if np.any(col_all_nan):
            bad = X[name].columns[col_all_nan].tolist()
            raise ValueError(
                f"Block '{name}' has columns with all values missing: {bad}. Drop these columns before fitting."
            )
        row_all_nan = np.all(np.isnan(values), axis=1)
        if np.any(row_all_nan):
            bad_rows = np.where(row_all_nan)[0].tolist()
            raise ValueError(
                f"Block '{name}' has rows with all values missing at positions {bad_rows}. "
                "Drop these observations or impute them before fitting."
            )


class _MBPCALoopContext(typing.NamedTuple):
    """Everything the component loop needs that does not change between components."""

    algo: str
    block_names: list[str]
    #: Algorithmic block weighting: X_b / sqrt(K_b), so blocks of unequal width contribute fairly.
    sqrt_kb: dict[str, float]
    tol: float
    max_iter: int
    n_samples: int

    @property
    def n_blocks(self) -> int:
        """How many X-blocks this fit is over."""
        return len(self.block_names)


class _MBPCAComponent(typing.NamedTuple):
    """One converged component, before deflation."""

    super_scores: np.ndarray
    super_loadings: np.ndarray
    block_loadings: dict[str, np.ndarray]
    block_scores: dict[str, np.ndarray]
    iterations: int


def _seed_super_score(x_def: dict[str, np.ndarray], context: _MBPCALoopContext) -> np.ndarray:
    """Deterministic start (#503): the column, across all blocks, with the largest sum of squares.

    Mirrors the single-block PCA / PLS seeding (#195). No RNG is involved, so the fit is
    reproducible without a ``random_state``, and the highest-variance column is closest to
    the leading component. The sign convention applied after convergence makes the fitted
    signs independent of this seed. NaN is replaced by 0 for the missing-data path.
    """
    col_ssq = {name: np.nansum(x_def[name] ** 2, axis=0) for name in context.block_names}
    start_block = max(context.block_names, key=lambda name: float(np.max(col_ssq[name], initial=0.0)))
    start_col = int(np.argmax(col_ssq[start_block]))
    return np.nan_to_num(x_def[start_block][:, start_col].astype(float).copy())


def _fit_one_component(x_def: dict[str, np.ndarray], context: _MBPCALoopContext) -> _MBPCAComponent:
    """Iterate the super-score to convergence on the currently deflated blocks."""
    t_super = _seed_super_score(x_def, context)
    prev = t_super + 1.0
    t_b_summary = np.zeros((context.n_samples, context.n_blocks))
    local_loadings: dict[str, np.ndarray] = {}
    local_scores: dict[str, np.ndarray] = {}
    p_s = np.zeros(context.n_blocks)
    itern = 0

    # Relative convergence criterion (#504): the change between two successive super-score
    # iterations is judged against the size of the current super-score, so the decision is
    # invariant to a global rescaling of the data. The denominator is floored via ``_nz``
    # so an all-zero super-score cannot divide by zero.
    while (
        np.linalg.norm(prev - t_super) / _nz(float(np.linalg.norm(t_super))) > context.tol and itern < context.max_iter
    ):
        prev = t_super
        if context.algo == "nipals":
            # Mask-aware NIPALS: each projection is a per-column (or per-row) regression
            # that uses only the entries that are not NaN, and divides by the masked sum of
            # squares. Reuses the same primitives as single-block PCA NIPALS.
            t_super_col = t_super.reshape(-1, 1)
            for b_idx, name in enumerate(context.block_names):
                p_b = quick_regress(x_def[name], t_super_col).flatten()
                p_b = p_b / _nz(float(np.sqrt(ssq(p_b.reshape(-1, 1)))))
                t_b = quick_regress(x_def[name], p_b.reshape(-1, 1)).flatten() / context.sqrt_kb[name]
                local_loadings[name] = p_b
                local_scores[name] = t_b
                t_b_summary[:, b_idx] = t_b
        else:
            for b_idx, name in enumerate(context.block_names):
                p_b = x_def[name].T @ t_super / _nz(float(t_super @ t_super))
                p_b = p_b / _nz(float(np.linalg.norm(p_b)))
                t_b = x_def[name] @ p_b / _nz(float(p_b @ p_b)) / context.sqrt_kb[name]
                local_loadings[name] = p_b
                local_scores[name] = t_b
                t_b_summary[:, b_idx] = t_b
        p_s = t_b_summary.T @ t_super / _nz(float(t_super @ t_super))
        p_s = p_s / _nz(float(np.linalg.norm(p_s)))
        t_super = t_b_summary @ p_s / _nz(float(p_s @ p_s))
        itern += 1

    # Sign convention: largest |super_loading| element positive.
    flip_idx = int(np.argmax(np.abs(p_s)))
    if p_s[flip_idx] < 0:
        p_s = -p_s
        t_super = -t_super
        for name in context.block_names:
            local_loadings[name] = -local_loadings[name]
            local_scores[name] = -local_scores[name]

    return _MBPCAComponent(
        super_scores=t_super,
        super_loadings=p_s,
        block_loadings=local_loadings,
        block_scores=local_scores,
        iterations=itern,
    )


def _deflate(
    x_def: dict[str, np.ndarray], component: _MBPCAComponent, context: _MBPCALoopContext
) -> dict[str, np.ndarray]:
    """Remove this component from every block, using the super-score and the scaled block loading."""
    deflated = {}
    for b_idx, name in enumerate(context.block_names):
        p_deflate = component.block_loadings[name] * component.super_loadings[b_idx] * context.sqrt_kb[name]
        deflated[name] = x_def[name] - np.outer(component.super_scores, p_deflate)
    return deflated


@dataclasses.dataclass
class _MBPCAArrays:
    """The numpy workspace a fit fills in, one column per component, wrapped in pandas at the end."""

    super_scores: np.ndarray
    super_loadings: np.ndarray
    block_scores: dict[str, np.ndarray]
    block_loadings: dict[str, np.ndarray]
    block_spe: dict[str, np.ndarray]
    r2_x_block_cum: np.ndarray
    r2_x_var_cum: dict[str, np.ndarray]
    timing: np.ndarray
    iterations: np.ndarray

    @classmethod
    def allocate(cls, block_widths: dict[str, int], n_samples: int, n_components: int) -> _MBPCAArrays:
        """Allocate the workspace for a fit of the given shape."""
        n_blocks = len(block_widths)
        per_block = {name: np.zeros((n_samples, n_components)) for name in block_widths}
        per_variable = {name: np.zeros((width, n_components)) for name, width in block_widths.items()}
        return cls(
            super_scores=np.zeros((n_samples, n_components)),
            super_loadings=np.zeros((n_blocks, n_components)),
            block_scores={name: values.copy() for name, values in per_block.items()},
            block_loadings={name: values.copy() for name, values in per_variable.items()},
            block_spe={name: values.copy() for name, values in per_block.items()},
            r2_x_block_cum=np.zeros((n_blocks, n_components)),
            r2_x_var_cum={name: values.copy() for name, values in per_variable.items()},
            timing=np.zeros(n_components),
            iterations=np.zeros(n_components, dtype=int),
        )

    def record_component(self, a: int, component: _MBPCAComponent, context: _MBPCALoopContext) -> None:
        """Store one converged component's scores and loadings."""
        for b_idx, name in enumerate(context.block_names):
            self.block_loadings[name][:, a] = component.block_loadings[name]
            self.block_scores[name][:, a] = component.block_scores[name]
            self.super_loadings[b_idx, a] = component.super_loadings[b_idx]
        self.super_scores[:, a] = component.super_scores
        self.iterations[a] = component.iterations

    def record_explained_variation(
        self,
        a: int,
        x_def: dict[str, np.ndarray],
        initial: _InitialSumsOfSquares,
        context: _MBPCALoopContext,
    ) -> None:
        """Store the cumulative R2X and SPE left after this component was removed."""
        for b_idx, name in enumerate(context.block_names):
            ssq_remain_per_var = np.nansum(x_def[name] ** 2, axis=0)
            # R^2 is undefined for a zero-variance block/column; report NaN rather than
            # dividing by zero (inf/nan + warning) or 1.0.
            self.r2_x_block_cum[b_idx, a] = (
                1 - np.sum(ssq_remain_per_var) / initial.per_block[name] if initial.per_block[name] > 0 else np.nan
            )
            per_var = initial.per_variable[name]
            self.r2_x_var_cum[name][:, a] = np.where(
                per_var > 0, 1 - ssq_remain_per_var / np.where(per_var > 0, per_var, 1.0), np.nan
            )
            self.block_spe[name][:, a] = np.sqrt(np.nansum(x_def[name] ** 2, axis=1))


class _InitialSumsOfSquares(typing.NamedTuple):
    """What each block held before any component was removed, the reference every R2X is against."""

    per_block: dict[str, float]
    per_variable: dict[str, np.ndarray]

    @classmethod
    def of(cls, x_blocks_pp: dict[str, np.ndarray]) -> _InitialSumsOfSquares:
        """Measure the preprocessed blocks before the first deflation."""
        return cls(
            per_block={name: float(np.nansum(values**2)) for name, values in x_blocks_pp.items()},
            per_variable={name: np.nansum(values**2, axis=0) for name, values in x_blocks_pp.items()},
        )


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
        Number of super-components (consensus latent variables) to extract.
    max_iter : int, default=500
        Maximum NIPALS iterations per component in the hierarchical outer loop.
    tol : float or None, default=None
        Relative convergence tolerance on the super-score change: the norm of
        the change between two successive super-score iterations, divided by
        the norm of the current super-score vector (#504). ``None`` uses
        ``epsqrt`` (about 1.49e-8), the same default as PCA / PLS / TPLS. The
        legacy absolute tolerance ``np.finfo(float).eps ** (9/10)`` (about
        8.2e-15) sits below the floating-point oscillation floor of a relative
        criterion, so it would never be reached in practice.
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
    block_names_ : list[str]
        Ordered list of X-block names (the keys of the input dict).
    block_widths_ : dict[str, int]
        Number of variables in each X-block.
    n_samples_ : int
        Number of rows fitted.
    n_features_in_ : int
        Total number of X variables summed across blocks.
    feature_names_in_ : np.ndarray
        Concatenated column names, one per feature, in block order.
    preproc_ : dict[str, MCUVScaler]
        Per-block preprocessors used to mean-centre and unit-variance
        scale each X-block.
    super_scores_ : pd.DataFrame, shape (n_samples, n_components)
        Super-block (consensus) scores ``T``.
    super_loadings_ : pd.DataFrame, shape (n_blocks, n_components)
        Super-block loadings ``p_super``; rows indexed by block name.
    super_hotellings_t2_ : pd.DataFrame, shape (n_samples, n_components)
        Cumulative Hotelling's T^2 on the super-scores per component.
    block_scores_ : dict[str, pd.DataFrame]
        Per-block scores ``t_b``, each shape ``(n_samples, n_components)``.
    block_loadings_ : dict[str, pd.DataFrame]
        Per-block loadings ``p_b``, each shape ``(K_b, n_components)``.
    block_spe_ : dict[str, pd.DataFrame]
        Per-block squared prediction error per sample and component.
    block_hotellings_t2_ : dict[str, pd.DataFrame]
        Per-block cumulative Hotelling's T^2 per sample and component.
    block_vip_ : dict[str, pd.Series]
        Per-block variable-importance in projection, indexed by variable
        name inside each block.
    r2_x_per_block_cumulative_ : pd.DataFrame, shape (n_blocks, n_components)
        Cumulative R^2X per block and component.
    r2_x_per_block_per_component_ : pd.DataFrame, shape (n_blocks, n_components)
        Incremental R^2X per block and component.
    r2_x_per_variable_ : dict[str, pd.DataFrame]
        Cumulative R^2X per variable within each block.
    explained_variance_ : np.ndarray, shape (n_components,)
        Variance of the super-score per component (ddof=1).
    scaling_factor_for_super_scores_ : pd.Series
        ``sqrt(explained_variance_)`` per component.
    fitting_info_ : dict
        Per-component iteration count and timing.
    has_missing_data_ : bool
        Whether any X-block had NaN values.
    algorithm_ : str
        The resolved algorithm actually used for the fit. With
        ``algorithm="auto"``, this is ``"dense"`` for complete data
        and ``"nipals"`` for NaN-containing data.

    Notes
    -----
    The deflation step is :math:`X_b \leftarrow X_b - t_{\rm super}\,
    (p_b\,p_s[b]\,\sqrt{K_b})^\top`, derived in Westerhuis et al. 1998.
    An earlier implementation of this method had this step marked as broken
    by its author; this implementation re-derives it directly from the paper
    and is independently validated against the pure-numpy reference oracles
    in the test suite.

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
    def fit(self, X: dict[str, pd.DataFrame], y: None = None) -> MBPCA:  # noqa: ARG002
        """Fit the multi-block PCA model.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            X-blocks. Keys are block names; values are DataFrames sharing the same row
            index (and row count). Each block is preprocessed independently.
        y : None
            Ignored; accepted so the transformer plugs into a sklearn Pipeline.
        """
        _validate_blocks(X)
        self._record_data_shape(X)
        algo = self._resolve_algorithm(X)
        # Resolve the iterative-algorithm settings. Only the validation inside is
        # load-bearing today: the resolved ``md_tol`` / ``md_max_iter`` do not yet reach
        # the NIPALS path, which uses ``tol`` and ``max_iter``.
        _resolve_missing_data_settings(self.missing_data_settings, algo)
        if algo == "nipals":
            _reject_degenerate_missingness(X, self.block_names_)

        x_blocks_pp = self._preprocess(X)
        context = _MBPCALoopContext(
            algo=algo,
            block_names=self.block_names_,
            sqrt_kb={name: float(np.sqrt(width)) for name, width in self.block_widths_.items()},
            tol=epsqrt if self.tol is None else float(self.tol),
            max_iter=self.max_iter,
            n_samples=self.n_samples_,
        )

        work = _MBPCAArrays.allocate(self.block_widths_, self.n_samples_, self.n_components_)
        initial = _InitialSumsOfSquares.of(x_blocks_pp)
        x_def = {name: values.copy() for name, values in x_blocks_pp.items()}

        for a in range(self.n_components_):
            start = time.time()
            component = _fit_one_component(x_def, context)
            x_def = _deflate(x_def, component, context)
            work.record_component(a, component, context)
            work.record_explained_variation(a, x_def, initial, context)
            work.timing[a] = time.time() - start

        self._store_latent_frames(work)
        self._report_convergence(work)
        self._store_explained_variation(work)
        self._store_diagnostics(work)
        return self

    def _record_data_shape(self, X: dict[str, pd.DataFrame]) -> None:
        """Record the block names, widths, row and column labels, and counts this fit is over."""
        self.block_names_: list[str] = list(X.keys())
        first = X[self.block_names_[0]]
        self.block_widths_: dict[str, int] = {name: int(X[name].shape[1]) for name in self.block_names_}
        self._sample_index = first.index
        self._block_columns: dict[str, pd.Index] = {name: X[name].columns for name in self.block_names_}
        self.n_samples_ = int(first.shape[0])
        self.n_features_in_ = int(sum(self.block_widths_.values()))
        # feature_names_in_: sklearn convention (#392). Flat concatenation of all blocks'
        # column names in block-iteration order. Lets ``Pipeline.get_feature_names_out``
        # and SHAP / eli5 / model-card tooling introspect a multiblock fit through the
        # same surface as a single-block estimator.
        self.feature_names_in_ = np.concatenate([self._block_columns[name].to_numpy() for name in self.block_names_])
        # Fitted mirror of the constructor parameter, so shared helpers (the T2 limit
        # mixin, spe_limit, the plot pre-checks) read one resolved attribute across
        # PCA / PLS / MBPCA / MBPLS (#505).
        self.n_components_ = int(self.n_components)

    def _resolve_algorithm(self, X: dict[str, pd.DataFrame]) -> str:
        """Pick between the dense and NIPALS paths, and record what was picked."""
        self.has_missing_data_ = any(np.any(X[name].isna().values) for name in self.block_names_)
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

    def _preprocess(self, X: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
        """Mean-centre and unit-variance scale each block independently."""
        self.preproc_: dict[str, MCUVScaler] = {name: MCUVScaler().fit(X[name]) for name in self.block_names_}
        return {name: self.preproc_[name].transform(X[name]).values.astype(float) for name in self.block_names_}

    def _component_names(self) -> list[int]:
        """Column labels for every per-component frame: 1-based component numbers."""
        return list(range(1, self.n_components_ + 1))

    def _store_latent_frames(self, work: _MBPCAArrays) -> None:
        """Wrap the scores and loadings in labelled pandas containers."""
        component_names = self._component_names()
        self.super_scores_ = pd.DataFrame(work.super_scores, index=self._sample_index, columns=component_names)
        self.super_loadings_ = pd.DataFrame(work.super_loadings, index=self.block_names_, columns=component_names)
        self.block_scores_ = {
            name: pd.DataFrame(work.block_scores[name], index=self._sample_index, columns=component_names)
            for name in self.block_names_
        }
        self.block_loadings_ = {
            name: pd.DataFrame(work.block_loadings[name], index=self._block_columns[name], columns=component_names)
            for name in self.block_names_
        }
        self.explained_variance_ = np.diag(work.super_scores.T @ work.super_scores) / max(1, self.n_samples_ - 1)
        self.scaling_factor_for_super_scores_ = pd.Series(
            np.sqrt(self.explained_variance_), index=component_names, name="Standard deviation per super-score"
        )

    def _report_convergence(self, work: _MBPCAArrays) -> None:
        """Record the per-component timing and iteration counts, and warn about any that did not converge."""
        converged = work.iterations < self.max_iter
        self.fitting_info_ = {"timing": work.timing, "iterations": work.iterations, "converged": converged}
        logger.debug("MBPCA (%s): iterations per component = %s", self.algorithm_, list(work.iterations))
        if np.all(converged):
            return
        failed = [int(i + 1) for i, ok in enumerate(converged) if not ok]
        warnings.warn(
            f"MBPCA NIPALS did not converge within max_iter={self.max_iter} for "
            f"component(s) {failed}; results for those components may be unreliable.",
            SpecificationWarning,
            stacklevel=3,
        )

    def _store_explained_variation(self, work: _MBPCAArrays) -> None:
        """Turn the cumulative R2X into the cumulative, per-component and per-variable frames."""
        component_names = self._component_names()
        r2_x_block_per_a = np.zeros_like(work.r2_x_block_cum)
        r2_x_block_per_a[:, 0] = work.r2_x_block_cum[:, 0]
        if self.n_components_ > 1:
            r2_x_block_per_a[:, 1:] = np.diff(work.r2_x_block_cum, axis=1)

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

        # VIPs (per-block: variance-of-X explanation; the super VIP is the same idea on
        # the super-loadings).
        self.block_vip_: dict[str, pd.Series] = {}
        for b_idx, name in enumerate(self.block_names_):
            r2 = r2_x_block_per_a[b_idx, :]
            if np.sum(r2) > 0:
                p = self.block_loadings_[name].values
                vip_b = np.sqrt(self.block_widths_[name] * np.sum(r2 * p**2, axis=1) / np.sum(r2))
            else:
                vip_b = np.zeros(self.block_widths_[name])
            self.block_vip_[name] = pd.Series(vip_b, index=self._block_columns[name], name=f"VIP[{name}]")

    def _store_diagnostics(self, work: _MBPCAArrays) -> None:
        """Build the per-block SPE and Hotelling's T2 frames, and the super-block T2."""
        component_names = self._component_names()
        self.block_spe_ = {
            name: pd.DataFrame(work.block_spe[name], index=self._sample_index, columns=component_names)
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
        super_t2 = np.cumsum((work.super_scores**2) / super_score_var, axis=1)
        self.super_hotellings_t2_ = pd.DataFrame(super_t2, index=self._sample_index, columns=component_names)

    def transform(self, X: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Project new data to super-scores using the fitted model."""
        check_is_fitted(self, "super_loadings_")
        return self._project(X).super_scores

    def diagnose(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Project new data; return super_scores, block_scores, block_spe, hotellings_t2.

        The rename (since 1.38.4, #395) matches :meth:`PCA.diagnose` and
        :meth:`PLS.diagnose`; :meth:`predict` is kept as a deprecation
        shim.
        """
        check_is_fitted(self, "super_loadings_")
        return self._project(X)

    def predict(self, X: dict[str, pd.DataFrame]) -> Bunch:
        """Forward to :meth:`diagnose`; emits a :class:`DeprecationWarning`.

        .. deprecated:: 1.38.4
            Use :meth:`MBPCA.diagnose` instead. ``predict`` matches the
            sklearn-convention name but MBPCA isn't a regressor; the
            historical return is a diagnostics Bunch. The rename aligns
            with :meth:`PCA.diagnose`. Will be removed in 2.0.0.
        """
        warnings.warn(
            "MBPCA.predict is deprecated and will be removed in 2.0.0; use MBPCA.diagnose instead.",
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

        x_pp: dict[str, np.ndarray] = {}
        sample_index = None
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
            t_b_row = np.column_stack(
                [
                    x_def[name]
                    @ self.block_loadings_[name].values[:, a]
                    / _nz(float(self.block_loadings_[name].values[:, a] @ self.block_loadings_[name].values[:, a]))
                    / sqrt_kb[name]
                    for name in self.block_names_
                ]
            )
            p_s = self.super_loadings_.values[:, a]
            t_super = t_b_row @ p_s / _nz(float(p_s @ p_s))
            for b_idx, name in enumerate(self.block_names_):
                p_b = self.block_loadings_[name].values[:, a]
                x_def[name] = x_def[name] - np.outer(t_super, p_b * p_s[b_idx] * sqrt_kb[name])

        assert sample_index is not None
        return x_def, sample_index

    def score_contributions(
        self,
        X: dict[str, pd.DataFrame],
        component: int = 1,
        scaling: str = "none",
    ) -> dict[str, pd.DataFrame]:
        r"""Per-block per-variable contributions to a super-score (MBPCA).

        The multi-block analogue of :meth:`PCA.score_contributions`. A super
        score is a weighted sum of the (deflated, preprocessed) variables across
        every block, so it splits exactly into one term per variable:

        .. math::

            c_{b,ij}^{(a)} = \tilde{x}_{b,ij}^{(a)}\,
                \frac{P_b[j, a]\, p_\mathrm{super}[b, a]}
                     {(p_b^\top p_b)(p_\mathrm{super}^\top p_\mathrm{super})\sqrt{K_b}},
            \qquad
            \sum_b \sum_j c_{b,ij}^{(a)} = t_{\mathrm{super},ia},

        where :math:`\tilde{x}^{(a)}` is the block data deflated through the
        first :math:`a-1` components.

        See :meth:`MBPLS.score_contributions` for the parameter and return
        descriptions; the API is identical.
        """
        check_is_fitted(self, "block_loadings_")
        deflated, sample_index = self._deflated_blocks(X, component)
        a = int(component) - 1
        p_s = self.super_loadings_.values[:, a]
        super_norm = _nz(float(p_s @ p_s))

        raw: dict[str, np.ndarray] = {}
        for b_idx, name in enumerate(self.block_names_):
            sqrt_kb = float(np.sqrt(self.block_widths_[name]))
            p_b = self.block_loadings_[name].values[:, a]
            weight = p_b / _nz(float(p_b @ p_b)) / sqrt_kb * (p_s[b_idx] / super_norm)
            raw[name] = deflated[name] * weight

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

        The multi-block analogue of :meth:`PCA.group_contributions`. See that
        method for the definition; the result is returned one Series per
        X-block, and the sum over every block equals the group's average super
        score (or the difference between the two groups' average super scores
        when ``reference`` is given).
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
