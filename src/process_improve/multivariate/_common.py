# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Shared primitives for the multivariate package (ENG-01).

This leaf module holds the small, dependency-free building blocks used across
the PCA / PLS / TPLS / multiblock implementations: the :data:`DataMatrix` type
alias, the :data:`epsqrt` tolerance, the NIPALS denominator floor helper
:func:`_nz`, and :class:`SpecificationWarning`. Nothing here imports any of the
sibling submodules, so it sits at the base of the dependency graph.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from scipy import sparse

# Names re-exported to the rest of the package. Declared explicitly so CodeQL
# does not flag the ``DataMatrix`` type alias (only ever referenced in lazy
# annotation strings under ``from __future__ import annotations``) as unused.
__all__ = [
    "Q2_MIN_INCREMENT",
    "DataMatrix",
    "NotEnoughVarianceError",
    "SelectionRule",
    "SpecificationWarning",
    "UncentredDataWarning",
    "epsqrt",
]

#: Component-selection rules supported by ``PLS.select_n_components``.
#:
#: ``"1se"`` (the current default) applies the one-standard-error rule of
#: Breiman, Friedman, Olshen & Stone (1984, *CART* sec.3.4.3), endorsed by
#: Hastie, Tibshirani & Friedman (*The Elements of Statistical Learning*,
#: sec.7.10): pick the smallest component count whose mean cross-validated
#: error is within one standard error of the lowest. Needs per-fold errors,
#: so repeated K-fold is recommended.
#:
#: ``"min"`` returns the component count with the lowest cross-validated error.
#: On data sets where the validated error keeps drifting down by noise-level
#: amounts after the systematic components are exhausted, this routinely runs
#: to (or near) the maximum component count.
#:
#: ``"q2_increment"`` keeps adding components while each one raises the
#: cumulative cross-validated :math:`Q^2` by at least ``min_q2_increase``;
#: it is a Wold's-R-style heuristic with an absolute (not relative) threshold.
#:
#: ``"randomization"`` is Van der Voet's (1994, *Chemom. Intell. Lab. Syst.*
#: 25(2):313-323) randomization test: under the null that two PLS models have
#: the same predictive ability, randomly flip the sign of each observation's
#: paired squared-residual difference between a candidate model and the
#: argmin-RMSECV reference. The smallest model whose *p*-value exceeds
#: ``alpha`` (i.e. fails to reject the null) is chosen - statistically
#: indistinguishable from the best, but more parsimonious.
SelectionRule = Literal["1se", "min", "q2_increment", "randomization"]

#: Default minimum increase in the cross-validated :math:`Q^2` for an extra
#: component to be judged worth keeping under the ``"q2_increment"`` selection
#: rule. ``0.01`` means "must add at least one percentage point of
#: cross-validated explained variance".
Q2_MIN_INCREMENT = 0.01

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)

#: Smallest positive float; used to floor NIPALS denominators away from zero.
_DENOM_FLOOR = float(np.finfo(float).tiny)


def _nz(denominator: float) -> float:
    """Floor a non-negative NIPALS denominator away from zero.

    Sum-of-squares and vector-norm denominators (``v @ v``, ``norm(v)``) are
    non-negative. When a score or loading vector collapses to (near) zero during
    NIPALS - a fully-deflated component, or a degenerate / perfectly collinear
    block - the denominator is ~0 and the division would yield ``inf``/``nan``
    that silently poisons the fitted model. Flooring to the smallest positive
    float leaves every well-conditioned value untouched (real denominators are
    far larger) while turning the degenerate ``0/0`` into a finite (~0)
    projection, since the numerator collapses with the same vector.
    """
    return max(_DENOM_FLOOR, denominator)


def _reject_sparse(X: object, estimator_name: str) -> None:
    """Raise if ``X`` is a SciPy sparse matrix, naming the remedy that actually helps.

    NIPALS centres and scales every column, which destroys sparsity, so there is nothing
    to gain from a sparse representation here and every estimator in this package sets
    ``accept_sparse=False``. sklearn's own message for that says to call ``.toarray()``,
    which is the expensive way round when the sparsity came from a one-hot block: the
    dense array is materialised anyway, but only after the sparse one has been built.

    The usual source is a :class:`~sklearn.compose.ColumnTransformer` whose one-hot block
    pushed the result over ``sparse_threshold`` (default 0.3), which flips the *whole*
    concatenated output to sparse. Two knobs at that level avoid the round trip
    altogether, and the message names both (#399).

    Parameters
    ----------
    X : object
        The candidate input. Anything SciPy does not consider sparse passes through.
    estimator_name : str
        Named in the message, so a Pipeline failure says which step rejected the input.

    Raises
    ------
    TypeError
        If ``X`` is sparse.
    """
    if not sparse.issparse(X):
        return
    msg = (
        f"{estimator_name} does not accept sparse input: NIPALS centres and scales every "
        "column, which destroys sparsity, so there is nothing to gain from it. This usually "
        "arrives from a ColumnTransformer whose one-hot block pushed the concatenated output "
        "over `sparse_threshold` (default 0.3). Pass `sparse_threshold=0` to the "
        "ColumnTransformer, or `OneHotEncoder(sparse_output=False)`, so it hands over a dense "
        "array directly. `X.toarray()` also works, but on a wide one-hot block it builds the "
        "sparse matrix first and then the dense one anyway."
    )
    raise TypeError(msg)


class SpecificationWarning(UserWarning):
    """Parent warning class."""


class UncentredDataWarning(SpecificationWarning):
    """Emitted when a model that fits no intercept is handed an un-centred block.

    Raised by :meth:`PLS.fit <process_improve.multivariate.PLS.fit>` under
    ``scale=False``, which centres nothing and fits no intercept, so a block
    carrying a non-zero mean displaces every prediction.

    It exists as its own class so a caller who fits un-centred data on purpose
    can permit *this* diagnostic without going blind to the rest. Under a
    ``filterwarnings = error`` policy::

        # pytest.ini / pyproject.toml, or @pytest.mark.filterwarnings on one test
        filterwarnings =
            error
            ignore::process_improve.multivariate.UncentredDataWarning

    Narrower still, and with no global filter at all, is the estimator flag:
    ``PLS(..., scale=False, warn_on_uncentred=False)`` silences the check for
    that one model and leaves every other ``SpecificationWarning`` in force.

    Subclasses :class:`SpecificationWarning`, so filters and ``pytest.warns``
    assertions written against the parent keep matching it.
    """


class NotEnoughVarianceError(RuntimeError):
    """Raised when more components are requested than the data's variance supports.

    During NIPALS extraction (PCA / PLS) the deflated data array can run out of
    variance before the requested number of components is reached, even when that
    count is within ``min(N, K)`` - for example with rank-deficient or perfectly
    collinear data. The model cannot compute any further components in that case.

    Subclasses :class:`RuntimeError` so existing ``except RuntimeError`` handlers
    keep working, while callers that want to react specifically (e.g. trimming the
    component count during cross-validation) can catch this narrower type.
    """


def _align_to_fit_features(X: pd.DataFrame, fit_feature_names: pd.Index) -> pd.DataFrame:
    """Validate and align new-data columns against the features seen during ``fit``.

    PCA / PLS only checked that data passed to ``transform`` / ``predict`` had the
    right *number* of columns. A correctly-shaped frame whose columns are renamed
    or reordered would otherwise be projected positionally (PCA, ``X.values @ P``)
    or silently label-aligned to all-``NaN`` (PLS, ``X @ direct_weights_``),
    producing wrong scores with no error raised (issue #195). This helper makes
    that consistency explicit, mirroring scikit-learn's ``feature_names_in_``
    handling:

    * If both the training data and ``X`` carry string feature names, the *set*
      of names must match (otherwise :class:`ValueError`); columns supplied in a
      different order are reordered to the training order.
    * If the training data had names but ``X`` does not (e.g. a bare ndarray was
      passed), the columns are taken to correspond positionally and are labelled
      with the training names, so downstream label-aligned arithmetic stays
      correct rather than collapsing to ``NaN``.
    * If the training data itself had no (string) feature names, there is nothing
      to validate and ``X`` is returned unchanged.

    The caller is expected to have already validated the column *count*.

    Parameters
    ----------
    X : pd.DataFrame
        New data passed to ``transform`` / ``predict`` (already coerced to a
        DataFrame and count-checked by the caller).
    fit_feature_names : pd.Index
        The ``X.columns`` captured during ``fit`` (``self._feature_names``).

    Returns
    -------
    pd.DataFrame
        ``X`` with columns aligned to the training feature order.

    Raises
    ------
    ValueError
        If both sides carry string names but the sets of names differ.
    """
    fit_names = list(fit_feature_names)
    if not (fit_names and all(isinstance(name, str) for name in fit_names)):
        # Training data had no string feature names (e.g. fitted from an ndarray);
        # only the column count is meaningful, which the caller already checked.
        return X

    new_names = list(X.columns)
    if not all(isinstance(name, str) for name in new_names):
        # New data carries default positional columns (e.g. came in as an
        # ndarray). Assume positional correspondence and label it with the
        # training names so label-aligned operations behave correctly.
        X = X.copy()
        X.columns = pd.Index(fit_names)
        return X

    if set(new_names) != set(fit_names):
        missing = [name for name in fit_names if name not in set(new_names)]
        unexpected = [name for name in new_names if name not in set(fit_names)]
        raise ValueError(
            "Feature names of the data passed to predict/transform do not match "
            "those seen during fit. "
            f"Missing columns: {missing}; unexpected columns: {unexpected}."
        )
    if new_names != fit_names:
        # Same names, different order: reorder to the training order so the
        # positional projection lines up.
        X = X[fit_names]
    return X


def _recommend_n_components_q2(
    q2_cumulative: np.ndarray | pd.Series | list[float],
    *,
    min_increment: float = Q2_MIN_INCREMENT,
) -> int:
    r"""Recommend a component count from a cumulative cross-validated :math:`Q^2` curve.

    Walks outward from one component, keeping component ``a`` only if it lifts
    the cumulative :math:`Q^2` by at least ``min_increment`` (with an implied
    :math:`Q^2` of ``0`` before any component). The first component that fails
    that test - a plateau, a drop, or a non-finite entry - stops the search,
    and the recommendation is the last component that passed (never fewer than
    one, since a model needs at least one component).

    This is a Wold's-R-style heuristic: parsimonious and cheap, but the
    threshold is absolute and hand-tuned. Prefer the 1-SE rule
    (:func:`_recommend_n_components_one_se`) when per-fold errors are available.
    """
    q2 = np.asarray(q2_cumulative, dtype=float)
    recommended = 1
    previous = 0.0
    for a, value in enumerate(q2, start=1):
        if not np.isfinite(value) or (value - previous) < min_increment:
            break
        recommended = a
        previous = float(value)
    return recommended


def _recommend_n_components_one_se(
    mean_error: np.ndarray | pd.Series | list[float],
    se_error: np.ndarray | pd.Series | list[float],
) -> int:
    r"""Recommend a component count by the one-standard-error rule.

    Given the mean cross-validated error per component count (``mean_error``,
    e.g. RMSECV across folds and repeats) and its standard error
    (``se_error``), find ``a* = nanargmin(mean_error)`` and return the
    *smallest* component count ``a`` whose ``mean_error[a]`` is no larger than
    ``mean_error[a*] + se_error[a*]``.

    The 1-SE rule (Breiman, Friedman, Olshen & Stone 1984, *CART* sec.3.4.3;
    Hastie, Tibshirani & Friedman, *ESL* sec.7.10) trades a tiny amount of
    fit for a more parsimonious model whose error is statistically
    indistinguishable from the best. It is the cheapest robustness upgrade for
    integer hyperparameters like the number of latent variables.

    Non-finite entries in ``mean_error`` are skipped. The selection always
    returns at least ``1``. If ``se_error`` at the minimum is non-finite or
    zero, the rule degenerates to the argmin.
    """
    mean = np.asarray(mean_error, dtype=float)
    se = np.asarray(se_error, dtype=float)
    if mean.shape != se.shape:
        raise ValueError(f"mean_error and se_error must have the same shape; got {mean.shape} and {se.shape}.")
    if mean.ndim != 1:
        raise ValueError("mean_error must be 1-D.")
    finite = np.isfinite(mean)
    if not finite.any():
        # Caller should already have raised before now; fall through to 1 so
        # this helper never returns a sentinel.
        return 1
    masked = np.where(finite, mean, np.inf)
    best = int(np.argmin(masked))
    se_best = se[best] if np.isfinite(se[best]) else 0.0
    threshold = masked[best] + se_best
    for a, value in enumerate(masked, start=1):
        if value <= threshold:
            return a
    return best + 1


def _select_n_components(
    rule: SelectionRule,
    *,
    mean_error: np.ndarray | pd.Series | list[float],
    se_error: np.ndarray | pd.Series | list[float] | None = None,
    q2_cumulative: np.ndarray | pd.Series | list[float] | None = None,
    min_q2_increase: float = Q2_MIN_INCREMENT,
) -> int:
    """Dispatch component selection to one of the supported rules.

    See :data:`SelectionRule` for the rule semantics. Raises ``ValueError`` if
    a rule is requested without the data it needs (``"1se"`` needs
    ``se_error``; ``"q2_increment"`` needs ``q2_cumulative``).
    """
    if rule == "min":
        mean = np.asarray(mean_error, dtype=float)
        if not np.isfinite(mean).any():
            return 1
        return int(np.nanargmin(mean)) + 1
    if rule == "1se":
        if se_error is None:
            raise ValueError("selection_rule='1se' requires per-component standard errors.")
        return _recommend_n_components_one_se(mean_error, se_error)
    if rule == "q2_increment":
        if q2_cumulative is None:
            raise ValueError("selection_rule='q2_increment' requires a cumulative Q^2 curve.")
        return _recommend_n_components_q2(q2_cumulative, min_increment=min_q2_increase)
    raise ValueError(
        f"Unknown selection_rule {rule!r}; expected one of "
        f"'1se', 'min', 'q2_increment', 'randomization' "
        f"('randomization' is handled by PLS.select_n_components, not this dispatcher)."
    )


def _equal_weight_r2_total(per_target: np.ndarray) -> np.ndarray:
    r"""Pool per-target validated :math:`R^2_Y` with every target weighted equally.

    The ``"total"`` column of a validated-:math:`R^2_Y` table is
    ``1 - sum_m PRESS_m / sum_m TSS_m`` on the *original* Y scale, so a target
    whose spread is two orders of magnitude larger than its neighbours' decides
    the number almost on its own. Autoscaling Y first fixes that, and costs
    nothing to compute here: after mean-centring and unit-variance scaling every
    target's TSS is the same, so the pooled ratio collapses to the arithmetic
    mean of the per-target values,

    .. math::

        1 - \frac{\sum_m \mathrm{PRESS}_m / s_m^2}{\sum_m \mathrm{TSS}_m / s_m^2}
        = \frac{1}{M} \sum_m \left(1 - \frac{\mathrm{PRESS}_m}{\mathrm{TSS}_m}\right).

    Targets with no spread contribute ``NaN`` per-target values and are left out
    of the mean rather than dragging it to ``NaN``.

    Parameters
    ----------
    per_target : np.ndarray
        Validated :math:`R^2` per target, shape ``(A, M)``.

    Returns
    -------
    np.ndarray
        The equal-weight pooled value per component count, shape ``(A,)``.
    """
    values = np.asarray(per_target, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"per_target must be 2-D (A x M); got shape {values.shape}.")
    with warnings.catch_warnings():
        # An all-NaN row (every target constant) means "nothing to pool"; NaN
        # is the right answer and the RuntimeWarning that says so is noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=1) if values.shape[1] else np.full(values.shape[0], np.nan)


def _model_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a module-level ``fn(model, ...)`` as an introspectable instance method.

    ENG-05: estimators (PCA / PLS / ...) expose convenience methods such as
    ``score_plot``, ``spe_limit`` and ``vip`` that forward to the standalone
    functions with ``self`` supplied as the ``model`` argument. Defining them
    via this factory at class-body time - rather than binding
    ``functools.partial`` instances in ``fit`` - means ``help`` and
    ``inspect.signature`` report the underlying function (minus ``self``), the
    fitted model stays picklable, and subclasses can override cleanly.
    """

    @functools.wraps(fn)
    def method(self: object, *args, **kwargs) -> object:
        return fn(self, *args, **kwargs)

    return method


def _scale_block_contributions(blocks: dict[str, np.ndarray], scaling: str) -> dict[str, np.ndarray]:
    """Apply a contribution-plot scaling jointly across every block.

    The scalings are those of Miller, Swanson and Heckler (1994). Both are taken
    over the blocks together, not one block at a time: a super score pools all
    the blocks, so scaling each block separately would make bars from different
    blocks incomparable, which is the one thing a multi-block contribution plot
    is read for.
    """
    if scaling == "none":
        return blocks
    if scaling == "maximum":
        largest = max((float(np.abs(v).max()) for v in blocks.values() if v.size), default=0.0)
        scalar = largest if largest > epsqrt else 1.0
        return {name: values / scalar for name, values in blocks.items()}
    if scaling == "within":
        totals = sum(np.abs(values).sum(axis=1) for values in blocks.values())
        per_row = np.where(totals > epsqrt, totals, 1.0).reshape(-1, 1)
        return {name: values / per_row for name, values in blocks.items()}
    msg = f"scaling must be one of 'none', 'maximum' or 'within', got {scaling!r}."
    raise ValueError(msg)


class BlockSet(dict):
    """A ``dict[str, pd.DataFrame]`` of equal-height blocks that can also be sliced by row (#193).

    :meth:`MBPCA.fit <process_improve.multivariate.methods.MBPCA.fit>` and
    :meth:`MBPLS.fit <process_improve.multivariate.methods.MBPLS.fit>` take a
    plain ``dict[str, pd.DataFrame]``, which is convenient to build and
    impossible to resample: a dict has no notion of "row 7 of every block". Any
    resampling or cross-validation pass needs exactly that. ``Resampler``, for
    instance, asks its data only for ``len(x)`` and ``x[indices]``.

    TPLS already has :class:`~process_improve.multivariate.methods.DataFrameDict`
    for this, but it is hardwired to the ``Z``/``F``/``Y`` block names and to a
    nested ``dict[str, dict[str, DataFrame]]`` layout, so the multi-block models
    could not borrow it. ``BlockSet`` is the flat equivalent: a real ``dict``
    subclass, so anything that already accepts the plain dict keeps working,
    plus row indexing.

    .. warning::
       ``len(blocks)`` is the number of **rows**, not the number of blocks. That
       is surprising for a dict, and it is deliberate: it is the convention
       ``DataFrameDict`` already set, and it is what the resampling code means
       by the length of a dataset. Use ``len(blocks.keys())`` to count blocks.

    Parameters
    ----------
    blocks : dict[str, pd.DataFrame]
        One entry per block. Every block must be a DataFrame with the same
        number of rows; widths may differ.

    Raises
    ------
    ValueError
        If ``blocks`` is empty, or the blocks disagree on their row count.
    TypeError
        If any value is not a DataFrame.

    Examples
    --------
    >>> blocks = BlockSet({"a": df_a, "b": df_b})   # doctest: +SKIP
    >>> len(blocks)                                 # rows, not blocks  # doctest: +SKIP
    40
    >>> blocks[[0, 1, 2]].keys()                    # a 3-row BlockSet  # doctest: +SKIP
    dict_keys(['a', 'b'])
    """

    def __init__(self, blocks: dict[str, pd.DataFrame]):
        if not isinstance(blocks, dict):
            raise TypeError(f"blocks must be a dict of DataFrames, one per block; got {type(blocks).__name__}.")
        if not blocks:
            raise ValueError("At least one block is required.")

        first_name, first = next(iter(blocks.items()))
        if not isinstance(first, pd.DataFrame):
            raise TypeError(f"Block {first_name!r} must be a pandas DataFrame; got {type(first).__name__}.")
        n_samples = first.shape[0]
        for name, block in blocks.items():
            if not isinstance(block, pd.DataFrame):
                raise TypeError(f"Block {name!r} must be a pandas DataFrame; got {type(block).__name__}.")
            if block.shape[0] != n_samples:
                raise ValueError(
                    f"Every block must have the same number of rows ({n_samples}, from block {first_name!r}). "
                    f"Block {name!r} has {block.shape[0]}."
                )

        super().__init__(blocks)
        self.n_samples = int(n_samples)
        self.shape = (self.n_samples, len(blocks))

    def __len__(self) -> int:
        """Return the number of rows; see the warning in the class docstring."""
        return self.n_samples

    def __getitem__(self, lookup: str | int | list | np.ndarray) -> pd.DataFrame | BlockSet:
        """Look a block up by name, or slice every block to the same rows."""
        if isinstance(lookup, str):
            return cast("pd.DataFrame", super().__getitem__(lookup))
        return BlockSet({name: _row_slice(block, lookup, name) for name, block in self.items()})


def _row_slice(block: pd.DataFrame, lookup: int | list | np.ndarray, name: str) -> pd.DataFrame:
    """Take rows from one block, keeping the result two-dimensional."""
    match lookup:
        case int() | np.integer():
            # A list, not a scalar: a single row still has to come back as a frame,
            # otherwise a one-row resample would arrive at ``fit`` as a Series.
            return block.iloc[[int(lookup)]]
        case list():
            return block.iloc[[int(index) for index in lookup]]
        case np.ndarray():
            return block.iloc[lookup.tolist()]
        case _:
            raise TypeError(
                f"Row lookup must be an int, a list of ints, or an ndarray; "
                f"got {type(lookup).__name__} while slicing block {name!r}."
            )
