# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Principal Component Analysis (PCA) estimator (ENG-01).

The sklearn-compatible :class:`PCA` transformer, with NIPALS, SVD and TSR
fitting paths and full missing-data support. Diagnostics, confidence limits and
plotting are pulled in from the sibling submodules and bound as convenience
methods after ``fit()``.
"""

from __future__ import annotations

import logging
import time
import typing
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.decomposition import PCA as _SkPCA  # noqa: N811
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data

from ..univariate.metrics import detect_outliers_esd
from ._base import _LatentVariableModel, _LazyFrame
from ._common import (
    Q2_MIN_INCREMENT,
    DataMatrix,
    NotEnoughVarianceError,
    SelectionRule,
    SpecificationWarning,
    _align_to_fit_features,
    _select_n_components,
    epsqrt,
)
from ._nipals import quick_regress, ssq, terminate_check
from ._preprocessing import MCUVScaler, _warn_scaling_traps
from ._projection import coerce_observed_mask, operator_for_pattern, project_rows

logger = logging.getLogger(__name__)


class _EkfPress(typing.NamedTuple):
    """What one element-wise k-fold pass measured. See :func:`_pca_ekf_press`."""

    press: np.ndarray
    per_fold_press: np.ndarray
    per_column_press: np.ndarray
    null_model_ss: float
    per_column_null_ss: np.ndarray
    press_input_units: np.ndarray


def _pca_ekf_press(  # noqa: PLR0913, PLR0915, PLR0912, C901
    X: np.ndarray,
    max_components: int,
    *,
    n_folds: int = 5,
    n_repeats: int = 1,
    n_iter: int = 50,
    tol: float = 1e-6,
    scale_inside_folds: bool = True,
    random_state: int | None = None,
) -> _EkfPress:
    """Element-wise k-fold (ekf) PCA cross-validation.

    Partitions the elements of ``X`` into ``n_folds`` element-folds (each cell
    is held out exactly once across folds); for each fold and each candidate
    component count, the held-out cells are masked, initialised from the
    in-fold column means, and refined by iterating SVD reconstruction
    (Expectation-Maximisation style). The fitted model never sees the
    held-out true values, so the squared error of the prediction is an honest
    out-of-sample PRESS - the independence requirement of Bro, Kjeldahl,
    Smilde & Kiers (2008, *Anal. Bioanal. Chem.* 390:1241-1251) that the
    row-wise CV scheme violates.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix. With the default ``scale_inside_folds=True`` the raw
        unscaled matrix may be passed; with ``False`` the caller is expected
        to have mean-centred (and usually unit-variance scaled) it.
    max_components : int
        Maximum number of components to evaluate; PRESS is computed for
        ``1 .. max_components``.
    n_folds : int, default 5
        Number of element-folds. Bro 2008 uses 7 as a typical default; 5 is
        a faster choice that still gives a stable curve.
    n_repeats : int, default 1
        Number of times to repeat the ekf pass with a fresh random fold
        permutation. Each repeat covers every cell exactly once; ``n_repeats
        > 1`` averages over different element-fold partitions, narrowing
        the per-component PRESS standard error at extra runtime.
    n_iter : int, default 50
        Maximum number of EM iterations per fold and component count.
    tol : float, default 1e-6
        Relative change in the held-out cell predictions below which EM
        stops early.
    scale_inside_folds : bool, default True
        If True, fit per-column mean and unit-variance constants on each
        fold's in-fold cells and apply them to the whole matrix before
        running EM. This removes the centring/scaling leakage of the
        previous default (which used a single set of constants iteratively
        recomputed from the imputed matrix). PRESS is then measured in that
        same space, so a variable's contribution reflects its correlation
        structure rather than its units; ``press_input_units`` carries the
        other scale for callers who need it. If False, the scheme reverts to
        the prior behaviour: the caller is responsible for scaling, the
        in-loop column-mean drifts with the EM imputation, and PRESS is in
        the units of whatever matrix was passed.
    random_state : int, optional
        Seed for the element-fold permutation, for reproducibility across
        repeats.

    Returns
    -------
    _EkfPress
        A named tuple with five fields. Under ``scale_inside_folds=True`` the
        first three live in the space each fold was fitted in, so every column
        contributes to the total in proportion to its correlation structure
        rather than its units; under ``False`` there is no in-fold scale and
        they are in the input units, unchanged from earlier releases.

        press : np.ndarray of shape (max_components,)
            PRESS per component count, averaged over ``n_repeats`` passes so
            the scale is comparable to a single-pass run.
        per_fold_press : np.ndarray of shape (max_components, n_folds * n_repeats)
            Per-fold PRESS contributions across every fold of every repeat;
            drives the 1-SE rule's standard error.
        per_column_press : np.ndarray of shape (max_components, n_features)
            The same total, split by variable, for the per-variable Q2.
        null_model_ss : float
            What the null model ("predict every held-out cell by the mean of
            the cells that were not held out") got wrong, measured the same
            way ``press`` is: the reference ``press`` is compared against.
            Computed here rather than by the caller because it needs each
            fold's own centring and scaling constants.
        per_column_null_ss : np.ndarray of shape (n_features,)
            That reference, split by variable.
        press_input_units : np.ndarray of shape (max_components,)
            PRESS in the units of the matrix that was passed in, for callers
            comparing prediction error against instrument error. Always in
            input units, whatever ``scale_inside_folds`` is.

    References
    ----------
    Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008).
    Cross-validation of component models: a critical look at current
    methods. *Anal. Bioanal. Chem.*, 390(5), 1241-1251. PMID 18214448.

    Camacho, J., & Ferrer, A. (2012). Cross-validation in PCA models with
    the element-wise k-fold (ekf) algorithm: theoretical aspects.
    *J. Chemometrics*, 26(7), 361-373. DOI 10.1002/cem.2440.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    # A cell the caller never measured has no true value to predict, so it is
    # never held out and never scored. It still has to be filled for the SVD,
    # which cannot see a NaN, so EM imputes it alongside the held-out cells.
    observed = ~np.isnan(X)

    total_folds = n_folds * n_repeats
    # NaN, not zero: a fold that holds out no cells has no PRESS, and leaving
    # a structural zero there let the caller's standard error treat it as a
    # real fold that happened to predict perfectly. On a matrix with fewer
    # cells than folds every cell lands in the last fold, so the empty folds
    # fabricated a standard error out of zeros. np.nansum below keeps the
    # PRESS total itself unchanged, since an empty fold contributes nothing.
    per_fold_press = np.full((max_components, total_folds), np.nan)
    # PRESS is what the model got wrong; the null reference is what predicting
    # the in-fold column mean would have got wrong on the same cells. Both are
    # accumulated here, in the same space, because only this loop knows each
    # fold's centring and scaling constants. The reference does not depend on
    # the component count, hence one value per fold rather than a grid.
    per_fold_null = np.zeros(total_folds)
    per_column_press = np.zeros((max_components, p))
    per_column_null = np.zeros(p)
    press_input_units = np.zeros(max_components)
    fold_counter = 0

    observed_positions = np.flatnonzero(observed)
    n_cells = observed_positions.size
    fold_size = n_cells // n_folds

    for _ in range(n_repeats):
        # Assign the observed cells to folds via a balanced random permutation
        # so every fold has ~n_cells/n_folds cells (not all the same column,
        # not the same row). Unmeasured cells stay at -1 and match no fold.
        perm = rng.permutation(n_cells)
        flat_fold = np.full(n * p, -1, dtype=np.int64)
        for k in range(n_folds):
            start = k * fold_size
            end = (k + 1) * fold_size if k < n_folds - 1 else n_cells
            flat_fold[observed_positions[perm[start:end]]] = k
        fold = flat_fold.reshape(n, p)

        for k in range(n_folds):
            mask = fold == k  # (n, p) bool, True where the cell is held out
            if not mask.any():
                fold_counter += 1
                continue

            # Fit per-column centring/scaling on the in-fold cells. The centre
            # is needed either way, as the null model's prediction for every
            # held-out cell in this fold; only the True path also centres and
            # scales the matrix it fits, and recomputes nothing inside EM.
            col_centre = np.zeros(p)
            col_scale = np.ones(p)
            # Held out this fold, or never measured: both are unknown to the
            # fit and both are filled by EM below. Only ``mask`` is scored.
            impute = mask | ~observed
            for j in range(p):
                in_fold = X[~impute[:, j], j]
                if in_fold.size > 1:
                    col_centre[j] = float(in_fold.mean())
                    if scale_inside_folds:
                        sd = float(in_fold.std(ddof=1))
                        col_scale[j] = sd if sd > epsqrt else 1.0
                elif in_fold.size == 1:
                    col_centre[j] = float(in_fold[0])

            # Initial imputation in original space: held-out cells take the
            # in-fold column mean (which is ``col_centre`` under
            # ``scale_inside_folds=True``, and the same value under False).
            Xtr = X.copy()
            for j in range(p):
                m_j = impute[:, j]
                if m_j.any():
                    Xtr[m_j, j] = col_centre[j]

            # The cells' own scale and centre for this fold, one value per
            # held-out cell. Under scale_inside_folds=False the scale is 1, so
            # every measurement below stays in the input units, as it was.
            cell_scale = np.broadcast_to(col_scale, X.shape)[mask]
            cell_centre = np.broadcast_to(col_centre, X.shape)[mask]
            column_of_cell = np.broadcast_to(np.arange(p), X.shape)[mask]
            null_residual = (X[mask] - cell_centre) / cell_scale
            per_fold_null[fold_counter] = float(np.sum(null_residual**2))
            per_column_null += np.bincount(column_of_cell, weights=null_residual**2, minlength=p)

            for a in range(1, max_components + 1):
                Xa = Xtr.copy()
                # Convergence is judged on the held-out cells in the space the
                # fold is fitted in, not in the input units. Otherwise a column
                # re-expressed in different units dominates the norm and changes
                # how many EM iterations are taken (#546). Under
                # scale_inside_folds=False the divisor is 1 throughout.
                prev_held = Xa[impute] / np.broadcast_to(col_scale, X.shape)[impute]
                for _iteration in range(n_iter):
                    if scale_inside_folds:
                        # Centre and scale by the FIXED in-fold constants; the
                        # in-fold cells now sit on the analysis scale and the
                        # held-out cells move under EM.
                        Xs = (Xa - col_centre) / col_scale
                        _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
                        rank = min(a, S.shape[0])
                        recon_s = (Xs @ Vt[:rank].T) @ Vt[:rank]
                        recon = recon_s * col_scale + col_centre
                    else:
                        # Prior behaviour: recompute the column mean each
                        # iteration. The mean drifts slightly as the held-out
                        # cells are updated.
                        col_mean = Xa.mean(axis=0)
                        Xc = Xa - col_mean
                        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                        rank = min(a, S.shape[0])
                        recon_centred = (Xc @ Vt[:rank].T) @ Vt[:rank]
                        recon = recon_centred + col_mean
                    Xa[impute] = recon[impute]
                    held = Xa[impute] / np.broadcast_to(col_scale, X.shape)[impute]
                    delta = np.linalg.norm(held - prev_held)
                    scale = max(1.0, float(np.linalg.norm(prev_held)))
                    if delta < tol * scale:
                        break
                    prev_held = held

                error_input_units = X[mask] - Xa[mask]
                error = error_input_units / cell_scale
                per_fold_press[a - 1, fold_counter] = float(np.sum(error**2))
                per_column_press[a - 1] += np.bincount(column_of_cell, weights=error**2, minlength=p)
                press_input_units[a - 1] += float(np.sum(error_input_units**2))
            fold_counter += 1

    # Average over repeats so PRESS stays on the per-cell scale. nansum so the
    # empty-fold NaNs above do not propagate into the total.
    repeats = max(1, n_repeats)
    press = np.nansum(per_fold_press, axis=1) / repeats
    return _EkfPress(
        press=press,
        per_fold_press=per_fold_press,
        per_column_press=per_column_press / repeats,
        null_model_ss=float(np.sum(per_fold_null)) / repeats,
        per_column_null_ss=per_column_null / repeats,
        press_input_units=press_input_units / repeats,
    )


def _ekf_null_reference(
    ekf: _EkfPress,
    X: np.ndarray,
    *,
    scale_inside_folds: bool,
) -> tuple[float, np.ndarray]:
    """Return the sum of squares :math:`Q^2` measures the ekf PRESS against.

    The reference has to be measured the same way PRESS was, or the ratio of the
    two is not a fraction of anything. Under in-fold scaling it therefore comes
    from the fold loop, which is the only place that knows each fold's constants;
    a reference computed out here could only be in the input units, and then a
    single high-variance column would set the whole curve however the folds were
    actually fitted (#546). Under ``scale_inside_folds=False`` there is no
    in-fold scale to match, so the centred total sum of squares of the matrix the
    caller passed stays the reference, unchanged from earlier releases.

    Parameters
    ----------
    ekf : _EkfPress
        What the fold loop measured.
    X : np.ndarray of shape (n_samples, n_features)
        The matrix as it was passed to the cross-validation.
    scale_inside_folds : bool
        Whether each fold standardised the matrix before fitting it.

    Returns
    -------
    tuple[float, np.ndarray]
        The pooled reference, and the same split by variable (shape
        ``(n_features,)``).
    """
    if scale_inside_folds:
        return ekf.null_model_ss, ekf.per_column_null_ss
    centred_ss = np.nansum((X - np.nanmean(X, axis=0)) ** 2, axis=0)
    return float(centred_ss.sum()), centred_ss


def _preprocess_for_cell_schemes(X: np.ndarray, scheme: str) -> np.ndarray:
    """Mean-centre and unit-variance scale the whole matrix, once.

    The schemes below are the classical ones, and all of them are defined on a
    single preprocessed matrix rather than on per-fold constants: there are no
    folds to fit constants on in ``"sacv"`` and ``"gcv"``, and the two SVD
    families of ``"ek"`` are both taken from the same preprocessed matrix. That
    is what ``FactoMineR`` and ``pcaMethods`` do, so a number computed here is
    comparable with theirs.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Raw data matrix. Missing cells are not supported by these schemes.
    scheme : str
        Name of the calling scheme, used only in the error message.

    Returns
    -------
    np.ndarray
        The centred, unit-variance scaled matrix.

    Raises
    ------
    ValueError
        If ``X`` holds missing cells. These schemes factorise ``X`` directly,
        and an SVD cannot see a NaN.
    """
    if np.isnan(X).any():
        raise ValueError(
            f"cv_scheme={scheme!r} cannot take a block with missing cells, because it "
            "factorises the matrix directly. Use cv_scheme='ekf', which imputes an "
            "unmeasured cell alongside the held-out ones and never scores it."
        )
    centre = X.mean(axis=0)
    spread = X.std(axis=0, ddof=1)
    spread[spread < epsqrt] = 1.0
    return (X - centre) / spread


def _leverage_corrected_press(
    X: np.ndarray,
    max_components: int,
    *,
    method: typing.Literal["sacv", "gcv"],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Leave-one-cell-out prediction error, approximated without refitting anything.

    Holding out one cell at a time and refitting costs :math:`n \times p` fits per
    component count. Josse and Husson (2012) avoid all of them: fit once, then
    inflate the ordinary residual by the leverage of the cell that produced it,
    the same device that turns a regression residual into its
    leave-one-out counterpart via :math:`e_i / (1 - h_{ii})`.

    With :math:`\mathbf{U}`, :math:`\mathbf{V}` the first ``a`` left and right
    singular vectors, the row and column leverages are
    :math:`a_i = \sum_k u_{ik}^2` and :math:`b_j = \sum_k v_{jk}^2`, and

    - ``"sacv"`` divides each residual by :math:`(1 - 1/n - a_i)(1 - b_j)`,
      cell by cell. This is the *smoothing approximation* of the
      cross-validation criterion.
    - ``"gcv"`` replaces both leverages by one averaged constant,
      :math:`np / [(n - 1)p - a(n + p - a - 1)]`, where the subtracted term
      counts the free parameters of a centred rank-``a`` bilinear model. This
      is *generalised* cross-validation, and it is the cheaper, blunter cousin.

    Neither holds any data out. They correct the circularity analytically
    instead, which is why they cost one SVD in total rather than one per fold.

    Both are first-order approximations and both degrade as ``a`` approaches
    the number of variables. A column's leverage is the share of its variance
    inside the retained subspace, so it reaches one when the components reach
    the variables, and the divisor reaches zero with it. Read these criteria
    over a component count well below the number of variables; ``FactoMineR``
    defaults to five for the same reason.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix, already centred and scaled.
    max_components : int
        Largest component count to evaluate; the criterion is returned for
        ``1 .. max_components``.
    method : {"sacv", "gcv"}
        Which of the two corrections to apply.

    Returns
    -------
    press : np.ndarray of shape (max_components,)
        The corrected sum of squared residuals per component count.
    per_column_press : np.ndarray of shape (max_components, n_features)
        The same total, split by variable.
    null_model_ss : float
        What predicting every cell by its column mean gets wrong. Since ``X``
        arrives centred, that is its total sum of squares.
    per_column_null_ss : np.ndarray of shape (n_features,)
        That reference, split by variable.

    References
    ----------
    Josse, J., & Husson, F. (2012). Selecting the number of components in
    principal component analysis using cross-validation approximations.
    *Computational Statistics & Data Analysis*, 56(6), 1869-1879.
    """
    n, p = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    press = np.full(max_components, np.nan)
    per_column_press = np.full((max_components, p), np.nan)

    for a in range(1, max_components + 1):
        rank = min(a, S.shape[0])
        residual = X - (U[:, :rank] * S[:rank]) @ Vt[:rank]
        if method == "sacv":
            row_leverage = np.sum(U[:, :rank] ** 2, axis=1)
            column_leverage = np.sum(Vt[:rank] ** 2, axis=0)
            denominator = np.outer(1.0 - 1.0 / n - row_leverage, 1.0 - column_leverage)
            # A cell whose leverage reaches one is reconstructed entirely by
            # itself, so its leave-one-out residual is not defined. Drop it
            # rather than let a division by ~0 dominate the total.
            usable = denominator > epsqrt
            corrected = np.where(usable, residual / np.where(usable, denominator, 1.0), np.nan)
        else:
            remaining = (n - 1) * p - a * (n + p - a - 1)
            if remaining <= 0:
                # More free parameters than degrees of freedom: the criterion
                # has nothing left to measure against.
                continue
            corrected = residual * (n * p / remaining)
        press[a - 1] = float(np.nansum(corrected**2))
        per_column_press[a - 1] = np.nansum(corrected**2, axis=0)

    return press, per_column_press, float(np.sum(X**2)), np.sum(X**2, axis=0)


def _eastment_krzanowski_press(
    X: np.ndarray,
    max_components: int,
    *,
    n_folds: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r"""Cross-validated prediction error by the two-model scheme of Eastment and Krzanowski.

    An element is predicted by a score taken from a model that never saw its
    **column** and a loading taken from a model that never saw its **row**, so
    :math:`x_{ij}` enters neither decomposition:

    .. math::
        \hat{x}_{ij} = \sum_{k=1}^{a} u^{(-j)}_{ik}\sqrt{d^{(-j)}_k}\,
                       \sqrt{d^{(-i)}_k}\, v^{(-i)}_{jk}

    This is what Simca-P reports as its PCA :math:`Q^2`, and what
    ``pcaMethods::Q2(type = "krzanowski")`` computes in R, so it is the scheme
    to use when a number has to line up with either of those.

    Rows and columns are each split into ``n_folds`` groups rather than deleted
    one at a time, which costs ``2 * n_folds`` decompositions instead of
    ``n + p``. The sign of every fold model's singular vectors is aligned to the
    full-data fit, because an SVD fixes each vector only up to sign and two fold
    models must agree before their product means anything.

    One caveat, which the classical scheme and the reference implementations
    share: centring and scaling are fitted once on the whole matrix, so a cell
    reaches the fold models through its own column's centre and spread even
    though it is absent from both decompositions. That path is of order
    :math:`1/n` and vanishes as rows are added, but it is not nothing, and it is
    why this scheme is not offered as the default. ``"ekf"`` fits its constants
    inside each fold and has no such path.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix, already centred and scaled, with no missing cells.
    max_components : int
        Largest component count to evaluate.
    n_folds : int
        Number of row groups, and of column groups.
    random_state : int, optional
        Seed for the row and column permutations.

    Returns
    -------
    press : np.ndarray of shape (max_components,)
        Prediction error sum of squares per component count.
    per_column_press : np.ndarray of shape (max_components, n_features)
        The same total, split by variable.
    null_model_ss : float
        What predicting every cell by its column mean gets wrong.
    per_column_null_ss : np.ndarray of shape (n_features,)
        That reference, split by variable.

    References
    ----------
    Eastment, H. T., & Krzanowski, W. J. (1982). Cross-validatory choice of the
    number of components from a principal component analysis. *Technometrics*,
    24(1), 73-77.
    """
    n, p = X.shape
    rng = np.random.default_rng(random_state)
    row_group = np.array_split(rng.permutation(n), n_folds)
    column_group = np.array_split(rng.permutation(p), n_folds)

    _, _, Vt_full = np.linalg.svd(X, full_matrices=False)
    U_full, _, _ = np.linalg.svd(X, full_matrices=False)

    def aligned(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Flip each column whose direction opposes the full-data fit."""
        width = min(vectors.shape[1], reference.shape[1])
        signs = np.sign(np.sum(vectors[:, :width] * reference[:, :width], axis=0))
        signs[signs == 0] = 1.0
        return vectors[:, :width] * signs

    # Loadings from models that never saw the rows of their group.
    loadings: list[np.ndarray] = []
    row_singular: list[np.ndarray] = []
    for rows in row_group:
        kept = np.setdiff1d(np.arange(n), rows)
        _, S_r, Vt_r = np.linalg.svd(X[kept], full_matrices=False)
        loadings.append(aligned(Vt_r.T, Vt_full.T))
        row_singular.append(S_r)

    # Scores from models that never saw the columns of their group.
    scores: list[np.ndarray] = []
    column_singular: list[np.ndarray] = []
    for columns in column_group:
        kept = np.setdiff1d(np.arange(p), columns)
        U_c, S_c, _ = np.linalg.svd(X[:, kept], full_matrices=False)
        scores.append(aligned(U_c, U_full))
        column_singular.append(S_c)

    available = min(
        *(v.shape[1] for v in loadings),
        *(u.shape[1] for u in scores),
        *(s.size for s in row_singular),
        *(s.size for s in column_singular),
    )
    if available < max_components:
        warnings.warn(
            f"cv_scheme='ek' can evaluate at most {available} component(s) with "
            f"{n_folds} folds on a {n} by {p} block, not {max_components}: each fold "
            "model is fitted on fewer rows or columns than the whole. Component counts "
            "above that are reported as NaN. Use fewer folds to evaluate more.",
            SpecificationWarning,
            stacklevel=2,
        )

    press = np.full(max_components, np.nan)
    per_column_press = np.full((max_components, p), np.nan)
    usable = min(available, max_components)
    for a in range(1, usable + 1):
        prediction = np.zeros_like(X)
        for rows, V_r, S_r in zip(row_group, loadings, row_singular, strict=True):
            for columns, U_c, S_c in zip(column_group, scores, column_singular, strict=True):
                block = np.ix_(rows, columns)
                weight = np.sqrt(S_c[:a]) * np.sqrt(S_r[:a])
                prediction[block] = (U_c[rows, :a] * weight) @ V_r[columns, :a].T
        squared = (X - prediction) ** 2
        press[a - 1] = float(np.sum(squared))
        per_column_press[a - 1] = np.sum(squared, axis=0)

    return press, per_column_press, float(np.sum(X**2)), np.sum(X**2, axis=0)


class PCA(_LatentVariableModel, TransformerMixin, BaseEstimator):
    """Principal Component Analysis with support for missing data.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to extract. ``None`` asks for as many
        components as the data supports: it is resolved at fit time to
        ``min(n_samples, n_features)``, which is also the ceiling an explicit
        request is clamped to (with a ``SpecificationWarning``). The resolved
        count is available on the fitted attribute ``n_components_``.

    algorithm : str, default="auto"
        Algorithm to use for fitting the model.
        - ``"auto"``: Uses SVD when data is complete, NIPALS when data has missing values.
        - ``"svd"``: Singular Value Decomposition. Requires complete data.
        - ``"nipals"``: Non-linear Iterative Partial Least Squares. Handles missing data.
        - ``"tsr"``: Trimmed Score Regression. Handles missing data.

    missing_data_settings : dict or None, default=None
        Settings for iterative missing data algorithms (NIPALS, TSR).
        Keys: ``md_tol`` (relative convergence tolerance on successive score
        vectors; see :func:`terminate_check`), ``md_max_iter`` (max iterations).

    Attributes (after fitting)
    --------------------------
    n_components_ : int
        The resolved number of components actually fitted (the constructor
        parameter clamped to ``min(n_samples, n_features)``; the parameter
        itself is left as the user set it, including ``None``).
    scores_ : pd.DataFrame of shape (n_samples, n_components)
        The score matrix (T).
    loadings_ : pd.DataFrame of shape (n_features, n_components)
        The loading matrix (P).
    r2_per_component_ : pd.Series of length n_components
        Fractional R² explained by each component.
    r2_cumulative_ : pd.Series of length n_components
        Cumulative R² after each component.
    r2_per_variable_ : pd.DataFrame of shape (n_features, n_components)
        Per-variable cumulative R² after each component.
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
    explained_variance_ : np.ndarray of shape (n_components,)
        Variance explained by each component.
    scaling_factor_for_scores_ : pd.Series of length n_components
        Standard deviation per score (sqrt of explained variance).
    has_missing_data_ : bool
        Whether the training data contained missing values.
    fitting_info_ : dict
        Timing and iteration info from the fitting algorithm.
    """

    _valid_algorithms: typing.ClassVar[list[str]] = ["auto", "svd", "nipals", "tsr"]

    _parameter_constraints: typing.ClassVar = {
        "n_components": [int, None],
        "algorithm": [str],
        "missing_data_settings": [dict, None],
    }

    def __init__(
        self,
        n_components: int,
        *,
        algorithm: str = "auto",
        missing_data_settings: dict | None = None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.missing_data_settings = missing_data_settings

    # ENG-17: the convenience methods (score_plot, vip, spe_limit, ...),
    # hotellings_t2_limit, ellipse_coordinates and the rename __getattr__ are
    # inherited from _LatentVariableModel. PCA supplies only its rename map.
    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {
        "x_scores": "scores_",
        "loadings": "loadings_",
        "x_loadings": "loadings_",
        "squared_prediction_error": "spe_",
        "R2": "r2_per_component_",
        "R2cum": "r2_cumulative_",
        "R2X_cum": "r2_per_variable_",
        "hotellings_t2": "hotellings_t2_",
        "scaling_factor_for_scores": "scaling_factor_for_scores_",
        "N": "n_samples_",
        "K": "n_features_in_",
        "A": "n_components",
        "extra_info": "fitting_info_",
    }
    _RENAME_CONTEXT: typing.ClassVar[str] = "PCA"

    # Fitted diagnostics: per-component arrays (NIPALS/TSR) or scalar totals (SVD).
    fitting_info_: dict[str, np.ndarray | int | float]

    # ENG-18: public DataFrame views built lazily from the private ndarrays.
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")
    loadings_ = _LazyFrame("_loadings", index="_feature_names", columns="_component_names")
    spe_ = _LazyFrame("_spe", index="_sample_index", columns="_component_names")

    def __sklearn_tags__(self):
        """Declare sklearn capability tags (sklearn 1.6+).

        ``allow_nan=True`` because the NIPALS / TSR algorithm paths
        thread missing data through; the SVD path raises explicitly
        when NaN is supplied. The default for any one ``PCA(...)``
        instance therefore allows NaN at the Pipeline / ``check_array``
        boundary and lets the fitting algorithm reject it later if
        needed (the user-facing error message is more informative than
        the sklearn check-array one).
        """
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ANN001, ARG002
        """Return the output column names of :meth:`transform`.

        :class:`PCA`'s ``transform`` produces scores, one column per
        component, named ``["PC1", "PC2", ..., "PC{n_components}"]``.
        The ``input_features`` argument is accepted (Pipeline
        introspection passes it through) but unused: the output column
        count is the fitted ``n_components``, not the input feature
        count.

        Used by :meth:`set_output` (sklearn 1.2+) to label the
        :class:`~pandas.DataFrame` view of the scores when
        ``set_output(transform="pandas")`` is on, and by Pipeline
        introspection.
        """
        check_is_fitted(self, "loadings_")
        return np.asarray([f"PC{a}" for a in self._component_names])

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataMatrix, y: DataMatrix | None = None) -> PCA:  # noqa: ARG002, PLR0912, PLR0915, C901
        """Fit a principal component analysis (PCA) model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. May contain NaN values for missing data (the
            NIPALS / TSR algorithms thread them through; the SVD path
            rejects them).
        y : ignored

        Returns
        -------
        self : PCA
        """
        # Capture the original DataFrame's index/columns before validate_data
        # converts X to an ndarray, then rebuild the DataFrame view downstream
        # code expects. validate_data also sets n_features_in_ / feature_
        # names_in_ and runs the sklearn input rejections (sparse, complex,
        # empty, dtype-object) with the standard error messages.
        sample_index = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if feature_columns is None:
            feature_columns = pd.RangeIndex(X_arr.shape[1])  # type: ignore[assignment]
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)

        N, K = X.shape
        self.n_samples_ = N
        # n_features_in_ already set by validate_data; reassert for clarity.
        self.n_features_in_ = K
        self._feature_names = X.columns
        self._sample_index = X.index

        # Clamp n_components
        min_dim = int(min(N, K))
        if self.n_components is not None and int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1; got {self.n_components}.")
        A = min_dim if self.n_components is None else int(self.n_components)
        if min_dim < A:
            warnings.warn(
                "The requested number of components is more than can be "
                "computed from data. The maximum number of components is "
                f"the minimum of either the number of rows ({N}) or "
                f"the number of columns ({K}).",
                SpecificationWarning,
                stacklevel=2,
            )
            A = min_dim
        # The resolved (possibly clamped) component count is a fitted attribute;
        # the constructor parameter n_components is left exactly as the user set
        # it, per the sklearn clone/get_params contract (#505).
        self.n_components_ = A

        # Detect missing data and resolve algorithm
        self.has_missing_data_ = bool(np.any(X.isna()))
        algo = self.algorithm.lower()
        if algo not in self._valid_algorithms:
            raise ValueError(
                f"Algorithm '{self.algorithm}' is not recognized. Must be one of {self._valid_algorithms}."
            )

        if algo == "auto":
            algo = "nipals" if self.has_missing_data_ else "svd"
        if algo == "svd" and self.has_missing_data_:
            raise ValueError("SVD algorithm cannot handle missing data. Use 'nipals', 'tsr', or 'auto'.")
        self.algorithm_ = algo

        # Build settings for iterative algorithms
        settings = {"md_tol": epsqrt, "md_max_iter": 1000}
        if isinstance(self.missing_data_settings, dict):
            settings.update(self.missing_data_settings)
        settings["md_max_iter"] = int(settings["md_max_iter"])

        if algo in ("nipals", "tsr"):
            if not settings["md_tol"] < 10:
                raise ValueError("Tolerance should not be too large.")
            if not settings["md_tol"] > epsqrt**1.95:
                raise ValueError("Tolerance must exceed machine precision.")

        # Storage for numpy results (set by _fit_* methods)
        X_values = np.asarray(X.copy())

        # Dispatch
        if algo == "svd":
            self._fit_svd(X_values, N, K, A)
        elif algo == "nipals":
            self._fit_nipals(X_values, N, K, A, settings)
        elif algo == "tsr":
            self._fit_tsr(X_values, N, K, A, settings)

        # --- Common post-fit path ---
        # ENG-18: scores_ / loadings_ / spe_ are stored as private ndarrays (the
        # source of truth); the public DataFrame views are built lazily by the
        # _LazyFrame descriptors from these arrays plus the index/column metadata
        # (self._sample_index, self._feature_names, self._component_names).
        self._component_names = list(range(1, A + 1))
        component_names = self._component_names
        self._loadings = self._loadings_np
        self._scores = self._scores_np
        self._spe = self._spe_np

        self.r2_per_component_ = pd.Series(
            self._r2_np,
            index=component_names,
            name="R² per component",
        )
        self.r2_cumulative_ = pd.Series(
            self._r2cum_np,
            index=component_names,
            name="Cumulative R²",
        )
        self.r2_per_variable_ = pd.DataFrame(
            self._r2_per_var_np,
            index=self._feature_names,
            columns=component_names,
        )

        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )

        # Hotelling's T² (cumulative across components). A component whose
        # score standard deviation is ~0 (a rank-deficient fit, e.g. asking
        # for min(N, K) components from mean-centred data, whose rank is at
        # most min(N-1, K)) carries no T² information: its standardized score
        # is 0/0. Skip such components rather than dividing by ~zero, which
        # previously poisoned T² (and everything downstream of it) with
        # inf/NaN for every observation.
        self.hotellings_t2_ = pd.DataFrame(
            np.zeros((N, A)),
            columns=component_names,
            index=self._sample_index,
        )
        scaling_factors = self.scaling_factor_for_scores_.to_numpy()
        t2_tol = epsqrt * max(1.0, float(np.max(scaling_factors, initial=0.0)))
        for a in range(A):
            contribution = (self._scores[:, a] / scaling_factors[a]) ** 2 if scaling_factors[a] > t2_tol else 0.0
            self.hotellings_t2_.iloc[:, a] = self.hotellings_t2_.iloc[:, max(0, a - 1)] + contribution

        # Clean up temporary numpy staging names (the kept ndarrays above alias
        # the same arrays, so they survive these del statements).
        del self._loadings_np, self._scores_np, self._r2_np, self._r2cum_np, self._r2_per_var_np, self._spe_np

        return self

    def _fit_svd(self, X_values: np.ndarray, N: int, K: int, A: int) -> None:
        """Fit PCA using SVD decomposition (complete data only)."""
        U, S, Vt = np.linalg.svd(X_values, full_matrices=False)

        # Loadings are the first A right singular vectors (transposed to K x A)
        self._loadings_np = Vt[:A, :].T
        # Scores are U * S for the first A components
        self._scores_np = U[:, :A] * S[:A]

        # Sign convention: flip so largest magnitude element in each loading is positive
        # (Wold, Esbensen, Geladi, PCA, CILS, 1987, p 42)
        for a in range(A):
            max_el_idx = np.argmax(np.abs(self._loadings_np[:, a]))
            if self._loadings_np[max_el_idx, a] < 0:
                self._loadings_np[:, a] *= -1.0
                self._scores_np[:, a] *= -1.0

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

        # Compute R2 and SPE via deflation
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))

        Xd = X_values.copy()
        prior_ssx_col = ssq(Xd, axis=0)
        base_variance = np.sum(prior_ssx_col)

        for a in range(A):
            Xd = Xd - self._scores_np[:, [a]] @ self._loadings_np[:, [a]].T
            row_ssx = ssq(Xd, axis=1)
            col_ssx = ssq(Xd, axis=0)

            self._spe_np[:, a] = np.sqrt(row_ssx)
            # Per-variable R^2 is undefined for a column with no variance to
            # explain; emit NaN there instead of letting RuntimeWarning
            # ``invalid value encountered in divide`` poison the output.
            # SEC-21 (#270) sub-item 4.
            self._r2_per_var_np[:, a] = np.where(
                prior_ssx_col > 0, 1 - col_ssx / np.where(prior_ssx_col > 0, prior_ssx_col, 1.0), np.nan
            )
            self._r2cum_np[a] = 1 - np.sum(row_ssx) / base_variance if base_variance > 0 else np.nan
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]

        self.fitting_info_ = {"timing": np.zeros(A) * np.nan, "iterations": np.zeros(A) * np.nan}

    def _fit_nipals(self, X_values: np.ndarray, N: int, K: int, A: int, settings: dict) -> None:
        """Fit PCA using the NIPALS algorithm (handles missing data)."""
        Xd = X_values.copy()
        base_variance = ssq(Xd)
        base_ss_col = ssq(Xd, axis=0)  # the denominator of the cumulative per-variable R2, fixed before deflation

        self._loadings_np = np.zeros((K, A))
        self._scores_np = np.zeros((N, A))
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))
        self.fitting_info_ = {"timing": np.zeros(A) * np.nan, "iterations": np.zeros(A) * np.nan}

        for a in np.arange(A):
            start_time = time.time()
            itern = 0
            start_ss_col = ssq(Xd, axis=0)

            if np.sum(start_ss_col) < epsqrt:
                emsg = (
                    "There is no variance left in the data array: cannot "
                    f"compute any more components beyond component {a}."
                )
                raise NotEnoughVarianceError(emsg)

            # Seed the score from the column of X with the greatest
            # sum-of-squares (variance, for mean-centred data) rather than the
            # arbitrary first column (#195). NIPALS converges to the same
            # component for any non-degenerate seed, but the highest-variance
            # column is closest to the leading component, so it needs fewer
            # iterations and is far more robust when the first column happens to
            # be near-orthogonal to it. The deterministic sign convention applied
            # below makes the fitted sign independent of this seed.
            #
            # ``Xd[:, [start_col]]`` (fancy indexing) already returns a copy in
            # current numpy, so the in-place ``isnan -> 0`` does not poison Xd
            # today. The explicit ``.copy()`` here is defensive: it mirrors the
            # PLS path and protects against any future numpy change that flips
            # fancy indexing to a view-returning variant. SEC-21 (#270) sub-item 2.
            start_col = int(np.argmax(start_ss_col))
            t_a_guess = Xd[:, [start_col]].copy()
            t_a_guess[np.isnan(t_a_guess)] = 0
            t_a = t_a_guess + 1.0
            p_a = np.zeros((K, 1))
            # ``itern == 0`` forces at least one NIPALS iteration. The ``+ 1.0``
            # offset that primes the loop can be negligible relative to a
            # large-magnitude seed column, and the relative criterion (#504)
            # would then report convergence at entry, silently returning the
            # unrefined seed as the score and the zero ``p_a`` as the loading.
            while itern == 0 or not terminate_check(t_a_guess, t_a, iterations=itern, settings=settings):
                t_a_guess = t_a.copy()

                # Regress X onto t_a to get loadings p_a
                p_a = quick_regress(Xd, t_a)
                p_a = p_a / np.sqrt(ssq(p_a))

                # Regress X onto p_a to get scores t_a
                t_a = quick_regress(Xd, p_a)

                itern += 1

            timing_arr = typing.cast("np.ndarray", self.fitting_info_["timing"])
            iterations_arr = typing.cast("np.ndarray", self.fitting_info_["iterations"])
            timing_arr[a] = time.time() - start_time
            iterations_arr[a] = itern
            # terminate_check stops the loop at ``iterations >= md_max_iter``;
            # warn on non-convergence (previously the PCA path was silent).
            if itern >= settings["md_max_iter"]:
                warnings.warn(
                    f"PCA NIPALS: component {a + 1} reached the maximum number of "
                    f"iterations ({settings['md_max_iter']}) without converging.",
                    SpecificationWarning,
                    stacklevel=2,
                )
            logger.debug(
                "PCA NIPALS: component %d converged in %d iterations (md_tol=%g)",
                a + 1,
                itern,
                settings["md_tol"],
            )

            # Deflate
            Xd = Xd - np.dot(t_a, p_a.T)
            row_ssx = ssq(Xd, axis=1)
            col_ssx = ssq(Xd, axis=0)

            self._spe_np[:, a] = np.sqrt(row_ssx)
            # Per-variable R^2 is undefined for a column with no variance to
            # explain; emit NaN there. SEC-21 (#270) sub-item 4.
            self._r2_per_var_np[:, a] = np.where(
                base_ss_col > 0, 1 - col_ssx / np.where(base_ss_col > 0, base_ss_col, 1.0), np.nan
            )
            self._r2cum_np[a] = 1 - np.sum(row_ssx) / base_variance if base_variance > 0 else np.nan
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]

            # Sign convention: largest magnitude element in loading is positive
            max_el_idx = np.argmax(np.abs(p_a))
            if np.sign(p_a[max_el_idx]) < 1:
                p_a *= -1.0
                t_a *= -1.0

            self._loadings_np[:, a] = p_a.flatten()
            self._scores_np[:, a] = t_a.flatten()

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

    def _fit_tsr(self, X_values: np.ndarray, N: int, K: int, A: int, settings: dict) -> None:
        """Fit PCA using the Trimmed Score Regression algorithm (handles missing data).

        See papers by Abel Folch-Fortuny and also DOI: 10.1002/cem.750
        """
        start_time = time.time()
        delta = 1e100
        Xd = X_values.copy()
        X_original = X_values.copy()
        base_variance = ssq(Xd)

        mmap = np.isnan(Xd)
        Xd[mmap] = 0.0
        itern = 0
        # No missing cells means nothing to impute: skip the EM loop entirely
        # (previously ``np.mean`` over the empty ``Xd[mmap]`` emitted a
        # RuntimeWarning every iteration and produced a NaN delta).
        while np.any(mmap) and (itern < settings["md_max_iter"]) and (delta > settings["md_tol"]):
            itern += 1
            missing_X = Xd[mmap]
            mean_X = np.mean(Xd, axis=0)
            S = np.cov(Xd, rowvar=False, ddof=1)
            Xc = Xd - mean_X
            # Both branches must end with V of shape (K, A): rows indexed by
            # feature, columns by component. svd(Xc) returns the loadings as
            # the ROWS of Vt (so transpose), while svd(Xc.T) returns them as
            # the COLUMNS of U (already oriented; do not transpose). The
            # previous shared ``V = V.T[:, 0:A]`` transposed the second branch
            # wrongly: an IndexError for N < K, and a silently wrong
            # imputation regression for N == K.
            if N > K:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                V = Vt.T[:, 0:A]
            else:
                U, _, _ = np.linalg.svd(Xc.T, full_matrices=False)
                V = U[:, 0:A]
            for n in range(N):
                row_mis = mmap[n, :]
                row_obs = ~row_mis
                if np.any(row_mis):
                    L = V[row_obs, 0 : min(A, sum(row_obs))]
                    S11 = S[row_obs, :][:, row_obs]
                    S21 = S[row_mis, :][:, row_obs]
                    z2 = (S21 @ L) @ np.linalg.pinv(L.T @ S11 @ L) @ L.T
                    Xc[n, row_mis] = z2 @ Xc[n, row_obs]
            Xd = Xc + mean_X
            delta = np.mean((Xd[mmap] - missing_X) ** 2)

        # Final decomposition
        S = np.cov(Xd, rowvar=False, ddof=1)
        _, _, V = np.linalg.svd(S, full_matrices=False)

        self._loadings_np = (V[0:A, :]).T  # K x A

        # Sign convention: flip so the largest-magnitude element in each
        # loading is positive, matching _fit_svd / _fit_nipals (previously
        # the TSR loading signs were LAPACK-dependent).
        for a in range(A):
            max_el_idx = np.argmax(np.abs(self._loadings_np[:, a]))
            if self._loadings_np[max_el_idx, a] < 0:
                self._loadings_np[:, a] *= -1.0

        # Project the imputed matrix as-is (no extra centring), exactly as
        # transform() will project new data: the package convention is that X
        # is preprocessed (e.g. MCUVScaler) before fit. The previous code
        # centred here, so scores_ disagreed with transform(X) on the very
        # same data, and the residuals below mixed centred reconstructions
        # with uncentred data (inflating SPE and deflating R2).
        self._scores_np = Xd @ self._loadings_np

        # R2 and SPE
        self._r2_np = np.zeros(A)
        self._r2cum_np = np.zeros(A)
        self._r2_per_var_np = np.zeros((K, A))
        self._spe_np = np.zeros((N, A))

        base_ss_col = ssq(X_original, axis=0)
        for a in range(A):
            residuals = self._scores_np[:, : a + 1] @ self._loadings_np[:, : a + 1].T - X_original
            self._r2cum_np[a] = 1 - ssq(residuals, axis=None) / base_variance
            self._r2_np[a] = self._r2cum_np[a] - self._r2cum_np[a - 1] if a > 0 else self._r2cum_np[a]
            self._r2_per_var_np[:, a] = np.where(
                base_ss_col > 0, 1 - ssq(residuals, axis=0) / np.where(base_ss_col > 0, base_ss_col, 1.0), np.nan
            )
            self._spe_np[:, a] = np.sqrt(ssq(residuals, axis=1))

        self.fitting_info_ = {"iterations": itern, "timing": time.time() - start_time}

        # Explained variance. ``max(1, N-1)`` mirrors the MBPLS / MBPCA
        # paths and prevents a division-by-zero / negative-divisor when
        # the caller fits a model on a single row. SEC-21 (#270) sub-item 6.
        self.explained_variance_ = np.diag(self._scores_np.T @ self._scores_np) / max(1, N - 1)

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Project new data onto the fitted PCA model to obtain scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to project. Must have the same number of features as
            the training data.

        Returns
        -------
        scores : pd.DataFrame of shape (n_samples, n_components)
        """
        check_is_fitted(self, "loadings_")
        # sklearn's validate_data(reset=False) checks feature-name *order*
        # strictly, while _align_to_fit_features only needs set equality.
        # Realign reordered DataFrame columns first so validate_data sees a
        # name-ordered view; ndarrays / un-named DataFrames pass straight
        # through.
        if isinstance(X, pd.DataFrame):
            X = _align_to_fit_features(X, self._feature_names)
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        scores = X_arr @ self._loadings
        return pd.DataFrame(scores, index=sample_index, columns=self._component_names)

    def fit_transform(self, X: DataMatrix, y: DataMatrix | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Fit the model and return the training scores."""
        self.fit(X)
        return self.scores_

    def diagnose(self, X: DataMatrix) -> Bunch:
        """Project new data and compute diagnostics (scores, Hotelling's T², SPE).

        The same logic that historically lived in :meth:`predict`. The rename
        (since 1.38.1, #396) matches :meth:`PLS.diagnose` and clears the
        ``predict`` name for a future return-type contract that does what its
        sklearn-convention name implies (a regression-style prediction).
        :meth:`predict` is kept as a deprecation shim for now.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys ``scores``, ``hotellings_t2``, ``spe``.
        """
        check_is_fitted(self, "loadings_")
        # diagnose() delegates to transform() which already runs validate_data;
        # call it once here so the rest can work with the aligned
        # DataFrame view (and so the validate_data error pathways fire on
        # the diagnose() call directly, not on the recursive transform() one).
        scores = self.transform(X)
        # transform's output is indexed by the validated X's row index;
        # recover an aligned DataFrame for the diagnostics below.
        sample_index: pd.Index = X.index if isinstance(X, pd.DataFrame) else scores.index
        feature_columns: pd.Index = X.columns if isinstance(X, pd.DataFrame) else self._feature_names
        X = pd.DataFrame(np.asarray(X, dtype=float), index=sample_index, columns=feature_columns)
        X = _align_to_fit_features(X, self._feature_names)

        # Hotelling's T² (cumulative)
        component_names = self._component_names
        t2 = pd.DataFrame(np.zeros((X.shape[0], self.n_components_)), columns=component_names, index=X.index)
        for a in range(self.n_components_):
            t2.iloc[:, a] = (
                t2.iloc[:, max(0, a - 1)] + (scores.iloc[:, a] / self.scaling_factor_for_scores_.iloc[a]) ** 2
            )

        # SPE: residual after reconstruction
        X_hat = scores.values @ self._loadings.T
        residuals = X.values - X_hat
        spe_values = pd.Series(np.sqrt(np.sum(residuals**2, axis=1)), index=X.index, name="SPE")

        return Bunch(scores=scores, hotellings_t2=t2, spe=spe_values)

    def predict(self, X: DataMatrix) -> Bunch:
        """Forward to :meth:`diagnose`; emits a :class:`DeprecationWarning`.

        .. deprecated:: 1.38.1
            Use :meth:`PCA.diagnose` instead. ``predict`` matches the
            sklearn-convention name (a regression-style prediction), but PCA
            isn't a regressor; the historical return is a diagnostics Bunch.
            The rename aligns with :meth:`PLS.diagnose` and frees the name
            for a future contract. Will be removed in 2.0.0.
        """
        warnings.warn(
            "PCA.predict is deprecated and will be removed in 2.0.0; use PCA.diagnose instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.diagnose(X)

    def project(self, X: DataMatrix, *, method: str = "tsr", ridge: float = 0.0) -> Bunch:
        """Estimate scores and diagnostics for rows that may contain missing values.

        Whereas :meth:`transform` and :meth:`diagnose` propagate NaN, this
        method estimates the scores of partially-observed rows from the
        observed columns only, using the missing-data estimators of Arteaga
        and Ferrer (2002): trimmed score regression (``"tsr"``, the default
        and the statistically strongest), single-component projection
        (``"scp"``), or projection to the model plane (``"pmp"``). Rows with
        no missing values take the standard complete-data path, so their
        scores are bitwise identical to :meth:`transform`.

        This is the "batch so far" primitive of online batch monitoring: the
        future part of an unfolded batch row is missing by construction, and
        the score estimate at each time sample is this projection (see
        Garcia-Munoz, Kourti and MacGregor, 2004).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data in the model's (centred and scaled) space; NaN marks a
            missing entry. Rows that are entirely NaN are rejected.
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
            With keys ``scores`` (DataFrame, n_samples x n_components),
            ``hotellings_t2`` (Series; total over all components, computed
            with the training score variances), ``spe`` (Series; the square
            root of the residual sum of squares over the *observed* columns
            only), ``condition_number`` (Series; the conditioning diagnostic
            of each row's estimator, 1.0 when nothing is missing) and
            ``n_observed`` (Series; observed features per row). SPE and T2
            for a partially-observed row must be compared against limits
            built from the same missingness pattern, not against the
            full-observation limits; see
            :class:`process_improve.batch.BatchMonitor`.
        """
        check_is_fitted(self, "loadings_")
        if isinstance(X, pd.DataFrame):
            X = _align_to_fit_features(X, self._feature_names)
        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])  # type: ignore[assignment]
        raw = project_rows(
            self._loadings,
            self._loadings,
            np.asarray(self.explained_variance_, dtype=float),
            np.asarray(X_arr, dtype=float),
            method=method,
            ridge=ridge,
        )
        scores = pd.DataFrame(raw.scores, index=sample_index, columns=self._component_names)
        s = self.scaling_factor_for_scores_.to_numpy(dtype=float)
        t2 = pd.Series(np.sum((raw.scores / s) ** 2, axis=1), index=sample_index, name="Hotelling's T2")
        return Bunch(
            scores=scores,
            hotellings_t2=t2,
            spe=pd.Series(raw.spe, index=sample_index, name="SPE"),
            condition_number=pd.Series(raw.condition_number, index=sample_index, name="condition_number"),
            n_observed=pd.Series(raw.n_observed, index=sample_index, name="n_observed"),
        )

    def projection_matrix(self, observed: object, *, method: str = "tsr", ridge: float = 0.0) -> Bunch:
        """Build the fixed linear operator mapping observed columns to score estimates.

        For a fixed missingness pattern, every estimator in :meth:`project`
        is a fixed linear map ``t_hat = M @ z_observed``. This method exposes
        that matrix so callers that reuse one pattern many times (an online
        monitor at time sample ``k``, or an optimiser treating candidate
        columns as observed) can precompute it once.

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
        check_is_fitted(self, "loadings_")
        mask = coerce_observed_mask(observed, self._feature_names)
        op = operator_for_pattern(
            self._loadings,
            self._loadings,
            np.asarray(self.explained_variance_, dtype=float),
            mask,
            method=method,
            ridge=ridge,
        )
        matrix = pd.DataFrame(
            op.matrix,
            index=self._component_names,
            columns=pd.Index(self._feature_names)[mask],
        )
        return Bunch(matrix=matrix, condition_number=op.condition_number, method=op.method)

    def score(self, X: DataMatrix, y: DataMatrix | None = None) -> float:  # noqa: ARG002
        """Negative mean squared reconstruction error (higher is better).

        Follows the sklearn convention where higher scores indicate better
        model fit. This makes PCA compatible with ``cross_val_score``,
        ``GridSearchCV``, and other sklearn model-selection utilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data to score.
        y : ignored

        Returns
        -------
        score : float
            Negative mean squared reconstruction error.

        Examples
        --------
        >>> from sklearn.model_selection import cross_val_score
        >>> scores = cross_val_score(PCA(n_components=2), X_scaled, cv=5)
        >>> print(f"Mean CV score: {scores.mean():.4f}")
        """
        check_is_fitted(self, "loadings_")
        # transform() runs validate_data; build a DataFrame view here for
        # the residual computation that matches its shape.
        scores = self.transform(X)
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_hat = scores.values @ self._loadings.T
        residuals = X_arr - X_hat
        return -float(np.mean(residuals**2))

    @classmethod
    def minka_mle(cls, X: DataMatrix) -> int:
        """Minka (2000) automatic-dimensionality estimate for PCA.

        Closed-form Bayesian model selection on the PPCA evidence (Minka,
        T. P. 2000. *Automatic Choice of Dimensionality for PCA*. NIPS 13,
        pp. 598-604). Operates only on the covariance eigenvalues of ``X``
        and is therefore very cheap; in the simulations Minka reports it
        beats cross-validation. Use it alongside the ekf-CV recommendation
        from :meth:`select_n_components` as a fast cross-check.

        Internally ``X`` is mean-centred before estimation (a PPCA
        assumption); it is **not** unit-variance scaled, because dividing
        each column by its standard deviation compresses the noise
        eigenvalues to near-zero values the MLE misreads as additional
        latent signal. If your columns are on wildly different scales,
        pass the analysis-scale ``X`` produced by your own preprocessing
        (e.g. SNV for spectral data) and accept the centring this method
        applies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.

        Returns
        -------
        n_components : int
            The MLE estimate of the effective dimensionality.

        References
        ----------
        Minka, T. P. (2000). Automatic Choice of Dimensionality for PCA.
        Advances in Neural Information Processing Systems, 13, 598-604.

        See Also
        --------
        parallel_analysis : Horn (1965) eigenvalue-vs-null retention.
        select_n_components : ekf cross-validation; pass
            ``return_consensus=True`` to report all three side by side.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_arr = np.asarray(X, dtype=float)
        Xc = X_arr - X_arr.mean(axis=0)
        # ``svd_solver="full"`` is the only solver that supports
        # ``n_components="mle"`` in sklearn.
        sk = _SkPCA(n_components="mle", svd_solver="full")
        sk.fit(Xc)
        return int(sk.n_components_)

    @classmethod
    def parallel_analysis(
        cls,
        X: DataMatrix,
        *,
        n_simulations: int = 200,
        quantile: float = 0.95,
        scale: bool = True,
        random_state: int | None = None,
    ) -> Bunch:
        """Horn (1965) parallel analysis component-count estimate.

        Generates ``n_simulations`` random matrices of the same shape as
        ``X``, computes their eigenvalues, and retains every observed
        component whose eigenvalue exceeds the ``quantile`` of the null
        distribution at the same rank. Widely regarded in psychometrics
        as the best simple retention rule for PCA.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data matrix.
        n_simulations : int, default 200
            Number of random matrices drawn to build the null
            eigenvalue distribution.
        quantile : float, default 0.95
            Quantile of the null eigenvalues used as the retention
            threshold. Horn's original proposal was the mean (0.5);
            the more conservative 95th-percentile threshold is the
            modern recommendation.
        scale : bool, default True
            Mean-centre and unit-variance scale ``X`` before estimation
            (matches :meth:`minka_mle`).
        random_state : int, optional
            Seed for the null-matrix simulations.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - number of components retained (can be 0
              on pure noise).
            - ``observed_eigenvalues`` - eigenvalues of ``X`` after
              centring/scaling (np.ndarray of length ``min(n, p)``).
            - ``null_threshold`` - per-rank ``quantile`` of the null
              eigenvalue distribution (same length as
              ``observed_eigenvalues``).

        References
        ----------
        Horn, J. L. (1965). A rationale and test for the number of
        factors in factor analysis. *Psychometrika*, 30(2), 179-185.

        See Also
        --------
        minka_mle : closed-form PPCA evidence rule.
        select_n_components : ekf cross-validation; pass
            ``return_consensus=True`` to report all three side by side.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if scale:
            X = MCUVScaler().fit_transform(X)
        X_arr = np.asarray(X, dtype=float)
        n, p = X_arr.shape
        k = min(n, p)

        # Observed eigenvalues from the centred X. Centring removes one DoF
        # so the smallest singular value is at-or-near zero; that's expected.
        Xc = X_arr - X_arr.mean(axis=0)
        _, S, _ = np.linalg.svd(Xc, full_matrices=False)
        observed = (S**2) / max(1, n - 1)

        rng = np.random.default_rng(random_state)
        null_eigs = np.zeros((n_simulations, k))
        for i in range(n_simulations):
            R = rng.standard_normal((n, p))
            Rc = R - R.mean(axis=0)
            _, S_r, _ = np.linalg.svd(Rc, full_matrices=False)
            null_eigs[i, : S_r.shape[0]] = (S_r**2) / max(1, n - 1)

        null_threshold = np.quantile(null_eigs, quantile, axis=0)
        # Standard PA: retain consecutive components from the top while
        # observed > null. Components past the first failure are not
        # retained even if their observed eigenvalue happens to exceed
        # the null (a rare numerical coincidence on real data).
        n_retained = 0
        for obs, thr in zip(observed, null_threshold, strict=False):
            if obs > thr:
                n_retained += 1
            else:
                break

        return Bunch(
            n_components=int(n_retained),
            observed_eigenvalues=observed,
            null_threshold=null_threshold,
        )

    @classmethod
    def select_n_components(  # noqa: PLR0913, PLR0915, PLR0912, C901
        cls,
        X: DataMatrix,
        *,
        max_components: int | None = None,
        cv: int | BaseCrossValidator = 5,
        cv_scheme: typing.Literal["ekf", "ek", "sacv", "gcv", "row_wise"] = "ekf",
        n_repeats: int = 1,
        selection_rule: SelectionRule = "min",
        min_q2_increase: float = Q2_MIN_INCREMENT,
        scale_inside_folds: bool = True,
        n_iter: int = 50,
        tol: float = 1e-6,
        random_state: int | None = None,
        return_consensus: bool = False,
        threshold: float | None = None,
        **pca_kwargs,
    ) -> Bunch:
        """Select the number of PCA components via cross-validation.

        Evaluates every component count ``1, 2, ..., max_components`` and
        recommends one via the configured ``selection_rule``. The default
        ``cv_scheme="ekf"`` is the element-wise k-fold algorithm of Bro,
        Kjeldahl, Smilde & Kiers (2008, *Anal. Bioanal. Chem.* 390:1241-1251),
        which holds out individual cells of ``X`` and predicts them via
        EM-style imputation from a model that never sees their true values.
        This restores the prediction-independence requirement the legacy
        row-wise scheme violates, fixing the trivial-fit pathology where
        PRESS shrinks monotonically with components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. With the default ``scale_inside_folds=True`` pass
            the raw, unscaled X; mean-centring and unit-variance scaling are
            fit on each fold's in-fold cells. Pre-scale it yourself only
            with ``scale_inside_folds=False``.
        max_components : int, optional
            Maximum number of components to evaluate. Default is
            ``min(n_samples - 1, n_features)``.
        cv : int or sklearn CV splitter, default 5
            For ``cv_scheme="ekf"``: the integer number of element-folds
            (splitter objects are ignored). For ``cv_scheme="ek"``: the number
            of row groups, and of column groups. For ``cv_scheme="row_wise"``:
            either an integer K (fed to ``KFold``) or any sklearn splitter.
            Ignored by ``"sacv"`` and ``"gcv"``, which hold nothing out.
        cv_scheme : {"ekf", "ek", "sacv", "gcv", "row_wise"}, default "ekf"
            How a held-out value is produced. Every one of these except
            ``"row_wise"`` keeps the prediction independent of the value being
            predicted; they differ in what they hold out and what they cost.

            - ``"ekf"``, the default: element-wise k-fold with EM imputation.
              Scattered cells are held out and each is imputed from a model
              that never saw it. Fits its centring and scaling inside every
              fold, and is the only scheme here that takes a block with
              missing cells. Cost: ``n_folds * n_repeats * max_components``
              decompositions.
            - ``"ek"``: the two-model scheme of Eastment and Krzanowski
              (1982). An element is predicted by a score from a model without
              its column and a loading from a model without its row. This is
              what Simca-P reports and what ``pcaMethods::Q2`` computes by
              default, so use it when a number has to line up with either.
              Cost: ``2 * n_folds`` decompositions.
            - ``"sacv"``: leave-one-cell-out, approximated by inflating each
              residual by the leverage of the cell that produced it, after
              Josse and Husson (2012). The cheap version of holding cells out
              one at a time. Cost: one decomposition.
            - ``"gcv"``: the same idea with a single averaged leverage instead
              of one per cell. Blunter, and it tends to keep more components.
              Cost: one decomposition. This and ``"sacv"`` are the defaults in
              ``FactoMineR`` and ``missMDA``.
            - ``"row_wise"``: **deprecated, removed in 2.0.** See the warning
              admonition below.

            ``"ek"``, ``"sacv"`` and ``"gcv"`` factorise the matrix directly,
            so they raise on a block with missing cells and they ignore
            ``scale_inside_folds``, ``n_repeats``, ``n_iter`` and ``tol``.
            They also report no per-fold spread, so ``selection_rule="1se"``
            has nothing to work with; ``"min"`` takes the global optimum,
            where ``FactoMineR`` instead stops at the first local worsening,
            which can return a smaller count on the same data.
        n_repeats : int, default 1
            Repeat the ekf pass with a fresh random fold permutation this
            many times. Each repeat covers every cell exactly once;
            ``n_repeats > 1`` narrows the per-component PRESS standard
            error (helpful when the 1-SE rule sits on a borderline) at
            roughly linear extra runtime. Ignored under
            ``cv_scheme="row_wise"``.
        selection_rule : {"min", "1se", "q2_increment"}, default "min"
            How the recommended component count is chosen. ``"min"`` is the
            GlobalMin criterion Bro 2008 pairs with ekf - the component
            count with the lowest pooled PRESS. ``"1se"`` is the one-
            standard-error rule (needs ``per_fold_press``, available under
            both schemes). ``"q2_increment"`` is the Wold's-R-style
            cumulative-:math:`Q^2` threshold from PR #371; ``min_q2_increase``
            sets the threshold.
        min_q2_increase : float, default 0.01
            Threshold used only when ``selection_rule="q2_increment"``.
        scale_inside_folds : bool, default True
            With the default, mean-centring and unit-variance scaling
            constants are fit on each fold's in-fold cells and applied to
            the whole matrix before EM, removing the centring/scaling
            leakage of the prior implementation. Set to ``False`` to
            reproduce the previous behaviour (column mean recomputed each
            EM iteration from the imputed matrix, no scaling); this is
            useful only when ``X`` is already pre-scaled, and a
            :class:`SpecificationWarning` is emitted because scaling
            constants fit on the full matrix leak into every element-fold.
            Ignored under ``cv_scheme="row_wise"``.

            Pass the **raw, unscaled** X under the default. In-fold
            re-standardisation overwrites whatever scaling the caller
            applied, so two deliberately different strategies (autoscale
            versus Pareto, say) become the same model and report the same
            PRESS: a comparison between them shows no difference for
            reasons that have nothing to do with the data. A
            :class:`SpecificationWarning` is emitted when ``X`` arrives
            already centred and unit-variance scaled, which is the
            detectable half of that case; a block scaled some other way
            cannot be recognised, so the rule is the caller's to keep.
            Same contract as :meth:`PLS.select_n_components`.
        n_iter, tol : int and float, default 50 and 1e-6
            EM iteration cap and convergence tolerance for the ekf imputation
            step. Ignored under ``cv_scheme="row_wise"``.
        random_state : int, optional
            Seed for the ekf element-fold permutation.
        return_consensus : bool, default False
            When ``True``, also cross-check the CV recommendation against
            two cheap alternative selectors: Minka's PPCA MLE
            (:meth:`minka_mle`) and Horn's parallel analysis
            (:meth:`parallel_analysis`). The result Bunch then gains the
            ``minka_n_components``, ``parallel_analysis_n_components``,
            ``consensus``, and ``consensus_counts`` keys (see Returns).
        threshold : float, optional
            Deprecated. The original Wold PRESS-ratio cutoff. Passing it
            emits a :class:`DeprecationWarning`; the value is ignored. Use
            ``selection_rule="q2_increment"`` (and tune ``min_q2_increase``)
            for a comparable parsimony preference.
        **pca_kwargs
            Additional keyword arguments passed to the ``PCA()`` constructor
            under ``cv_scheme="row_wise"`` (e.g. ``algorithm="nipals"``).
            Ignored under ``cv_scheme="ekf"`` because ekf runs its own SVD
            loop.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            - ``n_components`` - recommended number of components (int).
            - ``press`` - pooled PRESS per component count (pd.Series,
              indexed ``1..A_max``). Under ``cv_scheme="ekf"`` with
              ``scale_inside_folds=True`` this is measured in the space each
              fold was fitted in, so every variable weighs the same; see
              ``press_input_units`` for the other scale.
            - ``press_input_units`` - the same curve in the units of the
              matrix that was passed in, for comparing prediction error
              against instrument error (pd.Series, indexed ``1..A_max``).
            - ``per_fold_press`` - per-fold PRESS contributions
              (pd.DataFrame, ``A_max`` rows x ``n_folds`` columns).
            - ``se_press`` - standard error of the per-fold PRESS curve
              (pd.Series, indexed ``1..A_max``). Drives the 1-SE rule.
            - ``q2_se`` - the same standard error rescaled onto the Q2 scale
              (``se_press / null_model_ss``, pd.Series indexed ``1..A_max``),
              i.e. the half-width of a +/-1 SE band around ``q2``.
            - ``press_ratio`` - ``PRESS_a / PRESS_{a-1}`` for inspection
              (pd.Series, indexed ``2..A_max``).
            - ``q2`` - cross-validated :math:`R^2_X` per component count
              (pd.Series, indexed ``1..A_max``). Computed as
              ``1 - press / null_model_ss``, where the null model predicts
              each held-out cell by its in-fold column mean, measured the
              same way ``press`` is. Directly comparable to
              ``r2_cumulative_`` and to PLS's ``r2y_validated``.
            - ``q2_per_variable`` - that same quantity split by variable
              (pd.DataFrame, ``A_max`` rows x ``K`` columns), which is what
              shows whether one column is carrying the pooled figure. All
              ``NaN`` under ``cv_scheme="row_wise"``, which has no per-cell
              error to split.
            - ``cv_scores`` - alias of ``per_fold_press`` under ekf, or
              per-fold negative MSE from ``cross_val_score`` under row-wise
              (preserved for back-compat).
            - ``cv_scheme`` - the scheme used (``"ekf"`` or ``"row_wise"``).
            - ``selection_rule`` - the rule used to pick ``n_components``.

            When ``return_consensus=True``, the Bunch additionally carries:

            - ``minka_n_components`` - the Minka PPCA MLE estimate (int).
            - ``parallel_analysis_n_components`` - Horn's parallel-analysis
              estimate (int).
            - ``consensus`` - ``"agree"`` if the three integer estimates
              (CV recommendation, Minka, parallel analysis) span at most
              1, otherwise ``"disagree"``.
            - ``consensus_counts`` - the tuple
              ``(recommended, minka_n, parallel_analysis_n)``.

        References
        ----------
        Bro, R., Kjeldahl, K., Smilde, A. K., & Kiers, H. A. L. (2008).
        Cross-validation of component models: a critical look at current
        methods. *Anal. Bioanal. Chem.*, 390(5), 1241-1251.

        Camacho, J., & Ferrer, A. (2012). Cross-validation in PCA models
        with the element-wise k-fold (ekf) algorithm: theoretical aspects.
        *J. Chemometrics*, 26(7), 361-373.

        Eastment, H. T., & Krzanowski, W. J. (1982). Cross-validatory choice
        of the number of components from a principal component analysis.
        *Technometrics*, 24(1), 73-77.

        Josse, J., & Husson, F. (2012). Selecting the number of components in
        principal component analysis using cross-validation approximations.
        *Computational Statistics & Data Analysis*, 56(6), 1869-1879.

        .. warning::

           ``cv_scheme="row_wise"`` is **deprecated since 1.84 and will be
           removed in 2.0**. It emits a :class:`DeprecationWarning` and a
           :class:`SpecificationWarning`. It suffers from the *trivial-fit*
           problem: holding out whole rows and projecting them back via
           :meth:`transform` lets the held-out row's own values reach its
           prediction, so PRESS shrinks monotonically with the component
           count and reaches zero once the components equal the variables.
           It measures compression, not prediction, and cannot select a
           component count. Use ``"ekf"``, ``"ek"``, ``"sacv"`` or ``"gcv"``.
        """
        if threshold is not None:
            warnings.warn(
                "The `threshold` (Wold PRESS-ratio) argument of "
                "PCA.select_n_components is deprecated and ignored; the "
                "recommendation now uses `selection_rule`. Pass "
                "`selection_rule='q2_increment'` (and tune `min_q2_increase`) "
                "for a comparable parsimony preference.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        N, K = X.shape
        if max_components is None:
            max_components = min(N - 1, K)
        max_components = min(int(max_components), N - 1, K)
        if max_components < 1:
            raise ValueError("No components can be evaluated; the data is too small.")

        component_index = pd.Index(range(1, max_components + 1), name="n_components")
        X_arr = np.asarray(X, dtype=float)

        # ``press`` under ekf is the TOTAL PRESS per repeat (n_folds times the
        # mean per-fold PRESS), while row_wise's press is already a per-fold
        # mean; the per-fold SE computed further below must be rescaled by
        # this factor so the 1-SE rule compares like with like.
        press_scale_multiplier = 1
        if cv_scheme == "ekf":
            n_folds = cv if isinstance(cv, int) else 5
            press_scale_multiplier = n_folds
            if n_repeats < 1:
                raise ValueError(f"n_repeats must be >= 1; got {n_repeats}.")
            # Same two traps as PLS.select_n_components, checked only here
            # because row_wise ignores the flag.
            _warn_scaling_traps(X, scale_inside_folds=scale_inside_folds, fold="element-fold", metric="PRESS")
            ekf = _pca_ekf_press(
                X_arr,
                max_components,
                n_folds=n_folds,
                n_repeats=n_repeats,
                n_iter=n_iter,
                tol=tol,
                scale_inside_folds=scale_inside_folds,
                random_state=random_state,
            )
            press = pd.Series(ekf.press, index=component_index, name="PRESS")
            per_fold_press = pd.DataFrame(
                ekf.per_fold_press,
                index=component_index,
                columns=[f"fold_{i + 1}" for i in range(ekf.per_fold_press.shape[1])],
            )
            cv_scores = per_fold_press
            press_input_units = pd.Series(ekf.press_input_units, index=component_index, name="PRESS (input units)")
            null_model_ss, per_column_null_ss = _ekf_null_reference(ekf, X_arr, scale_inside_folds=scale_inside_folds)
            q2 = 1.0 - press / null_model_ss if null_model_ss > epsqrt else press * np.nan
            # A column with no spread across the held-out cells has nothing to
            # predict, so its Q^2 is undefined rather than zero.
            safe_null = np.where(per_column_null_ss > epsqrt, per_column_null_ss, np.nan)
            per_variable = 1.0 - ekf.per_column_press / safe_null
            q2_per_variable = pd.DataFrame(per_variable, index=component_index, columns=list(X.columns))
        elif cv_scheme in {"ek", "sacv", "gcv"}:
            Z = _preprocess_for_cell_schemes(X_arr, cv_scheme)
            if cv_scheme == "ek":
                n_folds = cv if isinstance(cv, int) else 5
                raw_press, raw_per_column, null_model_ss, per_column_null = _eastment_krzanowski_press(
                    Z, max_components, n_folds=n_folds, random_state=random_state
                )
            else:
                raw_press, raw_per_column, null_model_ss, per_column_null = _leverage_corrected_press(
                    Z, max_components, method=cv_scheme
                )
            press = pd.Series(raw_press, index=component_index, name="PRESS")
            # None of these three splits the data into folds that could disagree:
            # "ek" pools every cell into one total, and the other two hold nothing
            # out at all. A per-fold spread would be fabricated, so it is absent
            # rather than zero, and the 1-SE rule has nothing to work with.
            per_fold_press = pd.DataFrame(np.nan, index=component_index, columns=["fold_1"])
            cv_scores = per_fold_press
            press_input_units = press.rename("PRESS (input units)")
            q2 = 1.0 - press / null_model_ss if null_model_ss > epsqrt else press * np.nan
            safe_null = np.where(per_column_null > epsqrt, per_column_null, np.nan)
            q2_per_variable = pd.DataFrame(
                1.0 - raw_per_column / safe_null, index=component_index, columns=list(X.columns)
            )
        elif cv_scheme == "row_wise":
            warnings.warn(
                "cv_scheme='row_wise' is deprecated and will be removed in 2.0. It "
                "measures how well a held-out row reproduces itself, which is a "
                "compression error rather than a prediction error, so it cannot pick "
                "a component count. Use the default cv_scheme='ekf', or 'ek', 'sacv' "
                "or 'gcv'.",
                DeprecationWarning,
                stacklevel=2,
            )
            warnings.warn(
                "cv_scheme='row_wise' uses the legacy whole-row CV scheme that "
                "Bro et al. 2008 flagged as invalid: held-out row values flow "
                "back through transform() into their own prediction, so PRESS "
                "shrinks monotonically and the recommendation tends to run to "
                "the maximum component count. Prefer cv_scheme='ekf' (the new "
                "default).",
                SpecificationWarning,
                stacklevel=2,
            )
            press_values = {}
            all_cv_scores = {}
            for a in range(1, max_components + 1):
                scores_a = cross_val_score(cls(n_components=a, **pca_kwargs), X, cv=cv)
                all_cv_scores[a] = scores_a
                press_values[a] = -scores_a.mean()
            press = pd.Series(press_values, name="PRESS", index=component_index)
            cv_scores = pd.DataFrame(all_cv_scores).T
            cv_scores.index = component_index
            cv_scores.columns = [f"fold_{i + 1}" for i in range(cv_scores.shape[1])]
            # The row_wise per-fold values are negative MSE means, not raw
            # PRESS contributions, so don't pretend they are. Build a parallel
            # per_fold_press by undoing the negation.
            per_fold_press = -cv_scores
            null_model_ss = float(np.nanmean((X_arr - np.nanmean(X_arr, axis=0)) ** 2))
            q2 = 1.0 - press / null_model_ss if null_model_ss > epsqrt else press * np.nan
            # row_wise scores whole rows through the estimator, so there is no
            # per-cell error to split by variable and no in-fold scale to
            # report against. Both fields exist so the Bunch has one shape.
            press_input_units = press.rename("PRESS (input units)")
            q2_per_variable = pd.DataFrame(np.nan, index=component_index, columns=list(X.columns))
        else:
            raise ValueError(
                f"Unknown cv_scheme {cv_scheme!r}; expected one of 'ekf', 'ek', 'sacv', 'gcv' or 'row_wise'."
            )

        q2 = q2.rename("Q2")
        q2.index = component_index

        # PRESS ratio: still computable under either scheme, kept for inspection.
        # A block with no variation anywhere gives PRESS 0 at every component
        # count, and 0/0 should reach the caller as a NaN ratio rather than as a
        # numpy warning pointing into this function. The standard error below
        # already guards the same case.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_values = {a: press[a] / press[a - 1] for a in range(2, max_components + 1)}
        press_ratio = pd.Series(ratio_values, name="PRESS ratio")
        press_ratio.index.name = "n_components"

        # Per-component standard error across folds.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_fold_arr = per_fold_press.to_numpy()
            n_folds_per_a = np.maximum(1, np.sum(~np.isnan(per_fold_arr), axis=1))
            se_values = np.nanstd(per_fold_arr, axis=1, ddof=1) / np.sqrt(n_folds_per_a)
        # Rescale the per-fold-mean SE onto the same scale as ``press`` (see
        # press_scale_multiplier above). Previously the ekf band was ~n_folds
        # times too narrow, silently degenerating selection_rule="1se" to
        # "min".
        se_values = se_values * press_scale_multiplier
        se_press = pd.Series(se_values, index=component_index, name="SE(PRESS)")

        # The same constant null-model sum-of-squares that normalises Q2
        # (Q2 = 1 - PRESS / null_model_ss) also rescales the PRESS standard
        # error onto the Q2 scale, giving a directly usable +/-1 SE band around
        # the Q2 curve without callers having to re-derive the normalisation.
        q2_se_values = se_values / null_model_ss if null_model_ss > epsqrt else se_values * np.nan
        q2_se = pd.Series(q2_se_values, index=component_index, name="SE(Q2)")

        press_arr_final = press.to_numpy()
        if np.all(np.isnan(press_arr_final)):
            raise RuntimeError(
                "Cross-validation produced NaN PRESS for every component count; no recommendation can be made."
            )
        recommended = _select_n_components(
            selection_rule,
            mean_error=press_arr_final,
            se_error=se_values,
            q2_cumulative=q2.to_numpy(),
            min_q2_increase=min_q2_increase,
        )

        # A criterion that never turns over has not found an optimum: it has run
        # out of components to evaluate. The count returned is then the largest
        # one tried rather than an answer, and saying so is the difference
        # between a recommendation and a number. This is the failure mode that
        # makes cv_scheme="row_wise" useless, and the leverage approximations
        # can fall into it too on data whose R2 approaches one, where the
        # residual they inflate has almost nothing left in it.
        evaluated = q2.to_numpy()[~np.isnan(q2.to_numpy())]
        if evaluated.size > 1 and np.all(np.diff(evaluated) > 0) and int(recommended) == max_components:
            warnings.warn(
                f"cv_scheme={cv_scheme!r} did not turn over: its Q2 rises at every one of the "
                f"{max_components} component counts evaluated, so {recommended} is the largest "
                "count tried rather than an optimum. Evaluate more components, or use a scheme "
                "that holds data out (cv_scheme='ekf' or 'ek'), before reading this as an answer.",
                SpecificationWarning,
                stacklevel=2,
            )

        consensus_fields: dict[str, object] = {}
        if return_consensus:
            # Two cheap cross-checks: Minka's PPCA MLE (mean-centred input,
            # no unit-variance scaling) and Horn's parallel analysis (which
            # accepts the same scaling convention as the rest of the method).
            minka_n = cls.minka_mle(X)
            pa_result = cls.parallel_analysis(X, scale=scale_inside_folds, random_state=random_state)
            counts = (int(recommended), int(minka_n), int(pa_result.n_components))
            consensus = "agree" if max(counts) - min(counts) <= 1 else "disagree"
            consensus_fields = {
                "minka_n_components": int(minka_n),
                "parallel_analysis_n_components": int(pa_result.n_components),
                "consensus": consensus,
                "consensus_counts": counts,
            }

        return Bunch(
            n_components=recommended,
            press=press,
            per_fold_press=per_fold_press,
            se_press=se_press,
            q2_se=q2_se,
            press_ratio=press_ratio,
            q2=q2,
            q2_per_variable=q2_per_variable,
            press_input_units=press_input_units,
            cv_scores=cv_scores,
            cv_scheme=cv_scheme,
            selection_rule=selection_rule,
            **consensus_fields,
        )

    def detect_outliers(self, conf_level: float = 0.95) -> list[dict]:
        """Detect outlier observations using SPE and Hotelling's T² diagnostics.

        Combines two approaches:

        1. **Statistical limits** - observations exceeding the SPE or T² limit
           at ``conf_level`` are flagged.
        2. **Robust ESD test** - the generalized ESD test (with robust median/MAD
           variant) identifies observations that are unusual *relative to the
           rest of the data*, even if they fall below the statistical limit.

        An observation can be flagged for one or both reasons.

        Parameters
        ----------
        conf_level : float, default 0.95
            Confidence level in [0.8, 0.999]. Controls both the statistical
            limits and the ESD test's significance level (alpha = 1 - conf_level).

        Returns
        -------
        outliers : list of dict
            Sorted from most severe to least. Each dict contains:

            - ``observation`` - index label of the observation
            - ``outlier_types`` - list of ``"spe"`` and/or ``"hotellings_t2"``
            - ``spe`` - SPE value for this observation
            - ``hotellings_t2`` - T² value for this observation
            - ``spe_limit`` - SPE limit at the given confidence level
            - ``hotellings_t2_limit`` - T² limit at the given confidence level
            - ``severity`` - max(spe/spe_limit, t2/t2_limit)

        Examples
        --------
        >>> pca = PCA(n_components=3).fit(X_scaled)
        >>> outliers = pca.detect_outliers(conf_level=0.95)
        >>> for o in outliers:
        ...     print(f"{o['observation']}: {o['outlier_types']} (severity={o['severity']})")
        """
        check_is_fitted(self, "spe_")
        if not (0.8 <= conf_level <= 0.999):
            raise ValueError(f"conf_level must be between 0.8 and 0.999, got {conf_level}.")

        N = self.n_samples_

        # Full-model SPE and cumulative T² (last column)
        spe_values = self.spe_.iloc[:, -1]
        t2_values = self.hotellings_t2_.iloc[:, -1]

        # Statistical limits
        spe_lim = self.spe_limit(conf_level=conf_level)
        t2_lim = self.hotellings_t2_limit(conf_level=conf_level)

        # Robust ESD outlier detection on each series
        max_outliers = max(1, N // 5)
        alpha = 1 - conf_level

        spe_outlier_idx, _ = detect_outliers_esd(
            spe_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )
        t2_outlier_idx, _ = detect_outliers_esd(
            t2_values.to_numpy(), algorithm="esd", max_outliers_detected=max_outliers, alpha=alpha
        )

        # Collect all flagged observations: ESD outliers + above-limit
        spe_flagged = set(spe_outlier_idx)
        t2_flagged = set(t2_outlier_idx)

        # Also flag any observation above the statistical limit
        for i in range(N):
            if spe_values.iloc[i] > spe_lim:
                spe_flagged.add(i)
            if t2_values.iloc[i] > t2_lim:
                t2_flagged.add(i)

        # Merge into result dicts
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
            # floating-point noise. An infinite T2 limit (A == N) contributes
            # 0 already, but is made explicit here.
            # The ratio itself is scale-invariant (value and limit share units),
            # so the guard is only against a limit that carries no information:
            # zero (a perfect fit, which used to raise ZeroDivisionError) or
            # infinite (A == N). Deliberately NOT an absolute epsilon, which
            # would make severity depend on the units of the data.
            ratios = [
                spe_val / spe_lim if spe_lim > 0.0 else 0.0,
                t2_val / t2_lim if np.isfinite(t2_lim) and t2_lim > 0.0 else 0.0,
            ]
            severity = max(ratios)

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
