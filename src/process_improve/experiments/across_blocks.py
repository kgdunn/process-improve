# (c) Kevin Dunn, 2010-2026. MIT License.

r"""Models fitted across blocks: fit, choose the form, and predict the next block.

An experiment run in blocks (days, batches of raw material, sensory panel sessions) sees every response
shifted a little in each block. The unit of data is then one run's result in one block, and each model here
carries a block effect coded to sum to zero, so its intercept is the average block. Runs tested in more than one
block tie the blocks together.

:func:`fit_across_blocks` fits each candidate form (a model formula's right-hand side) to every response, chooses
the form with the lowest leave-one-run-out error (each run predicted by a fit without it, in every block it was
tested), and returns the chosen fit. Its :meth:`BlockedFit.predict` predicts new runs with prediction intervals in a
known block, in the average block, or in a coming block whose shift is read from runs tested again in it.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from process_improve.experiments._lm import validate_formula_is_safe
from process_improve.univariate.metrics import _check_probability


@dataclass(frozen=True)
class _ResponseFit:
    """One response's least-squares fit over [terms, block columns]: its blocks, (X'X)^-1, residual SD and df."""

    beta: np.ndarray
    xtx_inv: np.ndarray
    residual_sd: float
    df: int
    levels: list[Hashable]  # the blocks the response was measured in, in order of first appearance


def _block_columns(blocks: Sequence[Hashable], levels: list[Hashable]) -> np.ndarray:
    """Return the block effect coded to sum to zero: one column per block but the last, which is -1 in each."""
    columns = np.zeros((len(blocks), max(len(levels) - 1, 0)))
    for row, block in enumerate(blocks):
        index = levels.index(block)
        if index == len(levels) - 1:
            columns[row] = -1.0
        else:
            columns[row, index] = 1.0
    return columns


def _fit_rows(terms: np.ndarray, blocks: np.ndarray, y: np.ndarray, rows: np.ndarray) -> _ResponseFit | None:
    """Fit the rows given, with a block effect over the blocks they hold; None when the columns cannot be told apart
    or no degree of freedom is left.
    """
    levels = list(pd.unique(blocks[rows]))
    x = np.column_stack([terms[rows], _block_columns(list(blocks[rows]), levels)])
    n, p = x.shape
    if n <= p or np.linalg.matrix_rank(x) < p:
        return None
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y[rows]
    residuals = y[rows] - x @ beta
    return _ResponseFit(beta, xtx_inv, float(np.sqrt(residuals @ residuals / (n - p))), n - p, levels)


def _leave_one_run_out(terms: np.ndarray, blocks: np.ndarray, runs: np.ndarray, y: np.ndarray) -> tuple[float, list]:
    """
    Predict every run from a fit without it, in every block it was tested: the root mean squared error.

    Returns
    -------
    (float, list)
        The RMSE (NaN when no run could be predicted), and the runs that could not be: without them a term or a
        block has no data, so they alone carry it.
    """
    measured = np.isfinite(y)
    errors: list[float] = []
    alone = []
    for run in pd.unique(runs[measured]):
        held = measured & (runs == run)
        train = measured & (runs != run)
        fit = _fit_rows(terms, blocks, y, train)
        if fit is None or not set(blocks[held]) <= set(fit.levels):
            alone.append(run)
            continue
        x = np.column_stack([terms[held], _block_columns(list(blocks[held]), fit.levels)])
        errors += list(y[held] - x @ fit.beta)
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("nan")
    return rmse, alone


@dataclass(frozen=True)
class BlockedFit:
    """
    A model fitted across blocks by :func:`fit_across_blocks`: the chosen form, its fit, and how it was chosen.

    Attributes
    ----------
    chosen_form : str or None
        The name of the chosen form; None when no candidate could be fitted.
    formula : str
        The chosen form's right-hand side.
    selection : pandas.DataFrame
        One row per candidate form, best first: ``estimable`` (every response can be fitted), ``n_parameters``
        (terms and block columns), ``loo_rmse`` (the mean over the responses of each one's leave-one-run-out RMSE),
        one column per response with its own leave-one-run-out RMSE, and ``runs_alone`` (runs that alone carry a
        term or a block, so cannot be left out).
    coefficients : pandas.DataFrame
        The chosen form's coefficients, a row per term (``Intercept`` is the average block), a column per response.
    standard_errors : pandas.DataFrame
        Their standard errors, shaped as ``coefficients``.
    block_shift : pandas.DataFrame
        Each block's shift from the average block, a row per block, a column per response; the shifts sum to 0.
    block_shift_se : pandas.DataFrame
        Their standard errors, shaped as ``block_shift``.
    residual_sd : pandas.Series
        Each response's residual standard deviation.
    df : pandas.Series
        Each response's residual degrees of freedom.
    """

    chosen_form: str | None
    formula: str
    selection: pd.DataFrame
    coefficients: pd.DataFrame
    standard_errors: pd.DataFrame
    block_shift: pd.DataFrame
    block_shift_se: pd.DataFrame
    residual_sd: pd.Series
    df: pd.Series
    _design: patsy.DesignInfo | None = field(default=None, repr=False)
    _fits: Mapping[str, _ResponseFit] = field(default_factory=dict, repr=False)

    def _terms(self, frame: pd.DataFrame) -> np.ndarray:
        if self._design is None:
            msg = "No form could be fitted, so there is nothing to predict with."
            raise ValueError(msg)
        return np.asarray(patsy.build_design_matrices([self._design], frame, NA_action="raise")[0])

    def predict(
        self,
        new: pd.DataFrame,
        *,
        block: Hashable | None = None,
        anchors: pd.DataFrame | None = None,
        conf_level: float = 0.95,
    ) -> pd.DataFrame:
        r"""
        Predict new runs, with prediction intervals, in a known block, the average block or a coming block.

        Parameters
        ----------
        new : pandas.DataFrame
            The runs to predict: a column per factor the formula names.
        block : hashable or None
            A block of the fit: the prediction includes its shift.
        anchors : pandas.DataFrame or None
            Runs tested again in a coming block, with their factors and the responses measured there. That block's
            shift is read from them: the prediction is :math:`(x_0 - \bar{x}_a)\hat\beta + \bar{y}_a`, whose
            error variance is :math:`\sigma^2 (1 + 1/n_a + (x_0 - \bar{x}_a)^T (X^TX)^{-1} (x_0 - \bar{x}_a))`.
        conf_level : float
            The coverage of the intervals.

        Returns
        -------
        pandas.DataFrame
            Indexed as ``new``, with columns (response, ``prediction`` | ``lower`` | ``upper``).

        Notes
        -----
        With neither ``block`` nor ``anchors``, the prediction is for the average block, and the interval adds the
        spread of the fitted block shifts: what a block not yet tested may do.
        """
        _check_probability("conf_level", conf_level)
        if block is not None and anchors is not None:
            msg = "Give a block of the fit or the anchors of a coming block, not both."
            raise ValueError(msg)
        if block is not None and block not in self.block_shift.index:
            msg = f"{block!r} is not a block of the fit; its blocks are {list(self.block_shift.index)}."
            raise ValueError(msg)
        t0 = self._terms(new)
        columns: dict[tuple[str, str], np.ndarray] = {}
        for response, fit in self._fits.items():
            n_blocks = len(fit.levels) - 1
            x0 = np.column_stack([t0, np.zeros((len(t0), n_blocks))])
            extra = 0.0
            if block is not None and block not in fit.levels:  # the response was not measured in that block
                nan = np.full(len(t0), np.nan)
                columns |= {(response, "prediction"): nan, (response, "lower"): nan, (response, "upper"): nan}
                continue
            if anchors is not None and response in anchors and np.isfinite(anchors[response].to_numpy(float)).any():
                measured = np.isfinite(anchors[response].to_numpy(dtype=float))
                ta = self._terms(anchors.loc[measured])
                centre = np.concatenate([ta.mean(axis=0), np.zeros(n_blocks)])
                shift = float(anchors[response].to_numpy(dtype=float)[measured].mean() - centre @ fit.beta)
                x_var = x0 - centre
                extra = fit.residual_sd**2 / measured.sum()
            elif block is not None:
                x0[:, t0.shape[1] :] = _block_columns([block] * len(t0), fit.levels)
                shift, x_var = 0.0, x0
            else:
                # shortcut: the spread of the estimated shifts includes their estimation error, so it overstates the
                # block-to-block variance; a random-effects fit would separate the two, worth it once blocks are many.
                spread = self.block_shift[response].var(ddof=1) if n_blocks else fit.residual_sd**2
                shift, x_var, extra = 0.0, x0, float(spread)
            mean = x0 @ fit.beta + shift
            variance = fit.residual_sd**2 * (1 + np.einsum("ij,jk,ik->i", x_var, fit.xtx_inv, x_var)) + extra
            half = stats.t.ppf(0.5 + conf_level / 2, fit.df) * np.sqrt(variance)
            columns |= {
                (response, "prediction"): mean,
                (response, "lower"): mean - half,
                (response, "upper"): mean + half,
            }
        return pd.DataFrame(columns, index=new.index)


def _check(data: pd.DataFrame, responses: Sequence[str], forms: Mapping[str, str], block: str) -> None:
    problems = [f"{r!r} is not a column" for r in [*responses, block] if r not in data.columns]
    if not forms:
        problems.append("give at least one candidate form")
    if not responses:
        problems.append("give at least one response")
    factors = [c for c in data.columns if c not in {*responses, block}]
    for name, form in forms.items():
        if "~" in form:
            problems.append(f"form {name!r}: give the right-hand side only, without '~'")
            continue
        try:
            validate_formula_is_safe(form, factors, allow_transforms=True, allow_numpy=True)
        except ValueError as err:
            problems.append(f"form {name!r}: {err}")
    if problems:
        msg = "The data and forms cannot be fitted across blocks: " + "; ".join(problems) + "."
        raise ValueError(msg)


def _block_effects(fit: _ResponseFit, p_terms: int) -> tuple[pd.Series, pd.Series]:
    """Each block's shift from the average block, and its standard error."""
    contrasts = np.zeros((len(fit.levels), p_terms + len(fit.levels) - 1))
    contrasts[:, p_terms:] = _block_columns(fit.levels, fit.levels)
    se = fit.residual_sd * np.sqrt(np.einsum("ij,jk,ik->i", contrasts, fit.xtx_inv, contrasts))
    return pd.Series(contrasts @ fit.beta, index=fit.levels), pd.Series(se, index=fit.levels)


def fit_across_blocks(
    data: pd.DataFrame,
    responses: Sequence[str],
    forms: Mapping[str, str],
    *,
    block: str,
) -> BlockedFit:
    r"""
    Fit candidate model forms across blocks, choose one by leave-one-run-out error, and return its fit.

    Each form is fitted to each response by least squares, with a block effect coded to sum to zero. The rule,
    fixed before the data: the form that can be fitted to every response with the lowest mean (over the responses)
    leave-one-run-out RMSE. Leaving out a run removes it from every block it was tested in; the error is that of
    predicting it there from the other runs, so a form is judged on runs it has not seen.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per run per block. The index names the run, so a run tested in several blocks has several rows with
        one index value. The columns are the factors (numeric, or categorical as text), the block, and the responses;
        a response not measured in a row is NaN. A factor's value is the one the run was tested at.
    responses : sequence of str
        The response columns.
    forms : mapping of str to str
        Candidate forms, by name: each a model formula's right-hand side over the factors, such as
        ``"A + B + A:B + I(A**2)"``. A categorical factor can be written ``C(F, Sum)`` for effects coding; a square
        root as ``np.sqrt(A)``. Code the factors first when a form's meaning depends on their centre and scale.
    block : str
        The block column.

    Returns
    -------
    BlockedFit
        The chosen form, its coefficients and block shifts, the table of every candidate, and ``predict``.

    Raises
    ------
    ValueError
        If a column is missing, a form is not a safe right-hand side over the factors, or a factor is missing in a
        row (patsy's ``NA_action="raise"``).

    Examples
    --------
    >>> fit = fit_across_blocks(data, ["y"], {"linear": "A + B", "quadratic": "A + B + A:B + I(A**2)"}, block="day")
    >>> fit.chosen_form, fit.selection
    >>> fit.predict(new_runs, anchors=repeats_tested_today)
    """
    _check(data, responses, forms, block)
    runs = np.asarray(data.index)
    blocks = data[block].to_numpy()
    levels = list(pd.unique(blocks))
    rows, candidates = [], {}
    for name, form in forms.items():
        terms = patsy.dmatrix(form, data, NA_action="raise", return_type="dataframe")
        t = terms.to_numpy()
        row: dict = {"form": name, "n_parameters": t.shape[1] + len(levels) - 1}
        fits, alone = {}, set()
        for response in responses:
            y = data[response].to_numpy(dtype=float)
            fits[response] = _fit_rows(t, blocks, y, np.isfinite(y))
            row[response], runs_alone = _leave_one_run_out(t, blocks, runs, y) if fits[response] else (np.nan, [])
            alone |= set(runs_alone)
        row |= {"estimable": all(f is not None for f in fits.values()), "runs_alone": sorted(alone, key=str)}
        row["loo_rmse"] = float(np.nanmean([row[r] for r in responses])) if row["estimable"] else np.nan
        rows.append(row)
        candidates[name] = (terms.design_info, fits)
    selection = pd.DataFrame(rows).set_index("form")
    selection = selection[["estimable", "n_parameters", "loo_rmse", *responses, "runs_alone"]]
    usable = selection[selection["estimable"] & selection["loo_rmse"].notna()]
    chosen = str(usable["loo_rmse"].idxmin()) if len(usable) else None
    selection = selection.sort_values("loo_rmse", na_position="last", kind="stable")
    return _blocked_fit(chosen, forms[chosen] if chosen else "", selection, candidates, levels)


def _blocked_fit(
    chosen: str | None,
    formula: str,
    selection: pd.DataFrame,
    candidates: Mapping[str, tuple[patsy.DesignInfo, Mapping[str, _ResponseFit | None]]],
    levels: list[Hashable],
) -> BlockedFit:
    """Gather the chosen form's fit (``candidates`` holds each form's design and fit per response) into tables."""
    responses = list(next(iter(candidates.values()))[1])
    empty = pd.DataFrame(columns=responses, dtype=float)
    if chosen is None:
        nan = pd.Series(np.nan, index=responses)
        return BlockedFit(None, "", selection, empty, empty, empty, empty, nan, nan)
    design, fitted = candidates[chosen]
    fits = {r: f for r, f in fitted.items() if f is not None}
    p_terms = len(design.column_names)
    shifts = {r: _block_effects(f, p_terms) for r, f in fits.items()}
    blocks = pd.Index(levels, name="block")
    return BlockedFit(
        chosen_form=chosen,
        formula=formula,
        selection=selection,
        coefficients=pd.DataFrame({r: f.beta[:p_terms] for r, f in fits.items()}, index=design.column_names),
        standard_errors=pd.DataFrame(
            {r: f.residual_sd * np.sqrt(np.diag(f.xtx_inv)[:p_terms]) for r, f in fits.items()},
            index=design.column_names,
        ),
        block_shift=pd.DataFrame({r: s[0] for r, s in shifts.items()}).reindex(blocks),
        block_shift_se=pd.DataFrame({r: s[1] for r, s in shifts.items()}).reindex(blocks),
        residual_sd=pd.Series({r: f.residual_sd for r, f in fits.items()}),
        df=pd.Series({r: f.df for r, f in fits.items()}),
        _design=design,
        _fits=fits,
    )
