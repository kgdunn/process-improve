# (c) Kevin Dunn, 2010-2026. MIT License.
"""F tests between nested models: the extra sum of squares (#789 B2).

:func:`compare_models` tests a model against a fuller one fitted to the same runs, as R's
``anova(reduced, full)`` does. :func:`group_test` tests a group of a model's terms together,
with the smaller model built by leaving them out. Neither reads an ANOVA table, so a group of
terms is tested at once, however many rows of a table it would span.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
import statsmodels.formula.api as smf
from patsy import ModelDesc
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper

from process_improve.experiments._lm import Model, lm

__all__ = ["compare_models", "group_test"]

Fitted: TypeAlias = Model | RegressionResultsWrapper


def _results(model: Fitted) -> RegressionResultsWrapper:
    """Return the least-squares results of a model from :func:`lm`, or the results themselves."""
    return model._OLS if isinstance(model, Model) else model


def _spans(larger: np.ndarray, smaller: np.ndarray) -> bool:
    """Whether every column of ``smaller`` is a combination of the columns of ``larger``.

    Each column is scaled to unit length first, so the answer does not depend on the units
    the factors are in (a square in pascals beside an intercept of ones).
    """
    both = np.hstack([larger, smaller])
    norms = np.linalg.norm(both, axis=0)
    both = both / np.where(norms > 0, norms, 1.0)
    return bool(np.linalg.matrix_rank(both) == np.linalg.matrix_rank(both[:, : larger.shape[1]]))


def compare_models(reduced: Fitted, full: Fitted, error: Fitted | None = None) -> dict[str, float]:
    """Test a model against a fuller one fitted to the same runs: the extra-sum-of-squares F test.

    The sum of squares is the reduced model's residual sum of squares less the full model's, on
    the difference of their residual degrees of freedom. It is tested against the full model's
    residual mean square, or against that of ``error``, a third model that contains the full
    one, as Design-Expert tests a mixture's linear blending against the residual of the
    quadratic model. With the full model saturated in the design's factors, the test is the
    test of lack of fit against pure error.

    Parameters
    ----------
    reduced : Model or RegressionResultsWrapper
        The smaller model, from :func:`lm` or a statsmodels least-squares fit. Every column of
        its model matrix must be a combination of the full model's columns.
    full : Model or RegressionResultsWrapper
        The fuller model, fitted to the same runs of the same response.
    error : Model or RegressionResultsWrapper, optional
        A model that contains the full one, whose residual mean square is the denominator.

    Returns
    -------
    dict[str, float]
        ``ss``, ``df`` and ``ms`` of the difference; ``f`` and ``p_value``; and ``ss_error``,
        ``df_error`` and ``ms_error``, the denominator's.

    Raises
    ------
    ValueError
        If the models are not fitted to the same runs of the same response (a missing value in
        a column only one of them uses drops that run from it alone), the reduced model is not
        nested in the full one, or the full one in ``error``, the full model has no more
        estimable terms than the reduced one, or the denominator has no residual.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import compare_models, lm
    >>> runs = pd.DataFrame({"A": [-1, 1, -1, 1, 0, 0], "B": [-1, -1, 1, 1, 0, 0], "y": [3, 5, 4, 7, 4.6, 4.9]})
    >>> found = compare_models(lm("y ~ A + B", runs), lm("y ~ A * B", runs))
    >>> found["df"], round(found["ss"], 4), round(found["f"], 3)
    (1, 0.25, 11.111)
    """
    smaller, fuller = _results(reduced), _results(full)
    denominator = fuller if error is None else _results(error)
    if not all(np.array_equal(other.model.endog, smaller.model.endog) for other in (fuller, denominator)):
        raise ValueError(
            "The models must be fitted to the same runs of the same response; a missing value in a column "
            "only one model uses drops that run from that model alone."
        )
    if not _spans(fuller.model.exog, smaller.model.exog):
        why = (
            "the full model is nested in it, so pass the smaller model first"
            if _spans(smaller.model.exog, fuller.model.exog)
            else "some of its columns are not combinations of the full model's"
        )
        raise ValueError(f"The reduced model is not nested in the full one: {why}.")
    if not _spans(denominator.model.exog, fuller.model.exog):
        raise ValueError("The error model must contain the full model, or the two sums of squares are not independent.")
    df = round(float(smaller.df_resid - fuller.df_resid))
    if df < 1:
        raise ValueError("The full model estimates no more terms than the reduced one, so there is nothing to test.")
    ss_error, df_error = float(denominator.ssr), float(denominator.df_resid)
    if df_error < 1:
        raise ValueError("The model the difference is tested against has no residual degrees of freedom.")
    ss = float(smaller.ssr - fuller.ssr)
    f = ss / df / (ss_error / df_error)
    return {
        "ss": ss,
        "df": df,
        "ms": ss / df,
        "f": f,
        "p_value": float(stats.f.sf(f, df, df_error)),
        "ss_error": ss_error,
        "df_error": df_error,
        "ms_error": ss_error / df_error,
    }


def _factors(term: Any) -> frozenset[str]:  # noqa: ANN401 - a patsy Term
    """Return a patsy term's factors, so that ``A:B`` and ``B:A`` are the same term."""
    return frozenset(factor.code for factor in term.factors)


def _named_term(name: str) -> frozenset[str]:
    """Return the factors of one term named as in a formula: ``"A"``, ``"B:A"``, ``"I(A ** 2)"`` or ``"Intercept"``."""
    terms = ModelDesc.from_formula(f"~ 0 + {'1' if name == 'Intercept' else name}").rhs_termlist
    if len(terms) != 1:
        raise ValueError(f"{name!r} is not one term; name each term of the group on its own.")
    return _factors(terms[0])


def _refit(model: Fitted, formula: str) -> Fitted:
    """Fit ``formula`` the way ``model`` was fitted, to the runs it used: those with no missing value it reads."""
    data = _results(model).model.data
    used = np.ones(len(data.frame), dtype=bool)
    used[getattr(data, "missing_row_idx", None) or []] = False
    runs = data.frame.iloc[used]
    return lm(formula, runs) if isinstance(model, Model) else smf.ols(formula, data=runs).fit()


def group_test(model: Fitted, terms: str | Sequence[str]) -> dict[str, Any]:
    """Test a group of a model's terms together: the model against itself without them.

    The F test of :func:`compare_models`, with the reduced model built by leaving ``terms`` out
    of ``model`` and fitted to the same runs, such as the first-order terms of a quadratic
    model, or the interactions an analysis pools. A categorical factor's main effect left out
    beside its interaction tests nothing, since the interaction's columns then span it, as in R.

    Parameters
    ----------
    model : Model or RegressionResultsWrapper
        A model from :func:`lm`, or a statsmodels least-squares fit made from a formula.
    terms : str or Sequence[str]
        The terms to test, each named as in the formula: ``"A"``, ``"A:B"`` (or ``"B:A"``),
        ``"I(A ** 2)"``, ``"Intercept"``.

    Returns
    -------
    dict[str, Any]
        What :func:`compare_models` returns, with the ``reduced`` formula it built and the
        ``terms`` tested.

    Raises
    ------
    TypeError
        If the model was not fitted from a formula.
    ValueError
        If ``terms`` is empty, names a term the model does not have, or names several terms in
        one string; and as :func:`compare_models` raises.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments import group_test, lm
    >>> runs = pd.DataFrame({"A": [-1, 1, -1, 1, 0, 0], "B": [-1, -1, 1, 1, 0, 0], "y": [3, 5, 4, 7, 4.6, 4.9]})
    >>> found = group_test(lm("y ~ A * B", runs), ["A:B"])
    >>> found["reduced"], found["df"], round(found["ss"], 4)
    ('y ~ A + B', 1, 0.25)
    """
    terms = [terms] if isinstance(terms, str) else list(terms)
    if not terms:
        raise ValueError("Name at least one term to test.")
    formula = getattr(_results(model).model, "formula", None)
    if formula is None:
        raise TypeError("group_test needs a model fitted from a formula, by lm or the statsmodels formula API.")
    description = ModelDesc.from_formula(formula)
    present = {_factors(term) for term in description.rhs_termlist}
    wanted = {name: _named_term(name) for name in terms}
    if missing := [name for name, factors in wanted.items() if factors not in present]:
        names = [term.name() for term in description.rhs_termlist]
        raise ValueError(f"The model has no term {missing}; its terms are {names}.")
    kept = [term for term in description.rhs_termlist if _factors(term) not in wanted.values()]
    reduced = ModelDesc(description.lhs_termlist, kept).describe()
    return compare_models(_refit(model, reduced), model) | {"reduced": reduced, "terms": terms}
