# (c) Kevin Dunn, 2010-2026. MIT License.
"""Stepwise model selection with effect heredity, scored by AICc (ENG-02, #639).

The search moves one term at a time, taking whichever single addition or removal
improves the score most, and stops when no move helps. Two rules keep it sensible
for designed experiments:

- **Strong heredity.** An interaction ``A:B`` may enter only once ``A`` and ``B`` are
  in, and a square ``I(A ** 2)`` only once ``A`` is; a main effect may leave only
  when no interaction or square built on it remains. A caller may free the squares
  from it (``square_heredity=False``): a factor whose best setting is in the middle of
  its range has a square and no main effect.
- **AICc**, ``AIC + 2K(K + 1) / (N - K - 1)``, by default, where ``K = p + 1``
  counts the ``p`` regression coefficients and the error variance (Hurvich and
  Tsai 1989; Burnham and Anderson 2002, section 2.4). It equals AIC when the runs
  far outnumber the terms, and is infinite once ``N - p - 2 <= 0``, so in a
  supersaturated design (more factors than runs) the search cannot buy a smaller
  residual with ever more terms. AIC and BIC count ``K`` the same way.

The search starts from the full candidate model (backward) when that model leaves
at least two residual degrees of freedom, and from the intercept alone (forward)
otherwise, as in a screening or supersaturated design.

A Scheffe mixture model has no intercept, since the components sum to 1: every
linear blending term stays in, and the search chooses among the non-linear blending
terms (``x1:x2``, and ``x1:x2:x3`` for the special cubic). A formula analysed as a
mixture model is searched the same way, over its own terms. So are mixture components
beside process factors: a formula over them without an intercept is searched over its
own terms, and a standard model is the mixture-process model of that order
(:func:`~process_improve.experiments.designs_mixture_process.mixture_process_model`),
whose process main effects enter through the products ``x1:T``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from patsy import dmatrix

from process_improve.experiments._analyses.mixture import spans_constant
from process_improve.experiments.designs_mixture_constrained import (
    SCHEFFE_DEGREE,
    SCHEFFE_MODELS,
    cubic_difference_term,
)
from process_improve.experiments.designs_mixture_process import (
    PROCESS_MODELS,
    ReducedRankFormula,
    mixture_process_model,
)

_CRITERIA = ("aicc", "aic", "bic")


@dataclass(frozen=True)
class _Term:
    """A candidate model term and the main effects it needs (strong heredity)."""

    name: str
    parents: frozenset[str]


def _candidate_terms(factors: list[str], model: str | None, *, square_heredity: bool = True) -> list[_Term]:
    """Terms the search may use: main effects, plus 2FIs for interactions, plus squares for quadratic.

    For a Scheffe model, the non-linear blending terms (the linear ones are always in). A square
    needs its main effect in the model only with ``square_heredity``.
    """
    if model in SCHEFFE_MODELS:
        blends = [
            _Term(":".join(combo), frozenset())
            for size in range(2, SCHEFFE_DEGREE[model] + 1)
            for combo in itertools.combinations(factors, size)
        ]
        if model == "scheffe_cubic":
            blends += [_Term(cubic_difference_term(a, b), frozenset()) for a, b in itertools.combinations(factors, 2)]
        return blends
    terms = [_Term(f, frozenset()) for f in factors]
    if model == "main_effects":
        return terms
    terms += [_Term(f"{a}:{b}", frozenset({a, b})) for a, b in itertools.combinations(factors, 2)]
    if model == "quadratic":
        terms += [_Term(f"I({f} ** 2)", frozenset({f} if square_heredity else ())) for f in factors]
    return terms


def _mixture_process_terms(
    components: list[str], process: list[str], model: str, categorical: list[str]
) -> list[_Term]:
    """Terms the search may use on mixture-process data: the mixture-process ``model`` less its linear blending terms.

    They are searched free of heredity, as a Scheffe model's are: a process main effect enters
    through the products ``x_i:z_k``, so ``z_k`` itself is no term to need.
    """
    levels = dict.fromkeys(categorical, ())  # only which factors are categorical matters here: they have no square
    terms = mixture_process_model(components, process, model, levels).terms
    return [_Term(name, frozenset()) for name in (":".join(a.code for a in t) for t in terms) if name not in components]


def _reduced_rank(formula: str, data: pd.DataFrame) -> str:
    """Return ``formula``, which has no intercept, with each categorical column of ``data`` coded in ``L - 1`` columns.

    Patsy would give a categorical factor a column for every level wherever the term below it
    is missing, and those columns would add up to ones the blends already span.
    analyze_experiment codes the factors beside mixture components the same way (see
    :class:`~process_improve.experiments.designs_mixture_process.ReducedRankFormula`).
    """
    return ReducedRankFormula(formula, _categorical(data))


def _categorical(data: pd.DataFrame) -> list[str]:
    """Return the columns of ``data`` that patsy treats as categorical: booleans and text."""
    return [c for c in data.columns if is_bool_dtype(data[c]) or not is_numeric_dtype(data[c])]


class _Scorer:
    """Score subsets of terms from one model matrix, built once by patsy.

    ``fixed`` names the terms every model keeps (the linear blending terms of a Scheffe
    model, which has no intercept); without them, every model has an intercept.
    """

    def __init__(
        self,
        design_df: pd.DataFrame,
        y: np.ndarray,
        terms: list[_Term],
        criterion: str,
        fixed: list[str] | None = None,
    ) -> None:
        names = " + ".join([*(fixed or []), *(t.name for t in terms)])
        full = dmatrix(
            _reduced_rank(f"-1 + {names}", design_df) if fixed else names, design_df, return_type="dataframe"
        )
        self._x = full.to_numpy(dtype=float)
        slices = full.design_info.term_name_slices
        positions = np.arange(self._x.shape[1])
        self._fixed = list(itertools.chain.from_iterable(positions[slices[f]] for f in fixed)) if fixed else [0]
        self._columns = {t.name: positions[slices[t.name]] for t in terms}
        self._y, self._n, self._criterion = y, len(y), criterion

    @property
    def n_full(self) -> int:
        """Coefficients in the full candidate model, the always-included columns too."""
        return len(self._fixed) + sum(len(c) for c in self._columns.values())

    def __call__(self, included: frozenset[str]) -> float:
        """Score the model with the fixed columns and the ``included`` terms (lower is better)."""
        cols = [*self._fixed, *itertools.chain.from_iterable(self._columns[t] for t in sorted(included))]
        x = self._x[:, cols]
        beta, _, rank, _ = np.linalg.lstsq(x, self._y, rcond=None)
        rss = max(float(np.sum((self._y - x @ beta) ** 2)), 1e-300)
        n = self._n
        k = int(rank) + 1  # the coefficients and the error variance
        log_lik_term = n * np.log(2 * np.pi * rss / n) + n  # -2 log L for a Gaussian model
        if self._criterion == "bic":
            return float(log_lik_term + k * np.log(n))
        aic = log_lik_term + 2 * k
        if self._criterion == "aic":
            return float(aic)
        return float(aic + 2 * k * (k + 1) / (n - k - 1)) if n - k - 1 > 0 else np.inf


def _moves(current: frozenset[str], terms: list[_Term]) -> list[frozenset[str]]:
    """Every model one heredity-respecting addition or removal away from ``current``."""
    by_name = {t.name: t for t in terms}
    additions = [current | {t.name} for t in terms if t.name not in current and t.parents <= current]
    removals = [
        current - {name}
        for name in current
        if not any(name in by_name[other].parents for other in current if other != name)
    ]
    return additions + removals


def _stepwise(score: _Scorer, terms: list[_Term], start: frozenset[str]) -> tuple[frozenset[str], float]:
    """Take the best single move while it improves the score; each step strictly lowers it, so this ends."""
    current, best = start, score(start)
    while True:
        scored = [(score(candidate), candidate) for candidate in _moves(current, terms)]
        if not scored:
            return current, best
        value, candidate = min(scored, key=lambda pair: pair[0])
        if not value < best - 1e-9:
            return current, best
        current, best = candidate, value


def _search(score: _Scorer, terms: list[_Term], n: int) -> tuple[frozenset[str], float, str]:
    """Search backward from every term when they leave two residual degrees of freedom, else forward from none."""
    direction = "backward" if n - score.n_full >= 2 else "forward"
    start = frozenset(t.name for t in terms) if direction == "backward" else frozenset()
    selected, value = _stepwise(score, terms, start)
    return selected, value, direction


def _proposed_terms(design_df: pd.DataFrame, response_col: str, factor_cols: list[str], model: str) -> list[str]:
    """Return the terms of a standard ``model`` that the stepwise search (AICc) selects, squares free of heredity.

    The same search as :func:`_run_model_selection`, except that a square may enter before its
    main effect does: a factor whose best setting is in the middle of its range has a square and
    no main effect, and strong heredity would keep that square out for ever.
    """
    terms = _candidate_terms(factor_cols, model, square_heredity=False)
    data = design_df[design_df[[response_col, *factor_cols]].notna().all(axis=1)]
    y = data[response_col].to_numpy(dtype=float)
    selected, _, _ = _search(_Scorer(data, y, terms, "aicc"), terms, len(y))
    return [t.name for t in terms if t.name in selected]


def _run_model_selection(
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
    model: str | None = None,
    criterion: str = "aicc",
    mixture_terms: list[str] | None = None,
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Select the terms of ``model`` that the data support, by stepwise search with heredity.

    Parameters
    ----------
    design_df : pandas.DataFrame
        Factor settings and the response.
    response_col : str
        Name of the response column.
    factor_cols : list[str]
        Factor column names.
    model : str or None
        ``"main_effects"``, ``"interactions"`` (also for ``None``), ``"quadratic"`` or a
        Scheffé model: the largest model the search may reach. A formula analysed as a
        mixture model is searched over ``mixture_terms``; any other formula searches the
        interactions model and says so in a note.
    criterion : str
        ``"aicc"`` (default), ``"aic"`` or ``"bic"``.
    mixture_terms : list[str] or None
        The terms of ``model``, a formula analysed as a mixture model or a formula over
        ``components`` that holds the constant without an intercept, or None for any other
        model. As in a Scheffé model, its linear blending terms (the components) all stay
        in, and the search chooses among the others.
    components : list[str] or None
        The mixture components, when ``factor_cols`` holds them beside process factors;
        None otherwise. A standard model is then searched as the mixture-process model of
        that order, without an intercept: its linear blending terms stay in.

    Returns
    -------
    dict[str, Any]
        ``{"model_selection": {...}}`` with the selected formula and terms, the
        criterion and its value, the direction the search started in, and the fit
        statistics of the selected model. ``n_terms`` counts the selected model's
        columns, intercept included, as ``model_summary`` does.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(_CRITERIA)}; got {criterion!r}.")
    searchable = ("main_effects", "interactions", "quadratic", *SCHEFFE_MODELS)
    searched = model if mixture_terms is not None or model in searchable else "interactions"
    linear = components or factor_cols
    if mixture_terms is not None:
        fixed = [t for t in mixture_terms if t in linear]
        terms = [_Term(t, frozenset()) for t in mixture_terms if t not in linear]
    elif components and searched in PROCESS_MODELS:
        process = [f for f in factor_cols if f not in components]
        fixed = list(components)
        terms = _mixture_process_terms(components, process, searched, _categorical(design_df[process]))
    else:
        fixed = list(factor_cols) if searched in SCHEFFE_MODELS else []
        terms = _candidate_terms(factor_cols, searched)
    mixture = bool(fixed)
    data = design_df[design_df[[response_col, *factor_cols]].notna().all(axis=1)]
    y = data[response_col].to_numpy(dtype=float)
    selected, value, direction = _search(_Scorer(data, y, terms, criterion, fixed=fixed), terms, len(y))

    names = [t.name for t in terms if t.name in selected]  # in candidate order: mains, 2FIs, squares
    if mixture:
        formula = f"{response_col} ~ -1 + {' + '.join([*fixed, *names])}"
    else:
        formula = f"{response_col} ~ {' + '.join(names)}" if names else f"{response_col} ~ 1"
    coded = _reduced_rank(formula, data) if mixture else formula
    fit = smf.ols(coded, data=data).fit()
    if not fit.k_constant and spans_constant(fit):
        # As analyze_experiment does: statsmodels misses the constant of blends rounded for printing.
        fit = smf.ols(coded, data=data, hasconst=True).fit()
    result: dict[str, Any] = {
        "selected_formula": formula,
        "selected_terms": [*fixed, *names],
        "candidate_model": searched,
        "criterion": criterion,
        "criterion_value": float(value),
        "direction": direction,
        "r_squared": float(fit.rsquared),
        "r_squared_adj": float(fit.rsquared_adj),
        "n_terms": len(fit.params),
    }
    if model is not None and searched != model:
        result["note"] = f"model_selection searches the standard models; {model!r} was searched as 'interactions'."
    return {"model_selection": result}
