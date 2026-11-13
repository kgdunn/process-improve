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

The search starts from the full candidate model (backward) when the runs can
estimate every term of it and leave at least two residual degrees of freedom, and
from the intercept alone (forward) otherwise: in a screening or supersaturated
design, and wherever two candidates are aliased with each other, as ``A:B`` and
``C:D`` are in a replicated half fraction. Started from both, neither of an aliased
pair could leave, since the model fits as well without either one; started from
none, the first of the pair to enter is the one whose main effects are in.

A candidate that adds nothing to what every model holds is no candidate: an
interaction confounded with the blocks, or the square of a factor run at two levels,
which is the intercept again. Of two moves that score the same, the search takes the
one named first, so the result does not depend on rounding.

A Scheffe mixture model has no intercept, since the components sum to 1: every
linear blending term stays in, and the search chooses among the non-linear blending
terms (``x1:x2``, and ``x1:x2:x3`` for the special cubic). A formula analysed as a
mixture model is searched the same way, over its own terms.

Any other formula is searched over its own terms too, each one free to enter once
every term of the formula that it contains is in: ``A:B`` after ``A`` and ``B``,
``A:B:C`` after those and ``A:B``, ``A:C`` and ``B:C`` as well, and ``I(A ** 2)``
after ``A``. A categorical factor enters and leaves whole, all its contrasts at once,
and has no square. The blocks of a blocked design are in every model the search
scores: left out, their differences would go to the residual and hide real effects.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix

from process_improve.experiments._analyses.mixture import spans_constant
from process_improve.experiments.designs_mixture_constrained import (
    SCHEFFE_DEGREE,
    SCHEFFE_MODELS,
    cubic_difference_term,
)
from process_improve.experiments.designs_utils import BLOCK_COL

_CRITERIA = ("aicc", "aic", "bic")

#: A whole power of one column, as a formula writes it: ``I(A ** 2)``.
_POWER = re.compile(r"I\((\w+) \*\* \d+\)")

#: Two scores closer than this are the same score: the difference is rounding, not the data.
_SAME_SCORE = 1e-9


@dataclass(frozen=True)
class _Term:
    """A candidate model term and the terms it needs in the model before it (strong heredity)."""

    name: str
    parents: frozenset[str]


@dataclass(frozen=True)
class _Formula:
    """An explicit formula as the search needs it.

    ``terms`` holds each term of the formula, less its intercept and its blocks, as the codes of its factors:
    ``("A",)``, ``("A", "B")`` for ``A:B``, ``("I(A ** 2)",)``. ``mixture`` says the formula was analysed as a
    mixture model, so its linear blending terms stay in and it has no intercept; ``intercept`` says whether
    any other formula has one.
    """

    terms: list[tuple[str, ...]]
    mixture: bool = False
    intercept: bool = True


def _formula_terms(terms: list[tuple[str, ...]]) -> list[_Term]:
    """Give each term of a formula the terms of the formula it needs in the model before it.

    A term needs every other term whose factors are among its own, and a whole power of a column needs the
    column: so an interaction follows its main effects and the interactions inside it, and a square its main
    effect, wherever the formula has them. A term the formula leaves out is not needed.
    """
    names = [":".join(codes) for codes in terms]
    sets = [frozenset(codes) for codes in terms]
    candidates = []
    for name, codes in zip(names, sets, strict=True):
        below = {other for other, inside in zip(names, sets, strict=True) if inside < codes}
        power = _POWER.fullmatch(name)
        if power and power[1] in names:
            below.add(power[1])
        candidates.append(_Term(name, frozenset(below)))
    return candidates


def _candidate_terms(
    factors: list[str], model: str | None, *, square_heredity: bool = True, categorical: Collection[str] = ()
) -> list[_Term]:
    """Terms the search may use: main effects, plus 2FIs for interactions, plus squares for quadratic.

    For a Scheffe model, the non-linear blending terms (the linear ones are always in). A square
    needs its main effect in the model only with ``square_heredity``, and a ``categorical`` factor
    has none: a category has no square.
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
        numeric = [f for f in factors if f not in categorical]
        terms += [_Term(f"I({f} ** 2)", frozenset({f} if square_heredity else ())) for f in numeric]
    return terms


class _Scorer:
    """Score subsets of terms from one model matrix, built once by patsy.

    ``fixed`` names the terms every model keeps: the linear blending terms of a mixture model, and the
    blocks. Every model has an intercept unless ``intercept`` is false, as for a mixture model, whose
    blending terms make up the constant. A term of several columns, a categorical factor or an interaction
    with one, is scored whole.

    ``terms`` holds the candidates the search may use: those given, less any that adds nothing to what every
    model keeps. The runs cannot tell such a term from the blocks or the intercept, so no model is better
    for it, and left in it would hold its parents in the model with it.

    Without an intercept, a term of categorical factors alone holds the constant: patsy gives the first such
    term of a formula a column for every level, and codes the later ones a column short, the constant being
    there already. The matrix is built once, from every term, so a model with a later one and not the first
    is scored with the constant put back: it is the model patsy fits when given those terms alone.
    """

    def __init__(
        self,
        design_df: pd.DataFrame,
        y: np.ndarray,
        terms: list[_Term],
        criterion: str,
        fixed: Sequence[str] = (),
        *,
        intercept: bool = True,
    ) -> None:
        names = [*fixed, *(t.name for t in terms)]
        formula = " + ".join(names if intercept else ["-1", *names]) or "1"
        full = dmatrix(formula, design_df, return_type="dataframe")
        info = full.design_info
        slices = info.term_name_slices
        positions = np.arange(full.shape[1])
        self._x = np.column_stack([full.to_numpy(dtype=float), np.ones(len(full))])  # the constant, in the last column
        kept = itertools.chain.from_iterable(positions[slices[f]] for f in fixed)
        self._fixed = [*([0] if intercept else []), *kept]
        categories = {f for f, kind in info.factor_infos.items() if kind.type == "categorical"}
        self._constant = (
            frozenset()
            if intercept
            else frozenset(t.name() for t in info.terms if t.factors and categories.issuperset(t.factors))
        )
        self._y, self._n, self._criterion = y, len(y), criterion
        always = self._rank(self._fixed)
        told = {t.name for t in terms if self._rank([*self._fixed, *positions[slices[t.name]]]) > always}
        self.terms = [_Term(t.name, t.parents & told) for t in terms if t.name in told]
        self._columns = {t.name: positions[slices[t.name]] for t in self.terms}

    def _rank(self, columns: Sequence[int]) -> int:
        """Rank of some columns of the model matrix: how many coefficients the runs can estimate of them."""
        return int(np.linalg.matrix_rank(self._x[:, columns])) if len(columns) else 0

    @property
    def estimable(self) -> bool:
        """Whether the runs estimate every candidate at once, with two residual degrees of freedom to spare."""
        every = [*self._fixed, *itertools.chain.from_iterable(self._columns.values())]
        rank = self._rank(every)
        return rank - self._rank(self._fixed) == len(every) - len(self._fixed) and self._n - rank >= 2

    def __call__(self, included: frozenset[str]) -> float:
        """Score the model with the fixed columns and the ``included`` terms (lower is better)."""
        cols = [*self._fixed, *itertools.chain.from_iterable(self._columns[t] for t in sorted(included))]
        if not cols:
            return np.inf  # no intercept, nothing kept and no term: that is no model, so one term stays
        if self._constant & included:
            cols.append(self._x.shape[1] - 1)
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
    """Every model one heredity-respecting addition or removal away from ``current``, in the order of ``terms``."""
    held = [t for t in terms if t.name in current]
    needed = frozenset().union(*(t.parents for t in held))
    additions = [current | {t.name} for t in terms if t.name not in current and t.parents <= current]
    removals = [current - {t.name} for t in held if t.name not in needed]
    return additions + removals


def _stepwise(score: _Scorer, start: frozenset[str]) -> tuple[frozenset[str], float]:
    """Take the best single move while it improves the score; each step strictly lowers it, so this ends.

    Of moves that score the same, the first is taken: an addition before a removal, each in the order of the
    candidates. So of two aliased terms the one named first enters, whatever the rounding of their scores.
    """
    current, best = start, score(start)
    while True:
        scored = [(score(candidate), candidate) for candidate in _moves(current, score.terms)]
        lowest = min((value for value, _ in scored), default=np.inf)
        if not lowest < best - _SAME_SCORE:
            return current, best
        best, current = next(pair for pair in scored if pair[0] <= lowest + _SAME_SCORE)


def _search(score: _Scorer) -> tuple[frozenset[str], float, str]:
    """Search backward from every candidate when the runs estimate them all, else forward from none."""
    direction = "backward" if score.estimable else "forward"
    start = frozenset(t.name for t in score.terms) if direction == "backward" else frozenset()
    selected, value = _stepwise(score, start)
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
    selected, _, _ = _search(_Scorer(data, y, terms, "aicc"))
    return [t.name for t in terms if t.name in selected]


def _numeric(column: pd.Series) -> bool:
    """Whether patsy reads ``column`` as a number: a column of booleans or of labels is a category."""
    return pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)


def _candidates(
    design_df: pd.DataFrame, factor_cols: list[str], model: str | None, formula: _Formula | None
) -> tuple[str | None, list[str], list[_Term], bool]:
    """Return the model searched, the linear blending terms it keeps, its candidates, and whether it has an intercept.

    A named model is searched as itself, and None as the interactions model. A Scheffe model, and a formula
    analysed as a mixture model, keep the factors as their linear blending terms and have no intercept.
    """
    if formula is None:
        searched = model if model in ("main_effects", "interactions", "quadratic", *SCHEFFE_MODELS) else "interactions"
        linear = list(factor_cols) if searched in SCHEFFE_MODELS else []
        categorical = [f for f in factor_cols if not _numeric(design_df[f])]
        return searched, linear, _candidate_terms(factor_cols, searched, categorical=categorical), not linear
    names = [":".join(codes) for codes in formula.terms]
    if formula.mixture:
        linear = [name for name in names if name in factor_cols]
        return model, linear, [_Term(name, frozenset()) for name in names if name not in linear], False
    return model, [], _formula_terms(formula.terms), formula.intercept


def _run_model_selection(
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
    model: str | None = None,
    criterion: str = "aicc",
    formula: _Formula | None = None,
    blocks: Sequence[str] = (),
) -> dict[str, Any]:
    """Select the terms of ``model`` that the data support, by stepwise search with heredity.

    Parameters
    ----------
    design_df : pandas.DataFrame
        Factor settings and the response.
    response_col : str
        Name of the response column.
    factor_cols : list[str]
        Factor column names. One that is not numeric is a categorical factor: all its
        contrasts enter and leave together, and it has no square.
    model : str or None
        ``"main_effects"``, ``"interactions"`` (also for ``None``), ``"quadratic"`` or a
        Scheffé model: the largest model the search may reach. An explicit formula is
        searched over its own terms, which ``formula`` holds.
    criterion : str
        ``"aicc"`` (default), ``"aic"`` or ``"bic"``.
    formula : _Formula or None
        The terms of ``model``, an explicit formula, and how it was fitted; None for a
        named model. A formula analysed as a mixture model keeps its linear blending
        terms (the factors), as a Scheffé model does, and the search chooses among the
        others. Any other formula may lose any term that no remaining term contains.
    blocks : Sequence[str]
        The terms that fit the blocks, as the model was fitted: the contrast columns of
        ``design_df``, or a formula's own block terms. They are in every model scored.

    Returns
    -------
    dict[str, Any]
        ``{"model_selection": {...}}`` with the selected formula and terms, the
        criterion and its value, the direction the search started in, and the fit
        statistics of the selected model. ``selected_terms`` holds the terms chosen, and
        the linear blending terms of a mixture model; the blocks are not among them, and
        ``selected_formula`` names them as a formula given to ``analyze_experiment``
        does, by the ``Block`` column. ``n_terms`` counts the selected model's columns,
        intercept and blocks included, as ``model_summary`` does.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(_CRITERIA)}; got {criterion!r}.")
    searched, linear, terms, intercept = _candidates(design_df, factor_cols, model, formula)
    fixed = [*linear, *blocks]
    data = design_df[design_df[[response_col, *factor_cols]].notna().all(axis=1)]
    y = data[response_col].to_numpy(dtype=float)
    selected, value, direction = _search(_Scorer(data, y, terms, criterion, fixed, intercept=intercept))

    chosen = [t.name for t in terms if t.name in selected]  # in candidate order: mains, 2FIs, squares
    if not (intercept or fixed or chosen):
        raise ValueError(
            f"No term of the model can be scored by {criterion.upper()} with {len(y)} run(s) and no intercept: "
            "model selection has nothing to choose between."
        )
    rhs = " + ".join([*([] if intercept else ["-1"]), *fixed, *chosen]) or "1"
    fit = smf.ols(f"{response_col} ~ {rhs}", data=data).fit()
    if not fit.k_constant and spans_constant(fit):
        # As analyze_experiment does: statsmodels misses the constant of blends rounded for printing.
        fit = smf.ols(f"{response_col} ~ {rhs}", data=data, hasconst=True).fit()
    # The contrast columns analyze_experiment made for the blocks are no columns of the caller's own data: the
    # formula names the Block column in their place, as a formula given to analyze_experiment does. A term of
    # the caller's formula that names the blocks itself (``C(Block)``, ``Block:A``) is written as it was.
    own = [block for block in blocks if block not in design_df.columns]
    shown = [*linear, *([BLOCK_COL] if len(own) < len(blocks) else []), *own, *chosen]
    result: dict[str, Any] = {
        "selected_formula": f"{response_col} ~ {' + '.join([*([] if intercept else ['-1']), *shown]) or '1'}",
        "selected_terms": [*linear, *chosen],
        "candidate_model": searched,
        "criterion": criterion,
        "criterion_value": float(value),
        "direction": direction,
        "r_squared": float(fit.rsquared),
        "r_squared_adj": float(fit.rsquared_adj),
        "n_terms": len(fit.params),
    }
    return {"model_selection": result}
