# (c) Kevin Dunn, 2010-2026. MIT License.
"""Stepwise model selection with effect heredity, scored by AICc (ENG-02, #639).

The search moves one term at a time, taking whichever single addition or removal
improves the score most, and stops when no move helps. Two rules keep it sensible
for designed experiments:

- **Strong heredity.** An interaction ``A:B`` may enter only once ``A`` and ``B`` are
  in, and a square ``I(A ** 2)`` only once ``A`` is; a main effect may leave only
  when no interaction or square built on it remains.
- **AICc**, ``AIC + 2p(p + 1) / (N - p - 1)``, by default. It equals AIC when the
  runs far outnumber the terms, and grows without bound as the model nears
  saturation, so in a supersaturated design (more factors than runs) the search
  cannot buy a smaller residual with ever more terms.

The search starts from the full candidate model (backward) when that model leaves
at least two residual degrees of freedom, and from the intercept alone (forward)
otherwise, as in a screening or supersaturated design.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix

_CRITERIA = ("aicc", "aic", "bic")


@dataclass(frozen=True)
class _Term:
    """A candidate model term and the main effects it needs (strong heredity)."""

    name: str
    parents: frozenset[str]


def _candidate_terms(factors: list[str], model: str | None) -> list[_Term]:
    """Terms the search may use: main effects, plus 2FIs for interactions, plus squares for quadratic."""
    terms = [_Term(f, frozenset()) for f in factors]
    if model == "main_effects":
        return terms
    terms += [_Term(f"{a}:{b}", frozenset({a, b})) for a, b in itertools.combinations(factors, 2)]
    if model == "quadratic":
        terms += [_Term(f"I({f} ** 2)", frozenset({f})) for f in factors]
    return terms


class _Scorer:
    """Score subsets of terms from one model matrix, built once by patsy."""

    def __init__(self, design_df: pd.DataFrame, y: np.ndarray, terms: list[_Term], criterion: str) -> None:
        full = dmatrix(" + ".join(t.name for t in terms), design_df, return_type="dataframe")
        self._x = full.to_numpy(dtype=float)
        slices = full.design_info.term_name_slices
        self._columns = {t.name: np.arange(self._x.shape[1])[slices[t.name]] for t in terms}
        self._y, self._n, self._criterion = y, len(y), criterion

    @property
    def n_full(self) -> int:
        """Coefficients in the full candidate model, intercept included."""
        return 1 + sum(len(c) for c in self._columns.values())

    def __call__(self, included: frozenset[str]) -> float:
        """Score the model with an intercept and the ``included`` terms (lower is better)."""
        cols = [0, *itertools.chain.from_iterable(self._columns[t] for t in sorted(included))]
        x = self._x[:, cols]
        beta, _, rank, _ = np.linalg.lstsq(x, self._y, rcond=None)
        rss = max(float(np.sum((self._y - x @ beta) ** 2)), 1e-300)
        n, p = self._n, int(rank)
        log_lik_term = n * np.log(2 * np.pi * rss / n) + n  # -2 log L for a Gaussian model
        if self._criterion == "bic":
            return float(log_lik_term + p * np.log(n))
        aic = log_lik_term + 2 * p
        if self._criterion == "aic":
            return float(aic)
        return float(aic + 2 * p * (p + 1) / (n - p - 1)) if n - p - 1 > 0 else np.inf


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


def _run_model_selection(
    design_df: pd.DataFrame,
    response_col: str,
    factor_cols: list[str],
    model: str | None = None,
    criterion: str = "aicc",
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
        ``"main_effects"``, ``"interactions"`` (also for ``None``) or ``"quadratic"``:
        the largest model the search may reach. Any other model (a formula, a Scheffé
        model) searches the interactions model and says so in a note.
    criterion : str
        ``"aicc"`` (default), ``"aic"`` or ``"bic"``.

    Returns
    -------
    dict[str, Any]
        ``{"model_selection": {...}}`` with the selected formula and terms, the
        criterion and its value, the direction the search started in, and the fit
        statistics of the selected model.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {', '.join(_CRITERIA)}; got {criterion!r}.")
    searched = model if model in ("main_effects", "interactions", "quadratic") else "interactions"
    terms = _candidate_terms(factor_cols, searched)
    y = design_df[response_col].to_numpy(dtype=float)
    score = _Scorer(design_df, y, terms, criterion)

    direction = "backward" if len(y) - score.n_full >= 2 else "forward"
    start = frozenset(t.name for t in terms) if direction == "backward" else frozenset()
    selected, value = _stepwise(score, terms, start)

    names = [t.name for t in terms if t.name in selected]  # in candidate order: mains, 2FIs, squares
    formula = f"{response_col} ~ {' + '.join(names)}" if names else f"{response_col} ~ 1"
    fit = smf.ols(formula, data=design_df).fit()
    result: dict[str, Any] = {
        "selected_formula": formula,
        "selected_terms": names,
        "candidate_model": searched,
        "criterion": criterion,
        "criterion_value": float(value),
        "direction": direction,
        "r_squared": float(fit.rsquared),
        "r_squared_adj": float(fit.rsquared_adj),
        "n_terms": int(fit.df_model),
    }
    if model is not None and searched != model:
        result["note"] = f"model_selection searches the standard models; {model!r} was searched as 'interactions'."
    return {"model_selection": result}
