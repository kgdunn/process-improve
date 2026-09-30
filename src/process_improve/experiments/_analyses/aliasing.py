# (c) Kevin Dunn, 2010-2026. MIT License.
"""Fold exactly aliased model terms into one estimable effect per alias chain (#16).

In a fractional factorial some model columns are identical, or identical up to sign:
with D = ABC, the A:B column *is* the C:D column. Least squares cannot split their
joint effect, so the pseudo-inverse fit behind a rank-deficient model shares it out
evenly, and every aliased term reports half of it. A Pareto chart of those numbers
shows two half-size bars where the design measured one effect, and Lenth's method,
which assumes its effects are separate estimates, sees duplicates. What the design
does determine is the chain's combined coefficient, so that is what is reported,
under a name that says what it contains: ``"A:B + C:D"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import Bunch
from statsmodels.regression.linear_model import RegressionResultsWrapper

#: Two model columns whose cosine is within this of plus or minus one are aliased.
_ALIASED = 1e-9


def _word_length(term: str) -> int:
    """Return how many factors a term multiplies: 0 for the intercept, 1 for ``A``, 2 for ``A:B``."""
    return 0 if term == "Intercept" else term.count(":") + 1


def alias_chains(ols_result: RegressionResultsWrapper) -> list[list[tuple[str, float]]]:
    """Group the model's columns into chains of exact aliases, each led by its shortest term.

    Parameters
    ----------
    ols_result : RegressionResultsWrapper
        A fitted statsmodels OLS result.

    Returns
    -------
    list[list[tuple[str, float]]]
        One list per chain, in model order: ``(term, factor)`` pairs, the leader first,
        where each term's column is ``factor`` times the leader's (``-1`` for an
        anti-alias in a coded design). A term aliased with nothing is a chain of one.
    """
    exog = np.asarray(ols_result.model.exog, dtype=float)
    names = [str(name) for name in ols_result.model.exog_names]
    norms = np.linalg.norm(exog, axis=0)
    usable = norms > 0
    unit = np.divide(exog, norms, out=np.zeros_like(exog), where=usable)
    aliased = (np.abs(unit.T @ unit) >= 1 - _ALIASED) & np.outer(usable, usable)
    assigned = np.zeros(len(names), dtype=bool)
    chains = []
    for column in range(len(names)):
        if assigned[column]:
            continue
        members = [k for k in range(len(names)) if not assigned[k] and (k == column or aliased[column, k])]
        assigned[members] = True
        leader = min(members, key=lambda k: (_word_length(names[k]), k))
        ordered = sorted(members, key=lambda k: (k != leader, _word_length(names[k]), k))
        scale = float(exog[:, leader] @ exog[:, leader])
        chains.append(
            [(names[k], float(exog[:, k] @ exog[:, leader]) / scale if len(members) > 1 else 1.0) for k in ordered]
        )
    return chains


def estimable_effects(ols_result: RegressionResultsWrapper) -> Bunch:
    """Return one coefficient per alias chain, the part of the fit the design determines.

    A chain's coefficient is the sum of its members' coefficients, each weighted by
    its column's factor, and so does not depend on how the fit split it. Its
    standard error follows from the same weights. Without aliasing, this is exactly
    the fitted coefficients and their standard errors.

    Parameters
    ----------
    ols_result : RegressionResultsWrapper
        A fitted statsmodels OLS result.

    Returns
    -------
    Bunch
        ``coefficients`` (a Series, one entry per chain, the intercept excluded),
        ``std_errors`` (the same, or None without residual degrees of freedom),
        ``chains`` (chain name to its member terms, for chains of two or more) and
        ``confounded_with_mean`` (terms aliased with the intercept, which no
        effect can be estimated for).
    """
    params = ols_result.params.to_numpy(dtype=float)
    position = {str(name): k for k, name in enumerate(ols_result.model.exog_names)}
    covariance = ols_result.cov_params().to_numpy() if int(ols_result.df_resid) > 0 else None
    labels: list[str] = []
    coefficients: list[float] = []
    errors: list[float] = []
    chains: dict[str, list[str]] = {}
    confounded: list[str] = []
    for chain in alias_chains(ols_result):
        (leader, _), *rest = chain
        if leader == "Intercept":
            confounded.extend(term for term, _ in rest)
            continue
        label = leader + "".join(f" {'+' if factor > 0 else '-'} {term}" for term, factor in rest)
        members = [position[term] for term, _ in chain]
        weights = np.array([factor for _, factor in chain])
        labels.append(label)
        coefficients.append(float(weights @ params[members]))
        if covariance is not None:
            errors.append(float(np.sqrt(weights @ covariance[np.ix_(members, members)] @ weights)))
        if rest:
            chains[label] = [term for term, _ in chain]
    return Bunch(
        coefficients=pd.Series(coefficients, index=labels, dtype=float),
        std_errors=pd.Series(errors, index=labels, dtype=float) if covariance is not None else None,
        chains=chains,
        confounded_with_mean=confounded,
    )
