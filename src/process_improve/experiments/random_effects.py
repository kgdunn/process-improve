# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Random effects: variance components by REML, expected mean squares, and intervals of a one-way random model.

A factor is *random* when its levels are a sample of a larger population (the parts of a gauge
study, the batches of a raw material, the operators of a plant) and the question is how much
the response varies from level to level, not which level is best. The response then varies by
the sum of the components :math:`\sigma^2_k`, one per random term, and the error variance.

- :func:`variance_components` fits them by REML, with the fixed factors' effects and their F
  tests, and the best linear unbiased predictions of the random effects.
- :func:`expected_mean_squares` derives, for a balanced design, what each mean square of the
  analysis of variance estimates, under the restricted or the unrestricted mixed model, and so
  which mean square tests each term.
- :func:`intraclass_correlation_interval` gives the exact interval for the share of the variance
  that lies between the levels of a single random factor.

Examples
--------
>>> from process_improve.experiments.random_effects import expected_mean_squares
>>> ems = expected_mean_squares({"part": 20, "operator": 3}, random=["part", "operator"], replicates=2)
>>> ems["ems"]["operator"]
{'operator': 40, 'part:operator': 2, 'Error': 1}
>>> ems["error_term"]["operator"]
'part:operator'
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from patsy import ModelDesc
from scipy import stats

from process_improve.experiments.models import validate_formula_is_safe
from process_improve.regression._variance_components import fit_variance_components

ERROR = "Error"


# --------------------------------------------------------------------------- the model matrix of the fixed factors
def sum_coded(column: pd.Series, name: str) -> pd.DataFrame:
    """Return a factor's columns coded to sum to zero: one per level but the last, which takes -1 in each."""
    values = column.astype(str)
    levels = sorted(values.unique(), key=lambda v: (len(v), v))
    last = (values == levels[-1]).astype(float)
    return pd.DataFrame({f"{name}_{level}": (values == level).astype(float) - last for level in levels[:-1]})


def fixed_columns(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a fixed term's columns: a factor's sum-coded ones, or for ``a:b`` the products of a's and b's."""
    if ":" not in name:
        return sum_coded(table[name], name)
    first, second = (sum_coded(table[part], part) for part in name.split(":"))
    return pd.DataFrame(
        {
            f"{name}_{a.partition('_')[2]}.{b.partition('_')[2]}": first[a] * second[b]
            for a in first.columns
            for b in second.columns
        }
    )


def _group_columns(name: str) -> list[str]:
    """Return the columns whose combination labels a random term: ``"a:b"``, or ``"b(a)"`` for b nested in a."""
    if "(" in name:
        inner, _, outer = name.partition("(")
        return [*outer.rstrip(")").split(":"), *inner.split(":")]
    return name.split(":")


# --------------------------------------------------------------------------- REML variance components
def variance_components(
    design_matrix: pd.DataFrame,
    formula: str,
    random: Sequence[str],
    bounded: bool = True,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    r"""Fit a linear model with random intercepts by REML, and report its variance components.

    Parameters
    ----------
    design_matrix : DataFrame
        One row per observation, holding the response and the factors.
    formula : str
        The response and the fixed terms, ``"y ~ operator"`` (``"y ~ 1"`` for the mean alone): a
        plain formula of factors and ``a:b`` interactions of them (``a * b`` expands to both and
        their interaction), each coded to sum to zero over its levels. The overall mean is always
        fixed.
    random : sequence of str
        The random terms: a factor, ``"a:b"`` for the random interaction of two factors (each
        combination of their levels), or ``"b(a)"`` for ``b`` nested in ``a`` (each ``b`` within
        each ``a``, so labels may repeat from one ``a`` to the next).
    bounded : bool
        Keep every component at or above zero, as lme4 does (the default). ``False`` lets a
        component go negative, as JMP's unbounded estimates do: in a balanced design they are
        then the analysis-of-variance estimates.
    significance_level : float
        For the intervals and the tests' ``significant`` flags.

    Returns
    -------
    dict[str, Any]
        - ``components``: each random term's variance, and the ``"residual"``'s; ``percent``,
          each one's share of their total;
        - ``std_errors`` and ``intervals``: the asymptotic standard error of each component and its
          Wald interval, estimate :math:`\pm z\,` standard error (JMP's, for a random term), and for
          the residual the chi-squared interval :math:`\nu s^2 / \chi^2` on its degrees of freedom;
        - ``reml_criterion``: minus twice the restricted log-likelihood (JMP's "-2 LogLikelihood");
        - ``fixed_effects``: each fixed column's estimate, standard error, Satterthwaite degrees of
          freedom, t and p; ``tests``: each fixed term's F test on Satterthwaite's denominator
          degrees of freedom;
        - ``blups``: each random term's best linear unbiased predictions, by level;
        - ``r_squared`` and ``r_squared_adj``: one less the residual sum of squares about the
          fitted values with the random effects predicted, over the total about the mean, as JMP
          reports them for a REML fit, the adjusted one on the fixed parameters' count.
    """
    table = design_matrix.reset_index(drop=True)
    response_column, fixed = _response_and_terms(formula, [str(c) for c in table.columns])
    y = table[response_column].to_numpy(dtype=float)
    blocks = [pd.DataFrame({"Intercept": np.ones(len(y))}), *(fixed_columns(table, name) for name in fixed)]
    design = pd.concat(blocks, axis=1).astype(float)
    labels = [table[_group_columns(name)].astype(str).agg(":".join, axis=1) for name in random]
    codes = [pd.factorize(label) for label in labels]
    fit = fit_variance_components(y, design.to_numpy(), [c for c, _ in codes], bounded=bounded)
    names = [*random, "residual"]
    values = [*map(float, fit.variances), float(fit.sigma2)]
    components = dict(zip(names, values, strict=True))
    estimated = [name for name, value in components.items() if name == "residual" or value != 0]
    z = float(stats.norm.ppf(1 - significance_level / 2))
    std_errors = {name: float(np.sqrt(fit.vc_cov[i, i])) for i, name in enumerate(estimated)}
    intervals = {name: [components[name] - z * se, components[name] + z * se] for name, se in std_errors.items()}
    indicators = [np.eye(len(levels))[c] for c, levels in codes]
    df_residual = len(y) - int(np.linalg.matrix_rank(np.hstack([design.to_numpy(), *indicators])))
    if df_residual > 0:  # the chi-squared interval of the error variance, on the error's own degrees of freedom
        scaled = df_residual * components["residual"]
        intervals["residual"] = [
            scaled / float(stats.chi2.ppf(1 - significance_level / 2, df_residual)),
            scaled / float(stats.chi2.ppf(significance_level / 2, df_residual)),
        ]
    total = sum(values)
    starts = np.cumsum([0, *(len(levels) for _, levels in codes)])
    blups = {
        name: dict(zip(map(str, levels), map(float, fit.blups[start : start + len(levels)]), strict=True))
        for name, (_, levels), start in zip(random, codes, starts, strict=False)
    }
    n, p = len(y), design.shape[1]
    r_squared = 1 - float(np.sum((y - fit.fitted) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return {
        "components": components,
        "percent": {name: 100 * value / total for name, value in components.items()},
        "std_errors": std_errors,
        "intervals": intervals,
        "reml_criterion": float(fit.reml_criterion),
        "bounded": bounded,
        "fixed_effects": _fixed_effects(fit, list(design.columns)),
        "tests": _fixed_tests(fit, list(design.columns), fixed, significance_level),
        "blups": blups,
        "r_squared": r_squared,
        "r_squared_adj": 1 - (1 - r_squared) * (n - 1) / (n - p),
    }


def _response_and_terms(formula: str, columns: Sequence[str]) -> tuple[str, list[str]]:
    """Return a formula's response and its terms, ``a:b`` for an interaction, the intercept left out."""
    validate_formula_is_safe(formula, columns)
    description = ModelDesc.from_formula(formula)
    if len(description.lhs_termlist) != 1:
        raise ValueError(f"The formula {formula!r} must name one response on the left of '~'.")
    response = description.lhs_termlist[0].name()
    return response, [term.name() for term in description.rhs_termlist if term.factors]


def _fixed_effects(fit: Any, columns: list[str]) -> list[dict[str, float | str]]:  # noqa: ANN401 - the REML fit
    """Return each fixed column's estimate, standard error, Satterthwaite df, t and two-sided p."""
    rows = []
    for i, column in enumerate(columns):
        contrast = np.eye(len(columns))[i]
        se = float(np.sqrt(fit.cov_beta[i, i]))
        df = fit.satterthwaite_df(contrast)
        t = float(fit.beta[i]) / se
        p = float(2 * stats.t.sf(abs(t), df))
        rows.append({"term": column, "estimate": float(fit.beta[i]), "std_error": se, "df": df, "t": t, "p": p})
    return rows


def _fixed_tests(fit: Any, columns: list[str], fixed: Sequence[str], alpha: float) -> list[dict[str, Any]]:  # noqa: ANN401
    """Return each fixed term's F test, on Satterthwaite's denominator degrees of freedom."""
    tests = []
    for name in fixed:
        picked = [i for i, column in enumerate(columns) if column.startswith(f"{name}_")]
        f, df_den = fit.f_test(np.eye(len(columns))[picked])
        p = float(stats.f.sf(f, len(picked), df_den))
        tests.append({"term": name, "f": f, "df_num": len(picked), "df_den": df_den, "p": p, "significant": p < alpha})
    return tests


# --------------------------------------------------------------------------- expected mean squares
def _terms(levels: Mapping[str, int], nested: Mapping[str, str]) -> list[tuple[str, frozenset[str], frozenset[str]]]:
    """Return every term of the full model: its name, its own factors, and every subscript it carries."""

    def nests(factor: str) -> set[str]:
        outer = nested.get(factor)
        return set() if outer is None else {outer, *nests(outer)}

    found = []
    factors = list(levels)
    for size in range(1, len(factors) + 1):
        for own in itertools.combinations(factors, size):
            carried = set(own).union(*(nests(f) for f in own))
            if any(nested.get(g) in own for g in own):
                continue  # a factor and the factor it is nested in never interact
            outers = sorted(carried - set(own), key=factors.index)
            name = ":".join(own) + (f"({':'.join(outers)})" if outers else "")
            found.append((name, frozenset(own), frozenset(carried)))
    return found


_Term = tuple[str, frozenset[str], frozenset[str]]


def _ems_row(
    term: _Term,
    model: list[_Term],
    rules: tuple[Mapping[str, bool], set[str], bool],
    coefficient: Any,  # noqa: ANN401 - the multiple of a component, from the subscripts it carries
) -> tuple[dict[str, int], list[str]]:
    """Return one term's expected mean square, component by component, and the fixed forms it holds.

    *rules* are which terms are random, the random factors, and whether the model is restricted.
    """
    is_random, random_set, restricted = rules
    name, _, carried = term
    row: dict[str, int] = {}
    forms = []
    for other, other_own, other_carried in model:
        if not carried <= other_carried:
            continue  # a component appears only in the mean squares of the terms it carries
        if other == name:
            row[name] = coefficient(other_carried)
            if not is_random[name]:
                forms.append(name)
        elif is_random[other]:
            if restricted and any(f not in random_set for f in other_own - carried):
                continue  # it sums to zero over a fixed factor of its own the term does not carry
            row[other] = coefficient(other_carried)
        elif not restricted and not is_random[name]:
            forms.append(other)  # Minitab's Q[1,4]: the fixed terms that carry this one
    row[ERROR] = 1
    return row, forms


def _error_terms(ems: Mapping[str, Mapping[str, int]], names: list[str]) -> dict[str, str | None]:
    """Return, for each term, the term whose expected mean square is the term's less its own part."""
    found: dict[str, str | None] = {}
    for name in names:
        wanted = {component: c for component, c in ems[name].items() if component != name}
        found[name] = next((other for other in ems if other != name and ems[other] == wanted), None)
    return found


def expected_mean_squares(
    levels: Mapping[str, int],
    random: Sequence[str] = (),
    replicates: int = 1,
    restricted: bool = True,
    terms: Sequence[str] | None = None,
) -> dict[str, Any]:
    r"""Derive the expected mean squares of a balanced design's analysis of variance, and each term's error term.

    Each term's mean square estimates the error variance plus a multiple of every random component
    whose subscripts include the term's, and, for a fixed term, a multiple of its own quadratic
    form :math:`\sum \tau^2 / (a - 1)` (Montgomery, 2013, section 13.5). The multiple is the
    replicates times the levels of every factor the component does not carry. Under the
    *restricted* model, whose random interactions with a fixed factor sum to zero over it, a
    component is left out wherever it carries a fixed factor the term does not; under the
    *unrestricted* model it is kept, and a fixed term's quadratic form takes in those of the fixed
    terms that carry it, as Minitab writes ``Q[1,4]``. A factor a component is nested in does not
    count: its subscript is not free to sum over.

    Parameters
    ----------
    levels : mapping of str to int
        Each factor's number of levels, in the order the terms are to be listed. A nested factor
        is named with the factor it is nested in, ``{"supplier": 3, "batch(supplier)": 4}``, and
        its count is its levels within each level of that factor.
    random : sequence of str
        The random factors, by their own names (``"batch"``); the rest are fixed.
    replicates : int
        Observations in each cell.
    restricted : bool
        The restricted model (the default) or the unrestricted one.
    terms : sequence of str, optional
        The terms of a reduced model, named as the result names them; the others are pooled into
        the error, and their components leave every expected mean square. All of them by default.

    Returns
    -------
    dict[str, Any]
        ``terms``, in order (main effects, then interactions of two, three, ...; ``"b(a)"`` for a
        nested factor, ``"a:c(b)"`` for its interactions), then ``"Error"``; ``random``, which terms
        are random; ``ems``, for each term the coefficient of each component (a term's own fixed
        quadratic form under the term's own name), the error variance under ``"Error"``;
        ``fixed_forms``, for each fixed term the fixed terms whose quadratic forms its mean square
        holds; ``df``, each term's degrees of freedom (a fixed term's coefficient over them is the
        multiple of :math:`\sum \tau^2` that Montgomery writes); and ``error_term``, for each term
        the one whose expected mean square differs from it by the term's own part alone, or
        ``None`` when no single mean square does (an approximate F test is then needed).
    """
    levels, nested = _nesting(levels)
    every = _terms(levels, nested)
    if terms is not None:
        unknown = set(terms) - {name for name, _, _ in every}
        if unknown:
            raise ValueError(f"No such terms: {sorted(unknown)}; the full model's are {[n for n, _, _ in every]}.")
    model = [t for t in every if terms is None or t[0] in terms]
    random_set = set(random)
    is_random = {name: bool(own & random_set) for name, own, _ in model}

    def coefficient(carried: frozenset[str]) -> int:
        return replicates * int(np.prod([levels[f] for f in levels if f not in carried]))

    ems: dict[str, dict[str, int]] = {}
    fixed_forms: dict[str, list[str]] = {}
    for term in model:
        ems[term[0]], forms = _ems_row(term, model, (is_random, random_set, restricted), coefficient)
        if not is_random[term[0]]:
            fixed_forms[term[0]] = forms
    ems[ERROR] = {ERROR: 1}
    error_term = _error_terms(ems, [name for name, _, _ in model])
    df = {
        name: int(np.prod([levels[f] - 1 for f in own]) * np.prod([levels[f] for f in carried - own]))
        for name, own, carried in model
    }
    return {
        "terms": [name for name, _, _ in model] + [ERROR],
        "random": {name: is_random[name] for name, _, _ in model},
        "ems": ems,
        "fixed_forms": fixed_forms,
        "df": df,
        "error_term": error_term,
    }


def _nesting(levels: Mapping[str, int]) -> tuple[dict[str, int], dict[str, str]]:
    """Split ``"b(a)"`` names into the factor ``b`` and its nesting in ``a``; return the levels and the nesting."""
    plain: dict[str, int] = {}
    nested: dict[str, str] = {}
    for name, count in levels.items():
        factor, _, outer = name.partition("(")
        plain[factor.strip()] = count
        if outer:
            nested[factor.strip()] = outer.rstrip(")").strip()
    return plain, nested


# --------------------------------------------------------------------------- a single random factor
def intraclass_correlation_interval(
    ms_treatment: float,
    ms_error: float,
    df_treatment: int,
    n: int,
    conflevel: float = 0.95,
) -> dict[str, float]:
    r"""Return the exact interval for the intraclass correlation of a balanced one-way random model.

    With :math:`F_0 = MS_{Treatments}/MS_E`, the limits :math:`L` and :math:`U` of
    :math:`\sigma^2_\tau / \sigma^2` are :math:`\frac1n(F_0 / F_{\alpha/2} - 1)` and
    :math:`\frac1n(F_0 / F_{1-\alpha/2} - 1)`, with :math:`F_{\alpha/2}` the upper :math:`\alpha/2`
    point of F on ``df_treatment`` and the error's :math:`a(n - 1)` degrees of freedom, for
    :math:`a` = ``df_treatment`` + 1 levels of :math:`n` observations each; the intraclass
    correlation :math:`\sigma^2_\tau / (\sigma^2_\tau + \sigma^2)` lies between
    :math:`L/(1 + L)` and :math:`U/(1 + U)` (Montgomery, 2013, section 3.9).

    Returns
    -------
    dict[str, float]
        ``ratio_low`` (:math:`L`) and ``ratio_high`` (:math:`U`); ``icc_low`` and ``icc_high``;
        and ``error_low`` and ``error_high``, the chi-squared interval of the error variance.
    """
    alpha = 1 - conflevel
    df_error = (df_treatment + 1) * (n - 1)  # a levels of n observations each
    f0 = ms_treatment / ms_error
    upper = float(stats.f.ppf(1 - alpha / 2, df_treatment, df_error))
    lower = float(stats.f.ppf(alpha / 2, df_treatment, df_error))
    low, high = (f0 / upper - 1) / n, (f0 / lower - 1) / n
    scaled = df_error * ms_error
    return {
        "ratio_low": low,
        "ratio_high": high,
        "icc_low": low / (1 + low),
        "icc_high": high / (1 + high),
        "error_low": scaled / float(stats.chi2.ppf(1 - alpha / 2, df_error)),
        "error_high": scaled / float(stats.chi2.ppf(alpha / 2, df_error)),
    }
