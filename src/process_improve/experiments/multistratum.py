# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Multistratum analysis of variance: split-plot, strip-plot and other designs with several sizes of unit.

A design that restricts its randomisation (a hard-to-change factor set once for several runs, a
batch that holds four strips, a furnace load that crosses an implanter batch) has runs that fall
into *strata*: the whole plots, the subplots within them, and so on. Each stratum is a set of
contrasts among the runs with its own error, and each treatment term is estimated in the stratum
its contrasts fall in. This is R's ``aov(y ~ treatments + Error(blocks))``:

- the strata are given as error terms, ``"rep/plot"`` (replicates, then whole plots within them)
  or ``"row + column"`` (a strip plot); a last stratum, ``"Within"``, holds the contrasts left;
- each stratum's basis is the part of its error term's indicator columns that the strata before it
  do not span, so the strata split the n - 1 contrasts among the runs between them;
- in each stratum the treatment terms are fitted in order, each one's degrees of freedom the rank
  it adds there, and what is left is the stratum's residual.

Without a response it gives the skeleton analysis of variance: the sources and degrees of freedom a
design will have, before it is run. With one, each term's sum of squares, and its F test on the
residual of its stratum where that residual has degrees of freedom.

Examples
--------
>>> import itertools
>>> import pandas as pd
>>> from process_improve.experiments.multistratum import multistratum_anova
>>> runs = pd.DataFrame(list(itertools.product([0, 1], [0, 1], [0, 1], [0, 1])), columns=["S", "T", "rep", "A"])
>>> layout = multistratum_anova(runs, "S * T * A", "S:T:rep")
>>> layout["df"]["S:T:rep"]
{'S': 1, 'T': 1, 'S:T': 1, 'Residual': 4}
>>> layout["df"]["Within"]
{'A': 1, 'S:A': 1, 'T:A': 1, 'S:T:A': 1, 'Residual': 4}
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy import stats

from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

_RANK_TOLERANCE = 1e-9


def error_terms(strata: str | list[str]) -> list[str]:
    """Expand a block structure into its error terms, in order: ``"a/b"`` is ``a`` and ``a:b``, ``"a + b"`` is both.

    Parameters
    ----------
    strata : str or list[str]
        The block structure, ``"rep/plot"``, ``"row + column"``, or a list of error terms.

    Returns
    -------
    list[str]
        The error terms, each a factor or an interaction of factors such as ``"rep:plot"``.
    """
    parts = strata if isinstance(strata, list) else strata.split("+")
    terms: list[str] = []
    for part in parts:
        nested = [name.strip() for name in str(part).split("/")]
        terms += [":".join(nested[: i + 1]) for i in range(len(nested))]
    return list(dict.fromkeys(terms))


def _basis(columns: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis of the space the columns span."""
    if columns.size == 0:  # no columns, or a stratum with no units left: it spans nothing
        return columns[:, :0]
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return u[:, s > _RANK_TOLERANCE * max(1.0, float(s[0]))]


def _strata(table: pd.DataFrame, terms: list[str]) -> list[tuple[str, np.ndarray]]:
    """Return each stratum's name and orthonormal basis: one per error term, then ``"Within"``."""
    n = len(table)
    spanned = np.ones((n, 1)) / np.sqrt(n)  # the grand mean is a stratum of its own, and is not reported
    strata = []
    for term in terms:
        groups = table[term.split(":")].astype(str).agg(":".join, axis=1)
        indicators = pd.get_dummies(groups).to_numpy(dtype=float)
        basis = _basis(indicators - spanned @ (spanned.T @ indicators))
        strata.append((term, basis))
        spanned = np.hstack([spanned, basis])
    strata.append(("Within", _basis(np.eye(n) - spanned @ spanned.T)))
    return strata


def _stratum_rows(
    basis: np.ndarray, terms: list[tuple[str, np.ndarray]], y: np.ndarray | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the treatment terms in order within one stratum; return their rows and the stratum's residual."""
    projected_y = None if y is None else basis.T @ y
    fitted = np.empty((basis.shape[1], 0))
    rows: list[dict[str, Any]] = []
    for term, columns in terms:
        projected = basis.T @ columns
        added = _basis(projected - fitted @ (fitted.T @ projected))
        if added.shape[1] == 0:
            continue  # the term has no contrast in this stratum
        row: dict[str, Any] = {"term": term, "df": int(added.shape[1])}
        if projected_y is not None:
            row["ss"] = float(np.sum((added.T @ projected_y) ** 2))
            row["ms"] = row["ss"] / row["df"]
        rows.append(row)
        fitted = np.hstack([fitted, added])
    residual: dict[str, Any] = {"df": int(basis.shape[1] - fitted.shape[1])}
    if projected_y is not None:
        residual["ss"] = float(np.sum(projected_y**2) - np.sum((fitted.T @ projected_y) ** 2))
        residual["ms"] = residual["ss"] / residual["df"] if residual["df"] else float("nan")
        for row in rows:
            row["f"] = row["ms"] / residual["ms"] if residual["df"] else float("nan")
            row["p_value"] = float(stats.f.sf(row["f"], row["df"], residual["df"])) if residual["df"] else float("nan")
    return rows, residual


def multistratum_anova(
    design_matrix: pd.DataFrame,
    model: str,
    strata: str | list[str],
    response: str | None = None,
) -> dict[str, Any]:
    """Return the analysis of variance of a design with several strata of units, or its skeleton without a response.

    Parameters
    ----------
    design_matrix : DataFrame
        The runs: the treatment factors, the columns that name the units (replicate, whole plot,
        batch), and the response if there is one. Every factor is taken as levels.
    model : str
        The treatment terms, a plain formula over the columns: ``"S * T * A"``.
    strata : str or list[str]
        The block structure as error terms (see :func:`error_terms`): ``"rep/plot"`` for whole
        plots within replicates, ``"S:T"`` for a whole plot at each combination of S and T,
        ``"row + column"`` for a strip plot.
    response : str or None
        The response column; without one, the result holds degrees of freedom only.

    Returns
    -------
    dict[str, Any]
        ``strata``, one entry per stratum (each error term, then ``"Within"``) with its ``df``,
        its ``units`` (the number of groups its error term makes, the runs for ``"Within"``), the
        ``terms`` estimated in it, each with its ``df`` (and with a response ``ss``, ``ms``,
        ``f`` and ``p_value``), and its ``residual``; ``df``, each stratum's terms and
        ``"Residual"`` mapped to their degrees of freedom; and ``table``, every row as a
        DataFrame. A term whose contrasts fall in two strata, as in a design that is not
        orthogonal, has a row in each.

    Raises
    ------
    ValueError
        If the model or an error term names a column the runs do not have.
    """
    table = design_matrix.reset_index(drop=True)
    terms = error_terms(strata)
    for name in {factor for term in terms for factor in term.split(":")} | ({response} - {None}):
        validate_identifier_is_safe(name)
        if name not in table.columns:
            raise ValueError(f"The runs have no column {name!r}.")
    validate_formula_is_safe(f"{response or '_'} ~ {model}", [*table.columns, "_"])
    factors = [c for c in table.columns if c != response]
    x = dmatrix(model, table[factors].astype(str), return_type="dataframe")
    treatments = [
        (name, x.to_numpy()[:, columns])
        for name, columns in x.design_info.term_name_slices.items()
        if name != "Intercept"
    ]
    y = None if response is None else table[response].to_numpy(dtype=float)
    found: list[dict[str, Any]] = []
    for name, basis in _strata(table, terms):
        rows, residual = _stratum_rows(basis, treatments, y)
        units = len(table) if name == "Within" else int(table[name.split(":")].drop_duplicates().shape[0])
        found.append({"stratum": name, "df": int(basis.shape[1]), "units": units, "terms": rows, "residual": residual})
    flat = [
        {"stratum": s["stratum"], **row}
        for s in found
        for row in [*s["terms"], {"term": "Residual", **s["residual"]}]
        if row["df"] or row["term"] != "Residual"
    ]
    return {
        "strata": found,
        "df": {
            s["stratum"]: {row["term"]: row["df"] for row in s["terms"]} | {"Residual": s["residual"]["df"]}
            for s in found
        },
        "table": pd.DataFrame(flat),
    }
