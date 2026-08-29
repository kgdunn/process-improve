r"""(c) Kevin Dunn, 2010-2026. MIT License.

Permutation nulls that respond to signal, and a hypergeometric enrichment test.

At the sample sizes a sensory-to-chemistry study runs at, "is there anything
here at all" is the hard question, and the obvious ways to answer it do not
work. A high :math:`R^2` on twelve products with three components is close to
guaranteed. A count of how many variables exceed VIP 1 is not a test statistic
at all: VIP is normalised so that :math:`\sum_j \text{VIP}_j^2 = K` exactly, so
that count describes the shape of the VIP distribution rather than the presence
of a relationship, barely moves when the response is permuted, and yields a
false-discovery rate near 100% on data that does contain signal.

What does work is asking a permutation what it can achieve:

* :func:`permutation_q2` permutes the response across products, refits, and
  compares observed *out-of-sample* performance with what random reassignment
  reaches. Out-of-sample is what makes it bite; the same test on in-sample
  :math:`R^2` would mostly measure model capacity.
* :func:`pipeline_null` does the same for a whole selection procedure, counting
  discoveries under a permuted response, which gives an empirical
  false-discovery rate for the procedure as run rather than for a model in
  isolation.
* :func:`class_enrichment` asks whether a chemically expected class of compounds
  sits at the top of a ranking more often than chance allows. At small sample
  sizes this is frequently the stronger evidence: recovering the right class for
  an attribute is structure that noise does not produce.

References
----------
Westerhuis, Hoefsloot, Smit and others, "Assessment of PLSDA cross validation",
Metabolomics, 4, 81-89, 2008, doi:10.1007/s11306-007-0099-6.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from ._common import SpecificationWarning

__all__ = ["class_enrichment", "permutation_q2", "pipeline_null"]


def _as_frame(values: pd.DataFrame | pd.Series | np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    """Coerce a callable's return value to a frame with ``like``'s labels."""
    if isinstance(values, pd.DataFrame):
        return values
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.shape != like.shape:
        raise ValueError(
            f"fit_predict returned predictions of shape {array.shape}, but the response block is "
            f"{like.shape}. It must return one out-of-sample prediction per row and per attribute."
        )
    return pd.DataFrame(array, index=like.index, columns=like.columns)


def _q2_per_column(observed: pd.DataFrame, predicted: pd.DataFrame) -> np.ndarray:
    """Return 1 - PRESS / TSS per column, NaN where a column has no variance."""
    truth = observed.to_numpy(dtype=float)
    fitted = predicted.to_numpy(dtype=float)
    usable = np.isfinite(truth) & np.isfinite(fitted)
    out = np.full(truth.shape[1], np.nan)
    for position in range(truth.shape[1]):
        rows = usable[:, position]
        if rows.sum() < 2:
            continue
        column = truth[rows, position]
        press = float(np.sum((column - fitted[rows, position]) ** 2))
        total = float(np.sum((column - column.mean()) ** 2))
        if total <= 0:
            continue
        out[position] = 1.0 - press / total
    return out


def _permute_rows(y: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute whole response rows, keeping each product's full response vector intact.

    Permuting column by column would destroy the correlation structure among the
    attributes and inflate the null, making the test look more impressive than
    it is.
    """
    order = rng.permutation(len(y))
    return pd.DataFrame(y.to_numpy()[order], index=y.index, columns=y.columns)


def permutation_q2(
    fit_predict: Callable,
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_perm: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Test observed out-of-sample performance against a permuted response.

    This is the replacement for a VIP-exceedance null. The question a
    permutation test can answer is "does this model predict held-out products
    better than it would if the responses were shuffled", and unlike a VIP count
    the answer responds to whether signal is present.

    Parameters
    ----------
    fit_predict : callable
        ``fit_predict(x, y)`` returning **out-of-sample** predictions with one
        row per row of ``y`` and one column per attribute: a DataFrame, or an
        array of the same shape. Everything the pipeline does lives inside this
        callable, and two things about it are the caller's responsibility:

        * **The cross-validation scheme is yours to choose.** Leave-one-out is
          right when every product is precious and needlessly expensive at
          hundreds of rows, where a permutation null multiplies the cost by
          ``n_perm``. Pick deliberately.
        * **Re-derive the response constants inside each fold too.** It is easy
          to nest the predictor preprocessing and forget the response. Centre
          and scale the response on the training rows, predict, then
          back-transform with those same training constants before the
          prediction is returned; otherwise the score is computed on a scale the
          fold never saw.

    x : pandas.DataFrame
        Predictor block, one row per product. Not permuted.
    y : pandas.DataFrame
        Response block, one row per product and one column per attribute.
    n_perm : int, default 500
        Number of permutations. See the note on the p-value floor below.
    seed : int, default 0
        Seed for the permutation order, so a reported p-value can be reproduced.

    Returns
    -------
    pandas.DataFrame
        One row per attribute (the columns of ``y``), with columns:

        ``attribute``
            The response name.
        ``q2_observed``
            :math:`Q^2` from ``fit_predict(x, y)``.
        ``q2_null_mean``, ``q2_null_p95``
            The mean and 95th percentile of :math:`Q^2` under permutation. The
            95th percentile is the more useful of the two: it is roughly the
            performance a shuffled response reaches one time in twenty.
        ``p_value``
            The fraction of permutations reaching at least ``q2_observed``,
            computed as ``(1 + count) / (n_perm + 1)``.
        ``n_permutations``
            Permutations that produced a usable :math:`Q^2`.

    Raises
    ------
    ValueError
        If the blocks disagree on rows, ``n_perm`` is below 1, or
        ``fit_predict`` returns the wrong shape.

    Notes
    -----
    The p-value uses the ``(1 + count) / (n_perm + 1)`` form, which counts the
    observed statistic as one of its own null draws so the result can never be
    exactly zero. That also puts a floor on it: **the smallest attainable
    p-value is** ``1 / (n_perm + 1)``. The default 500 permutations cannot report
    anything below 0.002, so choose ``n_perm`` with the threshold you intend to
    apply in mind, and remember that a multiplicity correction over many
    attributes needs a floor well below the corrected threshold.

    Examples
    --------
    >>> def fit_predict(x, y):
    ...     return loo_predictions(x, y, n_components=2)  # your own CV loop
    >>> permutation_q2(fit_predict, chem_scaled, sensory_means, n_perm=999)
    """
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("x and y must both be pandas DataFrames, one row per product.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same number of rows; got {len(x)} and {len(y)}.")
    if len(y) < 2:
        raise ValueError(f"a permutation null needs at least 2 products; got {len(y)}.")
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm!r}.")

    observed = _q2_per_column(y, _as_frame(fit_predict(x, y), y))

    rng = np.random.default_rng(seed)
    null = np.full((int(n_perm), y.shape[1]), np.nan)
    for index in range(int(n_perm)):
        y_perm = _permute_rows(y, rng)
        null[index, :] = _q2_per_column(y_perm, _as_frame(fit_predict(x, y_perm), y_perm))

    rows: list[dict[str, object]] = []
    for position, attribute in enumerate(y.columns):
        draws = null[:, position]
        usable = draws[np.isfinite(draws)]
        n_used = int(usable.size)
        if n_used == 0 or not np.isfinite(observed[position]):
            mean, p95, p_value = np.nan, np.nan, np.nan
        else:
            mean = float(usable.mean())
            p95 = float(np.percentile(usable, 95))
            p_value = float((1 + int(np.sum(usable >= observed[position]))) / (n_used + 1))
        rows.append(
            {
                "attribute": str(attribute),
                "q2_observed": float(observed[position]),
                "q2_null_mean": mean,
                "q2_null_p95": p95,
                "p_value": p_value,
                "n_permutations": n_used,
            }
        )
    return pd.DataFrame(
        rows,
        columns=["attribute", "q2_observed", "q2_null_mean", "q2_null_p95", "p_value", "n_permutations"],
    )


def pipeline_null(
    select: Callable,
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_perm: int = 500,
    seed: int = 0,
) -> dict:
    """Count a selection procedure's discoveries against a permuted response.

    :func:`permutation_q2` tests one model; this tests the whole procedure that
    produced it. Pass a callable that runs filtering, transformation, scaling and
    selection end to end, and it is re-run in full on each permuted response.
    The average number of discoveries under permutation, divided by the number
    actually made, is an empirical false-discovery rate for the procedure **as
    run**, which is the only version of it worth quoting: a procedure whose
    selection step is honest but whose filtering step peeked at the response has
    an FDR that no formula recovers.

    The response-independent steps are deliberately not hoisted out of the loop.
    Re-running them per permutation is a no-op when they really are
    response-independent, and hoisting them would be an assumption about the
    caller's code that this function is in no position to make.

    Parameters
    ----------
    select : callable
        ``select(x, y)`` returning the names it selected: any iterable of
        hashable labels. It should be deterministic given its inputs; if it is
        not, the null counts pick up its own randomness and the reported FDR
        means nothing. A :class:`SpecificationWarning` is raised when a repeat
        call on the observed data disagrees with the first.
    x : pandas.DataFrame
        Predictor block, one row per product. Not permuted.
    y : pandas.DataFrame
        Response block, one row per product. Whole rows are permuted, so each
        product keeps its full response vector.
    n_perm : int, default 500
        Number of permutations.
    seed : int, default 0
        Seed for the permutation order.

    Returns
    -------
    dict
        With keys:

        ``observed``
            Number of names selected on the real response.
        ``null_mean``, ``null_p95``
            Mean and 95th percentile of the selection count under permutation.
        ``empirical_fdr``
            ``null_mean / observed``, clipped to ``[0, 1]``; ``NaN`` when
            nothing was selected. Read it as "of the names this procedure
            returned, about this fraction is what shuffling alone would have
            produced".
        ``null_counts``
            The per-permutation counts, as an ndarray, for plotting the null.
        ``selected``
            The names selected on the real response, in the order the callable
            returned them.

    Raises
    ------
    ValueError
        If the blocks disagree on rows or ``n_perm`` is below 1.

    Examples
    --------
    >>> def select(x, y):
    ...     kept, _dropped, _presence = trim_by_prevalence(x)
    ...     return names_above_threshold(kept, y)
    >>> result = pipeline_null(select, chem, sensory_means, n_perm=200)
    >>> result["empirical_fdr"]
    """
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("x and y must both be pandas DataFrames, one row per product.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same number of rows; got {len(x)} and {len(y)}.")
    if int(n_perm) < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm!r}.")

    def _names(result: Iterable) -> list:
        return list(result)

    selected = _names(select(x, y))
    # Cheap check rather than an assumption: a selector that disagrees with
    # itself makes every count below, and therefore the FDR, meaningless.
    if _names(select(x, y)) != selected:
        warnings.warn(
            "select() returned different names on two identical calls, so it carries its own "
            "randomness. The permutation counts below mix that randomness in with the null, and "
            "the empirical FDR cannot be read as a property of the data. Seed the selector (or "
            "make it deterministic) and re-run.",
            SpecificationWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    counts = np.empty(int(n_perm), dtype=float)
    for index in range(int(n_perm)):
        y_perm = _permute_rows(y, rng)
        counts[index] = len(_names(select(x, y_perm)))

    observed = len(selected)
    null_mean = float(counts.mean())
    empirical_fdr = float(np.clip(null_mean / observed, 0.0, 1.0)) if observed else float("nan")
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_p95": float(np.percentile(counts, 95)),
        "empirical_fdr": empirical_fdr,
        "null_counts": counts,
        "selected": selected,
    }


def class_enrichment(
    ranked: Sequence[str],
    all_names: Sequence[str],
    pattern: str,
    top_n: int = 12,
) -> dict:
    """Test whether a named class of compounds is over-represented at the top of a ranking.

    At small sample sizes this is frequently stronger evidence than
    :math:`R^2` or :math:`Q^2`. Recovering the chemically expected class for an
    attribute (the esters at the top of "fruity", the pyrazines at the top of
    "roasted") is structure that noise does not produce, whereas a high
    :math:`R^2` on few products with several components very nearly is.

    The test is hypergeometric: given ``n_compounds`` compounds of which
    ``class_size`` belong to the class, how surprising is it to draw ``in_top``
    of them in the top ``top_n``?

    .. note::

       Check where the ranking came from before reading much into a
       per-attribute result. A one-component PLS has a coefficient matrix
       ``outer(x_weights, y_loadings)``, so the absolute coefficients order
       identically for **every** attribute and the attributes differ only in
       sign and magnitude. One-component solutions are common at small sample
       sizes, so this is the normal case rather than an edge case: an enrichment
       that looks attribute-specific may be one ranking reported many times.

    Parameters
    ----------
    ranked : sequence of str
        Compound names, most important first. May be shorter than ``all_names``.
    all_names : sequence of str
        Every compound that could have been ranked: the population the draw
        comes from. Must not contain duplicates.
    pattern : str
        Regular expression defining the class, matched against each name with
        :func:`re.search` (so a plain substring works).
    top_n : int, default 12
        How much of the ranking counts as "the top". Truncated to the length of
        ``ranked`` when it is shorter.

    Returns
    -------
    dict
        With keys ``in_top`` (class members among the top ``n_drawn``),
        ``class_size`` (class members in the population), ``n_compounds``
        (population size), ``n_drawn`` (the effective number drawn),
        ``p_value`` (the hypergeometric upper-tail probability of seeing at
        least ``in_top``), and ``matched`` (the class members found, in ranked
        order).

    Raises
    ------
    ValueError
        If ``all_names`` has duplicates, ``ranked`` contains a name not in
        ``all_names``, ``top_n`` is below 1, or ``pattern`` is not a valid
        regular expression.

    Examples
    --------
    >>> class_enrichment(ranking_for_fruity, all_compounds, r"acetate|butanoate")
    {'in_top': 5, 'class_size': 9, 'n_compounds': 61, 'n_drawn': 12, ...}
    """
    names = [str(name) for name in all_names]
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"all_names must not contain duplicates; repeated: {duplicates}.")
    order = [str(name) for name in ranked]
    strangers = sorted(set(order) - set(names))
    if strangers:
        raise ValueError(
            f"ranked contains name(s) {strangers} that are not in all_names. The population has to "
            "contain the sample, or the hypergeometric probability is not defined."
        )
    if int(top_n) < 1:
        raise ValueError(f"top_n must be >= 1; got {top_n!r}.")
    try:
        matcher = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"pattern is not a valid regular expression: {exc}") from exc

    n_drawn = min(int(top_n), len(order))
    top = order[:n_drawn]
    matched = [name for name in top if matcher.search(name)]
    class_size = sum(1 for name in names if matcher.search(name))
    in_top = len(matched)

    if class_size == 0 or n_drawn == 0:
        p_value = float("nan")
    else:
        p_value = float(hypergeom.sf(in_top - 1, len(names), class_size, n_drawn))

    return {
        "in_top": in_top,
        "class_size": class_size,
        "n_compounds": len(names),
        "n_drawn": n_drawn,
        "p_value": p_value,
        "matched": matched,
    }
