"""(c) Kevin Dunn, 2010-2026. MIT License.

Mixed Assessor Model (MAM): per-assessor scaling and scale alignment.

The classical assessor-by-product interaction lumps together two different
things: a panelist who simply uses a wider or narrower part of the scale (a
multiplicative scaling difference), and a panelist who genuinely ranks the
products differently (real disagreement). The MAM separates them.

For each attribute, regress every panelist's product means on the panel
consensus product means. The slope is the panelist's scaling coefficient
``beta``:

* ``beta`` near 1: uses the scale like the panel;
* ``beta`` < 1: compresses (narrow range);
* ``beta`` > 1: expands (wide range).

What is left after removing the scaling part is the disagreement. Using the
disagreement (rather than the inflated raw interaction) as the error term gives
a more powerful product-effect F-test, and the ``beta`` coefficients let you
*align* the panel: rescale each panelist onto a common scale instead of
dropping them (:func:`align_scores`).

:func:`mixed_assessor_model` is the closed form on replicate-averaged cell means.
:func:`mixed_assessor_model_reml` is the random-effects version SensMixed fits:
REML with the panelist terms random, likelihood-ratio selection of the random
terms, and Satterthwaite F-tests, also for a product described by several
factors. Both are pure Python.

References
----------
Brockhoff, Schlich and Skovgaard, "Taking individual scaling differences into
account by analyzing profile data with the Mixed Assessor Model", Food Quality
and Preference, 39, 156-166, 2015.

Kuznetsova, Brockhoff and Christensen, "lmerTest package: tests in linear mixed
effects models", Journal of Statistical Software, 82(13), 2017.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_dist
from threadpoolctl import threadpool_limits

from process_improve.regression._robust_regression import repeated_median_slope
from process_improve.regression._variance_components import (
    VarianceComponentFit,
    fit_variance_components,
    independent_columns,
    type1_hypotheses,
)

#: Slopes at or below this are treated as unusable for rescaling (a panelist
#: who is flat or anti-correlated with the panel cannot be scale-corrected).
_MIN_SLOPE = 0.2

#: Columns :func:`mixed_assessor_model` reads out of the long-format panel.
_REQUIRED_COLUMNS: tuple[str, ...] = ("panelist_id", "product", "attribute", "score")

AlignMethod = str  # "both" | "location" | "scale"


@dataclass
class MAMResult:
    """Outcome of :func:`mixed_assessor_model`.

    Attributes
    ----------
    scaling : pandas.DataFrame
        One row per (attribute, panelist) with the scaling coefficient
        ``beta``, the panelist ``offset`` from the attribute grand mean, and the
        panelist ``mean``.
    ftests : pandas.DataFrame
        One row per attribute with the MAM and classical product-effect
        F-tests: ``f_product_mam`` / ``p_product_mam`` (disagreement as error)
        and ``f_product_classical`` / ``p_product_classical`` (raw interaction
        as error), plus the degrees of freedom.
    """

    scaling: pd.DataFrame
    ftests: pd.DataFrame


def _cell_means(panel: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Return the panelist-by-product matrix of replicate-averaged scores."""
    sub = panel[panel["attribute"] == attribute]
    return sub.pivot_table(index="panelist_id", columns="product", values="score", aggfunc="mean", observed=True)


def _assessor_scaling(matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series, np.ndarray, float]:
    """Return per-panelist slopes and means, the centred product effect, and its SSQ.

    ``matrix`` is panelists (rows) by products (columns) of cell means.
    """
    consensus = matrix.mean(axis=0)
    tau = (consensus - consensus.mean()).to_numpy()  # centred product effect
    ssq_tau = float(np.nansum(tau**2))
    means = matrix.mean(axis=1)
    if ssq_tau > 0:
        centred = matrix.to_numpy() - means.to_numpy()[:, None]
        slopes = np.nansum(centred * tau[None, :], axis=1) / ssq_tau
    else:
        slopes = np.ones(matrix.shape[0])
    beta = pd.Series(slopes, index=matrix.index)
    return beta, means, tau, ssq_tau


def mixed_assessor_model(panel: pd.DataFrame) -> MAMResult:
    """Fit the Mixed Assessor Model per attribute.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data.

    Returns
    -------
    MAMResult
        Per-panelist scaling coefficients and per-attribute F-tests; see the
        class docstring.

    Raises
    ------
    ValueError
        If a required column is missing, or the panel has no rows. An empty
        panel is reachable whenever an upstream filter removes every attribute;
        the frames built from it would have no columns at all, so the caller
        would meet the problem as a ``KeyError`` on ``ftests["f_product_mam"]``
        rather than here.

    Examples
    --------
    >>> mam = mixed_assessor_model(validated.normalized_df)
    >>> mam.scaling.query("attribute == 'saltiness'").sort_values("beta").head()
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(
            f"mixed_assessor_model needs the long-format panel columns {list(_REQUIRED_COLUMNS)}; "
            f"missing {missing}. Got columns {list(panel.columns)}."
        )
    if len(panel) == 0:
        raise ValueError(
            "mixed_assessor_model was given a panel with no rows, so there is no attribute "
            "to fit. This usually means an upstream filter (an attribute list, a panelist "
            "exclusion, a product subset) removed everything; check that filter rather than "
            "this call."
        )

    scaling_rows: list[dict[str, object]] = []
    ftest_rows: list[dict[str, object]] = []

    for attribute in sorted(panel["attribute"].unique()):
        matrix = _cell_means(panel, attribute)
        n_assessors, n_products = matrix.shape
        beta, means, _tau, ssq_tau = _assessor_scaling(matrix)
        grand = float(np.nanmean(matrix.to_numpy()))

        scaling_rows.extend(
            {
                "attribute": str(attribute),
                "panelist_id": str(pid),
                "beta": float(beta.loc[pid]),
                "offset": float(means.loc[pid] - grand),
                "mean": float(means.loc[pid]),
            }
            for pid in matrix.index
        )

        # Two-way decomposition of the cell-mean table.
        values = matrix.to_numpy()
        product_effect = np.nanmean(values, axis=0) - grand
        assessor_effect = np.nanmean(values, axis=1) - grand
        ss_product = n_assessors * float(np.nansum(product_effect**2))
        ss_assessor = n_products * float(np.nansum(assessor_effect**2))
        ss_total = float(np.nansum((values - grand) ** 2))
        ss_interaction = ss_total - ss_product - ss_assessor
        # MAM split of the interaction: scaling (slope deviations) + disagreement.
        ss_scaling = float(np.nansum((beta.to_numpy() - 1.0) ** 2)) * ssq_tau
        ss_disagreement = ss_interaction - ss_scaling

        df_product = n_products - 1
        df_interaction = (n_assessors - 1) * (n_products - 1)
        df_disagreement = (n_assessors - 1) * (n_products - 2)

        def _ftest(error_ss: float, error_df: int, *, ss_p: float = ss_product, df_p: int = df_product) -> tuple:
            if df_p <= 0 or error_df <= 0 or error_ss <= 0:
                return (float("nan"), float("nan"))
            f_value = (ss_p / df_p) / (error_ss / error_df)
            return (float(f_value), float(f_dist.sf(f_value, df_p, error_df)))

        f_mam, p_mam = _ftest(ss_disagreement, df_disagreement)
        f_classical, p_classical = _ftest(ss_interaction, df_interaction)
        ftest_rows.append(
            {
                "attribute": str(attribute),
                "f_product_mam": f_mam,
                "p_product_mam": p_mam,
                "f_product_classical": f_classical,
                "p_product_classical": p_classical,
                "df_product": int(df_product),
                "df_disagreement": int(df_disagreement),
            }
        )

    return MAMResult(scaling=pd.DataFrame(scaling_rows), ftests=pd.DataFrame(ftest_rows))


def align_scores(panel: pd.DataFrame, *, method: AlignMethod = "both", robust: bool = False) -> pd.DataFrame:
    """Harmonize every panelist's scores onto the common panel scale.

    For each attribute and panelist, the location lever removes the panelist's
    mean offset (so "rates everything high/low" goes away) and the scale lever
    divides by the panelist's scaling coefficient ``beta`` (so a compressor's
    narrow range is stretched toward the panel's). This rescales the whole panel
    (standard MAM practice), keeping panelists rather than dropping them.

    When a panelist is flat or anti-correlated with the panel (``beta`` is not
    finite or below ``_MIN_SLOPE``), the scale lever is skipped for that
    panelist and the fallback depends on ``method``:

    - ``"both"``: the panelist is left location-corrected only.
    - ``"scale"``: the panelist is left as-is (no correction applied).
    - ``"location"``: the slope is not consulted, so the location correction
      is always applied.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data.
    method : {"both", "location", "scale"}
        ``"location"`` recentres each panelist to the grand mean; ``"scale"``
        rescales the spread around the panelist's own mean; ``"both"`` (default)
        does both, the full MAM alignment.
    robust : bool
        Use the repeated-median slope for ``beta`` instead of least squares.

    Returns
    -------
    pandas.DataFrame
        A corrected copy of ``panel`` with the ``score`` column aligned.
    """
    if method not in ("both", "location", "scale"):
        raise ValueError(f"method must be 'both', 'location', or 'scale', got {method!r}.")

    out = panel.copy()
    for attribute in panel["attribute"].unique():
        matrix = _cell_means(panel, attribute)
        beta, means, _tau, ssq_tau = _assessor_scaling(matrix)
        grand = float(np.nanmean(matrix.to_numpy()))
        if robust and ssq_tau > 0:
            consensus = matrix.mean(axis=0)
            centred_tau = (consensus - consensus.mean()).to_numpy()
            beta = pd.Series(
                [repeated_median_slope(centred_tau, matrix.loc[pid].to_numpy(), nowarn=True) for pid in matrix.index],
                index=matrix.index,
            )

        mask = panel["attribute"] == attribute
        for pid in matrix.index:
            slope = float(beta.loc[pid])
            if not np.isfinite(slope) or slope < _MIN_SLOPE:
                slope = 1.0  # cannot rescale a flat / anti-correlated panelist
            mean_i = float(means.loc[pid])
            rows = mask & (panel["panelist_id"] == pid)
            raw = panel.loc[rows, "score"]
            if method == "location":
                out.loc[rows, "score"] = raw - mean_i + grand
            elif method == "scale":
                out.loc[rows, "score"] = mean_i + (raw - mean_i) / slope
            else:  # both
                out.loc[rows, "score"] = grand + (raw - mean_i) / slope
    return out


# ---------------------------------------------------------------------------
# The random-effects MAM (SensMixed)
# ---------------------------------------------------------------------------

#: SensMixed drops a random term whose estimated standard deviation is below this
#: before testing anything.
_ZERO_SD = 1e-7

#: Fewer product cells than this leave no disagreement once scaling is removed.
_MIN_PRODUCTS = 3


@dataclass
class MAMRemlResult:
    """Outcome of :func:`mixed_assessor_model_reml`.

    Attributes
    ----------
    anova : pandas.DataFrame
        The final model's sequential (Type I) F-tests, one row per attribute and term:
        each product term, then ``scaling``. Columns ``attribute``, ``term``, ``sum_sq``,
        ``mean_sq``, ``num_df``, ``den_df`` (Satterthwaite), ``f_value``, ``p_value``.
        ``scaling`` tests whether the panelists' scaling coefficients differ.
    random_effects : pandas.DataFrame
        One row per attribute and candidate random term: ``chi_sq``, ``chi_df`` and
        ``p_value`` of its likelihood-ratio test, ``status`` (``"kept"``,
        ``"eliminated"`` or ``"zero_variance"``) and ``step``, the order of elimination
        (``<NA>`` unless eliminated). An eliminated term shows the test that removed it;
        a kept term, its test in the final model; a zero-variance term was dropped
        before any test and has none.
    variance_components : pandas.DataFrame
        The final model's variances, one row per attribute and random term, then
        ``Residual``: columns ``attribute``, ``term``, ``variance``, ``std_dev``.
    scaling : pandas.DataFrame
        One row per attribute and panelist: the scaling coefficient ``beta``, scaled
        to average 1 over the panel as in :func:`mixed_assessor_model`.
    """

    anova: pd.DataFrame
    random_effects: pd.DataFrame
    variance_components: pd.DataFrame
    scaling: pd.DataFrame


def _term_label(columns: Sequence[str]) -> str:
    """Display name of a model term; ``panelist_id`` reads as ``panelist``."""
    return ":".join("panelist" if column == "panelist_id" else column for column in columns)


def _term_columns(frame: pd.DataFrame, factors: Sequence[str]) -> np.ndarray:
    """Model columns of a product term, coded as lmerTest codes them.

    Each factor gets an indicator column for every level but the last (SAS coding);
    an interaction is the products of its factors' columns, the first factor varying
    fastest, as in R's ``model.matrix``.
    """
    block = np.ones((len(frame), 1))
    for factor in factors:
        values = frame[factor].to_numpy()
        coded = (values[:, None] == np.sort(frame[factor].unique())[None, :-1]).astype(float)
        block = (coded[:, :, None] * block[:, None, :]).reshape(len(frame), -1)
    return block


def _fixed_design(
    frame: pd.DataFrame, product_terms: list[tuple[str, ...]], panelists: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed-effect matrix, each column's term number, and which scaling columns survive.

    The terms are the intercept, the product terms, and ``scaling``: the centred product
    cell means times each panelist's indicator, so each panelist gets a slope on the
    consensus. Those slopes sum to the consensus itself, which the product terms
    already span, so one panelist's column is aliased and dropped (the last, as lme4
    drops it); the slopes are identified only relative to each other.
    """
    y = frame["score"].to_numpy(dtype=float)
    blocks, assign = [np.ones((len(frame), 1))], [-1]
    for number, factors in enumerate(product_terms):
        columns = _term_columns(frame, factors)
        blocks.append(columns)
        assign.extend([number] * columns.shape[1])
    consensus = frame.groupby(list(product_terms[-1]), observed=True)["score"].transform("mean").to_numpy() - y.mean()
    blocks.append(consensus[:, None] * (frame["panelist_id"].to_numpy()[:, None] == panelists[None, :]))
    assign.extend([len(product_terms)] * len(panelists))
    X, assign_array = np.hstack(blocks), np.array(assign)
    keep = independent_columns(X)
    return X[:, keep], assign_array[keep], keep[-len(panelists) :]


def _random_terms(
    frame: pd.DataFrame, product_terms: list[tuple[str, ...]], replication: str | None
) -> list[tuple[str, ...]]:
    """SensMixed's candidate random terms, in its order.

    Every product term crossed with the panelist, the replicate and both, then those
    three on their own; a term with as many levels as observations is the residual
    and is left out.
    """
    random_sets: list[tuple[str, ...]] = [("panelist_id",)]
    if replication is not None:
        random_sets += [(replication,), ("panelist_id", replication)]
    candidates = [(*term, *random_set) for term in product_terms for random_set in random_sets]
    candidates += random_sets
    return [term for term in candidates if math.prod(frame[column].nunique() for column in term) < len(frame)]


class _AttributeModel:
    """The REML fits of one attribute's MAM, for any subset of its random terms."""

    def __init__(self, frame: pd.DataFrame, product_terms: list[tuple[str, ...]], replication: str | None) -> None:
        self.y = frame["score"].to_numpy(dtype=float)
        self.panelists = np.sort(frame["panelist_id"].unique())
        self.X, self.assign, self.scaling_kept = _fixed_design(frame, product_terms, self.panelists)
        self.terms = _random_terms(frame, product_terms, replication)
        self.codes = {term: frame.groupby(list(term), sort=True).ngroup().to_numpy() for term in self.terms}

    def fit(
        self, terms: list[tuple[str, ...]], start: dict[tuple[str, ...], float] | None = None
    ) -> VarianceComponentFit:
        """REML fit with the random ``terms``, from the variance ratios ``start`` when given.

        A zero ratio restarts at a small positive one, so the search can leave the boundary.
        """
        psi = None if start is None else np.array([max(start.get(term, 1.0), 1e-3) for term in terms])
        return fit_variance_components(self.y, self.X, [self.codes[term] for term in terms], start=psi)


def _ratios(terms: list[tuple[str, ...]], fit: VarianceComponentFit) -> dict[tuple[str, ...], float]:
    """Each term's variance relative to the residual: the fit's ``psi``, to warm-start another."""
    return dict(zip(terms, fit.variances / fit.sigma2, strict=True))


def _likelihood_ratio(
    model: _AttributeModel, terms: list[tuple[str, ...]], current: VarianceComponentFit
) -> dict[tuple[str, ...], tuple[float, float, VarianceComponentFit]]:
    """Test each term by refitting without it: the REML criterion's rise, on 1 df."""
    tests = {}
    start = _ratios(terms, current)
    for term in terms:
        reduced = model.fit([other for other in terms if other != term], start=start)
        chi_sq = max(reduced.reml_criterion - current.reml_criterion, 0.0)
        tests[term] = (chi_sq, float(chi2.sf(chi_sq, 1)), reduced)
    return tests


def _record(term: tuple[str, ...], status: str, test: tuple | None = None, step: int | None = None) -> dict:
    """One row of the random-effects table (without the attribute)."""
    return {
        "term": _term_label(term),
        "chi_sq": np.nan if test is None else test[0],
        "chi_df": 1,
        "p_value": np.nan if test is None else test[1],
        "status": status,
        "step": pd.NA if step is None else step,
    }


def _select_random(
    model: _AttributeModel, keep: set[tuple[str, ...]], alpha: float
) -> tuple[list[tuple[str, ...]], VarianceComponentFit, list[dict]]:
    """Drop zero-variance terms, then eliminate the least significant term while its p exceeds ``alpha``.

    This is SensMixed's selection (lmerTest's ``step``): terms in ``keep`` are tested
    but never eliminated, and the procedure stops when every remaining candidate is
    significant. Returns the final terms, their fit, and one record per candidate.

    The terms in ``keep`` are not dropped for a zero variance either. SensMixed does
    drop them, but only when lme4's search ends within 1e-7 of the boundary, which it
    often does not; an exact optimum sits on it. A kept term matters even at zero:
    when another term is tested, the panelist term can take up the variation it leaves.
    """
    full = model.fit(model.terms)
    zero = {
        term
        for term, variance in zip(model.terms, full.variances, strict=True)
        if math.sqrt(variance) < _ZERO_SD and term not in keep
    }
    records = [_record(term, "zero_variance") for term in model.terms if term in zero]
    active = [term for term in model.terms if term not in zero]
    current = model.fit(active, start=_ratios(model.terms, full)) if zero else full
    tests: dict = {}
    for step in range(1, len(active) + 1):
        tests = _likelihood_ratio(model, active, current)
        candidates = [term for term in active if term not in keep]
        worst = max(candidates, key=lambda term: tests[term][1], default=None)
        if worst is None or tests[worst][1] <= alpha:
            break
        records.append(_record(worst, "eliminated", tests[worst], step))
        current = tests[worst][2]
        active.remove(worst)
        tests = {}
    records += [_record(term, "kept", tests[term]) for term in active]
    return active, current, records


def _attribute_tables(
    attribute: str,
    model: _AttributeModel,
    terms: list[tuple[str, ...]],
    fit: VarianceComponentFit,
    product_terms: list[tuple[str, ...]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Type I tests, variance components and scaling coefficients of the final model."""
    labels = [":".join(term) for term in product_terms] + ["scaling"]
    anova = []
    for number, contrasts in type1_hypotheses(model.X, model.assign).items():
        f_value, den_df = fit.f_test(contrasts)
        num_df = contrasts.shape[0]
        anova.append(
            {
                "attribute": attribute,
                "term": labels[number],
                "sum_sq": f_value * fit.sigma2 * num_df,
                "mean_sq": f_value * fit.sigma2,
                "num_df": num_df,
                "den_df": den_df,
                "f_value": f_value,
                "p_value": float(f_dist.sf(f_value, num_df, den_df)),
            }
        )
    components = [
        {"attribute": attribute, "term": _term_label(term), "variance": float(variance)}
        for term, variance in zip(terms, fit.variances, strict=True)
    ]
    components.append({"attribute": attribute, "term": "Residual", "variance": fit.sigma2})
    slopes = np.zeros(len(model.panelists))
    slopes[model.scaling_kept] = fit.beta[model.assign == len(product_terms)]
    scaling = [
        {"attribute": attribute, "panelist_id": str(panelist), "beta": float(beta)}
        for panelist, beta in zip(model.panelists, 1.0 + slopes - slopes.mean(), strict=True)
    ]
    return anova, components, scaling


def _check_reml_inputs(panel: pd.DataFrame, product_factors: Sequence[str], replication: str | None) -> None:
    """Raise a ValueError naming what is missing or unusable."""
    needed = ["panelist_id", "attribute", "score", *product_factors]
    if replication is not None:
        needed.append(replication)
    missing = [column for column in needed if column not in panel.columns]
    if missing:
        hint = " Pass replication=None for a panel without replicates." if replication in missing else ""
        raise ValueError(
            f"mixed_assessor_model_reml needs the columns {needed}; missing {missing}.{hint} "
            f"Got columns {list(panel.columns)}."
        )
    if len(panel) == 0:
        raise ValueError("mixed_assessor_model_reml was given a panel with no rows, so there is no attribute to fit.")
    n_cells = panel.groupby(list(product_factors), observed=True).ngroups
    if n_cells < _MIN_PRODUCTS:
        raise ValueError(
            f"The MAM needs at least {_MIN_PRODUCTS} products to separate scaling from disagreement; "
            f"the product factors {list(product_factors)} define {n_cells}."
        )


def mixed_assessor_model_reml(
    panel: pd.DataFrame,
    *,
    product_factors: Sequence[str] = ("product",),
    replication: str | None = "replicate",
    alpha_random: float = 0.1,
) -> MAMRemlResult:
    """Fit the random-effects Mixed Assessor Model per attribute, as SensMixed does.

    The model for one attribute has the product terms (every main effect and
    interaction of ``product_factors``) and a ``scaling`` term as fixed effects. The
    scaling term gives each panelist a slope on the centred product means, so a
    panelist who uses a wider or narrower part of the scale is modelled, not counted
    as disagreement. The random terms are each product term crossed with the
    panelist, the replicate and both, and those three on their own.

    The random part is then reduced as SensMixed reduces it:

    1. terms whose standard deviation is estimated as zero are dropped;
    2. each remaining term is tested by a likelihood-ratio test (REML, 1 df), and
       the least significant is eliminated while its p-value exceeds
       ``alpha_random``. The panelist and the product-by-panelist terms are tested
       but always kept.

    The fixed terms of the final model are tested in sequence (Type I), with
    Satterthwaite's denominator degrees of freedom. For a balanced panel and a
    one-factor product this reproduces the closed form of
    :func:`mixed_assessor_model`; with several product factors each gets the error
    term the selected random effects imply, which the closed form cannot do.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long-format panel data with ``panelist_id``, ``attribute``, ``score`` and the
        product and replicate columns named below.
    product_factors : sequence of str
        The column(s) describing the products. The default is the ``product`` column;
        for products from a factorial design, name its factors (for example
        ``("tv_set", "picture")``) and every product term is tested separately.
    replication : str or None
        The replicate (session) column. A replicate is taken to be shared by the
        panelists, as a session is, so the replicate gets random effects of its own.
        ``None`` (or a column with one level) fits the panelist terms only.
    alpha_random : float
        A random term is eliminated while its likelihood-ratio p-value exceeds this.
        ``1.0`` keeps every term.

    Returns
    -------
    MAMRemlResult
        The F-tests, the random-term selection, the variance components and the
        scaling coefficients; see the class docstring.

    Raises
    ------
    ValueError
        If a column is missing, the panel is empty, or the products number fewer
        than three.

    Notes
    -----
    This reproduces ``sensmixed(..., MAM = TRUE)`` of the R package SensMixed 2.1
    with its defaults (all product interactions; replicate error structure; no
    separate scaling per product factor), which runs lme4 and lmerTest. The test
    suite checks it against SensMixed on the TVbo panel. The estimates are exact
    REML optima, where lme4's are numerical; they agree to about 1e-6.

    Examples
    --------
    >>> mam = mixed_assessor_model_reml(panel, product_factors=("tv_set", "picture"))
    >>> mam.anova.query("attribute == 'sharpness'")
    >>> mam.random_effects.query("status == 'eliminated'")
    """
    product_factors = tuple(product_factors)
    _check_reml_inputs(panel, product_factors, replication)
    product_terms = [
        terms for size in range(1, len(product_factors) + 1) for terms in combinations(product_factors, size)
    ]
    keep = {(*product_factors, "panelist_id"), ("panelist_id",)}

    tables: dict[str, list[dict]] = {"anova": [], "random": [], "components": [], "scaling": []}
    # Hundreds of small dense fits: waking a BLAS thread pool for each call costs more
    # than the arithmetic (ten times the run time, measured on four cores).
    with threadpool_limits(limits=1, user_api="blas"):
        for attribute in sorted(panel["attribute"].unique()):
            frame = panel.loc[panel["attribute"] == attribute].dropna(subset=["score"]).reset_index(drop=True)
            rep = replication if replication is not None and frame[replication].unique().size > 1 else None
            model = _AttributeModel(frame, product_terms, rep)
            terms, fit, records = _select_random(model, keep, alpha_random)
            tables["random"] += [{"attribute": str(attribute), **record} for record in records]
            anova, components, scaling = _attribute_tables(str(attribute), model, terms, fit, product_terms)
            tables["anova"] += anova
            tables["components"] += components
            tables["scaling"] += scaling

    random_effects = pd.DataFrame(tables["random"])
    random_effects["step"] = random_effects["step"].astype("Int64")
    variance_components = pd.DataFrame(tables["components"])
    variance_components["std_dev"] = np.sqrt(variance_components["variance"])
    return MAMRemlResult(
        anova=pd.DataFrame(tables["anova"]),
        random_effects=random_effects,
        variance_components=variance_components,
        scaling=pd.DataFrame(tables["scaling"]),
    )
