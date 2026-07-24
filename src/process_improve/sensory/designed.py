"""(c) Kevin Dunn, 2010-2026. MIT License.

Designed-mode comparison of descriptive panel data: factorial ANOVA of the
product (formulation) effect, with all-pairwise and against-a-control post-hoc
multiple comparisons.

The observational half of this subpackage relates attributes to *measured*
covariates of products whose formulation is unknown. This module covers the
complementary **designed** question: the products are controlled treatments
(formulations, aging conditions, ...), the same panelists score every treatment
(a randomized complete block design, with panelist as the block), and we want to
know which treatments differ, and by how much, on each attribute.

Per attribute the workhorse is a fixed-effects ANOVA

    score ~ C(factor_1) * C(factor_2) * ... + C(block)

fitted by ordinary least squares with **Type III** sums of squares, so an
unbalanced grid (missing cells, a panelist who skipped a sample) is handled
correctly and the interaction terms test whether one factor's effect depends on
another (e.g. does aging change some formulations more than others). When the
omnibus factor effect is real, two post-hoc procedures answer the follow-up
question:

* :func:`tukey_hsd` - all-pairwise Tukey HSD, using the blocked-model error
  mean square and the studentized-range distribution, so the block (panelist)
  variance is removed from the yardstick. Answers "which treatments differ from
  which". A compact-letter display groups treatments that are not separable.
* :func:`dunnett_vs_control` - Dunnett's two-sided test of every treatment
  against a single named control, controlling the family-wise error for the
  "many treatments, one control" comparison only (tighter than Tukey).

:func:`compare_products` runs the whole sequence and returns a
:class:`ComparisonResult`. Everything is generic in the factor column names, so
the same code serves a one-factor formulation screen or a
formulation-by-aging-condition stability study.

References
----------
Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
*Biometrics*, 5(2), 99-114.

Dunnett, C. W. (1955). A multiple comparison procedure for comparing several
treatments with a control. *JASA*, 50(272), 1096-1121.

Piepho, H.-P. (2004). An algorithm for a letter-based representation of
all-pairwise comparisons. *Journal of Computational and Graphical Statistics*,
13(2), 456-466.

Naes, T., Brockhoff, P. B. & Tomic, O. (2010). *Statistics for Sensory and
Consumer Science*. Wiley.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import dunnett as _scipy_dunnett
from scipy.stats import studentized_range
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from process_improve.univariate.metrics import confidence_interval

#: Regular expression that recovers the original column name from a patsy
#: ``C(Q('name'))`` term, so ANOVA tables read in the caller's own vocabulary.
_Q_TERM = re.compile(r"Q\('([^']+)'\)")


@dataclass
class ComparisonResult:
    """Outcome of :func:`compare_products`.

    Attributes
    ----------
    anova : pandas.DataFrame
        Type III ANOVA table, one row per (attribute, source) with ``df``,
        ``sum_sq``, ``mean_sq``, ``F`` and ``p_value``. The ``Residual`` row
        carries the error mean square used by the post-hoc tests.
    tukey : pandas.DataFrame
        All-pairwise Tukey HSD contrasts (see :func:`tukey_hsd`), prefixed with
        the stratum column when the comparison was stratified.
    dunnett : pandas.DataFrame
        Dunnett-vs-control contrasts (see :func:`dunnett_vs_control`); empty when
        no ``control`` was given.
    letters : pandas.DataFrame
        Compact-letter display: one row per (stratum, attribute, level) with a
        ``letters`` string. Treatments that share a letter are not separable at
        the chosen ``alpha``.
    means : pandas.DataFrame
        Per (stratum, attribute, level) mean, confidence interval and ``n``.
    config : dict
        The resolved call arguments, for provenance.
    """

    anova: pd.DataFrame
    tukey: pd.DataFrame
    dunnett: pd.DataFrame
    letters: pd.DataFrame
    means: pd.DataFrame
    config: dict[str, Any]


def _clean_term(term: str) -> str:
    """Turn a patsy term such as ``C(Q('a')):C(Q('b'))`` back into ``a:b``."""
    names = _Q_TERM.findall(term)
    return ":".join(names) if names else term


def _formula(factors: list[str], block: str | None, *, interactions: bool) -> str:
    """Build the OLS formula for one attribute."""
    joiner = " * " if (interactions and len(factors) > 1) else " + "
    terms = joiner.join(f"C(Q('{f}'))" for f in factors)
    if block is not None:
        terms = f"{terms} + C(Q('{block}'))"
    return f"score ~ {terms}"


def factorial_anova(
    panel: pd.DataFrame,
    *,
    factors: list[str],
    block: str | None = "panelist_id",
    interactions: bool = True,
) -> pd.DataFrame:
    """Type III factorial ANOVA of the panel scores, one model per attribute.

    Parameters
    ----------
    panel : pandas.DataFrame
        Descriptive-long panel data (columns ``attribute`` and ``score`` plus the
        factor and block columns named below).
    factors : list of str
        Column names of the fixed factors of interest (e.g.
        ``["formulation", "condition"]``). With more than one factor and
        ``interactions=True`` their full crossed model is fitted.
    block : str or None
        Column treated as a blocking factor (default ``"panelist_id"``). Pass
        ``None`` for no block.
    interactions : bool
        Include the factor-by-factor interaction terms (default ``True``).

    Returns
    -------
    pandas.DataFrame
        One row per (attribute, source) with ``df``, ``sum_sq``, ``mean_sq``,
        ``F`` and ``p_value``. The ``Residual`` row gives the error term. An
        attribute whose model cannot be fitted (too few observations, a singular
        design) yields a single ``source="(model failed)"`` row rather than
        aborting the sweep.

    Examples
    --------
    >>> factorial_anova(panel, factors=["formulation", "condition"]).head()
    """
    missing = [c for c in [*factors, *([block] if block else []), "attribute", "score"] if c not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required column(s): {missing}")

    formula = _formula(factors, block, interactions=interactions)
    rows: list[dict[str, Any]] = []
    for attribute in sorted(panel["attribute"].unique()):
        sub = panel[panel["attribute"] == attribute].dropna(subset=["score"])
        try:
            model = ols(formula, data=sub).fit()
            table = anova_lm(model, typ=3)
        except Exception as exc:  # noqa: BLE001 - degenerate design should not abort the sweep
            rows.append(
                {
                    "attribute": str(attribute),
                    "source": "(model failed)",
                    "df": float("nan"),
                    "sum_sq": float("nan"),
                    "mean_sq": float("nan"),
                    "F": float("nan"),
                    "p_value": float("nan"),
                    "note": str(exc).splitlines()[0][:120],
                }
            )
            continue
        for term, record in table.iterrows():
            if term == "Intercept":
                continue
            df_term = float(record["df"])
            ss_term = float(record["sum_sq"])
            rows.append(
                {
                    "attribute": str(attribute),
                    "source": _clean_term(str(term)),
                    "df": df_term,
                    "sum_sq": ss_term,
                    "mean_sq": ss_term / df_term if df_term > 0 else float("nan"),
                    "F": float(record["F"]) if not pd.isna(record["F"]) else float("nan"),
                    "p_value": float(record["PR(>F)"]) if not pd.isna(record["PR(>F)"]) else float("nan"),
                    "note": "",
                }
            )
    return pd.DataFrame(rows)


def _group_means(sub: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Per-level mean and non-missing count for one attribute subset."""
    grouped = sub.dropna(subset=["score"]).groupby(factor, observed=True)["score"]
    return pd.DataFrame({"mean": grouped.mean(), "n": grouped.count()})


def _blocked_error(sub: pd.DataFrame, factor: str, block: str | None) -> tuple[float, float]:
    """Return (mean-square error, error degrees of freedom) of ``score ~ factor + block``."""
    formula = _formula([factor], block, interactions=False)
    model = ols(formula, data=sub.dropna(subset=["score"])).fit()
    return float(model.mse_resid), float(model.df_resid)


def tukey_hsd(
    panel: pd.DataFrame,
    *,
    factor: str,
    block: str | None = "panelist_id",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """All-pairwise Tukey HSD of ``factor`` levels, one comparison per attribute.

    The critical difference uses the error mean square of the blocked model
    ``score ~ C(factor) + C(block)`` and the studentized-range distribution, so
    the block (panelist) variance is removed. Unequal group sizes use the
    Tukey-Kramer standard error.

    Parameters
    ----------
    panel : pandas.DataFrame
        Descriptive-long panel data.
    factor : str
        Column whose levels are compared all-pairwise.
    block : str or None
        Blocking column (default ``"panelist_id"``); ``None`` for no block.
    alpha : float
        Family-wise significance level (default ``0.05``).

    Returns
    -------
    pandas.DataFrame
        One row per (attribute, pair) with ``group1``, ``group2``, ``meandiff``
        (group1 minus group2), ``se``, ``q_stat``, ``p_value``, the
        ``(ci_low, ci_high)`` simultaneous interval and ``reject``.
    """
    rows: list[dict[str, Any]] = []
    for attribute in sorted(panel["attribute"].unique()):
        sub = panel[panel["attribute"] == attribute].dropna(subset=["score"])
        means = _group_means(sub, factor)
        levels = list(means.index)
        n_levels = len(levels)
        if n_levels < 2:
            continue
        try:
            mse, df_err = _blocked_error(sub, factor, block)
        except Exception:  # noqa: BLE001, S112 - degenerate attribute is skipped
            continue
        if not (df_err > 0 and mse > 0):
            continue
        q_crit = float(studentized_range.ppf(1.0 - alpha, n_levels, df_err))
        for a, b in combinations(levels, 2):
            mean_a, mean_b = float(means.loc[a, "mean"]), float(means.loc[b, "mean"])
            n_a, n_b = float(means.loc[a, "n"]), float(means.loc[b, "n"])
            se = float(np.sqrt(mse / 2.0 * (1.0 / n_a + 1.0 / n_b)))
            diff = mean_a - mean_b
            q_stat = abs(diff) / se if se > 0 else float("nan")
            p_value = float(studentized_range.sf(q_stat, n_levels, df_err)) if se > 0 else float("nan")
            half = q_crit * se
            rows.append(
                {
                    "attribute": str(attribute),
                    "group1": str(a),
                    "group2": str(b),
                    "meandiff": diff,
                    "se": se,
                    "q_stat": q_stat,
                    "p_value": p_value,
                    "ci_low": diff - half,
                    "ci_high": diff + half,
                    "reject": bool(p_value < alpha),
                }
            )
    return pd.DataFrame(rows)


def dunnett_vs_control(
    panel: pd.DataFrame,
    *,
    factor: str,
    control: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Dunnett's two-sided test of every ``factor`` level against ``control``.

    Uses :func:`scipy.stats.dunnett`, which pools the within-level variance and
    controls the family-wise error for the many-treatments-vs-one-control family
    (tighter than all-pairwise Tukey when the control is the only reference of
    interest). Unlike :func:`tukey_hsd` it does not remove a block term.

    Parameters
    ----------
    panel : pandas.DataFrame
        Descriptive-long panel data.
    factor : str
        Column whose levels are compared to the control.
    control : str
        The level of ``factor`` used as the reference.
    alpha : float
        Significance level (default ``0.05``).

    Returns
    -------
    pandas.DataFrame
        One row per (attribute, level) for every non-control level, with
        ``meandiff`` (level minus control), ``statistic``, ``p_value`` and
        ``reject``.
    """
    rows: list[dict[str, Any]] = []
    for attribute in sorted(panel["attribute"].unique()):
        sub = panel[panel["attribute"] == attribute].dropna(subset=["score"])
        if control not in set(sub[factor].unique()):
            continue
        control_scores = sub.loc[sub[factor] == control, "score"].to_numpy()
        others = [lvl for lvl in sorted(sub[factor].unique()) if lvl != control]
        samples = [sub.loc[sub[factor] == lvl, "score"].to_numpy() for lvl in others]
        usable = [(lvl, arr) for lvl, arr in zip(others, samples, strict=True) if arr.size >= 1]
        if len(control_scores) < 2 or not usable or any(arr.size < 2 for _, arr in usable):
            continue
        control_mean = float(np.mean(control_scores))
        try:
            result = _scipy_dunnett(*[arr for _, arr in usable], control=control_scores)
        except Exception:  # noqa: BLE001, S112 - degenerate attribute is skipped
            continue
        stats = np.atleast_1d(result.statistic)
        pvals = np.atleast_1d(result.pvalue)
        for (lvl, arr), stat, pval in zip(usable, stats, pvals, strict=True):
            rows.append(
                {
                    "attribute": str(attribute),
                    "level": str(lvl),
                    "control": str(control),
                    "meandiff": float(np.mean(arr)) - control_mean,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "reject": bool(pval < alpha),
                }
            )
    return pd.DataFrame(rows)


def _compact_letter_display(levels_by_mean: list[str], sig_pairs: set[frozenset[str]]) -> dict[str, str]:
    """Map each level to a letter string (Piepho 2004 insert-and-absorb).

    ``levels_by_mean`` is ordered by descending mean so the first letter tends to
    mark the highest group. ``sig_pairs`` holds the frozenset ``{a, b}`` for every
    pair judged significantly different. Levels sharing a letter are not separable.
    """
    columns: list[set[str]] = [set(levels_by_mean)]
    for pair in sig_pairs:
        a, b = tuple(pair)
        rebuilt: list[set[str]] = []
        for col in columns:
            if a in col and b in col:
                rebuilt.append(col - {a})
                rebuilt.append(col - {b})
            else:
                rebuilt.append(col)
        # Absorb: drop empties and any column that is a subset of another.
        rebuilt = [c for c in rebuilt if c]
        keep: list[set[str]] = []
        for col in rebuilt:
            if any(col < other for other in rebuilt if col is not other) or any(col == other for other in keep):
                continue
            keep.append(col)
        columns = keep

    # Order columns by the position of their highest-mean member, then letter them.
    order = {lvl: i for i, lvl in enumerate(levels_by_mean)}
    columns.sort(key=lambda col: min(order[lvl] for lvl in col))
    letters: dict[str, list[str]] = {lvl: [] for lvl in levels_by_mean}
    for idx, col in enumerate(columns):
        tag = chr(ord("a") + idx) if idx < 26 else f"({idx + 1})"
        for lvl in col:
            letters[lvl].append(tag)
    return {lvl: "".join(tags) for lvl, tags in letters.items()}


def _letters_table(tukey: pd.DataFrame, means: pd.DataFrame, factor: str, alpha: float) -> pd.DataFrame:
    """Build the compact-letter-display table from a Tukey frame and level means."""
    rows: list[dict[str, Any]] = []
    for attribute, grp in means.groupby("attribute", observed=True):
        ordered = list(grp.sort_values("mean", ascending=False)[factor].astype(str))
        pairs = tukey[tukey["attribute"] == attribute]
        sig = {frozenset((str(r.group1), str(r.group2))) for r in pairs.itertuples() if bool(r.reject)}
        mapping = _compact_letter_display(ordered, sig)
        rows.extend({"attribute": str(attribute), factor: lvl, "letters": mapping.get(lvl, "")} for lvl in ordered)
    return pd.DataFrame(rows)


def _means_table(panel: pd.DataFrame, factor: str, conf_level: float) -> pd.DataFrame:
    """Per (attribute, level) mean with a confidence interval and count."""
    rows: list[dict[str, Any]] = []
    for (attr, lvl), grp in panel.groupby(["attribute", factor], observed=True):
        scores = grp[["score"]].dropna()
        center = float(scores["score"].mean()) if not scores.empty else float("nan")
        if scores.shape[0] >= 2:
            lo, hi = confidence_interval(scores, "score", conflevel=conf_level, style="regular")
        else:
            lo = hi = float("nan")
        rows.append(
            {
                "attribute": str(attr),
                factor: str(lvl),
                "mean": center,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "n": int(scores.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def compare_products(  # noqa: PLR0913
    panel: pd.DataFrame,
    *,
    factors: list[str],
    block: str | None = "panelist_id",
    primary: str | None = None,
    within: str | None = None,
    control: str | None = None,
    interactions: bool = True,
    alpha: float = 0.05,
    conf_level: float = 0.95,
) -> ComparisonResult:
    """Compare product treatments per attribute: ANOVA plus post-hoc contrasts.

    Fits the factorial ANOVA over ``factors`` (with a blocking factor), then runs
    all-pairwise Tukey HSD and, when ``control`` is given, Dunnett vs the control,
    on the ``primary`` factor. When ``within`` is set the post-hoc tests are run
    as *simple effects* separately within each level of that factor (e.g. compare
    formulations within each aging condition), which is the right follow-up once
    the ``primary x within`` interaction is significant.

    Parameters
    ----------
    panel : pandas.DataFrame
        Descriptive-long panel data.
    factors : list of str
        Fixed factors for the ANOVA (e.g. ``["formulation", "condition"]``).
    block : str or None
        Blocking column (default ``"panelist_id"``).
    primary : str or None
        Factor whose levels the post-hoc tests compare. Defaults to
        ``factors[0]``.
    within : str or None
        If given, run the post-hoc tests separately within each level of this
        factor (simple effects). If ``None``, they pool over the other factors.
    control : str or None
        Level of ``primary`` used as the Dunnett reference. If ``None``, the
        Dunnett table is empty.
    interactions : bool
        Include interaction terms in the ANOVA (default ``True``).
    alpha : float
        Post-hoc significance level (default ``0.05``).
    conf_level : float
        Confidence level for the reported means (default ``0.95``).

    Returns
    -------
    ComparisonResult
        The ANOVA table, Tukey and Dunnett contrasts, compact-letter display and
        per-level means; see the class docstring.

    Examples
    --------
    >>> res = compare_products(
    ...     panel, factors=["formulation", "condition"], within="condition", control="Control"
    ... )
    >>> res.letters.query("condition == 'REF'").head()
    """
    primary = primary or factors[0]
    stratum_col = within or "stratum"
    anova = factorial_anova(panel, factors=factors, block=block, interactions=interactions)

    strata = [None] if within is None else sorted(panel[within].dropna().unique())
    tukey_frames: list[pd.DataFrame] = []
    dunnett_frames: list[pd.DataFrame] = []
    letter_frames: list[pd.DataFrame] = []
    means_frames: list[pd.DataFrame] = []
    for stratum in strata:
        sub = panel if stratum is None else panel[panel[within] == stratum]
        label = "all" if stratum is None else str(stratum)

        means = _means_table(sub, primary, conf_level)
        tukey = tukey_hsd(sub, factor=primary, block=block, alpha=alpha)
        letters = _letters_table(tukey, means, primary, alpha)
        for frame in (means, tukey, letters):
            frame.insert(0, stratum_col, label)
        means_frames.append(means)
        tukey_frames.append(tukey)
        letter_frames.append(letters)

        if control is not None:
            dunnett = dunnett_vs_control(sub, factor=primary, control=control, alpha=alpha)
            dunnett.insert(0, stratum_col, label)
            dunnett_frames.append(dunnett)

    empty_dunnett = pd.DataFrame(
        columns=[stratum_col, "attribute", "level", "control", "meandiff", "statistic", "p_value", "reject"]
    )
    return ComparisonResult(
        anova=anova,
        tukey=pd.concat(tukey_frames, ignore_index=True) if tukey_frames else pd.DataFrame(),
        dunnett=pd.concat(dunnett_frames, ignore_index=True) if dunnett_frames else empty_dunnett,
        letters=pd.concat(letter_frames, ignore_index=True) if letter_frames else pd.DataFrame(),
        means=pd.concat(means_frames, ignore_index=True) if means_frames else pd.DataFrame(),
        config={
            "factors": list(factors),
            "block": block,
            "primary": primary,
            "within": within,
            "control": control,
            "interactions": interactions,
            "alpha": alpha,
            "conf_level": conf_level,
        },
    )
