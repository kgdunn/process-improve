"""(c) Kevin Dunn, 2010-2026. MIT License.

Panel-anomaly scorecard for descriptive panel data.

Before any product conclusion is drawn, each panelist is scored on four axes:

* **discrimination** - does the panelist separate the products (mean
  eta-squared of the product effect across attributes; higher is better);
* **agreement** - does the panelist rank the products like the rest of the
  panel (mean correlation of the panelist's product means with the panel's,
  across attributes; higher is better);
* **scale use** - a location shift and a spread ratio relative to the panel;
* **drift** - association between session order and the panelist's mean score
  (only when more than one session is present).

Panelists that discriminate poorly, disagree with the panel, or use the scale
atypically are flagged so the caller can keep or drop them before relating the
attributes to the product.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from process_improve.univariate.metrics import detect_outliers_esd

#: A correlation at or below this with the panel is treated as disagreement
#: regardless of the panel size (anti-correlation is unambiguous).
_AGREEMENT_FLOOR = 0.0

#: A panelist is only flagged when it is both a relative outlier (ESD) and below
#: one of these absolute levels, so a tight, healthy cluster is not flagged just
#: because one member sits at its low edge.
_AGREEMENT_SUSPECT = 0.5
_DISCRIMINATION_SUSPECT = 0.5

#: Minimum number of panelists before the Extreme Studentised Deviate test is
#: used to flag low-tail outliers; below this the floor rules apply alone.
_MIN_PANELISTS_FOR_ESD = 4


@dataclass
class PanelScorecard:
    """Outcome of :func:`panel_scorecard`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per panelist, indexed by ``panelist_id``, with the columns
        ``discrimination``, ``agreement``, ``scale_shift``, ``scale_spread``,
        and ``drift``.
    flagged : list of str
        Panelist ids flagged as anomalous.
    reasons : dict
        Maps each flagged panelist id to the list of reasons it was flagged.
    """

    table: pd.DataFrame
    flagged: list[str]
    reasons: dict[str, list[str]] = field(default_factory=dict)


def _eta_squared(scores: pd.Series, groups: pd.Series) -> float:
    """Return the eta-squared of the ``groups`` effect on ``scores``."""
    valid = scores.notna()
    scores = scores[valid]
    groups = groups[valid]
    if scores.size < 2 or groups.nunique() < 2:
        return float("nan")
    grand = scores.mean()
    ss_total = float(((scores - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for _, grp in scores.groupby(groups):
        ss_between += grp.size * (grp.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _low_tail_outliers(values: pd.Series) -> set[str]:
    """Return the index labels of low-side outliers in ``values``."""
    clean = values.dropna()
    if clean.size < _MIN_PANELISTS_FOR_ESD:
        return set()
    arr = clean.to_numpy(dtype=float)
    indices, _ = detect_outliers_esd(
        arr,
        algorithm="esd",
        max_outliers_detected=max(1, clean.size // 4),
        alpha=0.05,
        robust_variant=True,
    )
    median = float(np.median(arr))
    return {str(clean.index[i]) for i in indices if arr[i] < median}


def panel_scorecard(panel: pd.DataFrame) -> PanelScorecard:
    """Score each panelist and flag anomalies.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data (the ``normalized_df`` of a
        :class:`~process_improve.sensory.validation.ValidationResult`).

    Returns
    -------
    PanelScorecard
        See the class docstring.

    Examples
    --------
    >>> card = panel_scorecard(validated.normalized_df)
    >>> card.flagged
    ['P7']
    """
    # Replicate-averaged score per (panelist, product, attribute).
    cell = (
        panel.groupby(["panelist_id", "product", "attribute"], observed=True)["score"]
        .mean()
        .reset_index()
    )
    # Panel consensus: mean across panelists per (product, attribute).
    consensus = cell.groupby(["product", "attribute"], observed=True)["score"].mean()

    grand_mean = panel["score"].mean()
    grand_sd = panel["score"].std()
    panelists = list(panel["panelist_id"].unique())

    records: dict[str, dict[str, float]] = {}
    for pid in panelists:
        pan_panel = panel[panel["panelist_id"] == pid]
        pan_cell = cell[cell["panelist_id"] == pid]

        # Discrimination: mean eta-squared of the product effect across
        # attributes (an attribute with no real product effect simply
        # contributes a low value to everyone, which does not bias flagging).
        etas = [
            _eta_squared(grp["score"], grp["product"])
            for _attr, grp in pan_panel.groupby("attribute", observed=True)
        ]
        # Agreement: a single correlation of the panelist's product-by-attribute
        # cell means with the panel consensus over the whole space. Stacking the
        # attributes lets those with real product variation dominate and avoids
        # the meaningless per-attribute correlations a near-flat attribute would
        # otherwise contribute.
        own_vec = pan_cell.set_index(["product", "attribute"])["score"]
        pair = pd.concat([own_vec, consensus], axis=1, join="inner").dropna()
        if pair.shape[0] >= 2 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
            agreement = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        else:
            agreement = float("nan")

        pan_scores = pan_panel["score"]
        # Drift: association of session order with the panelist's mean score.
        drift = float("nan")
        if pan_panel["session"].nunique() >= 2:
            by_session = pan_panel.groupby("session", observed=True)["score"].mean()
            order = pd.Series(range(by_session.size), index=by_session.index, dtype=float)
            if by_session.std() > 0:
                drift = float(by_session.reset_index(drop=True).corr(order.reset_index(drop=True)))

        records[str(pid)] = {
            "discrimination": float(np.nanmean(etas)) if etas else float("nan"),
            "agreement": agreement,
            "scale_shift": float(pan_scores.mean() - grand_mean),
            "scale_spread": float(pan_scores.std() / grand_sd) if grand_sd and grand_sd > 0 else float("nan"),
            "drift": drift,
        }

    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "panelist_id"

    # --- Flagging ------------------------------------------------------
    # Flag only the two axes that threaten product validity: a panelist who
    # disagrees with the panel, or who does not separate the products. Scale
    # use and drift are reported in the table as diagnostics, not auto-flagged,
    # because naturally variable scale use should not by itself drop a panelist.
    reasons: dict[str, list[str]] = {}
    low_agree = _low_tail_outliers(table["agreement"])
    low_discrim = _low_tail_outliers(table["discrimination"])

    for pid in table.index:
        why: list[str] = []
        agreement = table.loc[pid, "agreement"]
        discrimination = table.loc[pid, "discrimination"]
        if (pid in low_agree and agreement < _AGREEMENT_SUSPECT) or agreement <= _AGREEMENT_FLOOR:
            why.append("low agreement with the panel")
        if pid in low_discrim and discrimination < _DISCRIMINATION_SUSPECT:
            why.append("low discrimination between products")
        if why:
            reasons[str(pid)] = why

    return PanelScorecard(table=table, flagged=sorted(reasons), reasons=reasons)


#: Minimum products before a panelist's own-permutation null is meaningful.
_MIN_PRODUCTS_FOR_PERMUTATION = 3


@dataclass
class PanelConsistency:
    """Outcome of :func:`panel_consistency`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per panelist, indexed by ``panelist_id``, with the columns
        ``discrimination``, ``p_discrimination``, ``agreement``, ``p_agreement``,
        and ``consistent``.
    inconsistent : list of str
        Panelist ids whose product signal is not distinguishable from a reshuffle of
        their own scores (``consistent`` is ``False``).
    n_permutations : int
        Number of within-panelist permutations used for each null.
    """

    table: pd.DataFrame
    inconsistent: list[str]
    n_permutations: int


def _eta_squared_codes(scores: np.ndarray, codes: np.ndarray, n_groups: int) -> float:
    """Return the eta-squared of an integer-``codes`` grouping of ``scores``."""
    grand = float(scores.mean())
    ss_total = float(((scores - grand) ** 2).sum())
    if ss_total <= 0.0:
        return float("nan")
    sums = np.bincount(codes, weights=scores, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    means = np.divide(sums, counts, out=np.full(n_groups, grand), where=counts > 0)
    ss_between = float((counts * (means - grand) ** 2).sum())
    return ss_between / ss_total


def _group_means(scores: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Return the per-group mean of ``scores`` (NaN for an empty group)."""
    sums = np.bincount(codes, weights=scores, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    return np.divide(sums, counts, out=np.full(n_groups, np.nan), where=counts > 0)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return the Pearson correlation of two aligned vectors, or NaN if undefined."""
    if a.size < 2 or a.std() <= 0.0 or b.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _consistency_pvalues(  # noqa: PLR0913
    blocks: list[tuple[str, np.ndarray, np.ndarray, int]],
    locators: list[tuple[int, int]],
    consensus: np.ndarray,
    real: tuple[float, float],
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[float, float]:
    """Return one-sided permutation p-values for discrimination and agreement.

    Each permutation reshuffles the product labels within every attribute (breaking the
    panelist's product-to-score link while keeping their own score multiset), then
    recomputes the mean eta-squared and the agreement with the fixed leave-one-out
    consensus. The p-value is the add-one fraction of permutations reaching at least the
    real statistic (higher is better for both).
    """
    disc_real, agree_real = real
    ge_disc = 0
    ge_agree = 0
    for _ in range(n_permutations):
        # Permute the product labels once per attribute block and reuse that single
        # reshuffle for both the eta-squared and the agreement statistic.
        etas = []
        means_by_block = []
        for _attr, scores, codes, ng in blocks:
            pcodes = rng.permutation(codes)
            etas.append(_eta_squared_codes(scores, pcodes, ng))
            means_by_block.append(_group_means(scores, pcodes, ng))
        null_disc = float(np.nanmean(etas)) if etas else float("nan")
        if not np.isnan(null_disc) and null_disc >= disc_real:
            ge_disc += 1
        if not np.isnan(agree_real):
            null_own = np.array([means_by_block[bi][gi] for bi, gi in locators])
            null_agree = _correlation(null_own, consensus)
            if not np.isnan(null_agree) and null_agree >= agree_real:
                ge_agree += 1
    p_disc = (ge_disc + 1.0) / (n_permutations + 1.0)
    p_agree = (ge_agree + 1.0) / (n_permutations + 1.0) if not np.isnan(agree_real) else float("nan")
    return p_disc, p_agree


def panel_consistency(
    panel: pd.DataFrame,
    *,
    n_permutations: int = 299,
    random_state: int = 0,
    alpha: float = 0.05,
) -> PanelConsistency:
    """Test each panelist's product signal against a reshuffle of their own scores.

    For every panelist the product labels are permuted within each attribute (keeping the
    panelist's own score values, only reassigning which product got which), and the mean
    eta-squared product effect (:func:`_eta_squared_codes`) and the agreement with the
    leave-one-out panel consensus are recomputed. A panelist who carries a real,
    reproducible product signal beats almost every reshuffle of their own scores (small
    p-value); one whose scores are noise is indistinguishable from their own permutations
    (p near 1). The null is built entirely from the panelist's own data - no simulated or
    parametric reference.

    ``consistent`` gates on the discrimination p-value (does the panelist separate the
    products beyond their own noise). The agreement p-value is reported alongside but not
    gated: a panelist can discriminate reproducibly yet rank the products differently from
    the panel, which is disagreement rather than inconsistency.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data.
    n_permutations : int
        Within-panelist permutations per null.
    random_state : int
        Seed for the permutations.
    alpha : float
        Significance level for the ``consistent`` flag.

    Returns
    -------
    PanelConsistency
        See the class docstring.

    Examples
    --------
    >>> result = panel_consistency(validated.normalized_df, n_permutations=299)
    >>> result.inconsistent
    ['P8']
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be at least 1, got {n_permutations}.")

    cell = panel.groupby(["panelist_id", "product", "attribute"], observed=True)["score"].mean().reset_index()
    rng = np.random.default_rng(random_state)
    records: dict[str, dict[str, float | bool]] = {}
    for pid in panel["panelist_id"].unique():
        pan = panel[panel["panelist_id"] == pid]
        blocks: list[tuple[str, np.ndarray, np.ndarray, int]] = []
        block_labels: list[np.ndarray] = []
        etas_real: list[float] = []
        for attr, grp in pan.groupby("attribute", observed=True):
            scores = grp["score"].to_numpy(dtype=float)
            valid = ~np.isnan(scores)
            labels, codes = np.unique(grp["product"].to_numpy()[valid], return_inverse=True)
            if valid.sum() < 2 or labels.size < 2:
                continue
            blocks.append((str(attr), scores[valid], codes, labels.size))
            block_labels.append(labels)
            etas_real.append(_eta_squared_codes(scores[valid], codes, labels.size))

        # Agreement against the leave-one-out consensus, over the (product, attribute)
        # cells this panelist and the rest of the panel share.
        loo = cell[cell["panelist_id"] != pid].groupby(["product", "attribute"], observed=True)["score"].mean()
        loo_map = {(str(p), str(a)): float(v) for (p, a), v in loo.items()}
        locators: list[tuple[int, int]] = []
        consensus: list[float] = []
        real_own: list[float] = []
        for bi, (attr, scores, codes, ng) in enumerate(blocks):
            means = _group_means(scores, codes, ng)
            labels = block_labels[bi]
            for gi in range(ng):
                key = (str(labels[gi]), attr)
                if key in loo_map and not np.isnan(means[gi]):
                    locators.append((bi, gi))
                    consensus.append(loo_map[key])
                    real_own.append(float(means[gi]))
        finite_etas = [e for e in etas_real if not np.isnan(e)]
        discrimination = float(np.mean(finite_etas)) if finite_etas else float("nan")
        agreement = _correlation(np.array(real_own), np.array(consensus)) if len(real_own) >= 2 else float("nan")

        if not blocks or pan["product"].nunique() < _MIN_PRODUCTS_FOR_PERMUTATION or np.isnan(discrimination):
            records[str(pid)] = {
                "discrimination": discrimination,
                "p_discrimination": float("nan"),
                "agreement": agreement,
                "p_agreement": float("nan"),
                "consistent": False,
            }
            continue

        p_disc, p_agree = _consistency_pvalues(
            blocks, locators, np.array(consensus), (discrimination, agreement), rng, n_permutations
        )
        records[str(pid)] = {
            "discrimination": discrimination,
            "p_discrimination": p_disc,
            "agreement": agreement,
            "p_agreement": p_agree,
            "consistent": bool(p_disc <= alpha),
        }

    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "panelist_id"
    inconsistent = sorted(str(pid) for pid in table.index if not bool(table.loc[pid, "consistent"]))
    return PanelConsistency(table=table, inconsistent=inconsistent, n_permutations=n_permutations)


def apply_correction(panel: pd.DataFrame, drop: list[str]) -> pd.DataFrame:
    """Return ``panel`` with the listed panelists removed.

    Parameters
    ----------
    panel : pandas.DataFrame
        Validated ``descriptive_long`` panel data.
    drop : list of str
        Panelist ids to remove.

    Returns
    -------
    pandas.DataFrame
        The panel without the dropped panelists.
    """
    if not drop:
        return panel
    return panel[~panel["panelist_id"].isin(drop)].reset_index(drop=True)
