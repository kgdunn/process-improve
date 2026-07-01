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
        ``drift``, and the two-sided outlier bands ``scale_use_band`` and
        ``offset_band`` (each ``"low"`` / ``"normal"`` / ``"high"``).
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


def _tail_bands(values: pd.Series) -> pd.Series:
    """Classify each panelist as ``"low"`` / ``"normal"`` / ``"high"`` on ``values``.

    Uses two-sided Tukey fences: below ``Q1 - 1.5 * IQR`` is ``"low"``, above
    ``Q3 + 1.5 * IQR`` is ``"high"``, everything between is ``"normal"``. A
    caller (for example a front-end colouring a panel map) can act on this stable
    label instead of re-deriving thresholds from the raw numbers, and both tails
    are surfaced symmetrically so a compressor is reported just like an expander.
    The IQR fence is deliberately looser than the ESD test used for dropping,
    because scale use and offset are corrected by alignment rather than dropped,
    so the aim is to surface the genuine tails, not to withhold all but the most
    extreme.
    """
    bands = pd.Series("normal", index=values.index, dtype=object)
    clean = values.dropna()
    if clean.size < _MIN_PANELISTS_FOR_ESD:
        return bands
    q1, q3 = (float(clean.quantile(q)) for q in (0.25, 0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return bands
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    bands.loc[clean.index[clean < low_fence]] = "low"
    bands.loc[clean.index[clean > high_fence]] = "high"
    return bands


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

    # Two-sided outlier band per scale metric, so a caller can colour / label
    # a panelist without re-deriving thresholds. Scale use and offset are
    # correctable by alignment, so they are reported as bands here, not flagged.
    table["scale_use_band"] = _tail_bands(table["scale_spread"])
    table["offset_band"] = _tail_bands(table["scale_shift"])

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
