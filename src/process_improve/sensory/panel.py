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


def panel_scorecard(panel: pd.DataFrame) -> PanelScorecard:  # noqa: C901
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

        # Discrimination and agreement, averaged over attributes.
        etas: list[float] = []
        corrs: list[float] = []
        for _attr, grp in pan_panel.groupby("attribute", observed=True):
            etas.append(_eta_squared(grp["score"], grp["product"]))
        for attr, grp in pan_cell.groupby("attribute", observed=True):
            own = grp.set_index("product")["score"]
            ref = consensus.xs(attr, level="attribute")
            joined = pd.concat([own, ref], axis=1, join="inner").dropna()
            if joined.shape[0] >= 2 and joined.iloc[:, 0].std() > 0 and joined.iloc[:, 1].std() > 0:
                corrs.append(float(joined.iloc[:, 0].corr(joined.iloc[:, 1])))

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
            "agreement": float(np.nanmean(corrs)) if corrs else float("nan"),
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
        if pid in low_agree or table.loc[pid, "agreement"] <= _AGREEMENT_FLOOR:
            why.append("low agreement with the panel")
        if pid in low_discrim:
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
