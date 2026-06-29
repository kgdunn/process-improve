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

This is a pure-Python MAM. A later release may add the SensMixed / lmerTest
random-effects F-test via an R bridge; see the tracking issue.

References
----------
Brockhoff, Schlich and Skovgaard, "Taking individual scaling differences into
account by analyzing profile data with the Mixed Assessor Model", Food Quality
and Preference, 39, 156-166, 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

from process_improve.regression._robust_regression import repeated_median_slope

#: Slopes at or below this are treated as unusable for rescaling (a panelist
#: who is flat or anti-correlated with the panel cannot be scale-corrected).
_MIN_SLOPE = 0.2

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

    Examples
    --------
    >>> mam = mixed_assessor_model(validated.normalized_df)
    >>> mam.scaling.query("attribute == 'saltiness'").sort_values("beta").head()
    """
    scaling_rows: list[dict[str, object]] = []
    ftest_rows: list[dict[str, object]] = []

    for attribute in sorted(panel["attribute"].unique()):
        matrix = _cell_means(panel, attribute)
        n_assessors, n_products = matrix.shape
        beta, means, _tau, ssq_tau = _assessor_scaling(matrix)
        grand = float(matrix.to_numpy().mean())

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
    (standard MAM practice), keeping panelists rather than dropping them; a
    panelist who is flat or anti-correlated with the panel (``beta`` not usable)
    is left location-corrected only.

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
        grand = float(matrix.to_numpy().mean())
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
