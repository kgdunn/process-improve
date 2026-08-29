"""(c) Kevin Dunn, 2010-2026. MIT License.

Preconditions for panel analysis: can this attribute be modelled at all?

Every model in this subpackage assumes the attribute behaves like an intensity
that assessors read off a linear scale, and that what separates assessors is how
much of that scale they use. Two things break the assumption, and both of them
break it quietly:

* **The attribute is pinned against a bound.** In a region where everyone
  records the same value, no scaling difference is expressible, so the Mixed
  Assessor Model has nothing to estimate and reports the residue as
  disagreement. :func:`boundary_occupancy` measures how much of an attribute
  lives against the floor or the ceiling; :func:`detection_rate` gives the
  response that is appropriate instead, a probability of detection rather than
  an intensity.
* **Assessors differ in how noisy they are, not in how they scale.**
  :func:`assessor_variance_equality` tests that directly. Grossmann et al.
  (2023) show that the Mixed Assessor Model reads unequal assessor variance as
  a scaling effect, which shifts its F-test so that real disagreement is
  understated. A small p-value here means the model's scaling coefficients are
  measuring partly that, and the F-test should be read with the finding in mind.

All three take long-format panel data with ``panelist_id``, ``product``,
``attribute`` and ``score`` columns; the canonical ``descriptive_long`` schema
from :func:`~process_improve.sensory.validate_descriptive` satisfies that.

References
----------
Grossmann, Ellis, Hopfer and others, "The effect of unequal assessor variance
on the Mixed Assessor Model", Food Quality and Preference, 105, 104792, 2023,
doi:10.1016/j.foodqual.2022.104792.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import levene

#: Columns every function in this module reads.
_REQUIRED_COLUMNS: tuple[str, ...] = ("panelist_id", "product", "attribute", "score")

#: Levene needs at least this many observations in a group to say anything about
#: its spread.
_MIN_PER_ASSESSOR = 2


def _require_panel(panel: pd.DataFrame, caller: str) -> None:
    """Validate that ``panel`` carries the required columns and at least one row.

    An empty panel is reachable whenever an upstream filter removes every
    attribute, and every function here would otherwise return a frame with no
    columns at all, turning the problem into a ``KeyError`` several calls away
    from its cause.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(
            f"{caller} needs the long-format panel columns {list(_REQUIRED_COLUMNS)}; "
            f"missing {missing}. Got columns {list(panel.columns)}."
        )
    if len(panel) == 0:
        raise ValueError(
            f"{caller} was given a panel with no rows, so there is nothing to analyse. "
            "This usually means an upstream filter (an attribute list, a panelist "
            "exclusion, a product subset) removed everything; check that filter rather "
            "than this call."
        )


def _band_edges(lo: float, hi: float, band: float) -> tuple[float, float]:
    """Return the upper edge of the floor band and the lower edge of the ceiling band."""
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(f"lo must be finite and strictly below hi; got lo={lo!r}, hi={hi!r}.")
    if not (0.0 <= band < 0.5):
        raise ValueError(f"band must be a fraction of the scale in [0, 0.5); got {band!r}.")
    width = (hi - lo) * band
    return lo + width, hi - width


def boundary_occupancy(
    panel: pd.DataFrame,
    lo: float = 0.0,
    hi: float = 10.0,
    band: float = 0.10,
) -> pd.DataFrame:
    """Measure how much of each attribute sits against the ends of the scale.

    An attribute pinned against a scale bound violates the Mixed Assessor
    Model's premise that assessors compress or expand a linear scale: no scaling
    difference is expressible in a region where everyone records the same value.
    Use this before modelling to decide whether an attribute can be treated as
    an intensity at all, and :func:`detection_rate` when it cannot.

    Floor, ceiling and exact-zero occupancy are reported separately because they
    are different questions. In particular, a panel whose convention is to
    record "not perceived" as a small positive number rather than an exact zero
    will look floor-pinned when it is not, and the ``exact_zero`` column is what
    distinguishes the two.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long-format panel data with ``panelist_id``, ``product``, ``attribute``
        and ``score`` columns.
    lo : float, default 0.0
        Lower bound of the rating scale.
    hi : float, default 10.0
        Upper bound of the rating scale.
    band : float, default 0.10
        Width of the floor and ceiling bands, as a fraction of the scale range.
        The default counts a score as "at the floor" when it is within the
        bottom 10% of the scale. Must lie in ``[0, 0.5)``.

    Returns
    -------
    pandas.DataFrame
        One row per attribute, sorted by attribute, with columns:

        ``attribute``
            The attribute name.
        ``n``
            Number of non-missing scores.
        ``at_floor``, ``at_ceiling``
            Counts of scores inside the floor and ceiling bands.
        ``exact_zero``
            Count of scores exactly equal to ``lo``.
        ``frac_floor``, ``frac_ceiling``, ``frac_exact_zero``
            The same three as fractions of ``n``, which is what a keep/drop
            decision is actually made on.

    Raises
    ------
    ValueError
        If a required column is missing, the panel has no rows, or the scale
        bounds and ``band`` are not a usable combination.

    Examples
    --------
    >>> occupancy = boundary_occupancy(validated.normalized_df)
    >>> occupancy.query("frac_floor > 0.5")  # candidates for detection_rate instead
    """
    _require_panel(panel, "boundary_occupancy")
    floor_edge, ceiling_edge = _band_edges(lo, hi, band)

    rows: list[dict[str, object]] = []
    for attribute, group in panel.groupby("attribute", observed=True, sort=True):
        scores = pd.to_numeric(group["score"], errors="coerce").dropna().to_numpy()
        n = int(scores.size)
        at_floor = int(np.sum(scores <= floor_edge))
        at_ceiling = int(np.sum(scores >= ceiling_edge))
        exact_zero = int(np.sum(scores == lo))
        denominator = float(n) if n else np.nan
        rows.append(
            {
                "attribute": str(attribute),
                "n": n,
                "at_floor": at_floor,
                "at_ceiling": at_ceiling,
                "exact_zero": exact_zero,
                "frac_floor": at_floor / denominator,
                "frac_ceiling": at_ceiling / denominator,
                "frac_exact_zero": exact_zero / denominator,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "attribute",
            "n",
            "at_floor",
            "at_ceiling",
            "exact_zero",
            "frac_floor",
            "frac_ceiling",
            "frac_exact_zero",
        ],
    )


def detection_rate(
    panel: pd.DataFrame,
    lo: float = 0.0,
    band: float = 0.10,
    hi: float = 10.0,
) -> pd.DataFrame:
    """Report, per product and attribute, the fraction of assessments that detected it.

    This is the appropriate response for an attribute that
    :func:`boundary_occupancy` shows is pinned against the floor: the question
    "how intense is it" has no answer when most assessors record nothing, but
    "how often is it perceived at all" does.

    .. warning::

       A detection rate is **not comparable with an intensity score**. It is a
       probability on ``[0, 1]``, it does not share the attribute's units, and
       it must not be dropped into the same table, correlation matrix or PLS
       block as intensity-scored attributes without saying what it is. Two
       attributes with the same mean intensity can have very different detection
       rates, and vice versa.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long-format panel data with ``panelist_id``, ``product``, ``attribute``
        and ``score`` columns.
    lo : float, default 0.0
        Lower bound of the rating scale.
    band : float, default 0.10
        Width of the floor band, as a fraction of the scale range. A score
        strictly above ``lo + band * (hi - lo)`` counts as detected.
    hi : float, default 10.0
        Upper bound of the rating scale.

    Returns
    -------
    pandas.DataFrame
        Products (rows) by attributes (columns) of detection probabilities. A
        product-attribute pair that nobody assessed is ``NaN`` rather than 0:
        "never detected" and "never asked" are different answers.

    Raises
    ------
    ValueError
        If a required column is missing, the panel has no rows, or the scale
        bounds and ``band`` are not a usable combination.

    Examples
    --------
    >>> rates = detection_rate(validated.normalized_df)
    >>> rates["burnt"].sort_values(ascending=False)
    """
    _require_panel(panel, "detection_rate")
    floor_edge, _ceiling_edge = _band_edges(lo, hi, band)

    scores = pd.to_numeric(panel["score"], errors="coerce")
    detected = (scores > floor_edge).where(scores.notna())
    working = pd.DataFrame(
        {
            "product": panel["product"].astype(str),
            "attribute": panel["attribute"].astype(str),
            "detected": detected.astype(float),
        }
    )
    table = working.pivot_table(
        index="product",
        columns="attribute",
        values="detected",
        aggfunc="mean",
        observed=True,
        dropna=False,
    )
    table.index.name = "product"
    table.columns.name = "attribute"
    return table


def assessor_variance_equality(panel: pd.DataFrame) -> pd.DataFrame:
    """Test whether assessors are equally variable, per attribute.

    The Mixed Assessor Model splits the assessor-by-product interaction into a
    scaling part and a disagreement part, and reads the scaling part as "this
    assessor uses a wider or narrower range of the scale". Grossmann et al.
    (2023) show that an assessor who is simply *noisier* than the others loads
    onto that same scaling term, which shifts the MAM F-test so that real
    disagreement is understated. This function tests the precondition directly,
    so a caller can tell which of the two they are looking at.

    Method: take residuals as score minus the product mean, within each
    attribute, which removes the genuine product effects that would otherwise
    dominate the spread. Then apply Levene's test (median-centred, i.e. the
    Brown-Forsythe variant, for robustness against non-normal residuals) across
    assessors.

    A small ``p_equal_variance`` means the assessors genuinely differ in spread,
    and the MAM scaling coefficients for that attribute are measuring partly
    that rather than scale use alone.

    Parameters
    ----------
    panel : pandas.DataFrame
        Long-format panel data with ``panelist_id``, ``product``, ``attribute``
        and ``score`` columns. Replicates are used as-is; more replicates give
        the test more to work with, but it runs on unreplicated data too by
        drawing the spread from across products.

    Returns
    -------
    pandas.DataFrame
        One row per attribute, sorted by attribute, with columns:

        ``attribute``
            The attribute name.
        ``levene_stat``
            Levene's test statistic. ``NaN`` when fewer than two assessors have
            enough observations to have a spread.
        ``p_equal_variance``
            The p-value for the null "all assessors have the same residual
            spread". Small means they do not.
        ``spread_ratio_max_min``
            The largest assessor's residual standard deviation divided by the
            smallest, an effect size to read alongside the p-value. ``inf``
            where some assessor has no residual spread at all.
        ``n_assessors``
            Number of assessors contributing to the attribute.

    Raises
    ------
    ValueError
        If a required column is missing or the panel has no rows.

    Examples
    --------
    >>> equality = assessor_variance_equality(validated.normalized_df)
    >>> equality.query("p_equal_variance < 0.05")["attribute"].tolist()
    """
    _require_panel(panel, "assessor_variance_equality")

    working = pd.DataFrame(
        {
            "panelist_id": panel["panelist_id"].astype(str),
            "product": panel["product"].astype(str),
            "attribute": panel["attribute"].astype(str),
            "score": pd.to_numeric(panel["score"], errors="coerce"),
        }
    ).dropna(subset=["score"])

    rows: list[dict[str, object]] = []
    for attribute, group in working.groupby("attribute", observed=True, sort=True):
        # Remove the product effect first: without this a large, genuine product
        # separation inflates every assessor's spread equally and Levene is being
        # asked about the wrong quantity.
        residuals = group["score"] - group.groupby("product", observed=True)["score"].transform("mean")
        by_assessor = [residual.to_numpy() for _pid, residual in residuals.groupby(group["panelist_id"], sort=True)]
        n_assessors = len(by_assessor)

        usable = [values for values in by_assessor if values.size >= _MIN_PER_ASSESSOR]
        spreads = np.array([float(np.std(values, ddof=1)) for values in usable])
        if len(usable) < 2:
            stat, p_value, ratio = np.nan, np.nan, np.nan
        else:
            smallest = float(spreads.min())
            ratio = float(spreads.max() / smallest) if smallest > 0 else float("inf")
            if np.allclose(spreads, spreads[0]):
                # Levene returns NaN on groups that are identical to floating
                # point; the honest answer there is "no evidence of inequality".
                stat, p_value = 0.0, 1.0
            else:
                result = levene(*usable, center="median")
                stat, p_value = float(result.statistic), float(result.pvalue)

        rows.append(
            {
                "attribute": str(attribute),
                "levene_stat": stat,
                "p_equal_variance": p_value,
                "spread_ratio_max_min": ratio,
                "n_assessors": n_assessors,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["attribute", "levene_stat", "p_equal_variance", "spread_ratio_max_min", "n_assessors"],
    )
