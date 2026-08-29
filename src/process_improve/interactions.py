"""(c) Kevin Dunn, 2010-2026. MIT License.

Two-factor interaction terms, and whether the data can support them.

.. warning::

   **Provisional. These three functions are unvalidated on real data.** They
   exist to test conditions that a small observational study rarely meets: a
   pair of predictors whose observations populate all four corners of their
   plane, and a selection procedure stable enough for its choices to mean
   something. Because those conditions are rarely met, this code may go
   unreached for a long time after shipping, and its coverage is unit tests
   only. Nothing here has been run against a real product-by-compound block
   with a real sensory response. Treat the API as subject to change, and read
   any result from it as a hypothesis rather than a finding.

An interaction between two chemical predictors is a real thing to look for: a
compound whose perceptual effect depends on the level of another is exactly what
a linear additive model misses. But an interaction term is only estimable when
the data actually separates the four combinations of high and low, and in an
observational set of thirty products it usually does not.
:func:`pair_coverage` says so before a model is fitted rather than after.

The ordering inside :func:`interaction_terms` is not negotiable: transform,
centre and scale, multiply, then **re-centre and re-scale the products**. The
product of two standardised variables is not itself centred. For approximately
bivariate normal columns its mean is the parents' correlation :math:`r` and its
variance is :math:`1 + r^2`, so skipping the second centring leaks correlation
into the intercept and systematically inflates the columns belonging to
correlated pairs, exactly the pairs whose interactions are least trustworthy.

References
----------
Meinshausen and Buhlmann, "Stability selection", Journal of the Royal
Statistical Society: Series B, 72(4), 417-473, 2010,
doi:10.1111/j.1467-9868.2010.00740.x.

Shah and Samworth, "Variable selection with error control: another look at
stability selection", Journal of the Royal Statistical Society: Series B, 75(1),
55-80, 2013, doi:10.1111/j.1467-9868.2011.01034.x.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._preprocessing import _looks_prescaled

__all__ = ["interaction_terms", "pair_coverage", "stability_selection"]

#: Separator used to name a product term from its two parents.
_TERM_JOINER = "_x_"

#: A divisor at or below this is treated as no spread at all, mirroring
#: :class:`~process_improve.multivariate.MCUVScaler`.
_TINY = float(np.finfo(float).tiny) ** 0.5


def pair_coverage(
    x_a: np.ndarray,
    x_b: np.ndarray,
    min_per_corner: int = 4,
) -> tuple[bool, dict]:
    """Ask whether the observations populate all four corners of the ``(A, B)`` plane.

    An interaction term claims that the effect of A depends on the level of B.
    Estimating that claim needs products where A is high and B is low, and
    products where the reverse holds, as well as the two agreeing corners. With
    one corner empty the term is fitted from three points of support and will
    report whatever the noise there suggests.

    Two variables that co-vary populate only the agreeing corners and will fail
    this check. That is the correct answer, not a defect to work around: their
    interaction is not identifiable from these observations, and no amount of
    regularisation makes it so.

    Parameters
    ----------
    x_a, x_b : numpy.ndarray
        The two predictors, one value per product, the same length. Rows missing
        either value are dropped.
    min_per_corner : int, default 4
        How many observations a corner needs before it counts as populated.

    Returns
    -------
    (covered, detail) : tuple of (bool, dict)
        ``covered`` is True when every corner holds at least
        ``min_per_corner`` observations. ``detail`` carries ``low_low``,
        ``low_high``, ``high_low``, ``high_high`` (the corner counts, with the
        first word describing A), ``n`` (rows used), ``min_per_corner``,
        ``threshold_a`` and ``threshold_b`` (the median splits), and
        ``correlation`` (Pearson's r between the two, which is usually the
        explanation when the answer is False).

    Raises
    ------
    ValueError
        If the two inputs have different lengths, or ``min_per_corner`` is
        below 1.

    Notes
    -----
    Each variable is split at its own median, so the marginal split is balanced
    by construction and only the *joint* distribution can fail. A value exactly
    at the median counts as low.

    Examples
    --------
    >>> covered, detail = pair_coverage(x["linalool"].to_numpy(), x["geraniol"].to_numpy())
    >>> covered, detail["correlation"]
    (False, 0.91)
    """
    a = np.asarray(x_a, dtype=float).ravel()
    b = np.asarray(x_b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError(f"x_a and x_b must have the same length; got {a.size} and {b.size}.")
    if int(min_per_corner) < 1:
        raise ValueError(f"min_per_corner must be >= 1; got {min_per_corner!r}.")

    usable = np.isfinite(a) & np.isfinite(b)
    a, b = a[usable], b[usable]
    n = int(a.size)
    if n == 0:
        detail = {
            "low_low": 0,
            "low_high": 0,
            "high_low": 0,
            "high_high": 0,
            "n": 0,
            "min_per_corner": int(min_per_corner),
            "threshold_a": float("nan"),
            "threshold_b": float("nan"),
            "correlation": float("nan"),
        }
        return False, detail

    threshold_a = float(np.median(a))
    threshold_b = float(np.median(b))
    a_high = a > threshold_a
    b_high = b > threshold_b
    counts = {
        "low_low": int(np.sum(~a_high & ~b_high)),
        "low_high": int(np.sum(~a_high & b_high)),
        "high_low": int(np.sum(a_high & ~b_high)),
        "high_high": int(np.sum(a_high & b_high)),
    }
    correlation = float(np.corrcoef(a, b)[0, 1]) if n > 1 and a.std() > 0 and b.std() > 0 else float("nan")
    detail = {
        **counts,
        "n": n,
        "min_per_corner": int(min_per_corner),
        "threshold_a": threshold_a,
        "threshold_b": threshold_b,
        "correlation": correlation,
    }
    return all(count >= int(min_per_corner) for count in counts.values()), detail


def interaction_terms(
    x_log: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build centred, scaled product terms from centred, scaled predictors.

    The order matters and is not negotiable: the parents must already be
    transformed, centred and scaled; this function multiplies them and then
    **centres and scales the products again**. A product of two standardised
    columns has mean :math:`r` and variance :math:`1 + r^2` (for approximately
    bivariate normal parents), so a term skipping the second pass carries its
    parents' correlation into the intercept and arrives at the model with more
    variance than a genuine predictor would, inflating exactly the pairs whose
    interactions deserve the least trust.

    Check :func:`pair_coverage` first. This function will happily build a term
    for a pair that supports nothing.

    Parameters
    ----------
    x_log : pandas.DataFrame
        Predictors, already transformed, centred and scaled (see
        :mod:`process_improve.chemistry`). A :class:`SpecificationWarning` is
        raised when they do not look centred and unit-variance, because the
        reasoning above, and the ``parent_correlation`` column below, assume it.
    pairs : sequence of (str, str)
        The pairs to build. Both names must be columns of ``x_log``. A pair
        naming the same column twice gives a quadratic term, whose mean before
        re-centring is 1 rather than :math:`r`; that is allowed, but it is not
        an interaction.

    Returns
    -------
    (terms, constants) : tuple of pandas.DataFrame
        ``terms`` has one column per pair, named ``"a_x_b"``, on the rows of
        ``x_log``, centred and unit-variance scaled. ``constants`` has one row
        per term with ``term``, ``left``, ``right``, ``center``, ``divisor``
        and ``parent_correlation``. ``divisor`` is divided by, not multiplied
        by, and the pair lets a held-out block be built with training constants:
        multiply the same parents, then subtract ``center`` and divide by
        ``divisor``.

    Raises
    ------
    ValueError
        If ``pairs`` is empty, names a column that is not in ``x_log``, repeats
        a pair, or would produce a term name that collides with an existing
        column.

    Examples
    --------
    >>> terms, constants = interaction_terms(x_scaled, [("linalool", "geraniol")])
    >>> terms.mean().abs().max() < 1e-12
    True
    """
    if not isinstance(x_log, pd.DataFrame):
        raise TypeError("x_log must be a pandas DataFrame of products (rows) by predictors (columns).")
    pair_list = [(str(left), str(right)) for left, right in pairs]
    if not pair_list:
        raise ValueError("pairs is empty; there is nothing to build.")

    known = {str(column) for column in x_log.columns}
    strangers = sorted({name for pair in pair_list for name in pair if name not in known})
    if strangers:
        raise ValueError(f"pairs names column(s) {strangers} that are not in x_log.")
    if len(set(pair_list)) != len(pair_list):
        repeated = sorted({pair for pair in pair_list if pair_list.count(pair) > 1})
        raise ValueError(f"pairs repeats {repeated}, which would produce duplicate term columns.")

    names = [f"{left}{_TERM_JOINER}{right}" for left, right in pair_list]
    collisions = sorted(set(names) & known)
    if collisions:
        raise ValueError(
            f"the term name(s) {collisions} collide with existing columns of x_log. Rename the "
            f"parent column, or the joiner {_TERM_JOINER!r} will produce an ambiguous block."
        )

    if not _looks_prescaled(x_log):
        warnings.warn(
            "x_log does not look centred and unit-variance scaled. interaction_terms assumes it "
            "is: a product term's properties (mean equal to the parents' correlation, variance "
            "1 + r squared) and the parent_correlation column below both depend on it, and a "
            "product of un-standardised columns is dominated by whichever parent has the larger "
            "units. Transform, centre and scale the parents first.",
            SpecificationWarning,
            stacklevel=2,
        )

    values = x_log.astype(float)
    columns: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for (left, right), name in zip(pair_list, names, strict=True):
        parent_left = values[left].to_numpy(dtype=float)
        parent_right = values[right].to_numpy(dtype=float)
        product = parent_left * parent_right

        usable = np.isfinite(product)
        centre = float(np.mean(product[usable])) if usable.any() else 0.0
        spread = float(np.std(product[usable], ddof=1)) if usable.sum() > 1 else 0.0
        divisor = 1.0 if (not np.isfinite(spread) or spread <= _TINY) else spread

        columns[name] = (product - centre) / divisor
        both = np.isfinite(parent_left) & np.isfinite(parent_right)
        correlation = (
            float(np.corrcoef(parent_left[both], parent_right[both])[0, 1])
            if both.sum() > 1 and parent_left[both].std() > 0 and parent_right[both].std() > 0
            else float("nan")
        )
        rows.append(
            {
                "term": name,
                "left": left,
                "right": right,
                "center": centre,
                "divisor": divisor,
                "parent_correlation": correlation,
            }
        )

    terms = pd.DataFrame(columns, index=x_log.index)
    constants = pd.DataFrame(rows, columns=["term", "left", "right", "center", "divisor", "parent_correlation"])
    return terms, constants


def stability_selection(
    select: Callable,
    x: pd.DataFrame,
    y: pd.DataFrame,
    n_iter: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Report how often each predictor is selected across complementary half-samples.

    A selection made once on all the data is a selection made once. Repeating it
    on random halves, and reporting how often each name comes back, separates
    the choices the data supports from the ones that depended on which products
    happened to be in the set. The halves are complementary: each split is used
    in both directions, so every product appears in exactly half of the
    subsamples and the two runs of a split share no rows.

    Parameters
    ----------
    select : callable
        ``select(x, y)`` returning the names it selected, as any iterable of
        labels. Called ``2 * n_iter`` times, so keep it cheap.
    x : pandas.DataFrame
        Predictor block, one row per product. Its columns are the universe: a
        name the callable returns that is not a column here is an error.
    y : pandas.DataFrame
        Response block, one row per product. Sub-sampled with ``x``, row for
        row.
    n_iter : int, default 100
        Number of complementary splits, so ``2 * n_iter`` calls to ``select``.
    seed : int, default 0
        Seed for the splits, so a reported frequency can be reproduced.

    Returns
    -------
    pandas.DataFrame
        One row per column of ``x``, sorted by frequency then name, with
        ``name``, ``n_selected`` (subsamples that chose it), ``n_subsamples``
        (always ``2 * n_iter``) and ``selection_frequency`` (the ratio).

    Raises
    ------
    ValueError
        If the blocks disagree on rows, ``n_iter`` is below 1, there are fewer
        than four products to split, or ``select`` returns a name that is not a
        column of ``x``.

    Examples
    --------
    >>> frequencies = stability_selection(select, x_scaled, sensory_means, n_iter=50)
    >>> frequencies.query("selection_frequency > 0.6")["name"].tolist()
    """
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("x and y must both be pandas DataFrames, one row per product.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same number of rows; got {len(x)} and {len(y)}.")
    if int(n_iter) < 1:
        raise ValueError(f"n_iter must be >= 1; got {n_iter!r}.")
    if len(x) < 4:
        raise ValueError(
            f"a complementary half-sample split needs at least 4 products, so each half has 2; got {len(x)}."
        )

    universe = [str(column) for column in x.columns]
    tally = dict.fromkeys(universe, 0)

    def _record(chosen: Iterable) -> None:
        names = [str(name) for name in chosen]
        strangers = sorted(set(names) - set(universe))
        if strangers:
            raise ValueError(
                f"the select callable returned name(s) {strangers} that are not columns of x. "
                "The frequencies are over x's columns, so a name outside them cannot be counted."
            )
        for name in set(names):
            tally[name] += 1

    rng = np.random.default_rng(seed)
    half = len(x) // 2
    for _ in range(int(n_iter)):
        order = rng.permutation(len(x))
        for rows in (order[:half], order[half : 2 * half]):
            _record(select(x.iloc[rows], y.iloc[rows]))

    n_subsamples = 2 * int(n_iter)
    table = pd.DataFrame(
        {
            "name": universe,
            "n_selected": [tally[name] for name in universe],
            "n_subsamples": n_subsamples,
            "selection_frequency": [tally[name] / n_subsamples for name in universe],
        }
    )
    return table.sort_values(["selection_frequency", "name"], ascending=[False, True]).reset_index(drop=True)
