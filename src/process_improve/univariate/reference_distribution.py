# (c) Kevin Dunn, 2010-2026. MIT License.
r"""A reference distribution from a record of past results, to judge a new difference of averages.

A plant trial compares the average of :math:`n` runs by one method with the average of the next
:math:`n` by another. Whether the difference is unusual can be judged against the same process's
own history, without assuming that the observations are independent or normally distributed
(Box, Hunter and Hunter, chapter 2):

- **All differences.** Every difference of two adjacent averages of :math:`n` consecutive past
  observations (the later less the earlier) forms the reference set. The significance level of
  an observed difference is the share of the set at least as extreme. A member equal to it counts
  half, as Box, Hunter and Hunter count a tie.
- **Nearly independent differences.** Adjacent members of that set share observations, so they
  are correlated. Taking only the differences of groups of :math:`2n` a gap of one observation
  apart gives :math:`m` nearly independent differences. Their root mean square about zero,
  :math:`s = \sqrt{\sum d^2/m}`, scales the observed difference as a t statistic on :math:`m`
  degrees of freedom.

Examples
--------
Box, Hunter and Hunter's 210 past batch yields, against the plant trial's difference of 1.30
between averages of 10:

>>> import numpy as np
>>> from process_improve.experiments.reference_catalogue import reference_experiments
>>> [record] = [e for e in reference_experiments(source="box-hunter-hunter-1978")
...             if e["name"] == "bhh-production-record-210"]
>>> history = [row[1] for row in record["runs"]["rows"]]
>>> found = reference_distribution(history, 10, observed=1.30)
>>> found["n_differences"], found["n_beyond"], round(found["significance"], 3)
(191, 9, 0.047)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

#: How the observed difference is judged: ``"greater"`` when the later average is expected to be
#: larger, ``"less"`` when it is expected to be smaller.
ALTERNATIVES = ("greater", "less")

#: Differences closer than this share of the record's spread to the observed one are ties.
TIE_TOLERANCE = 1e-9


def reference_distribution(
    history: Sequence[float],
    window: int,
    observed: float | None = None,
    alternative: str = "greater",
    gap: int = 1,
) -> dict[str, Any]:
    r"""Build the reference distribution of differences of adjacent averages, and judge a difference by it.

    Parameters
    ----------
    history : Sequence[float]
        Past observations of the process, in time order.
    window : int
        :math:`n`, the number of observations in each average: as many as each method had in the
        trial.
    observed : float or None
        The trial's difference, the later average less the earlier; None gives the reference
        distribution alone.
    alternative : str
        ``"greater"`` (the later method is expected to give larger values) or ``"less"``.
    gap : int
        The number of past observations skipped between the groups of :math:`2n` that give the
        nearly independent differences.

    Returns
    -------
    dict[str, Any]
        - ``differences``: every difference of adjacent averages, in time order, and
          ``n_differences``, their number;
        - ``nearly_independent``: the differences of groups of :math:`2n` a ``gap`` apart, ``s``
          their root mean square about zero, and ``df`` their number;
        - with ``observed``: ``n_beyond`` and ``n_tied``, the members beyond it and equal to it,
          ``significance`` (``n_beyond`` plus half ``n_tied``, over ``n_differences``), ``t``
          (the observed difference over ``s``) and ``t_significance``, the share of a t
          distribution on ``df`` degrees of freedom beyond ``t``.

    Raises
    ------
    ValueError
        If the history is not finite numbers, ``window`` or ``gap`` is not a whole number of at
        least 1 (0 or more for ``gap``), the history is too short for one difference, or
        ``alternative`` is not one of :data:`ALTERNATIVES`.
    """
    record = np.asarray(history, dtype=float)
    _check(record, window, alternative, gap)
    averages = np.lib.stride_tricks.sliding_window_view(record, window).mean(axis=1)
    differences = averages[window:] - averages[:-window]
    starts = range(0, len(record) - 2 * window + 1, 2 * window + gap)
    independent = np.array([differences[start] for start in starts])
    s = float(np.sqrt(independent @ independent / len(independent)))
    found: dict[str, Any] = {
        "differences": differences.tolist(),
        "n_differences": len(differences),
        "nearly_independent": independent.tolist(),
        "s": s,
        "df": len(independent),
    }
    if observed is None:
        return found
    sign = 1.0 if alternative == "greater" else -1.0
    tolerance = TIE_TOLERANCE * max(float(np.ptp(record)), abs(observed), 1.0)
    tied = np.abs(differences - observed) <= tolerance
    beyond = (sign * (differences - observed) > 0) & ~tied
    t_statistic = observed / s if s > 0 else np.copysign(np.inf, observed)
    return found | {
        "n_beyond": int(beyond.sum()),
        "n_tied": int(tied.sum()),
        "significance": (int(beyond.sum()) + int(tied.sum()) / 2) / len(differences),
        "t": float(t_statistic),
        "t_significance": float(stats.t.sf(sign * t_statistic, len(independent))),
    }


def _check(record: np.ndarray, window: int, alternative: str, gap: int) -> None:
    """Refuse a history, window, alternative or gap the reference distribution cannot use."""
    if record.ndim != 1 or not np.all(np.isfinite(record)):
        msg = "history must be a sequence of finite numbers, in time order."
        raise ValueError(msg)
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        msg = f"window must be a whole number of at least 1; got {window!r}."
        raise ValueError(msg)
    if isinstance(gap, bool) or not isinstance(gap, (int, np.integer)) or gap < 0:
        msg = f"gap must be a whole number, 0 or more; got {gap!r}."
        raise ValueError(msg)
    if len(record) < 2 * window:
        msg = f"A history of {len(record)} observations is too short for one difference of averages of {window}."
        raise ValueError(msg)
    if alternative not in ALTERNATIVES:
        msg = f"alternative must be one of {ALTERNATIVES}; got {alternative!r}."
        raise ValueError(msg)
