# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Balanced incomplete block designs: the intrablock, interblock and combined estimates of the treatments.

In a balanced incomplete block design (BIBD) each of :math:`b` blocks holds :math:`k` of the
:math:`a` treatments, each treatment appears :math:`r` times and each pair of treatments shares
:math:`\lambda` blocks. A treatment total then carries the effects of the blocks it fell in, so the
treatments are compared within blocks (Montgomery, section 4.4; Yates 1940):

- the adjusted treatment total :math:`Q_i = y_{i.} - \tfrac{1}{k}\sum_j n_{ij}\,y_{.j}` removes them,
  with :math:`n_{ij}` 1 when treatment :math:`i` is in block :math:`j`;
- the intrablock estimate of the treatment effect is :math:`\hat\tau_i = k Q_i/(\lambda a)`, and the
  adjusted treatment sum of squares :math:`k\sum_i Q_i^2/(\lambda a)`.

When the blocks are a random sample, the block totals also carry information about the
treatments, Yates's *recovery of interblock information*:

- the interblock estimate :math:`\tilde\tau_i = [\sum_j n_{ij}\,y_{.j} - k r \bar y_{..}]/(r - \lambda)`;
- the combined estimate weights the two by their variances,

  .. math::

      \tau_i^* = \frac{k Q_i (\sigma^2 + k\sigma_\beta^2) + \left(\sum_j n_{ij}\,y_{.j}
                 - k r \bar y_{..}\right)\sigma^2}{(r - \lambda)\sigma^2 + \lambda a (\sigma^2 + k\sigma_\beta^2)},

  with :math:`\sigma^2` the intrablock mean square for error and
  :math:`\hat\sigma_\beta^2 = [MS_\text{blocks (adjusted)} - MS_E](b - 1)/[a(r - 1)]`, taken as 0
  when it comes out negative.

Examples
--------
Montgomery's Example 4.5, four catalysts in four batches of three:

>>> import pandas as pd
>>> runs = pd.DataFrame({
...     "catalyst": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
...     "batch": [1, 2, 4, 2, 3, 4, 1, 2, 3, 1, 3, 4],
...     "time": [73, 74, 71, 75, 67, 72, 73, 75, 68, 75, 72, 75],
... })
>>> found = bibd_analysis(runs, "catalyst", "batch", "time")
>>> found["lambda"], {t: round(v, 2) for t, v in found["interblock_effects"].items()}
(2, {'1': 10.5, '2': -3.5, '3': -0.5, '4': -6.5})
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


def bibd_analysis(data: pd.DataFrame, treatment: str, block: str, response: str) -> dict[str, Any]:
    r"""Analyse a balanced incomplete block design: intrablock, interblock and combined treatment estimates.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per run.
    treatment : str
        The column of the treatment labels.
    block : str
        The column of the block labels.
    response : str
        The column of the response.

    Returns
    -------
    dict[str, Any]
        The design's ``a``, ``b``, ``k``, ``r``, ``lambda`` and ``n``; by treatment label (as a
        string): ``treatment_totals`` :math:`y_{i.}`, ``adjusted_treatment_totals`` :math:`Q_i`,
        ``intrablock_effects`` :math:`\hat\tau_i`, ``totals_of_blocks_with`` :math:`\sum_j n_{ij}
        y_{.j}`, ``interblock_effects`` :math:`\tilde\tau_i` and ``combined_effects``
        :math:`\tau_i^*`; by block label: ``block_totals`` :math:`y_{.j}` and
        ``adjusted_block_totals`` :math:`Q'_j = y_{.j} - \tfrac{1}{r}\sum_i n_{ij}\,y_{i.}`;
        ``sums_of_squares`` (``total``, ``treatments_unadjusted``, ``treatments_adjusted``,
        ``blocks_unadjusted``, ``blocks_adjusted``, ``error``), ``df_error``, ``ms_error`` and
        ``sigma2_blocks`` (:math:`\hat\sigma_\beta^2`).

    Raises
    ------
    ValueError
        If a column is missing, a response is not a finite number, a treatment appears twice in a
        block, or the design is not a balanced incomplete block design: blocks of different sizes,
        a block holding every treatment, treatments run a different number of times, or pairs of
        treatments sharing different numbers of blocks.
    """
    y, incidence, treatments, blocks = _layout(data, treatment, block, response)
    a, b = incidence.shape
    k, r, lam = _balance(incidence, treatments)
    n = int(incidence.sum())
    treatment_totals, block_totals = y.sum(axis=1), y.sum(axis=0)
    grand = float(block_totals.sum())
    with_blocks = incidence @ block_totals  # sum over j of n_ij y_.j
    adjusted = treatment_totals - with_blocks / k
    adjusted_blocks = block_totals - incidence.T @ treatment_totals / r

    observed = y[incidence == 1]
    total = float(observed @ observed - grand**2 / n)
    treatments_adjusted = float(k * adjusted @ adjusted / (lam * a))
    treatments_unadjusted = float(treatment_totals @ treatment_totals / r - grand**2 / n)
    blocks_unadjusted = float(block_totals @ block_totals / k - grand**2 / n)
    error = total - treatments_adjusted - blocks_unadjusted
    blocks_adjusted = treatments_adjusted + blocks_unadjusted - treatments_unadjusted
    df_error = n - a - b + 1
    ms_error = error / df_error
    sigma2_blocks = max(0.0, (blocks_adjusted / (b - 1) - ms_error) * (b - 1) / (a * (r - 1)))

    between = with_blocks - k * r * grand / n
    within_weight = ms_error + k * sigma2_blocks
    combined = (k * adjusted * within_weight + between * ms_error) / ((r - lam) * ms_error + lam * a * within_weight)

    def by_treatment(values: np.ndarray) -> dict[str, float]:
        return dict(zip(treatments, values.tolist(), strict=True))

    return {
        "a": a,
        "b": b,
        "k": k,
        "r": r,
        "lambda": lam,
        "n": n,
        "treatment_totals": by_treatment(treatment_totals),
        "adjusted_treatment_totals": by_treatment(adjusted),
        "intrablock_effects": by_treatment(k * adjusted / (lam * a)),
        "totals_of_blocks_with": by_treatment(with_blocks),
        "interblock_effects": by_treatment(between / (r - lam)),
        "combined_effects": by_treatment(combined),
        "block_totals": dict(zip(blocks, block_totals.tolist(), strict=True)),
        "adjusted_block_totals": dict(zip(blocks, adjusted_blocks.tolist(), strict=True)),
        "sums_of_squares": {
            "total": total,
            "treatments_unadjusted": treatments_unadjusted,
            "treatments_adjusted": treatments_adjusted,
            "blocks_unadjusted": blocks_unadjusted,
            "blocks_adjusted": blocks_adjusted,
            "error": error,
        },
        "df_error": df_error,
        "ms_error": ms_error,
        "sigma2_blocks": sigma2_blocks,
    }


def _levels(column: pd.Series) -> list[str]:
    """Return a column's labels as strings, in natural order: by length, then by value."""
    return sorted(column.astype(str).unique(), key=lambda label: (len(label), label))


def _layout(
    data: pd.DataFrame, treatment: str, block: str, response: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return the responses as a treatment-by-block table (0 where not run), the incidence, and the labels."""
    missing = [name for name in (treatment, block, response) if name not in data.columns]
    if missing:
        msg = f"data has no column {missing}; its columns are {list(data.columns)}."
        raise ValueError(msg)
    values = pd.to_numeric(data[response], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        msg = f"Each value of {response!r} must be a finite number."
        raise ValueError(msg)
    treatments, blocks = _levels(data[treatment]), _levels(data[block])
    rows = data[treatment].astype(str).map({label: i for i, label in enumerate(treatments)}).to_numpy()
    columns = data[block].astype(str).map({label: j for j, label in enumerate(blocks)}).to_numpy()
    incidence = np.zeros((len(treatments), len(blocks)))
    np.add.at(incidence, (rows, columns), 1)
    if incidence.max() > 1:
        msg = "A treatment appears more than once in a block; a balanced incomplete block design runs it once."
        raise ValueError(msg)
    y = np.zeros_like(incidence)
    y[rows, columns] = values
    return y, incidence, treatments, blocks


def _balance(incidence: np.ndarray, treatments: list[str]) -> tuple[int, int, int]:
    """Return k, r and lambda, refusing a layout that is not a balanced incomplete block design."""
    a = incidence.shape[0]
    sizes, replicates = incidence.sum(axis=0), incidence.sum(axis=1)
    if len(set(sizes)) != 1 or sizes[0] >= a or len(set(replicates)) != 1:
        msg = (
            "This is not a balanced incomplete block design: the blocks must hold the same number of "
            f"treatments, fewer than all {a}, and each treatment must be run equally often; the block "
            f"sizes are {sorted(set(sizes.astype(int).tolist()))} and the replicates "
            f"{sorted(set(replicates.astype(int).tolist()))}."
        )
        raise ValueError(msg)
    concurrences = incidence @ incidence.T
    pairs = {(treatments[i], treatments[j]): int(concurrences[i, j]) for i, j in itertools.combinations(range(a), 2)}
    if len(set(pairs.values())) != 1:
        msg = (
            "This is not a balanced incomplete block design: pairs of treatments share different numbers of "
            f"blocks, {pairs}."
        )
        raise ValueError(msg)
    return int(sizes[0]), int(replicates[0]), next(iter(pairs.values()))
