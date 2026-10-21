# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Yates's algorithm for a two-level factorial, and the algorithm run in reverse (#803 H1).

Yates's algorithm turns the responses of a full :math:`2^k` factorial, written in standard
order, into the contrasts of its effects, by *k* passes of one rule: the first half of each new
column holds the sums of successive pairs of the column before, and the second half their
differences, the second of each pair less the first. After *k* passes the first entry is the
total, and each other entry is the contrast of one effect, in standard order (A, B, AB, C, ...).
An effect is its contrast divided by :math:`N/2`, and its sum of squares is the contrast squared
divided by *N*.

Each pass doubles the sum of squares of its column, so the sums of squares are the algorithm's
check on the arithmetic. Run in reverse, on the contrasts of the effects a model keeps, the same
passes give *N* times the model's fitted values (Box, Hunter and Hunter 1978, Appendix 10A).

Examples
--------
>>> import pandas as pd
>>> from process_improve.experiments import yates_table
>>> runs = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1], "y": [10.0, 14.0, 12.0, 20.0]})
>>> table = yates_table(runs, "y")
>>> table[["(1)", "(2)", "effect", "estimate"]].values.tolist()
[[24.0, 56.0, 'mean', 14.0], [32.0, 12.0, 'A', 6.0], [4.0, 8.0, 'B', 4.0], [8.0, 4.0, 'A:B', 2.0]]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["reverse_yates", "yates_table"]

#: The columns ``yates_table`` adds beside the factors and the response.
_ADDED = ("effect", "estimate", "sum_sq")


def _passes(values: np.ndarray, k: int) -> list[np.ndarray]:
    """Return the *k* columns of Yates's algorithm from ``values``: each the sums, then the differences, of pairs."""
    columns, column = [], values
    for _ in range(k):
        pairs = column.reshape(-1, 2)
        column = np.concatenate([pairs.sum(axis=1), pairs[:, 1] - pairs[:, 0]])
        columns.append(column)
    return columns


def _standard_order(design: pd.DataFrame, factors: list[str]) -> np.ndarray:
    """Return the positions of the design's runs in standard order, the first factor changing fastest.

    Raises
    ------
    ValueError
        If a factor is not coded -1 and +1, or the runs are not each run of the full factorial once.
    """
    signs = design[factors].to_numpy(dtype=float)
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError(f"Yates's algorithm needs every factor coded -1 and +1; {factors} are not.")
    k, index = len(factors), ((signs > 0) * (2 ** np.arange(len(factors)))).sum(axis=1)
    if len(index) != 2**k or len(np.unique(index)) != 2**k:
        raise ValueError(
            f"Yates's algorithm needs each of the {2**k} runs of the full 2^{k} factorial in {factors} once; "
            f"the design has {len(index)} runs, {len(np.unique(index))} of them different. Average replicates "
            "first, and give a fraction's basic factors as ``factors``."
        )
    return np.argsort(index)


def _effect_names(factors: list[str]) -> list[str]:
    """Return the effects in standard order: the mean, then each product of factors, as ``"A:B"``."""
    return ["mean"] + [
        ":".join(name for j, name in enumerate(factors) if i >> j & 1) for i in range(1, 2 ** len(factors))
    ]


def yates_table(design: pd.DataFrame, response: str, factors: Sequence[str] | None = None) -> pd.DataFrame:
    """Return Yates's algorithm for a full two-level factorial: its columns, its effects and their sums of squares.

    Parameters
    ----------
    design : pd.DataFrame
        The runs of a full :math:`2^k` factorial, each factor coded -1 and +1, each run once, in any order.
    response : str
        The column that holds the response.
    factors : Sequence[str], optional
        The factors, the first the one that changes fastest in standard order. By default every column of
        ``design`` but the response, in their order.

    Returns
    -------
    pd.DataFrame
        One row per run in standard order: the factors and the response; the algorithm's columns ``(1)`` to
        ``(k)``; ``effect``, the effect the row's last column estimates (``"mean"``, then ``"A"``, ``"B"``,
        ``"A:B"``, ...); its ``estimate``, the last column divided by N/2 (by N for the mean); and its
        ``sum_sq``, the last column squared and divided by N (for the mean, the correction for the mean).
        ``attrs["sum_of_squares"]`` holds the sum of squares of the response and of each column, each twice the
        one before.

    Raises
    ------
    ValueError
        If the response is not a column, a factor is not coded -1 and +1, the runs are not each run of the full
        factorial once, a response is missing, or a name would collide with a column of the table.
    """
    if response not in design.columns:
        raise ValueError(f"The response {response!r} is not a column of the design.")
    names = [str(c) for c in design.columns if c != response] if factors is None else [str(f) for f in factors]
    if not names:
        raise ValueError("Yates's algorithm needs at least one factor.")
    k = len(names)
    added = {f"({j})" for j in range(1, k + 1)} | set(_ADDED)
    clashes = sorted(added & {*names, response})
    if clashes:
        raise ValueError(f"The names {clashes} are columns the table adds; rename them first.")
    runs = design.iloc[_standard_order(design, names)]
    y = runs[response].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Every run needs its response.")
    columns = _passes(y, k)
    n = 2**k
    table = pd.DataFrame({name: runs[name].to_numpy() for name in names})
    table[response] = y
    for j, column in enumerate(columns, start=1):
        table[f"({j})"] = column
    table["effect"] = _effect_names(names)
    table["estimate"] = columns[-1] / np.where(np.arange(n) == 0, n, n / 2)
    table["sum_sq"] = columns[-1] ** 2 / n
    table.attrs["sum_of_squares"] = {response: float(y @ y)} | {
        f"({j})": float(column @ column) for j, column in enumerate(columns, start=1)
    }
    return table


def reverse_yates(table: pd.DataFrame, keep: Sequence[str]) -> pd.DataFrame:
    """Return the fitted values of a model that keeps some of the effects, by Yates's algorithm run in reverse.

    The contrasts of the kept effects, with the others set to zero, are written in reverse standard order
    (the highest interaction first, the total last) and passed through the algorithm *k* times; the last
    column, read from the bottom up, is N times the fitted values in standard order.

    Parameters
    ----------
    table : pd.DataFrame
        A table from :func:`yates_table`.
    keep : Sequence[str]
        The effects the model keeps, by their names in ``table["effect"]``, such as ``["A", "B", "A:C"]``. The
        mean is kept whether it is named or not.

    Returns
    -------
    pd.DataFrame
        One row per run in standard order, as in ``table``: the factors and the response; ``effect``, and
        ``kept``, its contrast, or 0 for an effect the model drops; the reverse passes' columns ``(1)`` to
        ``(k)``, each read from the bottom up, as the algorithm writes it; ``fitted``, the last column divided
        by N; and ``residual``. ``attrs["sum_of_squares"]`` holds the sum of squares of ``kept`` and of each
        column, each twice the one before, then of ``fitted`` (the crude sum of squares of the fitted values)
        and of ``residual``.

    Raises
    ------
    ValueError
        If ``table`` is not a table from :func:`yates_table`, or ``keep`` names an effect it does not have.
    """
    n = len(table)
    k = n.bit_length() - 1
    needed = [f"({j})" for j in range(1, k + 1)]
    if n < 2 or n != 2**k or not {*needed, *_ADDED} <= set(table.columns):
        raise ValueError("Pass a table from yates_table.")
    effects = [str(e) for e in table["effect"]]
    unknown = sorted(set(keep) - set(effects))
    if unknown:
        raise ValueError(f"The table has no effect {unknown}; its effects are {effects[1:]}.")
    kept = np.where(np.isin(effects, ["mean", *keep]), table[f"({k})"].to_numpy(dtype=float), 0.0)
    columns = [column[::-1] for column in _passes(kept[::-1], k)]
    first = list(table.columns[: k + 1])  # the factors and the response
    out = table[first].copy()
    out["effect"] = effects
    out["kept"] = kept
    for j, column in enumerate(columns, start=1):
        out[f"({j})"] = column
    out["fitted"] = columns[-1] / n
    out["residual"] = table[first[-1]].to_numpy(dtype=float) - out["fitted"].to_numpy()
    squares = {"kept": kept} | {f"({j})": column for j, column in enumerate(columns, start=1)}
    squares |= {"fitted": out["fitted"].to_numpy(), "residual": out["residual"].to_numpy()}
    out.attrs["sum_of_squares"] = {name: float(values @ values) for name, values in squares.items()}
    return out
