# (c) Kevin Dunn, 2010-2026. MIT License.
r"""Yates's algorithm for a two-level factorial, the algorithm run in reverse, and its form for a three-level one.

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

For a full :math:`3^k` factorial the passes take successive triples :math:`(w_1, w_2, w_3)`: the
first third of each new column holds their sums, the second third their linear contrasts
:math:`w_3 - w_1`, and the last third their quadratic contrasts :math:`w_1 - 2 w_2 + w_3` (Box and
Draper 2007, Appendix 7G). After *k* passes each entry is :math:`z'y` for one product of the
factors' orthogonal polynomials, :math:`x` and :math:`3x^2 - 2`, so the :math:`3^k - 1` effects
each have one degree of freedom.

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
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["reverse_yates", "yates_table", "yates_table_3k"]

#: The columns ``yates_table`` adds beside the factors and the response.
_ADDED = ("effect", "estimate", "sum_sq")

#: The columns ``yates_table_3k`` adds beside the factors, the response and the algorithm's columns.
_ADDED_3K = ("effect", "divisor", "estimate", "sum_sq", "standardized", "normal_score")

#: Each pass's contrasts of successive pairs (two levels) or triples (three levels), one row per third of the new
#: column: the sums, then the differences, or the linear contrasts w3 - w1 and the quadratic w1 - 2 w2 + w3.
_CONTRASTS = {2: np.array([[1, 1], [-1, 1]]), 3: np.array([[1, 1, 1], [-1, 0, 1], [1, -2, 1]])}

#: A factor's coded settings, lowest first, for a factorial of two and of three levels.
_CODED = {2: (-1.0, 1.0), 3: (-1.0, 0.0, 1.0)}


def _passes(values: np.ndarray, k: int, levels: int = 2) -> list[np.ndarray]:
    """Return the *k* columns of Yates's algorithm from ``values``: each the contrasts of its pairs or triples."""
    columns, column = [], values
    for _ in range(k):
        column = (_CONTRASTS[levels] @ column.reshape(-1, levels).T).ravel()
        columns.append(column)
    return columns


def _digits(i: int, k: int, levels: int) -> list[int]:
    """Return the base-``levels`` digits of ``i``, the first factor's first: a run's levels, or an effect's powers."""
    return [i // levels**j % levels for j in range(k)]


def _standard_order(design: pd.DataFrame, factors: list[str], levels: int = 2) -> np.ndarray:
    """Return the positions of the design's runs in standard order, the first factor changing fastest.

    Raises
    ------
    ValueError
        If a factor is not coded -1 and +1 (two levels) or -1, 0 and +1 (three), or the runs are not each run of the
        full factorial once.
    """
    settings, coded = design[factors].to_numpy(dtype=float), _CODED[levels]
    if not np.isin(settings, coded).all():
        written = "-1 and +1" if levels == 2 else "-1, 0 and +1"
        raise ValueError(f"Yates's algorithm needs every factor coded {written}; {factors} are not.")
    k, n = len(factors), levels ** len(factors)
    index = (np.searchsorted(coded, settings) * levels ** np.arange(k)).sum(axis=1)
    if len(index) != n or len(np.unique(index)) != n:
        raise ValueError(
            f"Yates's algorithm needs each of the {n} runs of the full {levels}^{k} factorial in {factors} once; "
            f"the design has {len(index)} runs, {len(np.unique(index))} of them different. Average replicates "
            "first, and give a fraction's basic factors as ``factors``."
        )
    return np.argsort(index)


def _effect_names(factors: list[str], levels: int = 2) -> list[str]:
    """Return the effects in standard order: the mean, then each product of factors, as ``"A:B"`` or ``"A^2:B"``."""

    def name(i: int) -> str:
        powers = _digits(i, len(factors), levels)
        return ":".join(f if p == 1 else f"{f}^2" for f, p in zip(factors, powers, strict=True) if p)

    return ["mean"] + [name(i) for i in range(1, levels ** len(factors))]


def _factor_names(design: pd.DataFrame, response: str, factors: Sequence[str] | None, added: set[str]) -> list[str]:
    """Return the factors' names, after checking the response is a column and no name is one the table adds.

    Raises
    ------
    ValueError
        If the response is not a column, there is no factor, the response is named as a factor, or a name would
        collide with a column of the table.
    """
    if response not in design.columns:
        raise ValueError(f"The response {response!r} is not a column of the design.")
    names = [str(c) for c in design.columns if c != response] if factors is None else [str(f) for f in factors]
    if not names:
        raise ValueError("Yates's algorithm needs at least one factor.")
    if response in names:
        raise ValueError(f"The response {response!r} cannot also be a factor.")
    clashes = sorted((added | {f"({j})" for j in range(1, len(names) + 1)}) & {*names, response})
    if clashes:
        raise ValueError(f"The names {clashes} are columns the table adds; rename them first.")
    return names


def _response(runs: pd.DataFrame, response: str) -> np.ndarray:
    """Return the response of the runs, which every run must have."""
    y = runs[response].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Every run needs its response.")
    return y


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
        factorial once, a response is missing or named as a factor, or a name would collide with a column of the
        table.

    Examples
    --------
    A 2^2 factorial, its runs in any order. The table puts them in standard order, and each pass of
    the algorithm doubles the sum of squares of the responses, 840.

    >>> runs = pd.DataFrame({"A": [1, -1, 1, -1], "B": [1, 1, -1, -1], "y": [20.0, 12.0, 14.0, 10.0]})
    >>> table = yates_table(runs, "y")
    >>> table["y"].tolist()
    [10.0, 14.0, 12.0, 20.0]
    >>> dict(zip(table["effect"], table["sum_sq"].tolist(), strict=True))
    {'mean': 784.0, 'A': 36.0, 'B': 16.0, 'A:B': 4.0}
    >>> table.attrs["sum_of_squares"]
    {'y': 840.0, '(1)': 1680.0, '(2)': 3360.0}
    """
    names = _factor_names(design, response, factors, set(_ADDED))
    k = len(names)
    runs = design.iloc[_standard_order(design, names)]
    y = _response(runs, response)
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

    Examples
    --------
    The 2^2 factorial of :func:`yates_table`, fitted with A alone: B and AB go into the residuals.

    >>> runs = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1], "y": [10.0, 14.0, 12.0, 20.0]})
    >>> fit = reverse_yates(yates_table(runs, "y"), keep=["A"])
    >>> fit[["effect", "kept", "fitted", "residual"]].values.tolist()
    [['mean', 56.0, 11.0, -1.0], ['A', 12.0, 17.0, -3.0], ['B', 0.0, 11.0, 1.0], ['A:B', 0.0, 17.0, 3.0]]
    >>> fit.attrs["sum_of_squares"]["fitted"], fit.attrs["sum_of_squares"]["residual"]
    (820.0, 20.0)
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


def yates_table_3k(design: pd.DataFrame, response: str, factors: Sequence[str] | None = None) -> pd.DataFrame:
    r"""Return Yates's algorithm for a full three-level factorial: its columns and its single-degree-of-freedom effects.

    Each effect is a product of the factors' orthogonal polynomials, the linear :math:`x` and the
    quadratic :math:`3x^2 - 2`, so the table splits the :math:`3^k - 1` degrees of freedom between the
    runs into one each, as Box and Draper (2007, Appendix 7G) work it. For the components of an
    interaction with two degrees of freedom each, see
    :func:`~process_improve.experiments.interaction_components`.

    Parameters
    ----------
    design : pd.DataFrame
        The runs of a full :math:`3^k` factorial, each factor coded -1, 0 and +1 (equally spaced settings), each
        run once, in any order.
    response : str
        The column that holds the response.
    factors : Sequence[str], optional
        The factors, the first the one that changes fastest in standard order. By default every column of
        ``design`` but the response, in their order.

    Returns
    -------
    pd.DataFrame
        One row per run in standard order: the factors and the response; the algorithm's columns ``(1)`` to
        ``(k)`` (Box and Draper number the response (1), so their columns (2) to (k + 1)); ``effect``, the effect
        the row's last column estimates (``"mean"``, then ``"A"``, ``"A^2"``, ``"B"``, ``"A:B"``, ``"A^2:B"``, ...,
        where ``"A^2"`` stands for the quadratic polynomial :math:`3x^2 - 2`; Box and Draper write ``"A^2:B"`` as
        1²20); and, with :math:`z'y` the last column and :math:`z'z` the sum of squares of the effect's polynomial:
        ``divisor``, :math:`z'z`; ``estimate``, :math:`z'y / z'z`, the coefficient of the polynomial in the
        least-squares fit (the average for the mean); ``sum_sq``, :math:`(z'y)^2 / z'z`, on one degree of freedom
        (for the mean, the correction for the mean); ``standardized``, :math:`z'y / (z'z)^{1/2}`, the signed
        square root of ``sum_sq``; and ``normal_score``, the standard normal quantile at :math:`(i - 1/2)/m` of the
        rank *i* of ``standardized`` among the :math:`m = 3^k - 1` effects, the position on a normal plot of the
        effects, empty for the mean.

    Raises
    ------
    ValueError
        If the response is not a column, a factor is not coded -1, 0 and +1, the runs are not each run of the full
        factorial once, a response is missing or named as a factor, or a name would collide with a column of the
        table.

    Examples
    --------
    One factor at three settings, its runs in any order: the passes give the total, the linear contrast
    and the quadratic contrast.

    >>> runs = pd.DataFrame({"A": [1, -1, 0], "y": [4.0, 2.0, 6.0]})
    >>> table = yates_table_3k(runs, "y")
    >>> table["(1)"].tolist(), table["divisor"].tolist()
    ([12.0, 2.0, -6.0], [3, 2, 6])
    >>> dict(zip(table["effect"], table["estimate"].tolist(), strict=True))
    {'mean': 4.0, 'A': 1.0, 'A^2': -1.0}
    """
    names = _factor_names(design, response, factors, set(_ADDED_3K))
    k, n = len(names), 3 ** len(names)
    runs = design.iloc[_standard_order(design, names, levels=3)]
    y = _response(runs, response)
    columns = _passes(y, k, levels=3)
    # z'z of a product of polynomials: 3 for the constant (1, 1, 1), 2 for x (-1, 0, 1) and 6 for (1, -2, 1).
    divisor = np.array([np.prod([(3, 2, 6)[power] for power in _digits(i, k, 3)]) for i in range(n)])
    table = pd.DataFrame({name: runs[name].to_numpy() for name in names})
    table[response] = y
    for j, column in enumerate(columns, start=1):
        table[f"({j})"] = column
    table["effect"] = _effect_names(names, levels=3)
    table["divisor"] = divisor
    table["estimate"] = columns[-1] / divisor
    table["sum_sq"] = columns[-1] ** 2 / divisor
    table["standardized"] = columns[-1] / np.sqrt(divisor)
    ranks = np.argsort(np.argsort(table["standardized"].to_numpy()[1:]))
    table["normal_score"] = np.concatenate([[np.nan], stats.norm.ppf((ranks + 0.5) / (n - 1))])
    return table
