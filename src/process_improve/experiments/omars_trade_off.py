# (c) Kevin Dunn, 2010-2026. MIT License.

r"""The OMARS trade-off: which model does a given run budget buy.

The two-level trade-off table in :mod:`process_improve.experiments.trade_off`
answers "I can afford N runs and want k factors, what do I give up?" with a
resolution and an alias structure. That currency does not transfer to OMARS
designs, because an OMARS design *always* has its main effects orthogonal to
each other and to every second-order term. Resolution is constant, so it cannot
be what the table reports.

What varies instead is **which model is estimable at all**, and that is set by
the foldover structure. A foldover is ``[H; -H; 0]``, and every second-order
term is an *even* function, so the quadratic and interaction columns of ``H``
and ``-H`` are identical. The even block therefore has at most ``h + 1``
distinct rows, against ``1 + k(k+1)/2`` columns for the full second-order model.
The main effects live in the odd block and contribute at most ``k`` more, so for
every foldover

.. math::

   \mathrm{rank}(X) \le k + \min\!\left(h + 1,\; 1 + \tfrac{k(k+1)}{2}\right)

with equality for half-designs in general position. Three capability classes
follow, and they are the OMARS analogue of resolution:

``Full``
    ``N >= k^2 + k + 1``. Main effects, pure quadratics and every two-factor
    interaction are jointly estimable, so a response surface can be fitted
    without a follow-up design.
``Quad``
    ``N >= 2k + 3``. Main effects and the pure quadratics, with degrees of
    freedom left to test them, so curvature can be judged factor by factor. The
    two-factor interactions are present in the design but not in the model.
``Satd``
    ``N = 2k + 1``. The definitive screening design at its minimal size:
    estimable but exactly saturated, so there are point estimates and no
    inference.

Alphabetically ``Full < Quad < Satd``, which is also decreasing capability, so
the ordering is easy to keep straight.

Every number here is closed-form, so the table is instant and exact: no integer
program, no solver, and no dependence on a search budget. The quality of a
*particular* design at a given size (its D-efficiency, its second-order
correlations) does need the ILP, and lives on
:func:`~process_improve.experiments.generate_omars` instead.

Also see
--------
process_improve.experiments.trade_off : the two-level counterpart.
process_improve.experiments.designs_omars_ilp.generate_omars : build the design.
process_improve.experiments.omars.analyze_omars : the staged analysis it feeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from process_improve.experiments.designs_omars_ilp import (
    _full_second_order_params,
    _min_runs,
)
from process_improve.experiments.strategy.budget import _BBD_RUNS

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Capability classes, best first. The four-character tags line up in a table,
#: and sort alphabetically in decreasing order of capability.
CAPABILITIES: tuple[str, ...] = ("full", "quad", "satd")

_TAGS = {"full": "Full", "quad": "Quad", "satd": "Satd"}

_MEANINGS = {
    "full": "main effects, quadratics and all two-factor interactions jointly estimable",
    "quad": "main effects and pure quadratics, with error degrees of freedom to test them",
    "satd": "saturated: estimable but no error degrees of freedom, so no inference",
}

# OMARS designs need at least three factors (see ``generate_omars``). The upper
# bound is generous because nothing here enumerates runs; it is arithmetic.
_MIN_FACTORS = 3
_MAX_FACTORS = 25

#: Default rows: odd run counts spanning the useful band for three to seven
#: factors. The last one is the smallest budget at which every one of those
#: factor counts reaches ``Full``.
DEFAULT_RUNS: tuple[int, ...] = (9, 13, 17, 21, 25, 31, 37, 43, 57)

#: Default columns.
DEFAULT_FACTORS: tuple[int, ...] = (3, 4, 5, 6, 7)

#: Named standard designs available as anchor rows on the table, smallest first.
#: The definitive screening design is the smallest member of the OMARS family
#: and the Box-Behnken design is among the largest, so a table carrying both
#: shows the span the family covers rather than only its middle.
REFERENCE_DESIGNS: tuple[str, ...] = ("dsd", "bbd")

_REFERENCE_TAGS = {"dsd": "DSD", "bbd": "BBD"}

#: Centre runs each Box-Behnken design conventionally carries. The design runs
#: themselves come from ``_BBD_RUNS``; these are the counts the published tables
#: quote alongside them, giving the familiar totals of 15, 27, 46, 54 and 62.
_BBD_CENTRE_RUNS: dict[int, int] = {3: 3, 4: 3, 5: 6, 6: 6, 7: 6}


@dataclass
class OmarsTradeOffTableEntry:
    """What one run budget buys for one factor count.

    Attributes
    ----------
    n_runs : int
        The run budget.
    n_factors : int
        Number of factors.
    exists : bool
        Whether any foldover OMARS design has this run count for this many
        factors. ``False`` for an even *n_runs* or one below ``2k + 1``.
    capability : str
        ``"full"``, ``"quad"``, ``"satd"``, or ``"none"`` when *exists* is
        ``False``.
    tag : str
        The four-character table tag: ``"Full"``, ``"Quad"``, ``"Satd"``, or
        the empty string.
    model : str or None
        The largest model this run count supports: ``"full_second_order"``,
        ``"main_quadratic"``, or ``None``.
    model_params : int
        Parameter count of that model; zero when there is no design.
    error_df : int
        Degrees of freedom left for error after fitting it. Zero for ``Satd``.
    label : str
        Self-contained cell text, e.g. ``"Full df=11"``.
    min_runs_full : int
        Smallest run count reaching ``Full`` for this many factors,
        ``k^2 + k + 1``.
    min_runs_quad : int
        Smallest run count reaching ``Quad``, ``2k + 3``.
    min_runs_satd : int
        Smallest run count that is a design at all, ``2k + 1``.
    reason : str
        Why a cell is empty; empty string when *exists* is ``True``.
    """

    n_runs: int
    n_factors: int
    exists: bool
    capability: str
    tag: str
    model: str | None
    model_params: int
    error_df: int
    label: str
    min_runs_full: int
    min_runs_quad: int
    min_runs_satd: int
    reason: str = ""


def _check_factors(n_factors: int) -> int:
    """Validate and return the factor count."""
    if int(n_factors) != n_factors:
        raise ValueError('The "n_factors" input must be an integer.')
    k = int(n_factors)
    if not (_MIN_FACTORS <= k <= _MAX_FACTORS):
        raise ValueError(f"OMARS designs need {_MIN_FACTORS} to {_MAX_FACTORS} factors; got {k}.")
    return k


def omars_minimum_runs(n_factors: int, capability: str = "full") -> int:
    """Return the smallest run count reaching *capability* for *n_factors*.

    Parameters
    ----------
    n_factors : int
        Number of factors, ``k``, between 3 and 25.
    capability : {"full", "quad", "satd"}, default "full"
        Which class to reach. See the module docstring.

    Returns
    -------
    int
        The (odd) run count: ``k^2 + k + 1`` for ``"full"``, ``2k + 3`` for
        ``"quad"``, ``2k + 1`` for ``"satd"``.

    Raises
    ------
    ValueError
        If *n_factors* is out of range or *capability* is not a known class.

    Examples
    --------
    >>> omars_minimum_runs(5)
    31
    >>> omars_minimum_runs(5, "quad")
    13
    >>> [omars_minimum_runs(k) for k in (3, 4, 5, 6, 7)]
    [13, 21, 31, 43, 57]
    """
    k = _check_factors(n_factors)
    if capability not in CAPABILITIES:
        raise ValueError(f"capability must be one of {CAPABILITIES}, got {capability!r}.")
    if capability == "full":
        return _min_runs(k)
    if capability == "quad":
        return 2 * k + 3
    return 2 * k + 1


def definitive_screening_runs(n_factors: int) -> int:
    """Return the run count of the definitive screening design for *n_factors*.

    Parameters
    ----------
    n_factors : int
        Number of factors, ``k``, between 3 and 25.

    Returns
    -------
    int
        ``2k + 1`` for an even *n_factors*, ``2k + 3`` for an odd one.

    Notes
    -----
    The construction folds a conference matrix of order ``k``, which exists for
    an even ``k``. For an odd ``k`` it uses one of order ``k + 1`` and drops the
    last column, so the design arrives with two runs to spare and lands one
    capability class above saturated.

    Examples
    --------
    >>> [definitive_screening_runs(k) for k in (3, 4, 5, 6, 7)]
    [9, 9, 13, 13, 17]
    """
    k = _check_factors(n_factors)
    return 2 * k + 1 if k % 2 == 0 else 2 * k + 3


def box_behnken_runs(n_factors: int) -> int | None:
    """Return the run count of the published Box-Behnken design for *n_factors*.

    Parameters
    ----------
    n_factors : int
        Number of factors, ``k``, between 3 and 25.

    Returns
    -------
    int or None
        Design runs plus the centre runs the published tables quote, or ``None``
        when Box and Behnken did not publish a design for that factor count.

    Examples
    --------
    >>> [box_behnken_runs(k) for k in (3, 4, 5, 6, 7)]
    [15, 27, 46, 54, 62]
    >>> box_behnken_runs(8) is None
    True
    """
    k = _check_factors(n_factors)
    design_runs = _BBD_RUNS.get(k)
    return None if design_runs is None else design_runs + _BBD_CENTRE_RUNS[k]


def _reference_entry(name: str, n_factors: int) -> OmarsTradeOffTableEntry | None:
    """Build the table entry for a named standard design at *n_factors*.

    Unlike :func:`get_omars_trade_off_table_entry` this does not require an odd
    run count. A named design carries whatever centre replication its published
    form specifies, and a Box-Behnken design carries three or six centre runs
    rather than one, so its total is even at five factors and above. The
    capability thresholds still apply: they are set by the number of distinct
    half-rows, which extra centre runs do not change.
    """
    k = _check_factors(n_factors)
    runs = definitive_screening_runs(k) if name == "dsd" else box_behnken_runs(k)
    if runs is None:
        return None

    satd, quad, full = (omars_minimum_runs(k, c) for c in ("satd", "quad", "full"))
    capability = "full" if runs >= full else "quad" if runs >= quad else "satd"
    if capability == "full":
        model, params = "full_second_order", _full_second_order_params(k)
    else:
        model, params = "main_quadratic", 1 + 2 * k

    tag = _TAGS[capability]
    return OmarsTradeOffTableEntry(
        n_runs=runs,
        n_factors=k,
        exists=True,
        capability=capability,
        tag=tag,
        model=model,
        model_params=params,
        error_df=runs - params,
        # Anchor cells lead with the run count, because unlike a normal row the
        # run count is not the row label and changes from column to column.
        label=f"{runs} {tag} df={runs - params}",
        min_runs_full=full,
        min_runs_quad=quad,
        min_runs_satd=satd,
    )


def get_omars_trade_off_table_entry(n_runs: int, n_factors: int, display: bool = True) -> OmarsTradeOffTableEntry:
    """Report which model a run budget buys, for a foldover OMARS design.

    The OMARS counterpart of
    :func:`~process_improve.experiments.trade_off.get_trade_off_table_entry`. Because OMARS main
    effects are always clear of the second-order terms, the answer is not a
    resolution: it is which model is estimable, and how much is left over to
    test it with.

    Parameters
    ----------
    n_runs : int
        Run budget. A foldover has ``2h + 1`` runs, so an even value is never a
        design.
    n_factors : int
        Number of factors, between 3 and 25.
    display : bool, default True
        Print a short report as well as returning it.

    Returns
    -------
    OmarsTradeOffTableEntry
        The capability class, the model it supports, and the error degrees of
        freedom. See the class for the full field list.

    Raises
    ------
    ValueError
        If *n_factors* is out of range, or *n_runs* is not a positive integer.

    Examples
    --------
    >>> get_omars_trade_off_table_entry(21, 4, display=False).label
    'Full df=6'
    >>> get_omars_trade_off_table_entry(17, 4, display=False).label
    'Quad df=8'
    >>> get_omars_trade_off_table_entry(9, 4, display=False).label
    'Satd df=0'

    Also see
    --------
    omars_trade_off_table : the same answer across a grid of budgets.
    """
    k = _check_factors(n_factors)
    if int(n_runs) != n_runs:
        raise ValueError('The "n_runs" input must be an integer.')
    runs = int(n_runs)
    if runs < 1:
        raise ValueError(f"The number of runs must be positive; got {runs}.")

    satd, quad, full = (omars_minimum_runs(k, c) for c in ("satd", "quad", "full"))

    # Which class the budget lands in, and why it might land in none of them.
    capability, reason = "none", ""
    if runs % 2 == 0:
        reason = f"{runs} is even; a foldover OMARS design has 2h + 1 runs."
    elif runs < satd:
        reason = f"{runs} runs is below the smallest OMARS design for {k} factors ({satd} runs)."
    elif runs >= full:
        capability = "full"
    elif runs >= quad:
        capability = "quad"
    else:
        capability = "satd"

    if capability == "none":
        model, params, error_df = None, 0, 0
    elif capability == "full":
        model, params = "full_second_order", _full_second_order_params(k)
        error_df = runs - params
    else:
        model, params = "main_quadratic", 1 + 2 * k
        # Satd is exactly saturated by construction, so runs - params is zero.
        error_df = runs - params

    tag = _TAGS.get(capability, "")
    result = OmarsTradeOffTableEntry(
        n_runs=runs,
        n_factors=k,
        exists=capability != "none",
        capability=capability,
        tag=tag,
        model=model,
        model_params=params,
        error_df=error_df,
        label=f"{tag} df={error_df}" if tag else "",
        min_runs_full=full,
        min_runs_quad=quad,
        min_runs_satd=satd,
        reason=reason,
    )

    if display:
        print(_format_result(result))  # noqa: T201
    return result


def _format_result(result: OmarsTradeOffTableEntry) -> str:
    """Render an :class:`OmarsTradeOffTableEntry` as the printed report."""
    lines = [f"OMARS: {result.n_runs} runs, {result.n_factors} factors"]
    if not result.exists:
        lines.append(f"  No design: {result.reason}")
        lines.append(f"  The smallest design for {result.n_factors} factors has {result.min_runs_satd} runs.")
        return "\n".join(lines) + "\n"

    lines.append(f"  {result.tag}: {_MEANINGS[result.capability]}")
    lines.append(f"  Model: {result.model} ({result.model_params} parameters), {result.error_df} error df")
    lines.append(
        f"  Thresholds for {result.n_factors} factors: "
        f"Satd {result.min_runs_satd}, Quad {result.min_runs_quad}, Full {result.min_runs_full} runs."
    )
    if result.capability != "full":
        short = result.min_runs_full - result.n_runs
        lines.append(f"  {short} more runs would reach Full (all two-factor interactions estimable).")
    return "\n".join(lines) + "\n"


def omars_trade_off_table(
    runs: Sequence[int] = DEFAULT_RUNS,
    factors: Sequence[int] = DEFAULT_FACTORS,
    display: bool = True,
    anchors: bool = False,
) -> pd.DataFrame:
    """Return the run-budget against factor-count table for OMARS designs.

    Each cell says which model that budget supports and how much error is left
    to test it with, for example ``"Full df=11"``. Blank cells are budgets that
    are not a foldover design at all.

    Parameters
    ----------
    runs : sequence of int, default :data:`DEFAULT_RUNS`
        Run budgets, one per row. Even values are always blank.
    factors : sequence of int, default :data:`DEFAULT_FACTORS`
        Factor counts, one per column.
    display : bool, default True
        Print the table left-aligned, with the ``df=`` label written once per
        column rather than in every cell.
    anchors : bool, default False
        Add a ``DSD`` row above the budgets and a ``BBD`` row below them, giving
        the two ends of the family a fixed place on the table. Their run counts
        change from column to column, so their cells lead with the run count,
        for example ``"46 Full df=25"``. Turning this on makes the index mix
        strings with the integer budgets.

    Returns
    -------
    pd.DataFrame
        Rows indexed by run count, columns by factor count. Cells are
        self-contained labels; the once-per-column compression applies only to
        the printed view.

    Examples
    --------
    >>> table = omars_trade_off_table(display=False)
    >>> table.loc[21, 4]
    'Full df=6'
    >>> table.loc[9, 3]
    'Quad df=2'
    >>> anchored = omars_trade_off_table(display=False, anchors=True)
    >>> anchored.loc["DSD", 4]
    '9 Satd df=0'
    >>> anchored.loc["BBD", 5]
    '46 Full df=25'

    Also see
    --------
    get_omars_trade_off_table_entry : the detail behind one cell.
    definitive_screening_runs, box_behnken_runs : the anchor run counts.
    """
    checked = [_check_factors(k) for k in factors]
    cells: dict[int, dict[object, str]] = {
        k: {n: get_omars_trade_off_table_entry(n, k, display=False).label for n in runs} for k in checked
    }
    index: list[object] = list(runs)

    if anchors:
        for name in REFERENCE_DESIGNS:
            tag = _REFERENCE_TAGS[name]
            for k in checked:
                entry = _reference_entry(name, k)
                cells[k][tag] = "" if entry is None else entry.label
        # The DSD is the smallest member of the family and the Box-Behnken
        # design among the largest, so they bracket the budgets.
        index = [_REFERENCE_TAGS["dsd"], *index, _REFERENCE_TAGS["bbd"]]

    table = pd.DataFrame(cells, index=index)
    table.index.name = "runs"
    table.columns.name = "factors"
    if display:
        print(_format_table(table))  # noqa: T201
    return table


def _format_table(table: pd.DataFrame) -> str:
    """Render the table left-aligned, with ``df=`` written once per column.

    Repeating ``df=`` in every cell buries the staircase where the capability
    class changes, which is the thing the table exists to show. The label is
    kept on the first live cell of each column, where a reader meets it first.
    """
    # Work off a plain list-of-columns rather than the frame, so the cells stay
    # ordinary strings all the way through the formatting.
    columns: list[list[str]] = []
    for column in table.columns:
        labelled = False
        cells: list[str] = []
        for label in (str(value) for value in table[column]):
            if not label:
                cells.append("")
            elif labelled:
                tag, _, value = label.partition(" df=")
                cells.append(f"{tag} {value}")
            else:
                cells.append(label)
                labelled = True
        columns.append(cells)

    width = max((len(cell) for column_cells in columns for cell in column_cells), default=0) + 2
    index_width = max(len("runs"), *(len(str(i)) for i in table.index)) + 2

    lines = ["runs".ljust(index_width) + "".join(f"k={c}".ljust(width) for c in table.columns)]
    lines.append("-" * (index_width + width * len(table.columns)))
    lines.extend(
        str(index).ljust(index_width) + "".join(column_cells[row].ljust(width) for column_cells in columns)
        for row, index in enumerate(table.index)
    )
    lines.append("")
    lines.append("  Full: " + _MEANINGS["full"])
    lines.append("  Quad: " + _MEANINGS["quad"])
    lines.append("  Satd: " + _MEANINGS["satd"])
    return "\n".join(lines) + "\n"
