# (c) Kevin Dunn, 2010-2026. MIT License.

"""The fractional-factorial trade-off: n_runs against n_factors.

The central question when screening many n_factors is how few n_runs you can get
away with, and what you pay for that saving. This module answers it in two
ways, mirroring the R ``pid`` package:

* :func:`get_trade_off_table_entry` reports, for one (runs, factors) pair, the design's
  resolution, its generators, its defining relation, and which effects end up
  aliased with which.
* :func:`trade_off_table` prints the whole grid at once, the Python
  counterpart of the trade-off table figure in the course notes.

Unlike the R version, which looks designs up in the ``FrF2`` catalogue, the
generators here are derived by a **minimum-aberration search**: for a given
number of n_runs and n_factors, every admissible set of generators is scored on
its word-length pattern and the best is kept. The search reproduces the table
in the course notes exactly, and extends past its printed edge.

Also see
--------
process_improve.experiments.designs.generate_design : builds the design matrix.
process_improve.experiments.evaluate.evaluate_design : evaluates one you have.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache
from string import ascii_uppercase
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

from process_improve.experiments.evaluate import (
    _defining_relation_from_generators,
    _multiply_words,
    _word_to_str,
)

_ROMAN = {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX", 10: "X"}

# "I" is the identity column in a defining relation, so it is never a factor name.
_FACTOR_NAMES = tuple(letter for letter in ascii_uppercase if letter != "I")

# Above this many candidate generator sets the exhaustive minimum-aberration
# search stops being interactive, so it refuses rather than hanging.
_MAX_CANDIDATE_SETS = 500_000


@dataclass
class TradeOffTableEntry:
    """What you get, and what you give up, at a given (runs, factors) pair.

    Attributes
    ----------
    n_runs : int
        Number of experiments in the design.
    n_factors : int
        Number of factors studied.
    n_generators : int
        The ``p`` in ``2^(k-p)``: how many factors are added on top of the
        ``k - p`` base n_factors. Zero for a full factorial.
    resolution : int or None
        Design resolution as an integer (3, 4, 5, ...), or ``None`` for a full
        factorial, which has no defining relation and so no resolution.
    roman : str or None
        The same resolution in roman numerals (``"III"``, ``"IV"``, ...), the
        way it is written as a subscript on ``2^(k-p)``.
    generators : list[str]
        Generators, e.g. ``["D=AB", "E=AC"]``. Empty for a full factorial.
        Every generator may be used with either sign; ``D=-AB`` gives the
        complementary fraction, which is equally valid.
    defining_relation : list[str]
        The full defining relation, e.g. ``["I=ABD", "I=ACE", "I=BCDE"]``.
        Empty for a full factorial.
    aliases : list[str]
        Alias chains for the main effects and the two-factor interactions,
        e.g. ``"A = BD + CE + ..."``. Empty for a full factorial.
    replicates : int
        How many times the full factorial fits into the run budget. ``1`` in
        the usual case; ``2`` means the budget pays for the full factorial
        twice over, and so on.
    label : str
        Compact description of the design, e.g. ``"2^(5-2) III"``,
        ``"2^3 (full)"`` or ``"2^3 (twice)"``.
    """

    n_runs: int
    n_factors: int
    n_generators: int
    resolution: int | None
    roman: str | None
    generators: list[str] = field(default_factory=list)
    defining_relation: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    replicates: int = 1
    label: str = ""


def _factor_names(k: int) -> list[str]:
    """Return the first *k* single-letter factor names, skipping ``I``."""
    if k > len(_FACTOR_NAMES):
        raise ValueError(
            f"At most {len(_FACTOR_NAMES)} n_factors can be given single-letter names; {k} were requested."
        )
    return list(_FACTOR_NAMES[:k])


def _candidate_words(n_base: int) -> list[str]:
    """Every interaction of two or more of the *n_base* base n_factors.

    These are the columns of the base factorial that an extra factor can be
    assigned to. Sorted by word length, then alphabetically, so that ties in
    aberration are broken deterministically and in favour of the shorter,
    more conventional generator.
    """
    base = _factor_names(n_base)
    words = ["".join(combo) for r in range(2, n_base + 1) for combo in itertools.combinations(base, r)]
    return sorted(words, key=lambda w: (len(w), w))


def _word_length_pattern(generators: tuple[str, ...], factor_names: list[str], k: int) -> tuple[int, ...]:
    """Word-length pattern ``(A3, A4, ...)`` of the defining relation.

    ``Ai`` counts the defining words of length ``i``. Comparing these tuples
    lexicographically is exactly the minimum-aberration criterion: fewer short
    words is better, and the first difference decides. Since the shortest word
    sets the resolution, the smallest pattern also has the highest resolution.
    """
    words = _defining_relation_from_generators(list(generators), factor_names)
    counts = [0] * (k + 2)
    for word in words:
        counts[len(word)] += 1
    return tuple(counts[3:])


@lru_cache(maxsize=256)
def minimum_aberration_generators(n_runs: int, n_factors: int) -> tuple[str, ...]:
    """Find the minimum-aberration generators for a ``2^(k-p)`` design.

    Every way of assigning the ``p`` extra n_factors to interaction columns of
    the base factorial is enumerated, scored by its word-length pattern, and
    the best-scoring one is returned. Ties are broken in favour of the
    generator set that comes first by word length and then alphabetically.

    Parameters
    ----------
    n_runs : int
        Number of runs; must be a power of two.
    n_factors : int
        Number of factors, greater than ``log2(n_runs)``.

    Returns
    -------
    tuple[str, ...]
        Generators such as ``("D=AB", "E=AC")``.

    Raises
    ------
    ValueError
        If *n_runs* is not a power of two, if the design is not fractional
        (``n_factors <= log2(n_runs)``), if there are too few interaction columns
        to hold the extra n_factors, or if the search space is too large to
        enumerate.

    Examples
    --------
    >>> minimum_aberration_generators(8, 5)
    ('D=AB', 'E=AC')
    >>> minimum_aberration_generators(16, 5)
    ('E=ABCD',)

    Notes
    -----
    A minimum-aberration design is unique only up to relabelling the factors,
    so a textbook may print a different but equivalent set of generators.
    """
    n_base = _check_runs(n_runs)
    n_extra = n_factors - n_base
    if n_extra <= 0:
        raise ValueError(
            f"{n_factors} n_factors in {n_runs} n_runs is not a fractional factorial: "
            f"{n_runs} runs accommodate a full 2^{n_base} factorial."
        )

    candidates = _candidate_words(n_base)
    if n_extra > len(candidates):
        raise ValueError(
            f"{n_runs} n_runs cannot accommodate {n_factors} n_factors: only "
            f"{n_base + len(candidates)} factors fit into {n_runs} n_runs."
        )

    n_sets = math.comb(len(candidates), n_extra)
    if n_sets > _MAX_CANDIDATE_SETS:
        raise ValueError(
            f"The minimum-aberration search for {n_factors} n_factors in {n_runs} runs would have to "
            f"score {n_sets:,} generator sets, above the limit of {_MAX_CANDIDATE_SETS:,}. "
            "Supply the generators yourself, via the `generators` argument of `generate_design`."
        )

    names = _factor_names(n_factors)
    extra_names = names[n_base:]

    best_pattern: tuple[int, ...] | None = None
    best_generators: tuple[str, ...] = ()
    for combo in itertools.combinations(candidates, n_extra):
        generators = tuple(f"{name}={word}" for name, word in zip(extra_names, combo, strict=True))
        pattern = _word_length_pattern(generators, names, n_factors)
        if best_pattern is None or pattern < best_pattern:
            best_pattern = pattern
            best_generators = generators

    return best_generators


def _check_runs(n_runs: int) -> int:
    """Validate *n_runs* as a power of two and return ``log2(n_runs)``."""
    if int(n_runs) != n_runs:
        raise ValueError('The "n_runs" input must be an integer.')
    n_runs = int(n_runs)
    if n_runs < 2 or (n_runs & (n_runs - 1)) != 0:
        raise ValueError(f"The number of runs must be a power of 2 (4, 8, 16, ...); got {n_runs}.")
    return n_runs.bit_length() - 1


def _alias_chains(generators: list[str], factor_names: list[str]) -> list[str]:
    """Alias chains for the main effects and the two-factor interactions."""
    words = _defining_relation_from_generators(generators, factor_names)
    if not words:
        return []

    k = len(factor_names)
    effects: list[frozenset[int]] = [frozenset([i]) for i in range(k)]
    effects += [frozenset(pair) for pair in itertools.combinations(range(k), 2)]

    chains = []
    for effect in effects:
        aliases = sorted(
            (_word_to_str(_multiply_words(effect, word), factor_names) for word in words),
            key=lambda s: (len(s), s),
        )
        chains.append(f"{_word_to_str(effect, factor_names)} = " + " + ".join(aliases))
    return chains


def get_trade_off_table_entry(n_runs: int = 8, n_factors: int = 7, display: bool = True) -> TradeOffTableEntry:
    """Report the resolution, generators and aliasing at a (runs, factors) pair.

    Answers the screening question "if I can afford *n_runs* experiments and I
    want to study *n_factors* n_factors, what do I lose?". The loss is aliasing:
    effects that the design cannot tell apart.

    Parameters
    ----------
    n_runs : int, default 8
        Number of experiments you can afford. Must be a power of two.
    n_factors : int, default 7
        Number of factors to study.
    display : bool, default True
        Print a human-readable report as well as returning it. Set to
        ``False`` to keep the function quiet.

    Returns
    -------
    TradeOffTableEntry
        Resolution, generators, defining relation and alias chains. See the
        class for the full field list.

    Raises
    ------
    ValueError
        If *n_runs* or *n_factors* is not an integer, if *n_runs* is not a power of
        two, if *n_factors* is below 2, or if the n_factors cannot fit into the
        run budget.

    Examples
    --------
    >>> result = get_trade_off_table_entry(n_runs=8, n_factors=5, display=False)
    >>> result.label
    '2^(5-2) III'
    >>> result.generators
    ['D=AB', 'E=AC']

    A run budget larger than the full factorial needs is reported as
    replication rather than as an error:

    >>> get_trade_off_table_entry(n_runs=16, n_factors=3, display=False).label
    '2^3 (twice)'

    Also see
    --------
    trade_off_table : the same information for a whole grid of designs.
    """
    if int(n_factors) != n_factors:
        raise ValueError('The "n_factors" input must be an integer.')
    n_factors = int(n_factors)
    n_base = _check_runs(n_runs)
    n_runs = int(n_runs)
    if n_factors < 2:
        raise ValueError(f"At least 2 factors are needed to design an experiment; got {n_factors}.")

    names = _factor_names(n_factors)

    if n_factors <= n_base:
        # The budget covers the full factorial, possibly several times over.
        replicates = 2 ** (n_base - n_factors)
        how_often = {1: "full", 2: "twice", 4: "4 times", 8: "8 times"}.get(replicates, f"{replicates} times")
        result = TradeOffTableEntry(
            n_runs=n_runs,
            n_factors=n_factors,
            n_generators=0,
            resolution=None,
            roman=None,
            replicates=replicates,
            label=f"2^{n_factors} ({how_often})",
        )
    else:
        generators = list(minimum_aberration_generators(n_runs, n_factors))
        words = _defining_relation_from_generators(generators, names)
        resolution = min(len(word) for word in words)
        roman = _ROMAN.get(resolution, str(resolution))
        result = TradeOffTableEntry(
            n_runs=n_runs,
            n_factors=n_factors,
            n_generators=n_factors - n_base,
            resolution=resolution,
            roman=roman,
            generators=generators,
            defining_relation=[f"I={_word_to_str(word, names)}" for word in words],
            aliases=_alias_chains(generators, names),
            label=f"2^({n_factors}-{n_factors - n_base}) {roman}",
        )

    if display:
        print(_format_entry(result))  # noqa: T201
    return result


def _format_entry(result: TradeOffTableEntry) -> str:
    """Render a :class:`TradeOffTableEntry` as the printed report."""
    lines = [f"With {result.n_runs} experiments, and {result.n_factors} factors:"]
    if not result.generators:
        lines.append(f"  A full 2^{result.n_factors} factorial fits, run {result.label.split('(')[1].rstrip(')')}.")
        lines.append("  No aliasing: every effect is estimated free of every other effect.")
        return "\n".join(lines) + "\n"

    lines.append(f"  Design: {result.label}")
    lines.append(f"  Resolution: {result.roman}")
    label = "Generator:" if len(result.generators) == 1 else "Generators:"
    lines.append(f"  {label}")
    lines.extend(f"      {generator}" for generator in result.generators)
    lines.append("      (each generator may be used with a + or a - sign)")
    lines.append("  Aliasing (main effects and 2-factor interactions only):")
    if result.resolution is not None and result.resolution > 3:
        lines.append("      Main effects are not aliased with 2-factor interactions.")
    lines.extend(f"      {chain}" for chain in result.aliases)
    return "\n".join(lines) + "\n"


def _cell_label(n_runs: int, n_factors: int) -> str:
    """Label for one cell of the trade-off table; empty when no design exists."""
    try:
        return get_trade_off_table_entry(n_runs=n_runs, n_factors=n_factors, display=False).label
    except ValueError:
        return ""


def trade_off_table(
    runs: Sequence[int] = (4, 8, 16, 32, 64),
    factors: Sequence[int] = (3, 4, 5, 6, 7, 8, 9),
    display: bool = True,
) -> pd.DataFrame:
    """Return the runs-against-factors trade-off table.

    The Python counterpart of R's ``tradeOffTable()``, which displays the
    table as a static image. Here it is computed, so it can be widened past
    the printed edge and the cells can be read programmatically.

    Reading the table: going down a column costs more experiments but buys
    resolution; going across a row studies more factors for the same money,
    at the cost of heavier aliasing. Blank cells are designs that do not
    exist (too many factors for that many runs).

    Parameters
    ----------
    runs : Sequence[int], default (4, 8, 16, 32, 64)
        Run budgets, one per row. Each must be a power of two.
    factors : Sequence[int], default (3, 4, 5, 6, 7, 8, 9)
        Factor counts, one per column.
    display : bool, default True
        Print the table as well as returning it. Set to ``False`` to keep the
        function quiet.

    Returns
    -------
    pd.DataFrame
        Rows indexed by run count, columns by factor count. Each cell is a
        label such as ``"2^(5-2) III"``, ``"2^3 (full)"`` or ``"2^3 (twice)"``;
        impossible combinations are the empty string.

    Examples
    --------
    >>> table = trade_off_table()
    >>> table.loc[8, 5]
    '2^(5-2) III'
    >>> table.loc[16, 4]
    '2^4 (full)'

    Also see
    --------
    get_trade_off_table_entry : generators and alias chains for a single cell of this table.
    """
    cells = {n_runs: {n_factors: _cell_label(n_runs, n_factors) for n_factors in factors} for n_runs in runs}

    table = pd.DataFrame(cells).T
    table.index.name = "runs"
    table.columns.name = "factors"
    if display:
        print(table.to_string())  # noqa: T201
    return table
