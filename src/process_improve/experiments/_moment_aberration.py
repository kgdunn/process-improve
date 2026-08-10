# (c) Kevin Dunn, 2010-2026. MIT License.

r"""Minimum moment aberration for two-level designs (Xu, 2003).

The classical minimum aberration criterion ranks two-level fractional
factorial designs by their word-length pattern, which is read off the
defining relation. That works only for *regular* designs, and only when the
generators are known. :func:`process_improve.experiments.evaluate_design`'s
``minimum_aberration`` metric therefore needs a
:class:`~process_improve.experiments.factor.DesignResult` carrying
``generators``; handed a bare matrix it has nothing to work from.

Minimum *moment* aberration (Xu, 2003) removes both restrictions. It ranks
designs by the moments of the distribution of pairwise similarities between
runs, so it needs nothing but the matrix itself. It is equivalent to minimum
aberration for regular designs, extends to non-regular ones, and costs
:math:`O(n^2 m^2)` rather than :math:`O(n 2^m)`.

That makes it the right tool for evaluating a design this library did **not**
construct: one pasted in from a spreadsheet, lifted from a paper, or emitted
by a language model. Vazquez et al. (2026) show the last case is not
hypothetical, and that the designs which come back are often a lower
resolution than they appear.

Definitions
-----------
For runs :math:`d_i` and :math:`d_j` of an :math:`n \times m` design, let
:math:`\delta(d_i, d_j)` count the columns in which they coincide. The
:math:`t`-th power moment is

.. math::

    K_t = \frac{\sum_{i<j} [\delta(d_i, d_j)]^t}{n(n-1)/2},

and the moment aberration pattern is :math:`(K_1, K_2, \ldots, K_m)`.
Minimum moment aberration sequentially minimises that vector.

Strength and resolution
-----------------------
Each :math:`K_t` has a lower bound :math:`K'_t` attained exactly when the
design is an orthogonal array of strength :math:`t`. Writing the runs in
:math:`\pm 1` coding, :math:`\delta = (m + h)/2` where
:math:`h_{ij} = \sum_k d_{ik} d_{jk}`, and

.. math::

    \sum_{i,j} h_{ij}^{\,u}
        = \sum_{k_1 \ldots k_u} \Big( \sum_i \prod_r d_{i k_r} \Big)^2
        \; \ge \; n^2 E_u(m),

because every term is a square, and a term equals :math:`n^2` exactly when
each column occurs an even number of times among :math:`k_1 \ldots k_u`.
:math:`E_u(m)` counts those tuples,

.. math::

    E_u(m) = u!\,[x^u] (\cosh x)^m = 2^{-m} \sum_{j=0}^{m} \binom{m}{j} (m - 2j)^u ,

and the remaining terms vanish precisely when the corresponding factorial
effect is balanced. Expanding :math:`\delta^t` binomially turns those bounds
into :math:`K'_t`. The design's *strength* is the largest :math:`s` with
:math:`K_t = K'_t` for all :math:`t \le s`, and its resolution is
:math:`s + 1`. For a regular design this reproduces the length of the
shortest word in the defining relation.

References
----------
Xu, H. (2003). Minimum moment aberration for nonregular designs and design
selection. *Statistica Sinica*, 13(3), 691-708.

Vazquez, A. R., Rother, K. M., and Charles-Gonzalez, M. V. (2026). A
systematic assessment of Large Language Models for constructing two-level
fractional factorial designs. arXiv:2512.17113.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Beyond this many factors the pattern is truncated. The exact arithmetic
#: below is integer-based, so the cost is in the size of the integers
#: (:math:`m^m` grows fast) rather than in precision. Strength never exceeds a
#: handful in practice, so the tail carries no decision-making information.
MAX_PATTERN_LENGTH = 32


class NotTwoLevelError(ValueError):
    """Raised when a design has a column that is not exactly two-level."""


@dataclass
class MomentAberrationResult:
    r"""Moment aberration pattern of a two-level design.

    Attributes
    ----------
    pattern : list[float]
        The moment aberration pattern :math:`(K_1, \ldots, K_T)`, where
        ``T = min(m, MAX_PATTERN_LENGTH)``.
    lower_bounds : list[float]
        The matching lower bounds, truncated to the first ``strength + 1``
        orders. A design attains :math:`K'_t` exactly when it is an
        orthogonal array of strength :math:`t`, so this prefix covers every
        order that carries a verdict: the attained bounds, plus the single
        order at which the design first falls short. Past that point the
        bound is still valid but increasingly slack (it can even go
        negative), because replacing each remaining sum of squares by its
        unconstrained minimum discards more and more of the structure that
        the higher moments are actually made of. Reporting it would invite
        a comparison that means nothing.
    strength : int
        Largest :math:`s` such that :math:`K_t = K'_t` for every
        :math:`t \le s`. Zero means the design is not level-balanced.
    resolution : int
        ``strength + 1``. For a regular fractional factorial this equals the
        length of the shortest word in the defining relation.
    n_runs, n_factors : int
        Shape of the evaluated design.
    is_orthogonal_array : bool
        True when ``strength >= 2``, i.e. every pair of columns is balanced.
    truncated : bool
        True when the pattern was cut off at :attr:`MAX_PATTERN_LENGTH`.
    """

    pattern: list[float]
    lower_bounds: list[float]
    strength: int
    resolution: int
    n_runs: int
    n_factors: int
    is_orthogonal_array: bool
    truncated: bool = False
    exact_pattern: list[Fraction] = field(default_factory=list, repr=False)
    exact_lower_bounds: list[Fraction] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (drops the exact Fractions)."""
        return {
            "pattern": self.pattern,
            "lower_bounds": self.lower_bounds,
            "strength": self.strength,
            "resolution": self.resolution,
            "n_runs": self.n_runs,
            "n_factors": self.n_factors,
            "is_orthogonal_array": self.is_orthogonal_array,
            "truncated": self.truncated,
        }

    def is_better_than(self, other: MomentAberrationResult) -> bool:
        r"""Return True when ``self`` beats ``other`` on minimum moment aberration.

        The criterion sequentially minimises the pattern, so the comparison is
        lexicographic on :math:`(K_1, K_2, \ldots)`. Uses the exact
        :class:`~fractions.Fraction` pattern, so designs that differ only in
        the far tail are still ordered correctly.

        Raises
        ------
        ValueError
            If the two designs differ in run size or factor count. The
            criterion ranks designs *of the same size*; :math:`K_t` scales
            with both :math:`n` and :math:`m`, so a cross-size comparison
            would silently favour the smaller design.
        """
        if (self.n_runs, self.n_factors) != (other.n_runs, other.n_factors):
            raise ValueError(
                "Minimum moment aberration compares designs of the same size; got "
                f"{self.n_runs}x{self.n_factors} and {other.n_runs}x{other.n_factors}."
            )
        mine = self.exact_pattern or [Fraction(v) for v in self.pattern]
        theirs = other.exact_pattern or [Fraction(v) for v in other.pattern]
        return mine < theirs


def _coded_pm1(design: pd.DataFrame) -> np.ndarray:
    """Return the design as a ``{-1, +1}`` integer array.

    Any two-level coding is accepted (``-1/1``, ``0/1``, ``"low"/"high"``,
    ``False/True``): the lower of the two sorted levels maps to ``-1``.

    Raises
    ------
    NotTwoLevelError
        If any column does not have exactly two distinct levels.
    """
    columns = []
    for name in design.columns:
        levels = pd.unique(design[name].dropna())
        if len(levels) != 2:
            raise NotTwoLevelError(
                f"Column {name!r} has {len(levels)} distinct level(s); "
                "moment aberration is defined for two-level designs only. "
                "Center points and mixed-level columns must be removed first."
            )
        low = min(levels)
        columns.append(np.where(design[name].to_numpy() == low, -1, 1))
    return np.column_stack(columns).astype(np.int64)


def _similarity_histogram(coded: np.ndarray) -> list[int]:
    r"""Count run pairs by their similarity :math:`\delta`.

    Returns a list of length ``m + 1`` whose ``d``-th entry is the number of
    unordered pairs ``i < j`` that agree in exactly ``d`` columns.

    Bucketing the pairs (there are at most ``m + 1`` distinct similarities)
    lets every later power sum be formed exactly in Python integers, at
    :math:`O(m)` per moment instead of :math:`O(n^2)`.
    """
    n_runs, n_factors = coded.shape
    # delta = (m + h) / 2, with h the Gram matrix of the +/-1 coding.
    gram = coded @ coded.T
    delta = (n_factors + gram) // 2
    iu = np.triu_indices(n_runs, k=1)
    return np.bincount(delta[iu], minlength=n_factors + 1).tolist()


def _even_tuple_counts(n_factors: int, max_order: int) -> list[int]:
    r"""Return ``E_u(m)`` for ``u = 0 .. max_order``.

    ``E_u(m)`` counts ordered ``u``-tuples of the ``m`` columns in which every
    column appears an even number of times, evaluated exactly as
    :math:`2^{-m} \sum_j \binom{m}{j} (m - 2j)^u`.
    """
    binomials = [comb(n_factors, j) for j in range(n_factors + 1)]
    deltas = [n_factors - 2 * j for j in range(n_factors + 1)]
    counts = []
    for order in range(max_order + 1):
        total = sum(b * d**order for b, d in zip(binomials, deltas, strict=True))
        # The sum counts each tuple 2**m times and is exactly divisible.
        count, remainder = divmod(total, 2**n_factors)
        if remainder:  # pragma: no cover - guards an arithmetic invariant
            raise ArithmeticError(f"E_{order}({n_factors}) is not integral; this is a bug.")
        counts.append(count)
    return counts


def _pattern_from_moments(
    raw_moments: list[int],
    n_runs: int,
    n_factors: int,
    max_order: int,
) -> list[Fraction]:
    r"""Convert :math:`\sum_{i,j} h^u` values into :math:`(K_1, \ldots)`.

    ``raw_moments[u]`` is :math:`\sum_{i,j} h_{ij}^u` over *ordered* pairs,
    including the diagonal. Uses the binomial expansion of
    :math:`\delta = (m + h)/2`.
    """
    pattern = []
    for order in range(1, max_order + 1):
        total = sum(comb(order, u) * n_factors ** (order - u) * raw_moments[u] for u in range(order + 1))
        sum_delta_power = Fraction(total, 2**order)
        # Strip the diagonal (delta_ii = m) and normalise by the pair count.
        pattern.append((sum_delta_power - n_runs * n_factors**order) / (n_runs * (n_runs - 1)))
    return pattern


def moment_aberration(design: pd.DataFrame | np.ndarray) -> MomentAberrationResult:
    r"""Compute the moment aberration pattern, strength and resolution.

    Works on any two-level design matrix, regular or not, with no need for
    generators or a defining relation.

    Parameters
    ----------
    design : DataFrame or ndarray
        An :math:`n \times m` two-level design. Any consistent two-level
        coding is accepted. Columns named ``"Run"``, ``"RunOrder"`` or
        ``"Block"`` are ignored.

    Returns
    -------
    MomentAberrationResult
        Pattern, lower bounds, strength and resolution.

    Raises
    ------
    NotTwoLevelError
        If a factor column does not have exactly two levels.
    ValueError
        If the design has fewer than two runs or no factor columns.

    Notes
    -----
    A design can have strength at most :math:`m`, so the reported resolution
    saturates at :math:`m + 1`. For a full factorial, which has no defining
    relation and hence no finite resolution in the classical sense, that
    saturated value is what you get: the criterion is telling you the design
    is aliasing-free as far as it can see.

    Examples
    --------
    >>> import itertools
    >>> import pandas as pd
    >>> from process_improve.experiments import moment_aberration
    >>> runs = list(itertools.product([-1, 1], repeat=3))
    >>> design = pd.DataFrame(runs, columns=["A", "B", "C"])
    >>> design["D"] = design["A"] * design["B"] * design["C"]  # the 2^(4-1), D = ABC
    >>> result = moment_aberration(design)
    >>> result.resolution
    4
    >>> [round(k, 2) for k in result.pattern]
    [1.71, 3.43, 6.86, 13.71]

    See Also
    --------
    process_improve.experiments.evaluate_design : the ``moment_aberration`` metric.
    """
    frame = pd.DataFrame(design).copy()
    for label in ("Run", "RunOrder", "Block", "run", "run_order", "block"):
        if label in frame.columns:
            frame = frame.drop(columns=[label])

    n_runs, n_factors = frame.shape
    if n_factors == 0:
        raise ValueError("Design has no factor columns.")
    if n_runs < 2:
        raise ValueError(f"Design has {n_runs} run(s); at least 2 are needed to form a pair.")

    coded = _coded_pm1(frame)
    max_order = min(n_factors, MAX_PATTERN_LENGTH)

    # Raw moments of the inner product h, formed from the similarity histogram
    # so the arithmetic stays exact: h = 2*delta - m.
    histogram = _similarity_histogram(coded)
    raw_moments = []
    for order in range(max_order + 1):
        # Ordered pairs including the diagonal, where h_ii = m.
        off_diagonal = 2 * sum(count * (2 * d - n_factors) ** order for d, count in enumerate(histogram) if count)
        raw_moments.append(off_diagonal + n_runs * n_factors**order)

    pattern = _pattern_from_moments(raw_moments, n_runs, n_factors, max_order)

    # Lower bounds: replace every sum of squares by its minimum, n^2 * E_u(m).
    even_counts = _even_tuple_counts(n_factors, max_order)
    bound_moments = [n_runs**2 * count for count in even_counts]
    bounds = _pattern_from_moments(bound_moments, n_runs, n_factors, max_order)

    strength = 0
    for observed, bound in zip(pattern, bounds, strict=True):
        if observed != bound:
            break
        strength += 1

    # Only the attained bounds and the first missed one are meaningful; see
    # ``MomentAberrationResult.lower_bounds``.
    informative = bounds[: min(strength + 1, max_order)]

    return MomentAberrationResult(
        pattern=[float(k) for k in pattern],
        lower_bounds=[float(k) for k in informative],
        strength=strength,
        resolution=strength + 1,
        n_runs=n_runs,
        n_factors=n_factors,
        is_orthogonal_array=strength >= 2,
        truncated=n_factors > MAX_PATTERN_LENGTH,
        exact_pattern=pattern,
        exact_lower_bounds=informative,
    )
