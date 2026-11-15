# (c) Kevin Dunn, 2010-2026. MIT License.

"""The setting of largest composite desirability, proven, when every model and every constraint is a straight line.

A response that is a straight line in the factors gives Derringer-Suich ramps that are
straight lines too, and the logarithm of the composite desirability is then a concave
function of those ramps. Over a region with straight constraints, factors that are left
out or used at no less than a minimum, and limits on how many factors are used, the
best setting solves a mixed-integer programme with that concave objective.

Each logarithm is replaced by tangents, which lie on or above it. What is left is a
mixed-integer *linear* programme, which HiGHS solves to proven optimality: its solution
says which factors to use, and its objective is an upper bound on the desirability of
every setting in the region. With those factors in use the best setting is a linear
programme's, refined with tangents where its own solution sits until bound and
desirability agree (Kelley's cutting planes, 1960). The tangents at that setting then
go to the mixed-integer programme, whose bound for this set of factors is now exact,
and it is solved again: it returns the same set, which proves the setting, or a set
that may still be better. No setting of the region scores higher than a proven one,
which a search from starting points cannot say.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: A ramp below this is taken as a desirability of 0, which this search does not look for.
_LOWEST_RAMP = 1e-3
#: Where the tangents to the logarithm of a ramp touch it, before any are added. A tangent at a low ramp is
#: steep, and steep rows beside flat ones are hard on the solver, so the low ones are added only where needed.
_FIRST_TANGENTS = np.geomspace(0.05, 1.0, 10)
#: The bound and the setting found agree when their composite desirabilities differ by less than this share.
#: HiGHS meets its constraints to about 1e-7, so a closer agreement is not to be had from it.
_AGREEMENT = 1e-6
#: The agreement asked of the best setting of one set of factors, whose tangents the bound is then built on.
_AGREEMENT_IN_SET = 1e-7
#: Two tangents whose points differ by less than this share are taken as the same tangent: the second would
#: lower the bound by less than 1e-8 of it, and HiGHS can hardly tell their two rows apart.
_SAME_TANGENT = 1e-4
#: The most rounds of tangents, and the seconds the search may take; one programme is given what is left of them.
_ROUNDS = 30
_SECONDS = 20.0
#: The status :func:`scipy.optimize.milp` gives a programme that HiGHS could not solve ("Solve error").
_SOLVE_ERROR = 4
#: How far inside the limits of an ``"in_range"`` goal the setting is kept, as a share of their width. A setting
#: on a limit is beyond it by rounding as often as not, and the goal then scores 0. HiGHS meets a row to about
#: 1e-7, so this leaves the setting inside by as much again. The bound is taken without it.
_INSIDE = 2e-7


@dataclass
class StraightRegion:
    """A region whose constraints are straight lines, over the factors of the models, in their order.

    Parameters
    ----------
    bounds : list[tuple[float, float]]
        The lowest and the highest setting of each factor.
    rows : list[tuple[np.ndarray, float]]
        Each ``(a, b)`` is the constraint ``a @ x <= b``.
    components : list[int]
        The factors that are proportions of a blend: they sum to 1. Empty without a blend.
    gaps : list[tuple[int, float, float]]
        ``(column, zero, minimum)`` for each factor that is left out (at ``zero``) or used
        at no less than ``minimum``.
    counts : list[tuple[list[int], int, int]]
        For each limit on how many factors are used: the positions in ``gaps`` of the
        factors it counts that can be left out, and the fewest and the most of those
        that may be used (the limit, less the factors it counts that are always used).
    """

    bounds: list[tuple[float, float]]
    rows: list[tuple[np.ndarray, float]] = field(default_factory=list)
    components: list[int] = field(default_factory=list)
    gaps: list[tuple[int, float, float]] = field(default_factory=list)
    counts: list[tuple[list[int], int, int]] = field(default_factory=list)


@dataclass
class ExactOptimum:
    """The setting found, and the composite desirability that no setting of the region exceeds."""

    x: np.ndarray
    #: The setting's own composite desirability.
    score: float
    #: The composite desirability no setting of the region exceeds, when ``holds``.
    bound: float
    #: Whether HiGHS solved the programme the bound is from to the end. A bound it did not is no bound.
    holds: bool

    def proves(self, score: float) -> bool:
        """Whether a setting that scores ``score`` is proven the best: it reaches the bound, which holds."""
        return self.holds and self.bound - score <= _AGREEMENT * self.bound

    @property
    def proven(self) -> bool:
        """Whether the setting found is proven the best of the region."""
        return self.proves(self.score)


def _ramps(
    goal: dict[str, Any], slope: np.ndarray, intercept: float
) -> tuple[list[tuple[np.ndarray, float, float]], list[tuple[np.ndarray, float]]]:
    """Return a goal's ramps and its limits, for the response ``slope @ x + intercept``.

    A ramp is ``(s, c, exponent)``: the desirability is ``min(1, s @ x + c) ** exponent``,
    the smaller of the two for a target. A limit is a row ``(a, b)`` of ``a @ x <= b``,
    in units of the width of the range, which an ``"in_range"`` goal has in place of a ramp.
    """
    low, high = float(goal["low"]), float(goal["high"])
    weight = float(goal.get("weight", 1.0))
    if goal["goal"] == "in_range":
        width = high - low
        return [], [(slope / width, (high - intercept) / width), (-slope / width, (intercept - low) / width)]
    rising = (slope / (high - low), (intercept - low) / (high - low), weight)
    falling = (-slope / (high - low), (high - intercept) / (high - low), weight)
    if goal["goal"] == "maximize":
        return [rising], []
    if goal["goal"] == "minimize":
        return [falling], []
    target = float(goal["target"])
    below = (slope / (target - low), (intercept - low) / (target - low), weight)
    above = (-slope / (high - target), (high - intercept) / (high - target), float(goal.get("weight_high", weight)))
    return [below, above], []


class _Programme:
    """The mixed-integer linear programme: its variables are the factors, the factors in use, the ramps and the scores.

    A score is the logarithm of one response's desirability, held under the tangents to
    it. The programme maximises the scores' weighted sum, the logarithm of the composite.
    """

    def __init__(
        self,
        region: StraightRegion,
        models: Sequence[tuple[np.ndarray, float]],
        goals: Sequence[dict[str, Any]],
        importances: Sequence[float] | None,
    ) -> None:
        self.k, self.n_gaps, self.gaps = len(region.bounds), len(region.gaps), region.gaps
        in_range: list[tuple[np.ndarray, float]] = []
        self.ramps: list[tuple[np.ndarray, float, float, int]] = []  # slope, constant, exponent, its score
        shares: list[float] = []
        for i, ((slope, intercept), goal) in enumerate(zip(models, goals, strict=True)):
            ramps, rows = _ramps(goal, slope, intercept)
            in_range += rows
            if ramps:
                self.ramps += [(s, c, exponent, len(shares)) for s, c, exponent in ramps]
                shares.append(1.0 if importances is None else float(importances[i]))
        self.first_ramp = self.k + self.n_gaps
        self.first_score = self.first_ramp + len(self.ramps)
        self.n = self.first_score + len(shares)
        self.rows: list[np.ndarray] = []
        self.low: list[float] = []
        self.high: list[float] = []
        for a, b in region.rows:
            self._add({j: a[j] for j in range(self.k)}, -np.inf, b)
        self.kept_inside = range(len(self.rows), len(self.rows) + len(in_range))  # the rows of the in-range limits
        for a, b in in_range:
            self._add({j: a[j] for j in range(self.k)}, -np.inf, b - _INSIDE)
        self._add_region(region)
        for q, (slope, constant, _exponent, _score) in enumerate(self.ramps):
            self._add({**{j: -slope[j] for j in range(self.k)}, self.first_ramp + q: 1.0}, -np.inf, constant)
        self.cost = np.zeros(self.n)
        self.cost[self.first_score :] = -np.array(shares) / sum(shares)
        self.touches: list[list[float]] = [[] for _ in self.ramps]
        for q in range(len(self.ramps)):
            for touch in _FIRST_TANGENTS:
                self._add_tangent(q, float(touch))

    def _add_region(self, region: StraightRegion) -> None:
        """Add the blend's sum, the factors left out or in use, and the limits on how many are, and set the bounds."""
        if region.components:
            self._add(dict.fromkeys(region.components, 1.0), 1.0, 1.0)
        lower, upper = (np.array(side, dtype=float) for side in zip(*region.bounds, strict=True))
        can_use = np.ones(self.n_gaps)
        for p, (j, zero, minimum) in enumerate(region.gaps):
            lower[j] = zero
            can_use[p] = float(upper[j] >= minimum)  # a search box that ends below the minimum leaves the factor out
            self._add({j: 1.0, self.k + p: -(upper[j] - zero)}, -np.inf, zero)  # at zero unless it is in use
            self._add({j: 1.0, self.k + p: -(minimum - zero)}, zero, np.inf)  # at its minimum or above when it is
        self.lower, self.upper = lower, upper
        for members, fewest, most in region.counts:
            self._add({self.k + p: 1.0 for p in members}, fewest, most)
        n_ramps, n_scores = len(self.ramps), self.n - self.first_score
        deepest = max((exponent for *_, exponent, _score in self.ramps), default=1.0) * np.log(_LOWEST_RAMP)
        self.bounds = Bounds(
            np.concatenate([lower, np.zeros(self.n_gaps), np.full(n_ramps, _LOWEST_RAMP), np.full(n_scores, deepest)]),
            np.concatenate([upper, can_use, np.ones(n_ramps), np.zeros(n_scores)]),
        )
        self.integrality = np.zeros(self.n)
        self.integrality[self.k : self.first_ramp] = 1

    def _add(self, coefficients: dict[int, float], low: float, high: float) -> None:
        row = np.zeros(self.n)
        for column, value in coefficients.items():
            row[column] = value
        self.rows.append(row)
        self.low.append(low)
        self.high.append(high)

    def _add_tangent(self, q: int, touch: float) -> bool:
        """Hold the score of ramp ``q`` under the tangent to ``exponent * log(ramp)`` at ``ramp = touch``.

        A tangent next to one already there adds nothing but two rows that HiGHS can
        hardly tell apart, so it is not added; the return says whether it was.
        """
        if any(abs(touch - other) <= _SAME_TANGENT * other for other in self.touches[q]):
            return False
        self.touches[q].append(touch)
        exponent, score = self.ramps[q][2], self.ramps[q][3]
        self._add(
            {self.first_score + score: 1.0, self.first_ramp + q: -exponent / touch},
            -np.inf,
            exponent * (np.log(touch) - 1.0),
        )
        return True

    def add_tangents(self, x: np.ndarray) -> bool:
        """Add the tangents at each ramp's value at the setting ``x``; return whether any was new."""
        values = np.clip([slope @ x + constant for slope, constant, _exponent, _score in self.ramps], _LOWEST_RAMP, 1.0)
        added = False
        for q, value in enumerate(values):
            added = self._add_tangent(q, float(value)) or added
        return added

    def solve(
        self, used: np.ndarray | None = None, margin: float = _INSIDE, seconds: float = _SECONDS
    ) -> tuple[np.ndarray, np.ndarray, float, bool] | None:
        """Return the programme's setting, the factors it uses, its bound on the desirability, and whether it is solved.

        With ``used``, which says for each factor that can be left out whether it is in
        use, the programme is the linear one over that set of factors. ``margin`` is how
        far inside the limits of the in-range goals the setting is kept; with 0 the bound
        is the one over the whole region. ``seconds`` is the time HiGHS may take: a
        programme it does not finish in them is not solved, and its bound proves nothing.
        """
        from process_improve.experiments.designs_omars_ilp import _run_milp  # noqa: PLC0415

        bounds, integrality = self.bounds, self.integrality
        if used is not None:
            lower, upper = bounds.lb.copy(), bounds.ub.copy()
            lower[self.k : self.k + self.n_gaps] = upper[self.k : self.k + self.n_gaps] = used
            bounds, integrality = Bounds(lower, upper), np.zeros(self.n)
        high = np.array(self.high)
        high[list(self.kept_inside)] += _INSIDE - margin
        rows = [LinearConstraint(np.array(self.rows), self.low, high)]
        options = {"time_limit": seconds, "mip_rel_gap": 0.0}
        solved = _run_milp(self.cost, rows, options, integrality=integrality, bounds=bounds)
        if solved.x is None and solved.status == _SOLVE_ERROR:
            # HiGHS's presolve fails on a programme now and then that HiGHS solves without it.
            solved = _run_milp(self.cost, rows, {**options, "presolve": False}, integrality=integrality, bounds=bounds)
        if solved.x is None:
            return None
        # HiGHS may stop within 1e-6 of the best objective: its own bound on that objective is the one that holds.
        dual = getattr(solved, "mip_dual_bound", None)  # None for a linear programme
        lowest = float(solved.fun) if dual is None else min(float(solved.fun), float(dual))
        in_use = np.round(solved.x[self.k : self.k + self.n_gaps])
        # HiGHS meets a bound to about 1e-7: a factor left out is put at its zero, one in use at its minimum or above.
        x = np.clip(solved.x[: self.k], self.lower, self.upper)
        for p, (j, zero, minimum) in enumerate(self.gaps):
            x[j] = max(x[j], minimum) if in_use[p] else zero
        return x, in_use, float(np.exp(-lowest)), solved.status == 0

    def best_with(self, used: np.ndarray, start: np.ndarray, desirability: Callable[[np.ndarray], float]) -> np.ndarray:
        """Return the setting of largest desirability that uses the factors ``used`` marks, from ``start`` on.

        The tangents this adds on the way are taken out again: only those at the setting
        returned are for the caller to keep, which leaves the programme's tangents apart
        from one another.
        """
        n_rows, n_touches = len(self.rows), [len(touches) for touches in self.touches]
        x = start
        for _ in range(_ROUNDS):
            if not self.add_tangents(x):
                break
            solved = self.solve(used)
            if solved is None:
                break
            x, bound = solved[0], solved[2]
            if bound - desirability(x) <= _AGREEMENT_IN_SET * bound:
                break
        del self.rows[n_rows:], self.low[n_rows:], self.high[n_rows:]
        for touches, n in zip(self.touches, n_touches, strict=True):
            del touches[n:]
        return x


def exact_optimum(
    region: StraightRegion,
    models: Sequence[tuple[np.ndarray, float]],
    goals: Sequence[dict[str, Any]],
    importances: Sequence[float] | None,
    desirability: Callable[[np.ndarray], float],
) -> ExactOptimum | None:
    """Return the setting of largest composite desirability in ``region``, or None when no setting scores above 0.

    Parameters
    ----------
    region : StraightRegion
        Where to look.
    models : sequence of tuple[np.ndarray, float]
        The slopes and the intercept of each response: ``y = slopes @ x + intercept``.
    goals : sequence of dict
        The goal of each response, as :func:`~process_improve.experiments._desirability.individual_desirability`
        takes it.
    importances : sequence of float or None
        The relative importance of each response in the composite; equal when None.
    desirability : callable
        The composite desirability of a setting, by which the setting found is judged.

    Returns
    -------
    ExactOptimum or None
        The best setting found, its composite desirability, and the composite desirability
        that no setting of the region exceeds; the setting is proven when the two agree.
        None when the programme has no solution: the region is empty, or every setting
        leaves a desirability at 0.
    """
    programme = _Programme(region, models, goals, importances)
    best_x, best_score, bound, holds = None, 0.0, 1.0, False
    began = time.monotonic()
    for _ in range(_ROUNDS):
        left = _SECONDS - (time.monotonic() - began)
        solved = programme.solve(seconds=max(left, 1.0))
        if solved is None:
            break
        start, used, bound, holds = solved
        x = programme.best_with(used, start, desirability)
        score = desirability(x)
        if best_x is None or score > best_score:
            best_x, best_score = x, score
        agree = bound - best_score <= _AGREEMENT * bound
        if agree or not holds or left <= 0 or not programme.add_tangents(x):
            break
    if best_x is None:
        return None
    if holds and len(programme.kept_inside):
        # The setting was kept inside the limits of the in-range goals: the bound is taken of the region up to them.
        whole = programme.solve(margin=0.0)
        holds = whole is not None and whole[3]
        bound = bound if whole is None else whole[2]
    return ExactOptimum(best_x, best_score, bound, holds)
