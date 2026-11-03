"""
Optimal inversion: the inputs a latent-variable model vouches for, at the least cost or closest to a target, under the
rules a product must meet.

:meth:`PLS.invert <process_improve.multivariate.methods.PLS.invert>` returns the minimum-norm inputs for a target, and
the null space along which every other answer lies. Jaeckle and MacGregor then picked a point in that window by hand,
"such as the most economical one". This module makes that choice an optimisation, and proves it optimal:

    minimise    the cost of the inputs, or their distance to target qualities (in training standard deviations)
    subject to  each predicted quality within its window;
                linear rules on the inputs: a mixture summing to 1, a safety or legal limit on a component;
                each input within its bounds, or held at a fixed value;
                Hotelling's T² and every block's SPE within their limits, so the answer looks like the data the model
                was built from: a latent-variable model's counterpart of "inside the tested region";
                at most k inputs used from a group, each used at a minimum amount or not at all.

A fitted PLS or TPLS model is a chain of centrings, scalings and projections, so its scores, predictions and residuals
are affine in the inputs. :func:`affine_model` reads those maps off the model's own ``diagnose`` once, and checks them
there, so the search is the model, not an approximation of it. Everything is then linear except T² and SPE, both
convex: T² a sum of squares of the scores, SPE the length of the residual. Without the counting rules the problem is
convex and one continuous search finds its global optimum. With them, outer approximation (Duran and Grossmann, 1986)
alternates a mixed-integer linear master (HiGHS), which chooses the inputs used, with that continuous search for each
choice.

Every answer carries the shadow price of each rule: how much the objective would change per unit the rule is moved,
read from the optimality conditions. :func:`inversion_sensitivity` checks them by solving again, traces the objective
as one limit moves, prices one more input per group, and finds how far each input's cost can move before the answer
changes.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

_ROW_TOLERANCE = 1e-6  # a linear rule met to within this, relative to its scale, is met
_LIMIT_TOLERANCE = 1e-4  # T² or an SPE within this fraction above its limit is within it
_ACTIVE = 1e-5  # a rule within this (relative) of its bound is binding
_MAX_ROUNDS = 200  # outer-approximation rounds before the search reports it is not proven optimal
_FIRST_TRY_ITERATIONS = 100  # SLSQP from the model centre settles in tens of iterations when an answer exists
_RESTARTS = 3  # SLSQP restarts from where it stopped, while the optimality conditions do not hold there
_KKT_TOLERANCE = 1e-6  # the optimality conditions' relative residual below which a point is the optimum
_PROOF_TOLERANCE = 1e-6  # a tangent-plane bound within this fraction of the answer's objective proves it

_NO_LINEAR = "No inputs within their bounds meet the linear rules and the windows, whatever their T² and SPE."
_NO_COUNT = "Without the counting rules inputs meet every rule, but none do with them."
_NOT_SETTLED = "The continuous search did not settle on a point that meets every rule."
_NO_OTHER = "No further answer leaves out {leave_out} of the counted inputs that each earlier answer uses."


# ---------------------------------------------------------------------------------------------------------------------
# The model as exact affine maps
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualBlock:
    """
    One block's residual, ``r = r0 + jacobian @ x``, whose length is its SPE.

    Attributes
    ----------
    name : str
        The block: ``"X"`` for PLS, ``"F/<group>"`` or ``"Z/<group>"`` for TPLS.
    r0 : np.ndarray
        The residual at ``x = 0``, in the model's scaled units.
    jacobian : np.ndarray
        The residual's change per unit of each input, shape ``(block width, n_inputs)``.
    limit : float
        The SPE limit at the model's confidence level.
    """

    name: str
    r0: np.ndarray
    jacobian: np.ndarray
    limit: float


@dataclass(frozen=True)
class AffineModel:
    """
    A fitted latent-variable model read as exact affine maps of one vector of inputs, in their original units.

    Attributes
    ----------
    names : tuple of str
        The inputs, in the order of the maps' columns.
    y_names : tuple of str
        The predicted qualities.
    groups : dict of str to tuple of str
        Natural groups of inputs (a TPLS model's materials, whose fractions sum to 1); empty for PLS.
    y0, y_map : np.ndarray
        Predictions: ``y = y0 + y_map @ x``.
    t0, t_map : np.ndarray
        Scores: ``t = t0 + t_map @ x``.
    score_sd : np.ndarray
        Each score's standard deviation, as the model's Hotelling's T² divides by it.
    t2_limit : float
        Hotelling's T² limit at the confidence level read.
    blocks : tuple of ResidualBlock
        Each block's residual, whose length is its SPE.
    y_scale : np.ndarray
        Each quality's training standard deviation: the unit of a distance to a target.
    centre, spread : np.ndarray
        Each input's training mean and standard deviation (1 where unknown): the start of a search and its scale.
    conf_level : float
        The confidence level the limits were read at.
    """

    names: tuple[str, ...]
    y_names: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    y0: np.ndarray
    y_map: np.ndarray
    t0: np.ndarray
    t_map: np.ndarray
    score_sd: np.ndarray
    t2_limit: float
    blocks: tuple[ResidualBlock, ...]
    y_scale: np.ndarray
    centre: np.ndarray
    spread: np.ndarray
    conf_level: float = 0.95

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted qualities of one input vector."""
        return self.y0 + self.y_map @ x

    def scores(self, x: np.ndarray) -> np.ndarray:
        """Return the scores of one input vector."""
        return self.t0 + self.t_map @ x

    def hotellings_t2(self, x: np.ndarray) -> float:
        """Return Hotelling's T² of one input vector."""
        return float(np.sum((self.scores(x) / self.score_sd) ** 2))

    def spe(self, x: np.ndarray) -> np.ndarray:
        """Return each block's SPE for one input vector."""
        return np.array([float(np.linalg.norm(b.r0 + b.jacobian @ x)) for b in self.blocks])

    def to_dict(self) -> dict:
        """
        Return the maps as plain JSON: lists and numbers, to store beside a project and search later without the model.

        Returns
        -------
        dict
            Every field; :meth:`from_dict` reads it back.
        """
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        out["groups"] = {g: list(names) for g, names in self.groups.items()}
        out["blocks"] = [{"name": b.name, "r0": b.r0.tolist(), "jacobian": b.jacobian.tolist(), "limit": b.limit}
                         for b in self.blocks]  # fmt: skip
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in out.items()}

    @classmethod
    def from_dict(cls, values: Mapping) -> AffineModel:
        """
        Read the maps back from :meth:`to_dict`.

        Parameters
        ----------
        values : dict
            As :meth:`to_dict` wrote it.

        Returns
        -------
        AffineModel
            The same maps, as arrays.
        """
        names = ("y0", "y_map", "t0", "t_map", "score_sd", "y_scale", "centre", "spread")
        arrays = {k: np.asarray(values[k], dtype=float) for k in names}
        blocks = tuple(
            ResidualBlock(
                b["name"], np.asarray(b["r0"], dtype=float), np.asarray(b["jacobian"], dtype=float), b["limit"]
            )
            for b in values["blocks"]
        )
        return cls(
            names=tuple(values["names"]), y_names=tuple(values["y_names"]),
            groups={g: tuple(n) for g, n in values["groups"].items()}, t2_limit=float(values["t2_limit"]),
            blocks=blocks, conf_level=float(values["conf_level"]), **arrays,
        )  # fmt: skip


@dataclass(frozen=True)
class Probe:
    """
    What a model says of rows of its inputs, and what is needed to read it as affine maps.

    Each model class that can be inverted builds its own, in ``_inversion_probe(conf_level)``: it knows its inputs,
    its blocks and its limits. This module then reads every model the same way.
    """

    names: tuple[str, ...]
    y_names: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    block_names: tuple[str, ...]
    block_limits: tuple[float, ...]
    score_sd: np.ndarray
    t2_limit: float
    y_scale: np.ndarray
    centre: np.ndarray
    spread: np.ndarray
    # rows -> (scores, predictions, [residual per block], T², SPE per block)
    diagnose: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]]


def finite_or(values: object, default: float) -> np.ndarray:
    """Return the values as floats, ``default`` where one is not finite or is 0: a spread to divide by or step along."""
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array) & (array != 0), array, default)


def affine_model(model: BaseEstimator, conf_level: float = 0.95) -> AffineModel:
    """
    Read a fitted PLS or TPLS model as exact affine maps of one input vector, checked against its ``diagnose``.

    The scores, predictions and residuals of a latent-variable model are affine in its inputs. One ``diagnose`` call,
    at the model's centre and one step along each input, gives every map exactly. A second call, at random points, is
    the check: the maps must reproduce the model's own predictions, Hotelling's T² and SPE there.

    Parameters
    ----------
    model : PLS or TPLS
        A fitted model. For TPLS the inputs are the formulas' fractions (each group's lots) and the process conditions.
    conf_level : float, default 0.95
        The confidence level of the T² and SPE limits.

    Returns
    -------
    AffineModel
        The maps, the limits, and each input's training mean and spread.

    Raises
    ------
    TypeError
        The model is not a PLS or TPLS model.
    ValueError
        The maps do not reproduce the model's ``diagnose``: the model is not one this reads exactly.
    """
    if not hasattr(model, "_inversion_probe"):
        msg = f"Optimal inversion reads PLS and TPLS models, not {type(model).__name__}."
        raise TypeError(msg)
    probe = model._inversion_probe(conf_level)
    centre, spread = probe.centre, probe.spread
    scores, y, residuals, _, _ = probe.diagnose(np.vstack([centre, centre + np.diag(spread)]))

    def read(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slope = ((values[1:] - values[0]) / spread[:, None]).T
        return values[0] - slope @ centre, slope

    t0, t_map = read(scores)
    y0, y_map = read(y)
    blocks = tuple(
        ResidualBlock(name, *read(r), limit)
        for name, r, limit in zip(probe.block_names, residuals, probe.block_limits, strict=True)
    )
    affine = AffineModel(
        names=probe.names, y_names=probe.y_names, groups=probe.groups, y0=y0, y_map=y_map, t0=t0, t_map=t_map,
        score_sd=probe.score_sd, t2_limit=probe.t2_limit, blocks=blocks, y_scale=probe.y_scale, centre=centre,
        spread=spread, conf_level=conf_level,
    )  # fmt: skip
    _check_affine(affine, probe)
    return affine


def _check_affine(affine: AffineModel, probe: Probe) -> None:
    """Refuse maps that do not reproduce the model's own predictions, T² and SPE at random points."""
    rng = np.random.default_rng(0)  # pragma: no mutate - any points do; fixed so a failure repeats
    points = affine.centre + rng.normal(size=(3, len(affine.names))) * affine.spread
    _, y, _, t2, spe = probe.diagnose(points)
    mine_y = np.array([affine.predict(p) for p in points])
    mine_t2 = np.array([affine.hotellings_t2(p) for p in points])
    mine_spe = np.array([affine.spe(p) for p in points])
    close = np.allclose(mine_y, y, rtol=1e-6, atol=1e-8 * (1 + np.abs(y).max()))
    close = (
        close and np.allclose(mine_t2, t2, rtol=1e-6, atol=1e-8) and np.allclose(mine_spe, spe, rtol=1e-6, atol=1e-8)
    )
    if not close:
        msg = "The model could not be read as affine maps: its predictions, T² or SPE differ from its diagnose()."
        raise ValueError(msg)


# ---------------------------------------------------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearRule:
    """
    A linear rule on the inputs: ``lower <= sum(coefficients[name] * x[name]) <= upper``.

    A mixture that sums to 1, a component's content below its legal limit (each ingredient's content as its
    coefficient), or a ratio held between two inputs, are each one rule.

    Attributes
    ----------
    name : str
        How the rule is reported, and how :func:`inversion_sensitivity` names it.
    coefficients : dict of str to float
        Each input's coefficient; inputs not named have 0.
    lower, upper : float or None
        The bounds; None for none.
    """

    name: str
    coefficients: Mapping[str, float]
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class InversionRules:
    """
    What the inputs must meet, and what to minimise.

    Give ``costs`` or ``targets``, not both. To keep a quality near its target while minimising cost, give it a
    window instead.

    Attributes
    ----------
    costs : dict of str to float
        Each input's cost per unit: minimise their sum. Inputs not named cost nothing.
    targets : dict of str to float
        Target qualities: minimise the summed distance to them, each in its training standard deviations.
    windows : dict of str to (float or None, float or None)
        Each quality's predicted value between a low and a high bound; None for no bound.
    bounds : dict of str to (float or None, float or None)
        Each input's lower and upper bound: its extreme limits, the values it can take. A TPLS formula's fractions
        are 0 to 1 unless given. Give every other input both: the T² and SPE limits alone let an answer set an input
        to a value it cannot take, such as a negative amount. An input without both is deprecated since 2.4.0 (a
        ``FutureWarning`` names it) and an error from 3.0.
    fixed : dict of str to float
        Inputs held at a value.
    rules : tuple of LinearRule
        Linear rules on the inputs: mixture totals, safety and legal limits.
    closure : bool
        For TPLS, each group of lots sums to 1, as formulas do.
    groups : dict of str to list of str
        Groups of inputs for the counting rules, over and above the model's own (a TPLS model's materials).
    max_count : dict of str to int
        The most inputs a group may use.
    min_amount : dict of str to float
        The least amount of an input of the group, when used.
    conf_level : float
        The confidence level of the T² and SPE limits.
    t2_limit : float or None
        A T² limit in place of the one at ``conf_level``.
    spe_limits : dict of str to float
        SPE limits in place of those at ``conf_level``, by block name (``"X"``, ``"F/Group1"``).
    """

    costs: Mapping[str, float] = field(default_factory=dict)
    targets: Mapping[str, float] = field(default_factory=dict)
    windows: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    bounds: Mapping[str, tuple[float | None, float | None]] = field(default_factory=dict)
    fixed: Mapping[str, float] = field(default_factory=dict)
    rules: tuple[LinearRule, ...] = ()
    closure: bool = True
    groups: Mapping[str, list[str]] = field(default_factory=dict)
    max_count: Mapping[str, int] = field(default_factory=dict)
    min_amount: Mapping[str, float] = field(default_factory=dict)
    conf_level: float = 0.95
    t2_limit: float | None = None
    spe_limits: Mapping[str, float] = field(default_factory=dict)


def _all_groups(affine: AffineModel, rules: InversionRules) -> dict[str, tuple[str, ...]]:
    return dict(affine.groups) | {g: tuple(names) for g, names in rules.groups.items()}


def _is_bound_pair(pair: object) -> bool:
    if not (isinstance(pair, tuple | list) and len(pair) == 2):
        return False
    low, high = pair
    numbers = all(b is None or (isinstance(b, int | float) and np.isfinite(b)) for b in pair)
    return numbers and (low is None or high is None or low <= high)


def _unknown_names(affine: AffineModel, rules: InversionRules) -> list[str]:
    inputs, qualities = set(affine.names), set(affine.y_names)
    groups = _all_groups(affine, rules)
    problems = [f"{q!r} is not a quality of the model" for q in [*rules.targets, *rules.windows] if q not in qualities]
    named = [*rules.costs, *rules.bounds, *rules.fixed, *(n for r in rules.rules for n in r.coefficients)]
    named += [n for names in rules.groups.values() for n in names]
    problems += [f"{n!r} is not an input of the model" for n in dict.fromkeys(named) if n not in inputs]
    problems += [f"{g!r} is not a group" for g in [*rules.max_count, *rules.min_amount] if g not in groups]
    block_names = {b.name for b in affine.blocks}
    return problems + [f"{b!r} is not a residual block" for b in rules.spe_limits if b not in block_names]


def _bad_values(rules: InversionRules) -> list[str]:
    problems = [f"{q}: a window is (low, high), numbers or None, low below high" for q, w in rules.windows.items()
                if not _is_bound_pair(w)]  # fmt: skip
    problems += [f"{n}: bounds are (low, high), numbers or None, low below high" for n, b in rules.bounds.items()
                 if not _is_bound_pair(b)]  # fmt: skip
    problems += [f"rule {r.name!r}: give a lower or an upper bound, lower below upper" for r in rules.rules
                 if not _is_bound_pair((r.lower, r.upper)) or (r.lower is None and r.upper is None)]  # fmt: skip
    problems += [f"{g}: allow at least one input, not {k!r}" for g, k in rules.max_count.items()
                 if not (isinstance(k, int) and k >= 1)]  # fmt: skip
    problems += [f"{g}: a minimum amount is above 0, not {m!r}" for g, m in rules.min_amount.items()
                 if not (isinstance(m, int | float) and m > 0)]  # fmt: skip
    if bool(rules.costs) == bool(rules.targets):
        problems.append("give costs or targets (one of them) as the objective; to hold a quality near a target "
                        "while minimising cost, give it a window")  # fmt: skip
    return problems


def check_rules(affine: AffineModel, rules: InversionRules) -> None:
    """
    Refuse rules that name what the model does not have, or that cannot be read.

    Parameters
    ----------
    affine : AffineModel
        The model, from :func:`affine_model`.
    rules : InversionRules
        The rules.

    Raises
    ------
    ValueError
        Naming every problem found.
    """
    problems = _unknown_names(affine, rules) + _bad_values(rules)
    if problems:
        msg = "The inversion rules cannot be used: " + "; ".join(problems) + "."
        raise ValueError(msg)


# ---------------------------------------------------------------------------------------------------------------------
# The problem: variables [x (inputs), u (a switch per counted input), d (a distance per target)]
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Limit:
    """A convex limit, ``value(x) <= limit``, with its gradient."""

    name: str
    value: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    limit: float


def _limits(affine: AffineModel, rules: InversionRules) -> list[_Limit]:
    """
    T² and each block's SPE, on arrays held once: the search calls them thousands of times.

    SPE stays a length, as the model reports it, not its square. The master's tangent plane of a squared limit, taken
    at a point rho times the limit away (as a length), sits (1 + rho²) / (2 rho) times the limit out, not at it. The
    box rows let rho reach the square root of the number of directions: one per component for T², one per input for
    SPE, so a squared SPE gives loose planes. Over 32 test problems a squared SPE took 28% more rounds; T² written as
    its square root changed nothing that held across them, so T² stays a sum of squares.
    """
    sd2 = affine.score_sd**2
    t2_limit = rules.t2_limit if rules.t2_limit is not None else affine.t2_limit

    def t2(x: np.ndarray) -> float:
        t = affine.t0 + affine.t_map @ x
        return float(t @ (t / sd2))

    def t2_gradient(x: np.ndarray) -> np.ndarray:
        return 2 * (((affine.t0 + affine.t_map @ x) / sd2) @ affine.t_map)

    out = [_Limit("T²", t2, t2_gradient, float(t2_limit))]
    for block in affine.blocks:

        def value(x: np.ndarray, b: ResidualBlock = block) -> float:
            return float(np.linalg.norm(b.r0 + b.jacobian @ x))

        def gradient(x: np.ndarray, b: ResidualBlock = block) -> np.ndarray:
            r = b.r0 + b.jacobian @ x
            norm = np.linalg.norm(r)
            return (r @ b.jacobian) / norm if norm > 0 else np.zeros(b.jacobian.shape[1])

        limit = rules.spe_limits.get(block.name, block.limit)
        out.append(_Limit(f"SPE {block.name}", value, gradient, float(limit)))
    return out


@dataclass
class _Row:
    """One linear row, ``low <= a @ z <= high``, and how it is reported."""

    a: np.ndarray
    low: float
    high: float
    name: str
    kind: str  # rule, closure, window, target, switch, count, box, cut, exclude, different
    shift: float = 0.0  # added to the row's value and bounds to report it in its own units (a window's y0)


class _Problem:
    """The variables, the linear rows, the objective, and T² and SPE as convex limits."""

    def __init__(self, affine: AffineModel, rules: InversionRules) -> None:
        check_rules(affine, rules)
        self.affine, self.rules = affine, rules
        self.names = list(affine.names)
        self.n = len(self.names)
        self.bounds = self._bounds()
        groups = _all_groups(affine, rules)
        self.counted = {g: groups[g] for g in dict.fromkeys([*rules.max_count, *rules.min_amount])}
        self.switched = [self.names.index(n) for names in self.counted.values() for n in names]
        self.group_of = {i: g for g, names in self.counted.items() for i in (self.names.index(n) for n in names)}
        self.targets = [k for k, q in enumerate(affine.y_names) if q in rules.targets]
        self.m = len(self.switched)
        self.width = self.n + self.m + len(self.targets)
        self.rows: list[_Row] = []
        self._linear_rules()
        self._windows_and_targets()
        self.core = len(self.rows)  # rows before this involve no switch: the continuous search takes them
        self._switches()
        self.objective = np.zeros(self.width)
        self.objective[: self.n] = [rules.costs.get(name, 0.0) for name in self.names]
        self.objective[self.n + self.m :] = 1.0
        self.limits = _limits(affine, rules)
        self.why = _NO_LINEAR
        # Every choice of switches solved, with its best point: a later search for alternatives starts from them, as
        # the master never offers a choice twice.
        self.evaluated: list[tuple[np.ndarray, float, tuple[int, ...]]] = []

    def _bounds(self) -> np.ndarray:
        """Each input's bounds: TPLS fractions 0 to 1 by default, then the rules' bounds, then fixed values."""
        fractions = {n for names in self.affine.groups.values() for n in names}
        bounds = np.array([[0.0, 1.0] if n in fractions else [-np.inf, np.inf] for n in self.names])
        for name, (low, high) in self.rules.bounds.items():
            bounds[self.names.index(name)] = [-np.inf if low is None else low, np.inf if high is None else high]
        for name, value in self.rules.fixed.items():
            bounds[self.names.index(name)] = [value, value]
        return bounds

    def add(self, row: _Row) -> None:
        """Add a linear row."""
        self.rows.append(row)

    def _vector(self, coefficients: Mapping[str, float]) -> np.ndarray:
        a = np.zeros(self.width)
        for name, c in coefficients.items():
            a[self.names.index(name)] = c
        return a

    def _linear_rules(self) -> None:
        if self.rules.closure:
            for group, names in self.affine.groups.items():
                self.add(_Row(self._vector(dict.fromkeys(names, 1.0)), 1.0, 1.0, f"{group} sums to 1", "closure"))
        for r in self.rules.rules:
            low, high = -np.inf if r.lower is None else r.lower, np.inf if r.upper is None else r.upper
            self.add(_Row(self._vector(r.coefficients), low, high, r.name, "rule"))

    def _windows_and_targets(self) -> None:
        y0, y_map = self.affine.y0, self.affine.y_map
        for name, (low, high) in self.rules.windows.items():
            k = self.affine.y_names.index(name)
            a = np.zeros(self.width)
            a[: self.n] = y_map[k]
            lo, hi = -np.inf if low is None else low - y0[k], np.inf if high is None else high - y0[k]
            self.add(_Row(a, lo, hi, name, "window", shift=float(y0[k])))
        for j, k in enumerate(self.targets):  # d_j >= |y_k - target| / sd_k
            name, scale = self.affine.y_names[k], self.affine.y_scale[k]
            for sign in (1.0, -1.0):
                a = np.zeros(self.width)
                a[: self.n], a[self.n + self.m + j] = sign * y_map[k] / scale, -1.0
                self.add(_Row(a, -np.inf, sign * (self.rules.targets[name] - y0[k]) / scale, name, "target"))

    def _switches(self) -> None:
        """Use an input only when it is switched on, then at its minimum or more; switch on at most k per group."""
        for k, i in enumerate(self.switched):
            u, minimum = self.n + k, self.rules.min_amount.get(self.group_of[i], 0.0)
            a = np.zeros(self.width)
            a[i], a[u] = 1.0, -self.bounds[i][1]
            self.add(_Row(a, -np.inf, 0.0, self.names[i], "switch"))
            if minimum > 0:
                a = np.zeros(self.width)
                a[i], a[u] = 1.0, -minimum
                self.add(_Row(a, 0.0, np.inf, self.names[i], "switch"))
        for group, most in self.rules.max_count.items():
            a = np.zeros(self.width)
            a[[self.n + k for k, i in enumerate(self.switched) if self.group_of[i] == group]] = 1.0
            self.add(_Row(a, 0.0, most, group, "count"))

    def check_counted(self) -> None:
        """Refuse a counted input whose bounds do not include 0 or have no upper end: 'unused' must be 0."""
        bad = [self.names[i] for i in self.switched if not (self.bounds[i][0] <= 0 <= self.bounds[i][1] < np.inf)]
        if bad:
            msg = f"A counted input must be allowed 0 and have an upper bound: {', '.join(bad[:5])}."
            raise ValueError(msg)

    # The continuous search, with the switches relaxed (on=None) or set.
    def keep(self) -> list[int]:
        """Return the indices of the continuous variables: x and d, no switches."""
        return [*range(self.n), *range(self.n + self.m, self.width)]

    def continuous_bounds(self, on: tuple[int, ...] | None) -> np.ndarray:
        """Return the bounds of x and d: an input switched off held at 0, one switched on at its minimum or more."""
        bounds = np.vstack([self.bounds, np.tile([0.0, np.inf], (len(self.targets), 1))])
        if on is None:
            return bounds
        for k, i in enumerate(self.switched):
            minimum = self.rules.min_amount.get(self.group_of[i], 0.0)
            bounds[i] = [max(minimum, self.bounds[i][0]), self.bounds[i][1]] if on[k] else [0.0, 0.0]
        return bounds

    def core_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the rows the continuous search keeps, over x and d."""
        keep = self.keep()
        if not self.core:
            return np.empty((0, len(keep))), np.empty(0), np.empty(0)
        rows = self.rows[: self.core]
        a = np.array([r.a for r in rows])[:, keep]
        return a, np.array([r.low for r in rows]), np.array([r.high for r in rows])

    def start(self, bounds: np.ndarray) -> np.ndarray:
        """Return the model's centre, within the bounds, and each target distance measured there."""
        x = np.clip(self.affine.centre, bounds[: self.n, 0], bounds[: self.n, 1])
        y = self.affine.predict(x)
        d = [abs(y[k] - self.rules.targets[self.affine.y_names[k]]) / self.affine.y_scale[k] for k in self.targets]
        return np.concatenate([x, d])

    def scale(self, bounds: np.ndarray) -> np.ndarray:
        """Each continuous variable's natural size: its range when finite, else its training spread."""
        span = bounds[:, 1] - bounds[:, 0]
        natural = np.concatenate([self.affine.spread, np.ones(len(self.targets))])
        return np.where(np.isfinite(span) & (span > 0), span, natural)


# ---------------------------------------------------------------------------------------------------------------------
# The continuous search
# ---------------------------------------------------------------------------------------------------------------------
def _linear_constraints(a: np.ndarray, low: np.ndarray, high: np.ndarray) -> list[dict]:
    """SLSQP's form of low <= a w <= high: an equality where the bounds meet, otherwise one inequality per bound."""
    equal = np.flatnonzero(low == high)
    lower = np.flatnonzero((low != high) & np.isfinite(low))
    upper = np.flatnonzero((low != high) & np.isfinite(high))
    return (
        [{"type": "eq", "fun": lambda w, r=r: a[r] @ w - low[r], "jac": lambda _w, r=r: a[r]} for r in equal]
        + [{"type": "ineq", "fun": lambda w, r=r: a[r] @ w - low[r], "jac": lambda _w, r=r: a[r]} for r in lower]
        + [{"type": "ineq", "fun": lambda w, r=r: high[r] - a[r] @ w, "jac": lambda _w, r=r: -a[r]} for r in upper]
    )


def _limit_constraints(limits: list[_Limit], n: int, extra: int, relax: bool = False) -> list[dict]:
    """
    T² and each SPE as SLSQP inequalities over [x, d] (and a trailing excess s when ``relax``:
    ``value <= limit * (1 + s)``).
    """
    tail = np.zeros(extra)
    out = []
    for lim in limits:
        if relax:
            out.append({"type": "ineq", "fun": lambda w, f=lim: f.limit * (1 + w[-1]) - f.value(w[:n]),
                        "jac": lambda w, f=lim: np.concatenate([-f.gradient(w[:n]), tail, [f.limit]])})  # fmt: skip
        else:
            out.append({"type": "ineq", "fun": lambda w, f=lim: f.limit - f.value(w[:n]),
                        "jac": lambda w, f=lim: np.concatenate([-f.gradient(w[:n]), tail])})  # fmt: skip
    return out


@dataclass(frozen=True)
class _Search:
    """One continuous search: a smooth objective with its gradient, a start, bounds, constraints, and a scale."""

    fun: Callable[[np.ndarray], float]
    jac: Callable[[np.ndarray], np.ndarray]
    start: np.ndarray
    bounds: np.ndarray
    constraints: list[dict]
    scale: np.ndarray


def _slsqp(search: _Search, maxiter: int) -> np.ndarray:
    """
    Run SLSQP over the variables free to move, each scaled by its natural size, and return the full point.

    A variable held fixed (an input switched off) is left out: SLSQP's cost grows with the cube of the variables.
    A fraction runs from 0 to 1 and a process condition can run to 1e5: unscaled, SLSQP's line search stops short.
    """
    bounds = search.bounds
    free = bounds[:, 1] > bounds[:, 0]
    scale = search.scale[free]
    point = np.clip(search.start, bounds[:, 0], bounds[:, 1])

    def full(v: np.ndarray) -> np.ndarray:
        point[free] = v * scale
        return point

    def scaled(c: dict) -> dict:
        return {"type": c["type"], "fun": lambda v: c["fun"](full(v)), "jac": lambda v: c["jac"](full(v))[free] * scale}

    found = optimize.minimize(
        lambda v: search.fun(full(v)), point[free] / scale, jac=lambda v: search.jac(full(v))[free] * scale,
        method="SLSQP", bounds=list(zip(bounds[free, 0] / scale, bounds[free, 1] / scale, strict=True)),
        constraints=[scaled(c) for c in search.constraints], options={"maxiter": maxiter, "ftol": 1e-12},
    )  # fmt: skip
    return np.clip(full(found.x).copy(), bounds[:, 0], bounds[:, 1])


def _linear_point(
    objective: np.ndarray, a: np.ndarray, low: np.ndarray, high: np.ndarray, bounds: np.ndarray
) -> np.ndarray | None:
    """
    Return a point meeting the linear rows within the bounds (HiGHS), or None when none does: proven at once.

    The point is the best of the linear rows alone, or, when that is unbounded (an input free in both directions
    the objective rewards), any point the rows allow: the start of a second search for the least excess.
    """
    constraints = optimize.LinearConstraint(a, low, high) if len(a) else ()
    box = optimize.Bounds(bounds[:, 0], bounds[:, 1])
    found = optimize.milp(objective, bounds=box, constraints=constraints)
    if found.status == 2:  # infeasible
        return None
    if found.x is None:  # unbounded: T² and SPE bound the inputs, but not the linear rows alone
        found = optimize.milp(np.zeros_like(objective), bounds=box, constraints=constraints)
    return np.clip(found.x, bounds[:, 0], bounds[:, 1])


def _meets(problem: _Problem, w: np.ndarray, rows: tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
    """Every linear row within its bounds, and T² and every SPE within their limits."""
    a, low, high = rows
    values = a @ w
    scale = 1 + np.abs(a) @ np.abs(w)
    rows_met = bool(np.all(values >= low - _ROW_TOLERANCE * scale) and np.all(values <= high + _ROW_TOLERANCE * scale))
    return rows_met and all(lim.value(w[: problem.n]) <= lim.limit * (1 + _LIMIT_TOLERANCE) for lim in problem.limits)


def _least_excess(problem: _Problem, rows: tuple, bounds: np.ndarray, start: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Minimise s with every limit relaxed to (1 + s) times itself and every linear row kept: always feasible.

    Returns
    -------
    (float, np.ndarray)
        s, above 0 when no point meets the linear rows inside the limits; and the point.
    """
    a, low, high = rows
    n, extra = problem.n, len(problem.targets)
    ratios = [lim.value(start[:n]) / lim.limit for lim in problem.limits]
    padded = np.hstack([a, np.zeros((len(a), 1))])
    unit = np.zeros(len(start) + 1)
    unit[-1] = 1.0
    search = _Search(
        fun=lambda v: v[-1], jac=lambda _v: unit, start=np.append(start, max(ratios) - 1),
        bounds=np.vstack([bounds, [-1.0, np.inf]]),
        constraints=_linear_constraints(padded, low, high) + _limit_constraints(problem.limits, n, extra, relax=True),
        scale=np.append(problem.scale(bounds), 1.0),
    )  # fmt: skip
    point = _slsqp(search, 1000)[:-1]
    return max(lim.value(point[:n]) / lim.limit for lim in problem.limits) - 1, point


def _proven_by_planes(problem: _Problem, w: np.ndarray, bounds: np.ndarray) -> bool:
    """
    Prove ``w`` optimal, for its choice of inputs, by one linear program when the optimality conditions cannot.

    With many inputs at a bound, the binding constraints are dependent, and the least-squares multipliers can miss
    even at the optimum. A convex limit's tangent plane at ``w`` holds at every point the limit allows, so the linear
    program over the rules, the bounds and the planes at ``w`` has an optimum no higher than the problem's. When that
    optimum reaches ``w``'s objective, no point that meets the rules does better.
    """
    n, objective = problem.n, problem.objective[problem.keep()]
    a, low, high = problem.core_rows()
    tail = np.zeros(len(w) - n)
    planes, tops = [], []
    for lim in problem.limits:  # value(w) + gradient (x - w) <= limit
        gradient = lim.gradient(w[:n])
        planes.append(np.concatenate([gradient, tail]))
        tops.append(lim.limit - lim.value(w[:n]) + gradient @ w[:n])
    scale = problem.scale(bounds)  # each input in its natural size, as the searches take it
    rows = optimize.LinearConstraint(
        np.vstack([a, *planes]) * scale,
        np.concatenate([low, np.full(len(tops), -np.inf)]),
        np.concatenate([high, tops]),
    )
    found = optimize.milp(objective * scale, bounds=optimize.Bounds(bounds[:, 0] / scale, bounds[:, 1] / scale),
                          constraints=rows)  # fmt: skip
    if found.x is None:  # not solved: no proof
        return False
    best = float(objective @ w)
    return bool(found.fun >= best - _PROOF_TOLERANCE * (1 + abs(best)))


def _settled(problem: _Problem, search: _Search, w: np.ndarray, rows: tuple) -> np.ndarray:
    """
    Return a point where the optimality conditions hold, starting again from ``w`` while they do not.

    SLSQP can stop at a point that meets every rule but is not the optimum: at its iteration limit, or (scipy 1.15
    more often than later releases) on a line search that fails. Feasibility alone cannot tell; the optimality
    conditions can. A restart resets SLSQP's estimate of the curvature, which is what usually stalled it.
    """
    for _ in range(_RESTARTS):
        if _kkt_residual(problem, w, search.bounds) < _KKT_TOLERANCE or _proven_by_planes(problem, w, search.bounds):
            break
        again = _slsqp(dataclasses.replace(search, start=w), 1000)
        if not _meets(problem, again, rows) or search.fun(again) > search.fun(w):
            break  # the restart found nothing better: the point in hand stands
        w = again
    return w


def _too_far(problem: _Problem, point: np.ndarray) -> str:
    """Say which limits stand in the way, and by how much, at the point of least excess."""
    ratios = {lim.name: lim.value(point[: problem.n]) / lim.limit for lim in problem.limits}
    over = ", ".join(f"{name} {r:.3g} times" for name, r in ratios.items() if r > 1 + _LIMIT_TOLERANCE)
    return (
        "Within their bounds the inputs can meet the linear rules and the windows, but only far from the data "
        f"the model was built from: at best, {over} its limit."
    )


def _continuous(problem: _Problem, on: tuple[int, ...] | None = None) -> tuple[np.ndarray, float] | None:
    """
    Search the inputs and target distances with every switch relaxed (``on=None``) or set.

    A linear program first checks the linear rows: when they cannot be met, that is proven at once. SLSQP then
    searches from the model's centre. Should it end outside a limit, the least excess over the limits any point
    meeting the linear rows can have tells "no answer" (``problem.why`` says which limit) from "start again there".

    Returns
    -------
    (np.ndarray, float) or None
        The point over every variable (switches read off the inputs) and its objective; None when nothing meets
        the rules.
    """
    keep = problem.keep()
    bounds = problem.continuous_bounds(on)
    rows = problem.core_rows()
    objective = problem.objective[keep]
    linear = _linear_point(objective, *rows, bounds)
    if linear is None:
        problem.why = _NO_LINEAR
        return None
    constraints = _linear_constraints(*rows) + _limit_constraints(problem.limits, problem.n, len(problem.targets))
    search = _Search(lambda w: objective @ w, lambda _w: objective, problem.start(bounds), bounds, constraints,
                     problem.scale(bounds))  # fmt: skip
    w = _slsqp(search, _FIRST_TRY_ITERATIONS)
    if not _meets(problem, w, rows):
        # The least excess decides "no answer" for this choice, so a second start guards against SLSQP stopping short.
        excess, point = _least_excess(problem, rows, bounds, w)
        if excess > _LIMIT_TOLERANCE:
            excess, point = min((excess, point), _least_excess(problem, rows, bounds, linear), key=lambda e: e[0])
        if excess > _LIMIT_TOLERANCE:
            problem.why = _too_far(problem, point)
            return None
        w = _slsqp(dataclasses.replace(search, start=point), 1000)
        if not _meets(problem, w, rows):
            problem.why = _NOT_SETTLED
            return None
    w = _settled(problem, search, w, rows)
    z = np.zeros(problem.width)
    z[keep] = w
    z[problem.n : problem.n + problem.m] = [1.0 if w[i] > 1e-9 else 0.0 for i in problem.switched]
    return z, float(objective @ w)


# ---------------------------------------------------------------------------------------------------------------------
# Outer approximation, for the counting rules
# ---------------------------------------------------------------------------------------------------------------------
class _SearchStoppedError(RuntimeError):
    """The counting rules' search had to stop before it proved anything; the message says why."""


def _master(problem: _Problem) -> tuple[np.ndarray, float] | None:
    """
    Choose the switches: the linear problem with every tangent plane so far; its optimum bounds the answer below.

    Each input's column is scaled by its natural size, as the continuous search is: a fraction runs from 0 to 1 and a
    process condition can run to 1e5. HiGHS's presolve is off: on these masters (tangent planes with coefficients
    over many orders of magnitude), the HiGHS of scipy 1.15 declared a master with feasible points infeasible, which
    would end the search with a false "no answer".
    """
    m, extra = problem.m, len(problem.targets)
    scale = np.concatenate([problem.scale(problem.continuous_bounds(None))[: problem.n], np.ones(m + extra)])
    integrality = np.concatenate([np.zeros(problem.n), np.ones(m), np.zeros(extra)])
    lower = np.concatenate([problem.bounds[:, 0], np.zeros(m), np.zeros(extra)]) / scale
    upper = np.concatenate([problem.bounds[:, 1], np.ones(m), np.full(extra, np.inf)]) / scale
    objective = problem.objective.copy()
    objective[problem.n : problem.n + m] = 1e-7  # among equal formulas, the one with fewer inputs
    rows = np.array([r.a for r in problem.rows]) * scale
    result = optimize.milp(
        objective * scale, integrality=integrality, bounds=optimize.Bounds(lower, upper),
        constraints=optimize.LinearConstraint(rows, [r.low for r in problem.rows], [r.high for r in problem.rows]),
        options={"time_limit": 120, "presolve": False},
    )  # fmt: skip
    if result.status == 2:  # infeasible: no choice of switches is left, which is the proof
        return None
    if result.x is None:  # a time limit or a numerical failure proves nothing
        raise _SearchStoppedError(f"The master problem could not be solved ({result.message}).")
    z = result.x * scale
    return z, float(problem.objective @ z)


def _box(problem: _Problem) -> None:
    """
    Add, for the master only, the linear consequences of the limits: T² within its limit puts every score within
    ±sd·√limit, and a block's SPE within its limit puts every component of its residual within ±that limit.

    They bound every input from the first round. Without them an input the objective ignores (a process condition,
    when only lots cost money) is free in the master, which puts it at an extreme; a tangent plane taken there has
    coefficients and a bound many orders apart, and HiGHS then rejects points that meet every rule.
    """
    affine, n = problem.affine, problem.n
    half = affine.score_sd * np.sqrt(problem.limits[0].limit)
    for a in np.flatnonzero(np.isfinite(half)):
        row = np.zeros(problem.width)
        row[:n] = affine.t_map[a]
        problem.add(_Row(row, -half[a] - affine.t0[a], half[a] - affine.t0[a], f"t{a + 1}", "box"))
    for block, lim in zip(affine.blocks, problem.limits[1:], strict=True):
        for j in np.flatnonzero(np.abs(block.jacobian).max(axis=1) > 0):  # a column with no variance has no residual
            row = np.zeros(problem.width)
            row[:n] = block.jacobian[j]
            problem.add(_Row(row, -lim.limit - block.r0[j], lim.limit - block.r0[j], block.name, "box"))


def _cut(problem: _Problem, x: np.ndarray, every: bool = False) -> int:
    """Add the tangent plane of each limit at x (each broken one, or every one near its limit): valid, as convex."""
    added = 0
    for lim in problem.limits:
        value = lim.value(x)
        if value > lim.limit * (1 + _LIMIT_TOLERANCE) or (every and value >= 0.5 * lim.limit):
            gradient = lim.gradient(x)
            size = float(np.abs(gradient).max()) or 1.0  # each plane at unit scale: HiGHS's tolerances are absolute
            a = np.zeros(problem.width)
            a[: problem.n] = gradient / size
            problem.add(_Row(a, -np.inf, (lim.limit - value + gradient @ x) / size, lim.name, "cut"))
            added += 1
    return added


def _exclude(problem: _Problem, on: tuple[int, ...]) -> None:
    """Rule out one choice of switches: at least one must differ next time."""
    a = np.zeros(problem.width)
    for k, value in enumerate(on):
        a[problem.n + k] = -1.0 if value else 1.0
    problem.add(_Row(a, 1 - sum(on), np.inf, "", "exclude"))


@dataclass
class _Solution:
    """The best point found, its objective, the switches, the planes used, and whether it is proven optimal."""

    z: np.ndarray | None
    objective: float = np.nan
    on: tuple[int, ...] | None = None
    cuts: int = 0
    proven: bool = False
    message: str = ""


def _solve(problem: _Problem) -> _Solution:
    """Solve the relaxed problem; with counting rules, run outer approximation until its bound meets the best."""
    relaxed = _continuous(problem)
    if relaxed is None:  # out of reach of the linear rows (a linear program), or of the limits (a convex search)
        return _Solution(None, proven=problem.why != _NOT_SETTLED, message=problem.why)
    if not problem.switched:
        return _Solution(relaxed[0], relaxed[1], None, 0, proven=True)
    problem.check_counted()
    _box(problem)
    cuts = _cut(problem, relaxed[0][: problem.n], every=True)
    return _outer_approximation(problem, cuts, _NO_COUNT)


def _allowed(problem: _Problem, z: np.ndarray) -> bool:
    """Whether a point's choice of inputs meets every 'different' row added for the alternatives."""
    return all(r.a @ z <= r.high + 1e-9 for r in problem.rows if r.kind == "different")


def _outer_approximation(problem: _Problem, cuts: int, none: str) -> _Solution:
    """
    Alternate the master's choice of switches with the continuous search for that choice, until the master's bound
    meets the best answer. A choice solved before still counts when the rows added since allow it: the master never
    offers it again. ``none`` is the message when no choice meets the rules.
    """
    best = min((e for e in problem.evaluated if _allowed(problem, e[0])), key=lambda e: e[1], default=None)
    proven = False  # True once the master's bound meets the best answer, or the master has nothing left
    stopped = ""
    for _ in range(_MAX_ROUNDS):
        try:
            chosen = _master(problem)
        except _SearchStoppedError as err:
            stopped = str(err)
            break
        if chosen is None or (best is not None and chosen[1] >= best[1] - 1e-9 * (1 + abs(best[1]))):
            proven = True
            break
        on = tuple(round(v) for v in chosen[0][problem.n : problem.n + problem.m])
        found = _continuous(problem, on)
        _exclude(problem, on)  # each choice is solved once; its best point is now known
        cuts += _cut(problem, chosen[0][: problem.n])  # the master's own point, where it breaks a limit
        if found is not None:
            cuts += _cut(problem, found[0][: problem.n], every=True)
            problem.evaluated.append((found[0], found[1], on))
            best = min([c for c in (best, problem.evaluated[-1]) if c is not None], key=lambda c: c[1])
    unproven = stopped or f"The search ran {_MAX_ROUNDS} rounds."
    if best is None:
        message = none if proven else f"{unproven} No answer was found, and none is ruled out."
        return _Solution(None, cuts=cuts, proven=proven, message=message)
    message = "" if proven else f"{unproven} This is the best answer found; it is not proven optimal."
    return _Solution(best[0], best[1], best[2], cuts, proven=proven, message=message)


def _check_alternatives(problem: _Problem, alternatives: int, leave_out: int) -> None:
    """Refuse a request for alternatives that cannot be read, or that has no counted inputs to differ in."""
    problems = []
    if not (isinstance(alternatives, int) and alternatives >= 0):
        problems.append(f"alternatives is a whole number, 0 or more, not {alternatives!r}")
    if not (isinstance(leave_out, int) and leave_out >= 1):
        problems.append(f"leave_out is a whole number, 1 or more, not {leave_out!r}")
    if alternatives and not problem.switched:
        problems.append("alternatives differ in the inputs they use, so they need counted inputs: give a group a "
                        "max_count, or a min_amount (which also says what 'used' means)")  # fmt: skip
    if problems:
        msg = "The alternatives cannot be searched: " + "; ".join(problems) + "."
        raise ValueError(msg)


def _differ(problem: _Problem, z: np.ndarray, leave_out: int, number: int) -> None:
    """Add a no-good cut: every later answer leaves out at least ``leave_out`` of the counted inputs ``z`` uses."""
    used = [k for k, i in enumerate(problem.switched) if abs(z[i]) > 1e-9]
    a = np.zeros(problem.width)
    a[[problem.n + k for k in used]] = 1.0
    problem.add(_Row(a, -np.inf, len(used) - leave_out, f"differs from answer {number}", "different"))


def _alternatives(problem: _Problem, first: _Solution, count: int, leave_out: int) -> list[Bunch]:
    """
    Search the next answers in turn: each the best that leaves out at least ``leave_out`` of the counted inputs each
    earlier answer uses. The list ends early, with an infeasible result saying why, when no further answer exists.
    """
    found: list[Bunch] = []
    solution = first
    while len(found) < count and solution.z is not None:
        _differ(problem, solution.z, leave_out, len(found) + 1)
        none = _NO_OTHER.format(leave_out=leave_out)
        solution = _most_typical(problem, _outer_approximation(problem, solution.cuts, none))
        found.append(_result(problem, solution))
    return found


def _most_typical(problem: _Problem, solution: _Solution) -> _Solution:
    """
    Among the answers as close to the targets as the best, return the one with the least T².

    A target pins down only as many score directions as it has qualities, so equally close answers fill a null
    space (Jaeckle and MacGregor, 2000); the most typical of them is the one the data best support.
    """
    if solution.z is None or problem.rules.costs:
        return solution
    keep, n = problem.keep(), problem.n
    bounds, rows = problem.continuous_bounds(solution.on), problem.core_rows()
    objective = problem.objective[keep]
    cap = solution.objective + 1e-6 * (1 + abs(solution.objective))
    a, low, high = rows
    capped = (np.vstack([a, objective]), np.append(low, -np.inf), np.append(high, cap))
    t2 = problem.limits[0]
    tail = np.zeros(len(problem.targets))
    search = _Search(
        fun=lambda w: t2.value(w[:n]), jac=lambda w: np.concatenate([t2.gradient(w[:n]), tail]),
        start=solution.z[keep], bounds=bounds,
        constraints=_linear_constraints(*capped) + _limit_constraints(problem.limits, n, len(tail)),
        scale=problem.scale(bounds),
    )  # fmt: skip
    w = _slsqp(search, 1000)
    if not _meets(problem, w, capped) or t2.value(w[:n]) > t2.value(solution.z[:n]):
        return solution  # the best answer found stands
    z = solution.z.copy()
    z[keep] = w
    return dataclasses.replace(solution, z=z, objective=float(objective @ w))


# ---------------------------------------------------------------------------------------------------------------------
# Shadow prices, from the optimality conditions
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Active:
    """A constraint as reported: its value and bounds, its gradient over [x, d], and the sign of its multiplier."""

    name: str
    kind: str
    value: float
    lower: float
    upper: float
    gradient: np.ndarray
    side: str  # "upper", "lower", "equal" or "" (not binding)


def _side(value: float, lower: float, upper: float, scale: float) -> str:
    tolerance = _ACTIVE * max(1.0, scale)
    if lower == upper:
        return "equal"
    if np.isfinite(upper) and value >= upper - tolerance:
        return "upper"
    if np.isfinite(lower) and value <= lower + tolerance:
        return "lower"
    return ""


def _reported(problem: _Problem, w: np.ndarray, bounds: np.ndarray) -> list[_Active]:
    """Every rule, window and limit, and each input at a bound, as constraints over [x, d]."""
    keep = problem.keep()
    out = []
    for r in problem.rows[: problem.core]:  # a target's two rows enter the conditions but are not reported
        a = r.a[keep]
        value = float(a @ w)
        side = _side(value, r.low, r.high, float(np.abs(a) @ np.abs(w)))
        out.append(_Active(r.name, r.kind, value + r.shift, r.low + r.shift, r.high + r.shift, a, side))
    for lim in problem.limits:
        g = np.concatenate([lim.gradient(w[: problem.n]), np.zeros(len(problem.targets))])
        value = lim.value(w[: problem.n])
        out.append(
            _Active(lim.name, "limit", value, -np.inf, lim.limit, g, _side(value, -np.inf, lim.limit, lim.limit))
        )
    for i, name in enumerate(problem.names):
        low, high = bounds[
            i
        ]  # as the search had them: a counted input switched off is held at 0, one on at its minimum
        side = _side(w[i], low, high, float(problem.affine.spread[i]))
        if not side:
            continue
        g = np.zeros(len(w))
        g[i] = 1.0
        own_low, own_high = problem.bounds[i]
        if name in problem.rules.fixed or low == high:
            kind = "fixed" if name in problem.rules.fixed else "unused"
        elif side == "lower" and low > own_low:
            kind = "min_amount"  # its group's minimum amount, shared by every input of the group: not one input's bound
        else:
            kind, low, high = "bound", own_low, own_high  # the input's own bounds, as the rules state them
        out.append(_Active(name, kind, float(w[i]), float(low), float(high), g, side))
    return out


# The order in which binding constraints claim a multiplier when several are linearly dependent: the rules a person
# sets, then the limits, then what holds a single input. A lot at 100% of its material is held both by its upper bound
# and by the material's total with the other lots at 0; the total keeps the price, the bound is marked as implied.
_PRIORITY = ("target", "window", "rule", "closure", "limit", "count", "fixed", "unused", "min_amount", "bound")


def _independent(columns: np.ndarray, order: list[int]) -> list[int]:
    """Return the columns, taken in ``order``, that each add a direction to those kept before them (Gram-Schmidt)."""
    basis: list[np.ndarray] = []
    kept = []
    for k in order:
        r = columns[:, k] - sum((q @ columns[:, k]) * q for q in basis)
        if np.linalg.norm(r) > 1e-9 * max(1.0, float(np.linalg.norm(columns[:, k]))):
            basis.append(r / np.linalg.norm(r))
            kept.append(k)
    return kept


def _multipliers(objective: np.ndarray, active: list[_Active]) -> tuple[np.ndarray, float]:
    """
    Solve the optimality conditions for each binding constraint's multiplier: the objective's gradient is a
    non-negative mix of the binding constraints' outward gradients (any sign for an equality).

    When binding constraints are linearly dependent, their multipliers are not unique: only the independent ones,
    taken in ``_PRIORITY`` order, get one; the others are implied, with a multiplier of NaN.

    Returns
    -------
    (np.ndarray, float)
        The multiplier of each constraint (0 when not binding, NaN when implied), and the relative residual of the
        conditions: near 0 at an optimum.
    """
    binding = [k for k, c in enumerate(active) if c.side]
    lam = np.zeros(len(active))
    if not binding:
        return lam, float(np.linalg.norm(objective) / max(1.0, np.linalg.norm(objective)))
    sign = {"upper": 1.0, "lower": -1.0, "equal": 1.0}
    g = np.column_stack([sign[active[k].side] * active[k].gradient for k in binding])
    order = sorted(range(len(binding)), key=lambda j: _PRIORITY.index(active[binding[j]].kind))
    kept = _independent(g, order)
    lower = np.array([-np.inf if active[binding[j]].side == "equal" else 0.0 for j in kept])
    found = optimize.lsq_linear(g[:, kept], -objective, bounds=(lower, np.full(len(kept), np.inf)))
    lam[binding] = np.nan
    lam[[binding[j] for j in kept]] = found.x
    residual = float(np.linalg.norm(objective + g[:, kept] @ found.x) / max(1.0, np.linalg.norm(objective)))
    return lam, residual


def _kkt_residual(problem: _Problem, w: np.ndarray, bounds: np.ndarray) -> float:
    """How nearly the optimality conditions hold at ``w``: near 0 at an optimum."""
    return _multipliers(problem.objective[problem.keep()], _reported(problem, w, bounds))[1]


def _constraint_table(problem: _Problem, w: np.ndarray, bounds: np.ndarray) -> tuple[pd.DataFrame, float]:
    """
    Every rule, window and limit, and each input at a bound, with its shadow price: the change in the optimal
    objective per unit increase of the bound it sits at (negative for an upper bound that saves when raised).
    """
    objective = problem.objective[problem.keep()]
    active = _reported(problem, w, bounds)
    lam, residual = _multipliers(objective, active)
    rows = []
    for c, mu in zip(active, lam, strict=True):
        # d objective / d bound: -mu for an upper bound or an equality, +mu for a lower bound; NaN when implied.
        price = 0.0 if not c.side else (mu if c.side == "lower" else -mu)
        at = {"upper": c.upper, "lower": c.lower, "equal": c.upper}.get(c.side, np.nan)
        rows.append({
            "constraint": c.name, "kind": c.kind, "value": c.value, "lower": c.lower, "upper": c.upper,
            "binding": c.side, "shadow_price": price,
            "per_percent": price * abs(at) / 100 if np.isfinite(at) and at != 0 else np.nan,
        })  # fmt: skip
    table = pd.DataFrame(rows)
    return table[table["kind"] != "target"].reset_index(drop=True), residual


# ---------------------------------------------------------------------------------------------------------------------
# The public functions
# ---------------------------------------------------------------------------------------------------------------------
def _result(problem: _Problem, solution: _Solution) -> Bunch:
    """Gather the answer, its predictions and limits, every rule with its shadow price, and how it was found."""
    affine = problem.affine
    if solution.z is None:
        return Bunch(feasible=False, message=solution.message, proven_optimal=solution.proven, n_cuts=solution.cuts)
    x = solution.z[: problem.n]
    bounds = problem.continuous_bounds(solution.on)
    table, residual = _constraint_table(problem, solution.z[problem.keep()], bounds)
    settled = residual < 1e-3 or _proven_by_planes(problem, solution.z[problem.keep()], bounds)
    used = {
        g: [n for n in names if abs(x[problem.names.index(n)]) > 1e-9]
        for g, names in _all_groups(affine, problem.rules).items()
    }
    limits = {lim.name: lim.limit for lim in problem.limits}
    return Bunch(
        feasible=True, message=solution.message, x_new=pd.Series(x, index=affine.names, name="x_new"),
        y_hat=pd.Series(affine.predict(x), index=affine.y_names, name="y_hat"),
        scores=pd.Series(affine.scores(x), index=[f"t{a + 1}" for a in range(len(affine.t0))], name="scores"),
        hotellings_t2=affine.hotellings_t2(x), t2_limit=limits["T²"],
        spe=pd.Series(affine.spe(x), index=[b.name for b in affine.blocks], name="SPE"),
        spe_limit=pd.Series([limits[f"SPE {b.name}"] for b in affine.blocks], index=[b.name for b in affine.blocks]),
        objective=solution.objective, used=used, constraints=table, kkt_residual=residual,
        proven_optimal=solution.proven and settled, n_cuts=solution.cuts,
    )  # fmt: skip


def _read(model: BaseEstimator | AffineModel, conf_level: float) -> AffineModel:
    """Return the maps: read from a fitted model, or the stored maps given, if read at the same confidence level."""
    if not isinstance(model, AffineModel):
        return affine_model(model, conf_level)
    if model.conf_level != conf_level:
        msg = f"The model was read at conf_level {model.conf_level}, the rules ask for {conf_level}."
        raise ValueError(msg)
    return model


def _warn_unbounded(problem: _Problem) -> None:
    """Name, until 3.0, each input without a lower or an upper bound: only T² and SPE then hold it."""
    free = [
        n for n, (low, high) in zip(problem.names, problem.bounds, strict=True) if not np.isfinite([low, high]).all()
    ]
    if not free:
        return
    named = ", ".join(free[:5]) + (f" and {len(free) - 5} more" if len(free) > 5 else "")
    warnings.warn(
        f"process_improve.multivariate.optimal_inversion: {named} {'has' if len(free) == 1 else 'have'} no lower or "
        "no upper bound. Only the T² and SPE limits then hold an input, and an answer can set it to a value it cannot "
        "take, such as a negative amount. Searching an input without both bounds is deprecated since 2.4.0 and an "
        "error from 3.0: give each one its extreme limits in InversionRules(bounds=...).",
        FutureWarning,
        stacklevel=3,
    )


def optimal_inversion(
    model: BaseEstimator | AffineModel, rules: InversionRules, *, alternatives: int = 0, leave_out: int = 1
) -> Bunch:
    """
    Find the inputs a PLS or TPLS model vouches for that meet the rules, at the least cost or closest to the targets.

    Parameters
    ----------
    model : PLS, TPLS or AffineModel
        A fitted model, or its maps as :func:`affine_model` read them at ``rules.conf_level`` (pass the maps to solve
        many problems on one model, or to search with maps stored by :meth:`AffineModel.to_dict`).
    rules : InversionRules
        The objective (costs or targets), the windows, bounds, fixed inputs, linear rules, counting rules and limits.
    alternatives : int
        How many further answers to find, each with a different set of inputs: the best answer that leaves out at least
        ``leave_out`` of the counted inputs used by the answer and by every alternative before it (a no-good cut on the
        master problem). Needs counted inputs: a group with a ``max_count`` or a ``min_amount``. Give the
        ``min_amount``, so that a trace of an input does not count as using it.
    leave_out : int
        How many of each earlier answer's counted inputs every alternative must leave out.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys:

        ``feasible`` : bool
            Whether any inputs meet every rule. When False, ``message`` says which rule is in the way.
        ``x_new`` : pd.Series
            The inputs, in their original units.
        ``y_hat`` : pd.Series
            The predicted qualities.
        ``scores``, ``hotellings_t2``, ``t2_limit``, ``spe``, ``spe_limit``
            How far the answer sits from the data the model was built from, against its limits.
        ``objective`` : float
            The cost, or the summed distance to the targets in training standard deviations.
        ``used`` : dict of str to list of str
            The inputs used from each group.
        ``constraints`` : pd.DataFrame
            Every rule, window and limit, and each input at a bound: its value, bounds, the side it binds on, and its
            ``shadow_price``, the change in the optimal objective per unit increase of that bound (``per_percent``:
            per 1% of it). A rule that does not bind has a shadow price of 0. A binding constraint implied by others
            (a lot at 100% of its material, held both by its bound and by the material's total) has NaN: its price
            is not its own.
        ``kkt_residual`` : float
            How nearly the optimality conditions hold at the answer; near 0 at an optimum.
        ``proven_optimal`` : bool
            The search proved no better answer exists, and the optimality conditions hold. With no answer: the
            search proved none exists; False when it stopped before it could.
        ``n_cuts`` : int
            The tangent planes the counting rules needed.
        ``message`` : str
            Why there is no answer, or why it is not proven optimal; empty otherwise.
        ``alternatives`` : list of sklearn.utils.Bunch
            Only when ``alternatives`` is asked for: each alternative as a result with the keys above, best first. Its
            ``proven_optimal`` means no input set the earlier answers allow does better. The list ends early, with a
            result whose ``feasible`` is False and whose ``message`` says why, when no further input set meets the
            rules; it is empty when there is no answer at all.

    Raises
    ------
    ValueError
        The rules name what the model does not have, or cannot be read; the maps given were read at another
        confidence level; or alternatives are asked for without counted inputs, or with a count that is not a whole
        number.

    See Also
    --------
    inversion_sensitivity : check the shadow prices by solving again, and trace a limit, counts and prices.

    References
    ----------
    C. M. Jaeckle and J. F. MacGregor, "Industrial applications of product design through the inversion of latent
    variable models", Chemometrics and Intelligent Laboratory Systems, 50 (2000): 199-210.

    M. A. Duran and I. E. Grossmann, "An outer-approximation algorithm for a class of mixed-integer nonlinear
    programs", Mathematical Programming, 36 (1986): 307-339.

    Examples
    --------
    >>> rules = InversionRules(
    ...     costs={"Acetic": 1.0, "H2S": 3.0, "Lactic": 8.0}, windows={"Taste": (20.0, 30.0)},
    ...     bounds={"Acetic": (4.0, 7.0), "H2S": (3.0, 10.0), "Lactic": (0.8, 2.1)},
    ... )
    >>> result = optimal_inversion(pls, rules)
    >>> result.x_new, result.objective, result.constraints
    """
    problem = _Problem(_read(model, rules.conf_level), rules)
    _warn_unbounded(problem)
    _check_alternatives(problem, alternatives, leave_out)
    solution = _most_typical(problem, _solve(problem))
    result = _result(problem, solution)
    if alternatives:
        result.alternatives = _alternatives(problem, solution, alternatives, leave_out)
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------------------------------------------------
def _set_bound(rules: InversionRules, row: pd.Series, side: str, value: float) -> InversionRules:
    """
    Return the rules with one reported constraint's ``side`` bound set to ``value``: a window, a linear rule, the T²
    limit, an SPE limit, or an input's bound.
    """
    lower = float(row["lower"]) if np.isfinite(row["lower"]) else None
    upper = float(row["upper"]) if np.isfinite(row["upper"]) else None
    low, high = (value, upper) if side == "lower" else (lower, value)
    kind, name = row["kind"], row["constraint"]
    if kind == "window":
        return dataclasses.replace(rules, windows=dict(rules.windows) | {name: (low, high)})
    if kind == "rule":
        moved = tuple(dataclasses.replace(r, lower=low, upper=high) if r.name == name else r for r in rules.rules)
        return dataclasses.replace(rules, rules=moved)
    if name == "T²":
        return dataclasses.replace(rules, t2_limit=value)
    if kind == "limit":
        return dataclasses.replace(rules, spe_limits=dict(rules.spe_limits) | {name[len("SPE ") :]: value})
    return dataclasses.replace(rules, bounds=dict(rules.bounds) | {name: (low, high)})


def _relaxed(rules: InversionRules, row: pd.Series, step: float) -> InversionRules:
    """Return the rules with one binding constraint moved by ``step`` the way that relaxes it."""
    side = row["binding"]
    value = row["upper"] + step if side == "upper" else row["lower"] - step
    return _set_bound(rules, row, side, float(value))


def _step(row: pd.Series, fraction: float) -> float:
    at = row["upper"] if row["binding"] in {"upper", "equal"} else row["lower"]
    return fraction * abs(at) if np.isfinite(at) and at != 0 else fraction


def _check_prices(rules: InversionRules, affine: AffineModel, result: Bunch, fraction: float) -> pd.DataFrame:
    """Each binding rule moved a little in the direction that relaxes it, solved again: the change, and the forecast."""
    rows = []
    table = result.constraints
    movable = table["binding"].isin(["upper", "lower"]) & table["kind"].isin(["window", "rule", "limit", "bound"])
    movable &= table["shadow_price"].notna()  # an implied constraint has no price of its own to check
    for _, row in table[movable].iterrows():
        step = _step(row, fraction)
        again = optimal_inversion(affine, _relaxed(rules, row, step))
        change = again.objective - result.objective if again.feasible else np.nan
        signed = step if row["binding"] == "upper" else -step
        rows.append({
            "constraint": row["constraint"], "kind": row["kind"], "moved_by": signed,
            "forecast_change": row["shadow_price"] * signed, "actual_change": change,
        })  # fmt: skip
    return pd.DataFrame(rows, columns=["constraint", "kind", "moved_by", "forecast_change", "actual_change"])


def _trace(rules: InversionRules, affine: AffineModel, front: tuple[str, list[float]], result: Bunch) -> pd.DataFrame:
    """
    Return the optimal objective as one rule's bound takes each value: the T² limit, an SPE limit (``"SPE X"``), a
    window or a linear rule. A two-sided rule moves the side that binds (its lower side if neither does).
    """
    name, values = front
    table = result.constraints
    found = table[(table["constraint"] == name) & table["kind"].isin(["window", "rule", "limit"])]
    if found.empty:
        msg = f"{name!r} is neither 'T²', an SPE limit, a window nor a rule: it cannot be traced."
        raise ValueError(msg)
    row = found.iloc[0]
    side = (
        row["binding"] if row["binding"] in {"upper", "lower"} else ("lower" if np.isfinite(row["lower"]) else "upper")
    )
    rows = []
    other = row["upper"] if side == "lower" else row["lower"]
    for value in values:
        # A bound moved past the rule's other bound leaves nothing to meet: no answer, without a search.
        crossed = value > other if side == "lower" else value < other
        again = Bunch(feasible=False) if crossed else optimal_inversion(affine, _set_bound(rules, row, side, value))
        rows.append(
            {"value": value, "feasible": again.feasible, "objective": again.objective if again.feasible else np.nan}
        )
    return pd.DataFrame(rows)


def _counts(rules: InversionRules, affine: AffineModel, result: Bunch) -> pd.DataFrame:
    """For each counted group, the optimal objective with one input fewer and one more allowed."""
    rows = []
    for group, most in rules.max_count.items():
        for k in (most - 1, most + 1):
            if k < 1:
                continue
            found = optimal_inversion(affine, dataclasses.replace(rules, max_count=dict(rules.max_count) | {group: k}))
            rows.append({
                "group": group, "max_count": k, "feasible": found.feasible,
                "objective": found.objective if found.feasible else np.nan,
                "change": found.objective - result.objective if found.feasible else np.nan,
            })  # fmt: skip
    return pd.DataFrame(rows, columns=["group", "max_count", "feasible", "objective", "change"])


def _usage(result: Bunch) -> frozenset[str]:
    """Return the answer's shape: the inputs used, and the rules, limits and bounds that bind; or that there is none."""
    if not result.feasible:  # a search that fails at another price is a change of shape, not a crash
        return frozenset({"no answer"})
    used = {f"uses {n}" for names in result.used.values() for n in names}
    table = result.constraints
    return frozenset(used | set(table.loc[(table["binding"] != "") & (table["kind"] != "unused"), "constraint"]))


def _price_range(rules: InversionRules, affine: AffineModel, name: str, base: frozenset[str]) -> tuple[float, float]:
    """
    How far one input's cost can fall and rise, as a fraction of itself, before the answer changes shape: another
    input is used, or another rule binds. Within that range the answer moves smoothly with the cost.

    Bisection on each side over [-100%, +100%] (a cost of 0 to twice itself), to 1% of the cost.
    """

    def same(fraction: float) -> bool:
        costs = dict(rules.costs) | {name: rules.costs[name] * (1 + fraction)}
        return _usage(optimal_inversion(affine, dataclasses.replace(rules, costs=costs))) == base

    edges = []
    for far in (-1.0, 1.0):
        if same(far):
            edges.append(far)
            continue
        inside, outside = 0.0, far
        while abs(outside - inside) > 0.01:
            middle = (inside + outside) / 2
            inside, outside = (middle, outside) if same(middle) else (inside, middle)
        edges.append(inside)
    return edges[0], edges[1]


def _prices(rules: InversionRules, affine: AffineModel, result: Bunch, names: list[str]) -> pd.DataFrame:
    """Each named input's cost range, as fractions of its cost, over which the answer keeps its shape."""
    base = _usage(result)
    rows = []
    for name in names:
        low, high = _price_range(rules, affine, name, base)
        used = bool(abs(result.x_new[name]) > 1e-9)
        rows.append({"input": name, "cost": rules.costs[name], "used": used, "falls_by": low, "rises_by": high})
    return pd.DataFrame(rows, columns=["input", "cost", "used", "falls_by", "rises_by"])


def inversion_sensitivity(
    model: BaseEstimator | AffineModel,
    rules: InversionRules,
    *,
    front: tuple[str, list[float]] | None = None,
    prices: list[str] | None = None,
    fraction: float = 0.01,
) -> Bunch:
    """
    How the optimal inversion moves with its rules: shadow prices checked by solving again, a traced limit, the
    value of one input more or fewer per counted group, and how far costs can move before the answer changes.

    Parameters
    ----------
    model : PLS, TPLS or AffineModel
        A fitted model, or its maps, as for :func:`optimal_inversion`.
    rules : InversionRules
        The rules, as for :func:`optimal_inversion`.
    front : (str, list of float), optional
        A rule to trace, with the values its bound takes: ``"T²"``, an SPE limit (``"SPE X"``), a quality with a
        window, or a linear rule's name. A two-sided rule moves the side that binds. The optimal objective at each.
    prices : list of str, optional
        Inputs whose cost to range (``costs`` objective only): how far, as a fraction of itself, each can fall or
        rise before the answer changes shape (another input used, another rule binding). Each costs about a dozen
        solves.
    fraction : float, default 0.01
        How far each binding rule is moved to check its shadow price, as a fraction of its bound.

    Returns
    -------
    result : sklearn.utils.Bunch
        With keys ``inversion`` (the :func:`optimal_inversion` result), ``checked`` (each binding rule: moved by,
        forecast change from its shadow price, actual change on solving again), ``front`` (value, feasible,
        objective), ``counts`` (group, max_count, feasible, objective, change) and ``prices`` (input, cost, used,
        falls_by, rises_by).

    Examples
    --------
    >>> report = inversion_sensitivity(pls, rules, front=("T²", [4.0, 6.0, 8.0]), prices=["Lactic"])
    >>> report.checked   # shadow prices against solving again
    >>> report.front     # the cost as the T² limit moves
    """
    affine = _read(model, rules.conf_level)
    result = optimal_inversion(affine, rules)
    empty = pd.DataFrame()
    if not result.feasible:
        return Bunch(inversion=result, checked=empty, front=empty, counts=empty, prices=empty)
    if prices and not rules.costs:
        msg = "Prices can only be ranged when the objective is cost."
        raise ValueError(msg)
    return Bunch(
        inversion=result,
        checked=_check_prices(rules, affine, result, fraction),
        front=_trace(rules, affine, front, result) if front else empty,
        counts=_counts(rules, affine, result),
        prices=_prices(rules, affine, result, prices) if prices else empty,
    )
