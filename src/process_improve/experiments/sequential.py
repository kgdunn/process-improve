# (c) Kevin Dunn, 2010-2026. MIT License.

"""The sequential method of response surfaces: a path of steepest ascent, and the next design on it (#798).

A first design finds the direction in which the response improves. The experimenter steps along it,
runs a few settings on the path, and centres the next design on the best of them. Two functions set
that out from a fitted model:

- :func:`steepest_path` steps along the steepest ascent (or descent) as sources set it out: a round
  amount of one leading factor, in its own units, with the other factors in proportion to the slopes;
- :func:`next_design` centres a new design on a step of that path, each factor's range kept or rescaled.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from process_improve.experiments.designs import generate_design
from process_improve.experiments.factor import Factor
from process_improve.experiments.optimization import (
    _build_model_evaluator,
    _coded_to_actual,
    _parsed_terms,
    _ranges_from_coding,
)

if TYPE_CHECKING:
    from process_improve.experiments.factor import DesignResult

#: The way a path goes for each goal.
_DIRECTIONS = {"maximize": "ascent", "minimize": "descent"}
#: A slope this small, next to the terms that add up to it or to the largest slope, is zero: what is left
#: when terms cancel (at a stationary point, say), or a factor the path barely moves, which cannot lead it.
_NO_SLOPE = 1e-9


class NoSteepestDirectionError(ValueError):
    """The fitted model is flat at the base of the path, so no direction improves it fastest.

    Raised by :func:`steepest_path` when the gradient of the model at the base is zero: at the
    design centre, when every first-order coefficient is zero.
    """


def steepest_path(
    model: dict[str, Any],
    step: float | Mapping[str, float] = 0.5,
    n_steps: int = 10,
    *,
    base: Mapping[str, float] | None = None,
    goal: str = "maximize",
) -> dict[str, Any]:
    """Return the path of steepest ascent (or descent) of a fitted model, as a table of steps (#798 S1).

    The path leaves ``base`` along the gradient of the fitted model there, which is the first-order
    coefficients when the base is the centre of the design, and moves every factor in proportion
    to its slope in coded units. A curved surface turns away from this straight line; ridge
    analysis (``optimize_responses(..., method="ridge_analysis")``) follows the curve instead.

    Parameters
    ----------
    model : dict
        A fitted model, as :func:`~process_improve.experiments.analysis.analyze_experiment`
        returns it, or written out by hand: its ``coefficients`` on the coded factors, its
        ``factor_names``, and optionally its ``factor_coding`` (each factor's ``low`` and
        ``high``), which gives each step in actual units too.
    step : float or dict
        How far each step goes.

        - A number: the distance between successive points, in coded units (default 0.5).
        - ``{lead: amount}``: the lead factor moves ``amount`` a step, in its own units (its
          actual units when the model has its coding, else coded units), up or down as its slope
          says, and the others move in proportion. This is how sources set a path out: Davies
          (Table 11-22) steps 10 c.c. of solvent, Box, Hunter and Hunter (Table 15.2) 5 minutes of
          reaction time, and Schmidt and Launsby one coded unit of the factor with the smaller slope.
    n_steps : int
        The number of steps away from the base (default 10). The table holds ``n_steps + 1``
        rows: step 0 is the base.
    base : dict or None
        Where the path starts, each factor in its own units, as for ``step``. A factor left out
        is at the centre of its range; by default the path starts at the centre of the design.
    goal : {"maximize", "minimize"}
        ``"maximize"`` follows the steepest ascent, ``"minimize"`` the steepest descent.

    Returns
    -------
    dict
        - ``goal``, and ``direction``: ``"ascent"`` or ``"descent"``.
        - ``lead``: the factor that sets each step, or ``None`` for a distance in coded units.
        - ``step_size``: the distance between successive points, in coded units.
        - ``direction_vector``: the unit vector of the path, in coded units.
        - ``increment``: each factor's change a step, ``coded`` and, with the coding,
          ``actual``.
        - ``factor_ranges``: the ranges the coding gives, which :func:`next_design` reads.
        - ``steps``: one row per step, with its ``step`` number, its ``coded`` settings, its
          ``actual`` settings when the model has its coding, and its ``predicted_response``.

        The intervals on the predictions need the fit itself: pass the coded settings of the
        steps as ``new_points`` to ``analyze_experiment(..., analysis_type="prediction")``.

    Raises
    ------
    NoSteepestDirectionError
        If the gradient of the model is zero at the base.
    ValueError
        If ``step``, ``n_steps``, ``base`` or ``goal`` is not one the path can take, or the lead
        factor has no slope at the base, so stepping it cannot set the path.

    Examples
    --------
    >>> from process_improve.experiments import steepest_path
    >>> model = {
    ...     "coefficients": [
    ...         {"term": "Intercept", "coefficient": 75.9},
    ...         {"term": "x1", "coefficient": 4.55},
    ...         {"term": "x2", "coefficient": -1.45},
    ...     ],
    ...     "factor_names": ["x1", "x2"],
    ...     "factor_coding": {"x1": {"low": 100, "high": 200}, "x2": {"low": 125, "high": 175}},
    ... }
    >>> path = steepest_path(model, step={"x2": 25}, n_steps=1)
    >>> round(path["steps"][1]["actual"]["x1"], 1), path["steps"][1]["actual"]["x2"]
    (306.9, 125.0)
    """
    if goal not in _DIRECTIONS:
        raise ValueError(f"goal must be 'maximize' or 'minimize'; got {goal!r}.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, numbers.Integral) or n_steps < 1:
        raise ValueError(f"n_steps must be a whole number of steps, at least 1; got {n_steps!r}.")
    missing = [key for key in ("coefficients", "factor_names") if key not in model]
    if missing:
        raise ValueError(f"model has no {missing}: pass the result of analyze_experiment, or write them out.")
    names = list(model["factor_names"])
    ranges = _ranges_from_coding([model], None) or {}
    start = _coded(base or {}, names, ranges)
    slopes = _gradient(model["coefficients"], names, start)
    if not np.any(slopes):
        raise NoSteepestDirectionError(
            "The fitted model is flat at the base of the path: every slope there is zero, so no direction "
            "improves it fastest."
        )
    way = slopes if goal == "maximize" else -slopes
    lead, increment = _increment(way, step, names, ranges)
    evaluate = _build_model_evaluator(model["coefficients"], names)
    steps = []
    for number in range(int(n_steps) + 1):
        at = start + number * increment
        row: dict[str, Any] = {"step": number, "coded": dict(zip(names, map(float, at), strict=True))}
        if ranges:
            row["actual"] = _coded_to_actual(row["coded"], ranges)
        row["predicted_response"] = float(evaluate(at))
        steps.append(row)
    change = {"coded": dict(zip(names, map(float, increment), strict=True))}
    if ranges:
        change["actual"] = {name: value * _unit(name, ranges) for name, value in change["coded"].items()}
    return {
        "goal": goal,
        "direction": _DIRECTIONS[goal],
        "lead": lead,
        "step_size": float(np.linalg.norm(increment)),
        "direction_vector": dict(zip(names, map(float, way / np.linalg.norm(way)), strict=True)),
        "increment": change,
        "factor_ranges": ranges,
        "steps": steps,
    }


def next_design(
    path: dict[str, Any],
    at_step: int,
    *,
    scale: Mapping[str, float] | None = None,
    **design: Any,  # noqa: ANN401
) -> DesignResult:
    """Return a new design centred on a step of a path of steepest ascent (#798 S2).

    The design's centre is the path's setting at ``at_step``, and each factor keeps the width of
    its range, times ``scale[factor]`` where given: Davies (11.42) enlarges the units of the
    factors whose effects were small, Box, Hunter and Hunter widen the ranges near a maximum, and
    others narrow them. A factor whose range the path does not know (a model without its coding)
    is set out in the coded units of the first design, from -1 to +1 about its new centre.

    Parameters
    ----------
    path : dict
        A path from :func:`steepest_path`.
    at_step : int
        The step to centre on: 0 is the base of the path.
    scale : dict or None
        A factor's new range width over its old one; 1 for a factor left out.
    **design
        The design to make, as :func:`~process_improve.experiments.designs.generate_design`
        takes it: ``design_type="full_factorial", n_center_points=3``, say.

    Returns
    -------
    DesignResult
        The design, with ``design_actual`` in the factors' units about the new centre.

    Raises
    ------
    ValueError
        If ``at_step`` is not a step of the path, or ``scale`` names a factor the path does not
        hold or gives a factor a width that is not a positive number.

    Examples
    --------
    >>> from process_improve.experiments import next_design, steepest_path
    >>> model = {
    ...     "coefficients": [
    ...         {"term": "Intercept", "coefficient": 40.0},
    ...         {"term": "A", "coefficient": 2.0},
    ...         {"term": "B", "coefficient": 1.0},
    ...     ],
    ...     "factor_names": ["A", "B"],
    ...     "factor_coding": {"A": {"low": 100, "high": 120}, "B": {"low": 1, "high": 3}},
    ... }
    >>> path = steepest_path(model, step={"A": 10}, n_steps=3)
    >>> new = next_design(path, at_step=3, scale={"B": 2.0}, design_type="full_factorial", n_center_points=3)
    >>> sorted(set(new.design_actual["A"])), sorted(set(new.design_actual["B"]))
    ([130.0, 140.0, 150.0], [1.5, 3.5, 5.5])
    """
    steps = path["steps"]
    if isinstance(at_step, bool) or not isinstance(at_step, numbers.Integral) or not 0 <= at_step < len(steps):
        raise ValueError(f"at_step must be a step of the path, 0 to {len(steps) - 1}; got {at_step!r}.")
    ranges = path.get("factor_ranges") or {}
    centre = steps[at_step]["coded"]
    widths = dict(scale or {})
    unknown = sorted(set(widths) - set(centre))
    if unknown:
        raise ValueError(f"scale names {unknown}, which the path does not hold; its factors are {list(centre)}.")
    bad = {name: value for name, value in widths.items() if not (math.isfinite(value) and value > 0)}
    if bad:
        raise ValueError(f"scale must give each factor a positive, finite width ratio; got {bad}.")
    factors = []
    for name, coded in centre.items():
        middle = _coded_to_actual({name: coded}, ranges)[name]
        half = _unit(name, ranges) * widths.get(name, 1.0)
        factors.append(Factor(name=name, low=middle - half, high=middle + half))
    return generate_design(factors, **design)


def _unit(name: str, ranges: Mapping[str, Mapping[str, float]]) -> float:
    """Return a factor's half-range, the actual amount one coded unit is, or 1 without its range."""
    if name not in ranges:
        return 1.0
    return (float(ranges[name]["high"]) - float(ranges[name]["low"])) / 2.0


def _coded(point: Mapping[str, float], names: list[str], ranges: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """Return a setting given in the factors' own units as a coded vector; a factor left out is at its centre."""
    unknown = sorted(set(point) - set(names))
    if unknown:
        raise ValueError(f"base names {unknown}, which are not factors of the model; its factors are {names}.")
    coded = np.zeros(len(names))
    for i, name in enumerate(names):
        if name in point:
            low, high = (float(ranges[name][end]) for end in ("low", "high")) if name in ranges else (-1.0, 1.0)
            coded[i] = (float(point[name]) - (low + high) / 2.0) / ((high - low) / 2.0)
    return coded


def _gradient(coefficients: list[dict[str, Any]], names: list[str], at: np.ndarray) -> np.ndarray:
    """Return the fitted model's slopes at the coded point ``at``, term by term.

    A term is a product of factors (``A:I(B ** 2)`` is A, B and B), so its slope along a factor is
    the product of the others, once for each time the factor appears in it. A slope that is only
    what is left when its terms cancel is set to zero, so the judgement does not depend on the
    units of the response.
    """
    index = {name: i for i, name in enumerate(names)}
    slopes, sizes = np.zeros(len(names)), np.zeros(len(names))
    for parts, coefficient in _parsed_terms(coefficients, names):
        for k, name in enumerate(parts):
            term = coefficient * math.prod(at[index[other]] for other in parts[:k] + parts[k + 1 :])
            slopes[index[name]] += term
            sizes[index[name]] += abs(term)
    slopes[np.abs(slopes) <= _NO_SLOPE * sizes] = 0.0
    return slopes


def _increment(
    way: np.ndarray, step: float | Mapping[str, float], names: list[str], ranges: Mapping[str, Mapping[str, float]]
) -> tuple[str | None, np.ndarray]:
    """Return the lead factor (``None`` for a distance) and the coded change a step makes, along ``way``."""
    if isinstance(step, Mapping):
        if len(step) != 1:
            raise ValueError(f"step must name one lead factor and its amount, as {{'x1': 10}}; got {dict(step)}.")
        ((lead, amount),) = step.items()
        if lead not in names:
            raise ValueError(f"step leads with {lead!r}, which is not a factor of the model; its factors are {names}.")
        if not (isinstance(amount, numbers.Real) and math.isfinite(amount) and amount > 0):
            raise ValueError(f"The step of {lead!r} must be a positive number, in its own units; got {amount!r}.")
        slope = way[names.index(lead)]
        if abs(slope) <= _NO_SLOPE * np.abs(way).max():
            movers = [name for name, s in zip(names, way, strict=True) if abs(s) > _NO_SLOPE * np.abs(way).max()]
            raise ValueError(
                f"The path does not move {lead!r}: its slope at the base is zero, so a step in it cannot set the "
                f"path. Lead with a factor the path moves: {movers}."
            )
        return lead, way * (float(amount) / _unit(lead, ranges)) / abs(slope)
    if isinstance(step, bool) or not (isinstance(step, numbers.Real) and math.isfinite(step) and step > 0):
        raise ValueError(f"step must be a positive number of coded units, or {{lead: amount}}; got {step!r}.")
    return None, way * float(step) / np.linalg.norm(way)
