# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response optimization for designed experiments (Tool 4).

Find optimal factor settings for one or multiple responses after fitting
a model with :func:`analyze_experiment` (Tool 3).

Implemented methods
-------------------
- **desirability** - Derringer-Suich desirability functions (single and
  multi-response) with ``scipy.optimize.minimize`` (SLSQP).
- **steepest_ascent** / **steepest_descent** - Move along the gradient
  of a first-order model from the design centre.
- **stationary_point** - Locate the stationary point of a second-order
  model via ``numpy.linalg.solve``.
- **canonical_analysis** - Eigenvalue decomposition of the *B* matrix
  to classify the stationary point (max / min / saddle).

Stubs (not yet implemented)
---------------------------
- **ridge_analysis** - Trace the optimum along increasing radii.
- **pareto_front** - Multi-objective Pareto frontier (NSGA-II).
"""

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from patsy import PatsyError
from scipy import optimize

from process_improve.experiments._desirability import composite_desirability, individual_desirability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_METHODS = {
    "desirability",
    "steepest_ascent",
    "steepest_descent",
    "stationary_point",
    "canonical_analysis",
    "ridge_analysis",
    "pareto_front",
}

# ---------------------------------------------------------------------------
# Model evaluation layer
# ---------------------------------------------------------------------------


def _parse_term(term: str) -> tuple[str, ...]:
    """Classify a coefficient term name into its components.

    Returns
    -------
    tuple[str, ...]
        Empty tuple for ``"Intercept"``, single-element for linear,
        ``("A", "B")`` for interaction ``"A:B"``, ``("A", "A")`` for
        quadratic ``"I(A ** 2)"``.
    """
    if term == "Intercept":
        return ()

    # Quadratic: ``I(A ** 2)`` (older statsmodels) or
    # ``np.power(A, 2)`` / ``power(A, 2)`` (newer). Both spellings
    # appear in the wild depending on the installed statsmodels /
    # patsy version. SEC-27 (#276): if either is missed, the term
    # silently falls through to the linear branch and the
    # downstream surface / optimisation produces wrong results.
    m = re.match(r"I\((\w+)\s*\*\*\s*2\)", term) or re.match(r"(?:np\.)?power\((\w+)\s*,\s*2\)", term)
    if m:
        name = m.group(1)
        return (name, name)

    # Interaction: A:B
    if ":" in term:
        parts = term.split(":")
        return tuple(parts)

    # Linear: plain factor name
    return (term,)


def _build_model_evaluator(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
) -> Callable[[np.ndarray], float]:
    """Return a function ``f(point) -> float`` that evaluates the model.

    Parameters
    ----------
    coefficients : list[dict]
        Each dict has ``"term"`` and ``"coefficient"`` keys, as returned
        by ``analyze_experiment(..., analysis_type="coefficients")``.
    factor_names : list[str]
        Ordered factor names (e.g. ``["A", "B"]``).

    Returns
    -------
    callable
        ``f(x)`` where *x* is a 1-D array of coded factor values in the
        same order as *factor_names*.
    """
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    parsed: list[tuple[tuple[str, ...], float]] = []
    for entry in coefficients:
        term = entry["term"]
        coef = float(entry["coefficient"])
        parsed.append((_parse_term(term), coef))

    def _eval(x: np.ndarray) -> float:
        y = 0.0
        for components, coef in parsed:
            if len(components) == 0:
                # Intercept
                y += coef
            elif len(components) == 1:
                # Linear
                y += coef * x[name_to_idx[components[0]]]
            elif len(components) == 2:
                # Interaction or quadratic
                y += coef * x[name_to_idx[components[0]]] * x[name_to_idx[components[1]]]
            else:
                # Higher-order (unusual but handle gracefully)
                val = 1.0
                for c in components:
                    val *= x[name_to_idx[c]]
                y += coef * val
        return y

    return _eval


def evaluate_model(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    point: dict[str, float],
) -> float:
    """Evaluate predicted response at an arbitrary coded point.

    Parameters
    ----------
    coefficients : list[dict]
        Coefficient list from ``analyze_experiment``.
    factor_names : list[str]
        Ordered factor names.
    point : dict[str, float]
        Factor settings in coded units, e.g. ``{"A": 0.5, "B": -1.0}``.

    Returns
    -------
    float
        Predicted response value.
    """
    f = _build_model_evaluator(coefficients, factor_names)
    x = np.array([point[n] for n in factor_names], dtype=float)
    return float(f(x))


# ---------------------------------------------------------------------------
# Extract b vector and B matrix from second-order model
# ---------------------------------------------------------------------------


def _extract_b_and_B(  # noqa: N802
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Extract intercept, linear vector *b* and quadratic matrix *B*.

    For a second-order model ``y = b0 + b'x + x'Bx``, returns
    ``(b0, b, B)`` where *B* is symmetric with off-diagonal elements
    equal to half the interaction coefficients.
    """
    k = len(factor_names)
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    b0 = 0.0
    b = np.zeros(k)
    B = np.zeros((k, k))

    for entry in coefficients:
        term = entry["term"]
        coef = float(entry["coefficient"])
        components = _parse_term(term)

        if len(components) == 0:
            b0 = coef
        elif len(components) == 1:
            b[name_to_idx[components[0]]] = coef
        elif len(components) == 2:
            i = name_to_idx[components[0]]
            j = name_to_idx[components[1]]
            if i == j:
                # Quadratic term: coefficient is the diagonal of B
                B[i, i] = coef
            else:
                # Interaction: split equally across B[i,j] and B[j,i]
                B[i, j] = coef / 2.0
                B[j, i] = coef / 2.0

    return b0, b, B


# ---------------------------------------------------------------------------
# Stationary point
# ---------------------------------------------------------------------------


def _find_stationary_point(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Find the stationary point of a second-order response surface model.

    Solves ``2*B*x_s + b = 0`` for ``x_s``.

    Parameters
    ----------
    coefficients : list[dict]
        Model coefficients.
    factor_names : list[str]
        Ordered factor names.
    factor_ranges : dict or None
        Maps factor name to ``{"low": float, "high": float}`` in actual
        units.  Used to convert coded -> actual.
    search_bounds : tuple[float, float] or dict[str, tuple[float, float]] or None
        Coded-unit region used to decide ``inside_design_space``. Pass a
        single ``(low, high)`` tuple to apply the same bounds to every
        factor, or a per-factor dict to give each factor its own bounds.
        Defaults to the factorial cube ``(-1, 1)`` for each factor when
        ``None``; supply a wider region for e.g. a central composite
        design's axial distance.

    Returns
    -------
    dict
        ``stationary_point_coded``, ``predicted_response``, ``classification``,
        ``eigenvalues`` (list of floats, spectrum of the pure-quadratic
        matrix ``B``), and ``inside_design_space`` (bool, whether the
        stationary point falls inside ``search_bounds``). Also includes
        ``stationary_point_actual`` when ``factor_ranges`` is provided.
        Returns a dict with a single ``error`` key instead when the model
        has no quadratic/interaction terms or ``B`` is singular.
    """
    b0, b, B = _extract_b_and_B(coefficients, factor_names)

    # Check that B has quadratic terms (not purely first-order)
    if np.allclose(B, 0):
        return {"error": "Model has no quadratic or interaction terms - cannot find stationary point."}

    try:
        # Solve 2*B*x_s = -b
        x_s = np.linalg.solve(2.0 * B, -b)
    except np.linalg.LinAlgError:
        return {"error": "Singular B matrix - stationary point does not exist."}

    # Predicted response at stationary point
    y_s = float(b0 + b @ x_s + x_s @ B @ x_s)

    # Classification from eigenvalues
    eigenvalues = np.linalg.eigvalsh(B)
    if np.all(eigenvalues < 0):
        classification = "maximum"
    elif np.all(eigenvalues > 0):
        classification = "minimum"
    else:
        classification = "saddle_point"

    # Is the stationary point inside the region the experiment covered? The
    # default region is the factorial cube; a central composite design reaches
    # further, so its axial distance can be supplied via search_bounds.
    region = _resolve_search_bounds(search_bounds, factor_names)
    inside_design_space = bool(all(low <= value <= high for value, (low, high) in zip(x_s, region, strict=True)))

    result: dict[str, Any] = {
        "stationary_point_coded": {n: float(x_s[i]) for i, n in enumerate(factor_names)},
        "predicted_response": y_s,
        "classification": classification,
        "eigenvalues": [float(e) for e in eigenvalues],
        "inside_design_space": inside_design_space,
    }

    if factor_ranges:
        actual = {}
        for i, name in enumerate(factor_names):
            if name in factor_ranges:
                lo = factor_ranges[name]["low"]
                hi = factor_ranges[name]["high"]
                center = (lo + hi) / 2.0
                half_range = (hi - lo) / 2.0
                actual[name] = center + x_s[i] * half_range
            else:
                actual[name] = float(x_s[i])
        result["stationary_point_actual"] = actual

    return result


# ---------------------------------------------------------------------------
# Canonical analysis
# ---------------------------------------------------------------------------


def _canonical_analysis(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
) -> dict[str, Any]:
    """Canonical analysis of a second-order response surface model.

    Computes eigenvalues and eigenvectors of the *B* matrix to determine
    the shape and orientation of the response surface.

    Returns
    -------
    dict
        ``eigenvalues``, ``eigenvectors``, ``classification``,
        ``canonical_form_description``.
    """
    _b0, _b, B = _extract_b_and_B(coefficients, factor_names)

    if np.allclose(B, 0):
        return {"error": "Model has no quadratic or interaction terms - canonical analysis not applicable."}

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by absolute value (largest first)
    order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.all(eigenvalues < 0):
        classification = "maximum"
    elif np.all(eigenvalues > 0):
        classification = "minimum"
    else:
        classification = "saddle_point"

    desc_parts = []
    for i, ev in enumerate(eigenvalues):
        w_name = f"W{i + 1}"
        direction = "concave" if ev < 0 else "convex"
        desc_parts.append(f"{w_name}: eigenvalue={ev:.4f} ({direction})")

    return {
        "eigenvalues": [float(e) for e in eigenvalues],
        "eigenvectors": [[float(v) for v in eigenvectors[:, i]] for i in range(len(eigenvalues))],
        "classification": classification,
        "canonical_form_description": desc_parts,
        "factor_names": factor_names,
    }


# ---------------------------------------------------------------------------
# Steepest ascent / descent
# ---------------------------------------------------------------------------


def _steepest_path(  # noqa: PLR0913
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    step_size: float = 0.5,
    n_steps: int = 10,
    direction: str = "ascent",
    factor_ranges: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Generate a table of steps along the steepest ascent (or descent).

    Uses only the first-order (linear) coefficients to determine
    direction.  Steps start at the design centre (all coded = 0).

    Parameters
    ----------
    coefficients : list[dict]
        Model coefficients.
    factor_names : list[str]
        Ordered factor names.
    step_size : float
        Step magnitude in coded units (default 0.5).
    n_steps : int
        Number of steps to take away from the design centre (default 10).
        The returned ``steps`` list has ``n_steps + 1`` entries because it
        also includes step 0 at the centre.
    direction : str
        ``"ascent"`` or ``"descent"``.
    factor_ranges : dict or None
        For coded → actual conversion.

    Returns
    -------
    dict
        ``steps`` list and ``direction_vector``.
    """
    evaluator = _build_model_evaluator(coefficients, factor_names)

    # Extract linear coefficients only
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    b = np.zeros(len(factor_names))
    for entry in coefficients:
        components = _parse_term(entry["term"])
        if len(components) == 1 and components[0] in name_to_idx:
            b[name_to_idx[components[0]]] = float(entry["coefficient"])

    if np.allclose(b, 0):
        return {"error": "All linear coefficients are zero - no steepest direction."}

    # Direction: normalize, then scale by step_size
    norm = np.linalg.norm(b)
    direction_vec = b / norm
    if direction == "descent":
        direction_vec = -direction_vec

    steps = []
    for step_num in range(n_steps + 1):
        x_coded = direction_vec * step_size * step_num
        predicted = float(evaluator(x_coded))

        step_entry: dict[str, Any] = {
            "step": step_num,
            "coded": {n: float(x_coded[i]) for i, n in enumerate(factor_names)},
            "predicted_response": predicted,
        }

        if factor_ranges:
            actual = {}
            for i, name in enumerate(factor_names):
                if name in factor_ranges:
                    lo = factor_ranges[name]["low"]
                    hi = factor_ranges[name]["high"]
                    center = (lo + hi) / 2.0
                    half_range = (hi - lo) / 2.0
                    actual[name] = center + x_coded[i] * half_range
                else:
                    actual[name] = float(x_coded[i])
            step_entry["actual"] = actual

        steps.append(step_entry)

    return {
        "direction": direction,
        "direction_vector": {n: float(direction_vec[i]) for i, n in enumerate(factor_names)},
        "step_size": step_size,
        "steps": steps,
    }


def _resolve_search_bounds(
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None,
    factor_names: list[str],
) -> list[tuple[float, float]]:
    """Return per-factor coded bounds for the region to search.

    The default of (-1, 1) is the factorial cube, which is the right region for
    a two-level design. It is not the right region for a central composite
    design, whose axial runs sit at plus or minus alpha: restricting the search
    to the cube there would refuse to consider settings the experiment actually
    covered. Pass the design's axial distance to search the whole region.

    Parameters
    ----------
    search_bounds : tuple, dict, or None
        A single ``(low, high)`` pair applied to every factor, or a mapping from
        factor name to its own pair. Factors absent from the mapping fall back to
        (-1, 1). ``None`` means (-1, 1) throughout.
    factor_names : list[str]
        Ordered factor names.

    Returns
    -------
    list[tuple[float, float]]
        One ``(low, high)`` pair per factor, in *factor_names* order.

    Raises
    ------
    ValueError
        If a pair is malformed, non-finite, or has low >= high, or if the
        mapping names a factor the model does not have.
    """
    default = (-1.0, 1.0)

    def _check(pair: Sequence[float], where: str) -> tuple[float, float]:
        try:
            low, high = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            msg = f"search_bounds{where} must be a (low, high) pair of numbers; got {pair!r}."
            raise ValueError(msg) from exc
        if not (np.isfinite(low) and np.isfinite(high)):
            msg = f"search_bounds{where} must be finite; got ({low}, {high})."
            raise ValueError(msg)
        if low >= high:
            msg = f"search_bounds{where} must have low < high; got ({low}, {high})."
            raise ValueError(msg)
        return low, high

    if search_bounds is None:
        return [default] * len(factor_names)

    if isinstance(search_bounds, dict):
        unknown = set(search_bounds) - set(factor_names)
        if unknown:
            msg = f"search_bounds names unknown factor(s) {sorted(unknown)}; the model has {factor_names}."
            raise ValueError(msg)
        return [
            _check(search_bounds[name], f"[{name!r}]") if name in search_bounds else default for name in factor_names
        ]

    return [_check(search_bounds, "")] * len(factor_names)


def _align_goals_to_models(
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return *goals* reordered to match *fitted_models*.

    Goals were previously consumed in list order while ``goal["response"]`` was
    documented as the key that ties a goal to its model. Passing the two lists
    in different orders therefore optimised the wrong thing without complaint.

    When every model names its response and every goal names a matching one, the
    goals are reordered by name. Otherwise the original positional order is kept,
    with a warning, since that is the only interpretation left.

    Parameters
    ----------
    fitted_models : list[dict]
        Each optionally has ``"response_name"``.
    goals : list[dict]
        Each optionally has ``"response"``.

    Returns
    -------
    list[dict]
        Goals in the same order as *fitted_models*.

    Raises
    ------
    ValueError
        If the two lists differ in length.
    """
    if len(goals) != len(fitted_models):
        msg = f"Got {len(fitted_models)} fitted model(s) but {len(goals)} goal(s); they must correspond one to one."
        raise ValueError(msg)

    model_names = [m.get("response_name") for m in fitted_models]
    goal_names = [g.get("response") for g in goals]

    if any(n is None for n in model_names) or any(n is None for n in goal_names):
        logger.warning(
            "Matching goals to fitted models by position: not every model has 'response_name' and not every "
            "goal has 'response'. Name both to have them matched by name instead."
        )
        return goals

    by_name = {str(g["response"]): g for g in goals}
    if len(by_name) != len(goals) or set(by_name) != {str(n) for n in model_names}:
        logger.warning(
            "Matching goals to fitted models by position: the goal 'response' names %s do not correspond "
            "one to one with the model 'response_name' values %s.",
            sorted(str(n) for n in goal_names),
            sorted(str(n) for n in model_names),
        )
        return goals

    return [by_name[str(n)] for n in model_names]


def _optimize_desirability(  # noqa: PLR0913
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
    importances: list[float] | None = None,
    random_state: int | np.random.Generator | None = 42,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Optimise composite desirability using scipy SLSQP.

    Parameters
    ----------
    fitted_models : list[dict]
        Each has ``"coefficients"`` and ``"response_name"``.
    goals : list[dict]
        Per-response goals. Matched to *fitted_models* by response name when
        both sides supply one, otherwise by position.
    factor_names : list[str]
        Ordered factor names.
    factor_ranges : dict or None
        Factor bounds in actual units.
    importances : list[float] or None
        Relative importance of each response in the composite. This is not the
        same as a goal's ``weight``, which shapes that response's own ramp.
    search_bounds : tuple, dict, or None
        Coded region to search. Defaults to the factorial cube, (-1, 1).

    Returns
    -------
    dict
        Optimal settings, predicted responses, individual and composite
        desirability.
    """
    goals = _align_goals_to_models(fitted_models, goals)
    evaluators = [_build_model_evaluator(m["coefficients"], factor_names) for m in fitted_models]

    def neg_composite(x: np.ndarray) -> float:
        """Return the negated composite desirability at coded settings ``x``, for minimization."""
        d_vals = []
        for evaluator, goal in zip(evaluators, goals, strict=True):
            y_pred = evaluator(x)
            d = individual_desirability(y_pred, goal)
            d_vals.append(d)
        return -composite_desirability(d_vals, importances)

    bounds = _resolve_search_bounds(search_bounds, factor_names)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])

    # Multi-start: try centre + random points.
    # SEC-33 (#282): the hard-coded ``42`` moved to the public signature
    # ``random_state=42`` (default preserves the previous deterministic
    # behaviour). Resolved via the ENG-08 helper.
    from process_improve._random import check_random_state  # noqa: PLC0415

    rng = check_random_state(random_state)
    best_result = None
    best_value = np.inf

    # Start from the centre of the searched region, then sample across it, so
    # that widening the bounds actually widens where the search looks.
    centre = (lows + highs) / 2.0
    starting_points = [centre, *[rng.uniform(lows, highs) for _ in range(9)]]

    for x0 in starting_points:
        res = optimize.minimize(neg_composite, x0, method="SLSQP", bounds=bounds)
        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    if best_result is None:
        msg = "optimization produced no result"
        raise RuntimeError(msg)

    x_opt = best_result.x
    composite_d = -best_value

    # Evaluate individual responses and desirabilities at optimum
    predictions = {}
    individual_d = {}
    for evaluator, model_dict, goal in zip(evaluators, fitted_models, goals, strict=True):
        resp_name = model_dict.get("response_name", "response")
        y_pred = float(evaluator(x_opt))
        predictions[resp_name] = y_pred
        individual_d[resp_name] = individual_desirability(y_pred, goal)

    result: dict[str, Any] = {
        "optimal_coded": {n: float(x_opt[i]) for i, n in enumerate(factor_names)},
        "predicted_responses": predictions,
        "individual_desirability": individual_d,
        "composite_desirability": composite_d,
        "optimizer_success": bool(best_result.success),
    }

    if factor_ranges:
        actual = {}
        for i, name in enumerate(factor_names):
            if name in factor_ranges:
                lo = factor_ranges[name]["low"]
                hi = factor_ranges[name]["high"]
                center = (lo + hi) / 2.0
                half_range = (hi - lo) / 2.0
                actual[name] = center + x_opt[i] * half_range
            else:
                actual[name] = float(x_opt[i])
        result["optimal_actual"] = actual

    return result


# ---------------------------------------------------------------------------
# Stubs for future implementation
# ---------------------------------------------------------------------------


def _ridge_analysis(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Ridge analysis - trace the optimum along increasing radii.

    .. note::
        Not yet implemented.  Planned: constrained eigenvalue computation
        tracing the optimum on spheres of increasing radius from the
        design centre when the stationary point lies outside the design
        space.
    """
    return {
        "error": (
            "Ridge analysis is not yet implemented. Use 'stationary_point' or 'canonical_analysis' as alternatives."
        ),
        "status": "stub",
    }


def _pareto_front(
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Multi-objective Pareto frontier.

    .. note::
        Not yet implemented.  Planned: multi-start ``scipy.optimize``
        wrapper or ``pymoo`` NSGA-II for true Pareto frontiers with
        many responses.
    """
    return {
        "error": ("Pareto front is not yet implemented. Use 'desirability' for multi-response optimization instead."),
        "status": "stub",
    }


# ---------------------------------------------------------------------------
# Coded ↔ actual conversion helpers
# ---------------------------------------------------------------------------


def _coded_to_actual(coded: dict[str, float], factor_ranges: dict[str, dict[str, float]]) -> dict[str, float]:
    """Convert coded factor settings to actual units."""
    actual = {}
    for name, coded_val in coded.items():
        if name in factor_ranges:
            lo = factor_ranges[name]["low"]
            hi = factor_ranges[name]["high"]
            center = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            actual[name] = center + coded_val * half_range
        else:
            actual[name] = coded_val
    return actual


# ---------------------------------------------------------------------------
# Public API - dispatcher
# ---------------------------------------------------------------------------


def _intervals_at_point(
    fitted_results: list[Any],
    fitted_models: list[dict[str, Any]],
    factor_names: list[str],
    point_coded: dict[str, float],
    significance_level: float,
) -> dict[str, Any]:
    """Confidence and prediction intervals for each response at one point.

    The optimizer works from coefficients alone, which is enough to locate an
    optimum but not to say how well it is known. The residual variance and the
    design's leverage at that point are needed for that, and both live on the
    fitted model object rather than in its coefficients.

    Parameters
    ----------
    fitted_results : list
        Statsmodels results objects, aligned with *fitted_models*, fitted on
        the coded factors.
    fitted_models : list[dict]
        Used only for the response names.
    factor_names : list[str]
        Ordered factor names, matching the columns the models were fitted on.
    point_coded : dict[str, float]
        Coded factor settings at which to report the intervals.
    significance_level : float
        Alpha. 0.05 gives 95% intervals.

    Returns
    -------
    dict
        Keyed by response name. Each entry has ``predicted``,
        ``confidence_interval``, ``prediction_interval``, and
        ``confidence_level``. A response whose model cannot be evaluated
        carries an ``error`` string instead, so one failure does not discard
        the intervals for the others.
    """
    from process_improve.experiments._analyses.prediction import _run_prediction  # noqa: PLC0415

    if len(fitted_results) != len(fitted_models):
        msg = (
            f"Got {len(fitted_models)} fitted model(s) but {len(fitted_results)} fitted result(s); "
            "they must correspond one to one and be in the same order."
        )
        raise ValueError(msg)

    new_point = pd.DataFrame([{name: point_coded[name] for name in factor_names}])

    intervals: dict[str, Any] = {}
    for i, (results_obj, model) in enumerate(zip(fitted_results, fitted_models, strict=True)):
        resp_name = model.get("response_name", f"Response {i + 1}")
        try:
            record = _run_prediction(results_obj, new_point, alpha=significance_level)["predictions"][0]
        except (AttributeError, KeyError, TypeError, ValueError, PatsyError) as exc:
            logger.warning("Could not compute intervals for response %r: %s", resp_name, exc)
            intervals[resp_name] = {"error": str(exc)}
            continue

        intervals[resp_name] = {
            "predicted": record["predicted"],
            "confidence_interval": [record["ci_low"], record["ci_high"]],
            "prediction_interval": [record["pi_low"], record["pi_high"]],
            "confidence_level": 1.0 - significance_level,
        }
    return intervals


def _desirability_result(  # noqa: PLR0913
    *,
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None,
    response_importance: list[float] | None,
    fitted_results: list[Any] | None,
    significance_level: float,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Assemble the full desirability result: optimum, intervals, and plot input.

    Returns
    -------
    dict
        The optimum from :func:`_optimize_desirability`, plus
        ``"response_intervals"`` when *fitted_results* is supplied, plus
        ``"responses"``, which pairs each model's coefficients with its
        specification limits so the result can be passed straight to the
        overlay plot.
    """
    aligned_goals = _align_goals_to_models(fitted_models, goals)
    importances = response_importance
    if importances is None:
        importances = [g.get("importance", 1.0) for g in aligned_goals]

    desirability = _optimize_desirability(
        fitted_models, aligned_goals, factor_names, factor_ranges, importances, search_bounds=search_bounds
    )

    if fitted_results is not None:
        desirability["response_intervals"] = _intervals_at_point(
            fitted_results, fitted_models, factor_names, desirability["optimal_coded"], significance_level
        )

    carried = ("goal", "low", "high", "target", "weight", "weight_high", "importance")
    desirability["responses"] = [
        {
            "name": model.get("response_name", f"Response {i + 1}"),
            "coefficients": model.get("coefficients", []),
            **{key: goal[key] for key in carried if key in goal},
        }
        for i, (model, goal) in enumerate(zip(fitted_models, aligned_goals, strict=True))
    ]
    return desirability


def optimize_responses(  # noqa: PLR0913, C901
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]] | None = None,
    method: str = "desirability",
    factor_ranges: dict[str, dict[str, float]] | None = None,
    step_size: float = 0.5,
    n_steps: int = 10,
    response_importance: list[float] | None = None,
    fitted_results: list[Any] | None = None,
    significance_level: float = 0.05,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    desirability_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Find optimal factor settings for one or multiple responses.

    Parameters
    ----------
    fitted_models : list[dict]
        Each dict describes a fitted model with keys:

        - ``"response_name"`` (str) - name of the response.
        - ``"coefficients"`` (list[dict]) - coefficient list, each with
          ``"term"`` and ``"coefficient"`` keys as returned by
          ``analyze_experiment(..., analysis_type="coefficients")``.
        - ``"factor_names"`` (list[str]) - ordered factor names.
        - ``"mse_residual"`` (float, optional) - mean squared error.
        - ``"r_squared"`` (float, optional) - model R-squared.

    goals : list[dict] or None
        Per-response optimisation goals.  Each dict has keys:

        - ``"response"`` (str) - response name. Matched against each model's
          ``"response_name"``; when both sides name their responses the goals
          are reordered to match, so the two lists need not be in the same
          order. When either side omits a name, goals are taken in list order.
        - ``"goal"`` (str) - ``"maximize"``, ``"minimize"``, or
          ``"target"``.
        - ``"target"`` (float, optional) - target value (required when
          ``goal="target"``).
        - ``"low"`` (float) - lower acceptable bound.
        - ``"high"`` (float) - upper acceptable bound.
        - ``"weight"`` (float, default 1) - the exponent shaping *this*
          response's desirability ramp between ``low`` and ``high``. Above 1
          concentrates desirability near the good end; below 1 flattens it.
        - ``"weight_high"`` (float, optional) - a separate exponent for the
          falling side of a ``"target"`` goal. Defaults to ``"weight"``.
        - ``"importance"`` (float, default 1) - how much this response counts
          relative to the others when the composite is formed. Unlike
          ``weight``, it has no effect on this response's own ramp.

    method : str
        Optimisation method: ``"desirability"``,
        ``"steepest_ascent"``, ``"steepest_descent"``,
        ``"stationary_point"``, ``"canonical_analysis"``,
        ``"ridge_analysis"`` (stub), ``"pareto_front"`` (stub).
    factor_ranges : dict or None
        Maps factor name to ``{"low": float, "high": float}`` in actual
        units.  Used for coded ↔ actual conversion.
    step_size : float
        Step magnitude for steepest ascent/descent (coded units).
    n_steps : int
        Number of steps for steepest ascent/descent.
    response_importance : list[float] or None
        Relative importance per response, overriding the per-goal
        ``"importance"`` values. Aligned with *fitted_models*.
    fitted_results : list or None
        Optional statsmodels results objects, one per entry in *fitted_models*
        and in the same order, as returned by ``lm()`` or by
        ``analyze_experiment``. When supplied, a confidence interval and a
        prediction interval for each response are reported at the optimum.
        The models must have been fitted on the coded factors, since the
        optimum is located in coded units.
    significance_level : float
        Alpha for those intervals. The default of 0.05 gives 95% intervals.
    search_bounds : tuple, dict, or None
        The coded region to search, and the region against which a stationary
        point is judged inside or outside. Defaults to the factorial cube,
        ``(-1, 1)`` on every factor.

        That default suits a two-level design but understates a central
        composite design, whose axial runs sit at plus or minus alpha: leaving
        it at the cube would refuse to consider settings the experiment
        actually covered. Pass ``(-1.41, 1.41)`` for a two-factor rotatable
        central composite design, or a mapping such as
        ``{"T": (-1.41, 1.41)}`` to widen one factor only. Factors left out of
        a mapping keep the (-1, 1) default.
    desirability_weights : list[float] or None
        Deprecated alias for *response_importance*. The name was misleading:
        these values are importances, not the ``weight`` that shapes an
        individual ramp.

    Returns
    -------
    dict[str, Any]
        Results keyed by method.  Always includes ``"method"`` and
        ``"factor_names"``.

    Raises
    ------
    ValueError
        If *method* is unknown, if *fitted_models* is empty, if a method that
        needs goals is called without them, or if both *response_importance*
        and *desirability_weights* are given.

    Examples
    --------
    >>> from process_improve.experiments.optimization import optimize_responses
    >>> model = {
    ...     "response_name": "yield",
    ...     "coefficients": [
    ...         {"term": "Intercept", "coefficient": 40.0},
    ...         {"term": "A", "coefficient": 5.25},
    ...         {"term": "B", "coefficient": -2.0},
    ...         {"term": "I(A ** 2)", "coefficient": -3.0},
    ...         {"term": "I(B ** 2)", "coefficient": -1.5},
    ...         {"term": "A:B", "coefficient": 1.5},
    ...     ],
    ...     "factor_names": ["A", "B"],
    ... }
    >>> result = optimize_responses(
    ...     fitted_models=[model],
    ...     method="stationary_point",
    ... )
    >>> result["stationary_point"]["classification"]
    'maximum'
    """
    logger.debug("optimize_responses: method=%r, %d fitted model(s)", method, len(fitted_models))
    if method not in _METHODS:
        available = sorted(_METHODS)
        msg = f"Unknown method {method!r}. Available: {available}"
        raise ValueError(msg)

    if not fitted_models:
        msg = "At least one fitted model is required."
        raise ValueError(msg)

    if desirability_weights is not None:
        if response_importance is not None:
            msg = (
                "Pass either 'response_importance' or the deprecated 'desirability_weights', not both. "
                "They set the same thing: how much each response counts in the composite."
            )
            raise ValueError(msg)
        warnings.warn(
            "'desirability_weights' is deprecated; use 'response_importance'. The values are importances, "
            "which set how much each response counts in the composite, not the per-goal 'weight' that shapes "
            "an individual desirability ramp.",
            DeprecationWarning,
            stacklevel=2,
        )
        response_importance = desirability_weights

    # Use factor_names from the first model as the canonical ordering
    factor_names = fitted_models[0]["factor_names"]
    coefficients = fitted_models[0]["coefficients"]

    result: dict[str, Any] = {"method": method, "factor_names": factor_names}

    if method == "stationary_point":
        result["stationary_point"] = _find_stationary_point(coefficients, factor_names, factor_ranges, search_bounds)

    elif method == "canonical_analysis":
        result["canonical_analysis"] = _canonical_analysis(coefficients, factor_names)
        # Also include the stationary point for context
        result["stationary_point"] = _find_stationary_point(coefficients, factor_names, factor_ranges, search_bounds)

    elif method in ("steepest_ascent", "steepest_descent"):
        direction = "ascent" if method == "steepest_ascent" else "descent"
        result["steepest_path"] = _steepest_path(
            coefficients, factor_names, step_size, n_steps, direction, factor_ranges
        )

    elif method == "desirability":
        if goals is None:
            msg = "Goals are required for desirability optimization."
            raise ValueError(msg)
        result["desirability"] = _desirability_result(
            fitted_models=fitted_models,
            goals=goals,
            factor_names=factor_names,
            factor_ranges=factor_ranges,
            response_importance=response_importance,
            fitted_results=fitted_results,
            significance_level=significance_level,
            search_bounds=search_bounds,
        )

    elif method == "ridge_analysis":
        result["ridge_analysis"] = _ridge_analysis(coefficients, factor_names, factor_ranges)

    elif method == "pareto_front":
        if goals is None:
            msg = "Goals are required for Pareto front optimization."
            raise ValueError(msg)
        result["pareto_front"] = _pareto_front(fitted_models, goals, factor_names, factor_ranges)

    return result
