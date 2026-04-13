# (c) Kevin Dunn, 2010-2026. MIT License.

"""Response optimization for designed experiments (Tool 4).

Find optimal factor settings for one or multiple responses after fitting
a model with :func:`analyze_experiment` (Tool 3).

Implemented methods
-------------------
- **desirability** — Derringer-Suich desirability functions (single and
  multi-response) with ``scipy.optimize.minimize`` (SLSQP).
- **steepest_ascent** / **steepest_descent** — Move along the gradient
  of a first-order model from the design centre.
- **stationary_point** — Locate the stationary point of a second-order
  model via ``numpy.linalg.solve``.
- **canonical_analysis** — Eigenvalue decomposition of the *B* matrix
  to classify the stationary point (max / min / saddle).

Stubs (not yet implemented)
---------------------------
- **ridge_analysis** — Trace the optimum along increasing radii.
- **pareto_front** — Multi-objective Pareto frontier (NSGA-II).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import optimize

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

    # Quadratic: I(A ** 2) or np.power(A, 2) patterns
    m = re.match(r"I\((\w+)\s*\*\*\s*2\)", term)
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
) -> callable:
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
            elif len(components) == 2:  # noqa: PLR2004
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
        elif len(components) == 2:  # noqa: PLR2004
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
        units.  Used to convert coded → actual.

    Returns
    -------
    dict
        ``stationary_point_coded``, ``stationary_point_actual``,
        ``predicted_response``, ``classification``.
    """
    b0, b, B = _extract_b_and_B(coefficients, factor_names)

    # Check that B has quadratic terms (not purely first-order)
    if np.allclose(B, 0):
        return {"error": "Model has no quadratic or interaction terms — cannot find stationary point."}

    try:
        # Solve 2*B*x_s = -b
        x_s = np.linalg.solve(2.0 * B, -b)
    except np.linalg.LinAlgError:
        return {"error": "Singular B matrix — stationary point does not exist."}

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

    # Check if stationary point is inside the design space (coded [-1, 1])
    inside_design_space = bool(np.all(np.abs(x_s) <= 1.0))

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
        return {"error": "Model has no quadratic or interaction terms — canonical analysis not applicable."}

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


def _steepest_path(
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
        Number of steps to generate (default 10).
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
        return {"error": "All linear coefficients are zero — no steepest direction."}

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


# ---------------------------------------------------------------------------
# Derringer-Suich desirability functions
# ---------------------------------------------------------------------------


def _desirability_maximize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """One-sided desirability for maximisation.

    d = 0 if y <= low, d = ((y - low) / (high - low))^weight if
    low < y < high, d = 1 if y >= high.
    """
    if y <= low:
        return 0.0
    if y >= high:
        return 1.0
    return ((y - low) / (high - low)) ** weight


def _desirability_minimize(y: float, low: float, high: float, weight: float = 1.0) -> float:
    """One-sided desirability for minimisation.

    d = 1 if y <= low, d = ((high - y) / (high - low))^weight if
    low < y < high, d = 0 if y >= high.
    """
    if y <= low:
        return 1.0
    if y >= high:
        return 0.0
    return ((high - y) / (high - low)) ** weight


def _desirability_target(
    y: float,
    low: float,
    target: float,
    high: float,
    weight_low: float = 1.0,
    weight_high: float = 1.0,
) -> float:
    """Two-sided desirability for a target value.

    Ramps up from *low* to *target*, then ramps down from *target* to
    *high*.  Returns 0 outside ``[low, high]``.
    """
    if y <= low or y >= high:
        return 0.0
    if y <= target:
        return ((y - low) / (target - low)) ** weight_low
    return ((high - y) / (high - target)) ** weight_high


def _individual_desirability(y: float, goal: dict[str, Any]) -> float:
    """Compute individual desirability for a single response value."""
    goal_type = goal["goal"]
    low = goal.get("low", 0.0)
    high = goal.get("high", 1.0)
    weight = goal.get("weight", 1.0)

    if goal_type == "maximize":
        return _desirability_maximize(y, low, high, weight)
    if goal_type == "minimize":
        return _desirability_minimize(y, low, high, weight)
    if goal_type == "target":
        target = goal["target"]
        return _desirability_target(y, low, target, high, weight, weight)

    msg = f"Unknown goal type: {goal_type!r}. Use 'maximize', 'minimize', or 'target'."
    raise ValueError(msg)


def _composite_desirability(d_values: list[float], importances: list[float] | None = None) -> float:
    """Weighted geometric mean of individual desirability values.

    D = (d1^w1 * d2^w2 * ... * dk^wk) ^ (1 / sum(wi))

    If any d_i is zero, the composite is zero.
    """
    if not d_values:
        return 0.0

    weights = importances if importances else [1.0] * len(d_values)

    # If any desirability is zero, composite is zero
    if any(d == 0.0 for d in d_values):
        return 0.0

    log_d = sum(w * np.log(d) for d, w in zip(d_values, weights))
    w_sum = sum(weights)
    if w_sum == 0:
        return 0.0
    return float(np.exp(log_d / w_sum))


def _optimize_desirability(
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
    importances: list[float] | None = None,
) -> dict[str, Any]:
    """Optimise composite desirability using scipy SLSQP.

    Parameters
    ----------
    fitted_models : list[dict]
        Each has ``"coefficients"`` and ``"response_name"``.
    goals : list[dict]
        Per-response goals.
    factor_names : list[str]
        Ordered factor names.
    factor_ranges : dict or None
        Factor bounds in actual units.
    importances : list[float] or None
        Relative importance weights for composite desirability.

    Returns
    -------
    dict
        Optimal settings, predicted responses, individual and composite
        desirability.
    """
    evaluators = [_build_model_evaluator(m["coefficients"], factor_names) for m in fitted_models]

    def neg_composite(x: np.ndarray) -> float:
        d_vals = []
        for evaluator, goal in zip(evaluators, goals):
            y_pred = evaluator(x)
            d = _individual_desirability(y_pred, goal)
            d_vals.append(d)
        return -_composite_desirability(d_vals, importances)

    k = len(factor_names)
    bounds = [(-1.0, 1.0)] * k

    # Multi-start: try centre + random points
    rng = np.random.default_rng(42)
    best_result = None
    best_value = np.inf

    starting_points = [np.zeros(k)]
    for _ in range(9):
        starting_points.append(rng.uniform(-1, 1, size=k))

    for x0 in starting_points:
        res = optimize.minimize(neg_composite, x0, method="SLSQP", bounds=bounds)
        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    x_opt = best_result.x
    composite_d = -best_value

    # Evaluate individual responses and desirabilities at optimum
    predictions = {}
    individual_d = {}
    for evaluator, model_dict, goal in zip(evaluators, fitted_models, goals):
        resp_name = model_dict.get("response_name", "response")
        y_pred = float(evaluator(x_opt))
        predictions[resp_name] = y_pred
        individual_d[resp_name] = _individual_desirability(y_pred, goal)

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
    """Ridge analysis — trace the optimum along increasing radii.

    .. note::
        Not yet implemented.  Planned: constrained eigenvalue computation
        tracing the optimum on spheres of increasing radius from the
        design centre when the stationary point lies outside the design
        space.
    """
    return {
        "error": (
            "Ridge analysis is not yet implemented. "
            "Use 'stationary_point' or 'canonical_analysis' as alternatives."
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
        "error": (
            "Pareto front is not yet implemented. "
            "Use 'desirability' for multi-response optimization instead."
        ),
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
# Public API — dispatcher
# ---------------------------------------------------------------------------


def optimize_responses(  # noqa: PLR0912, PLR0913, C901
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]] | None = None,
    method: str = "desirability",
    factor_ranges: dict[str, dict[str, float]] | None = None,
    step_size: float = 0.5,
    n_steps: int = 10,
    desirability_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Find optimal factor settings for one or multiple responses.

    Parameters
    ----------
    fitted_models : list[dict]
        Each dict describes a fitted model with keys:

        - ``"response_name"`` (str) — name of the response.
        - ``"coefficients"`` (list[dict]) — coefficient list, each with
          ``"term"`` and ``"coefficient"`` keys as returned by
          ``analyze_experiment(..., analysis_type="coefficients")``.
        - ``"factor_names"`` (list[str]) — ordered factor names.
        - ``"mse_residual"`` (float, optional) — mean squared error.
        - ``"r_squared"`` (float, optional) — model R-squared.

    goals : list[dict] or None
        Per-response optimisation goals.  Each dict has keys:

        - ``"response"`` (str) — response name (must match a model).
        - ``"goal"`` (str) — ``"maximize"``, ``"minimize"``, or
          ``"target"``.
        - ``"target"`` (float, optional) — target value (required when
          ``goal="target"``).
        - ``"low"`` (float) — lower acceptable bound.
        - ``"high"`` (float) — upper acceptable bound.
        - ``"weight"`` (float, default 1) — desirability shape parameter.
        - ``"importance"`` (float, default 1) — relative importance for
          composite desirability.

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
    desirability_weights : list[float] or None
        Importance weights for composite desirability (overrides per-goal
        ``"importance"`` values).

    Returns
    -------
    dict[str, Any]
        Results keyed by method.  Always includes ``"method"`` and
        ``"factor_names"``.

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
    if method not in _METHODS:
        available = sorted(_METHODS)
        msg = f"Unknown method {method!r}. Available: {available}"
        raise ValueError(msg)

    if not fitted_models:
        msg = "At least one fitted model is required."
        raise ValueError(msg)

    # Use factor_names from the first model as the canonical ordering
    factor_names = fitted_models[0]["factor_names"]
    coefficients = fitted_models[0]["coefficients"]

    result: dict[str, Any] = {"method": method, "factor_names": factor_names}

    if method == "stationary_point":
        result["stationary_point"] = _find_stationary_point(coefficients, factor_names, factor_ranges)

    elif method == "canonical_analysis":
        result["canonical_analysis"] = _canonical_analysis(coefficients, factor_names)
        # Also include the stationary point for context
        result["stationary_point"] = _find_stationary_point(coefficients, factor_names, factor_ranges)

    elif method in ("steepest_ascent", "steepest_descent"):
        direction = "ascent" if method == "steepest_ascent" else "descent"
        result["steepest_path"] = _steepest_path(
            coefficients, factor_names, step_size, n_steps, direction, factor_ranges
        )

    elif method == "desirability":
        if goals is None:
            msg = "Goals are required for desirability optimization."
            raise ValueError(msg)

        importances = desirability_weights
        if importances is None:
            importances = [g.get("importance", 1.0) for g in goals]

        result["desirability"] = _optimize_desirability(
            fitted_models, goals, factor_names, factor_ranges, importances
        )

    elif method == "ridge_analysis":
        result["ridge_analysis"] = _ridge_analysis(coefficients, factor_names, factor_ranges)

    elif method == "pareto_front":
        if goals is None:
            msg = "Goals are required for Pareto front optimization."
            raise ValueError(msg)
        result["pareto_front"] = _pareto_front(fitted_models, goals, factor_names, factor_ranges)

    return result
