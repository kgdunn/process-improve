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
- **ridge_analysis** - Trace the constrained optimum along spheres of
  increasing radius, by solving Draper's secular equation for the
  Lagrange multiplier.
- **pareto_front** - The non-dominated set over several responses, via
  augmented Chebyshev scalarisation on a Das-Dennis weight lattice.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import math
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from patsy import PatsyError
from scipy import optimize

from process_improve._random import check_random_state
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
# Ridge analysis
# ---------------------------------------------------------------------------

#: Below this, ``b`` counts as having no component along the leading eigenspace.
_RIDGE_HARD_CASE_TOL = 1e-10
#: Initial distance from the spectrum when bracketing the Lagrange multiplier.
_RIDGE_BRACKET_START = 1e-6
_RIDGE_BRACKET_MAX = 1e12


def _ridge_point(
    b: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    radius: float,
    *,
    maximise: bool,
) -> tuple[np.ndarray, float]:
    r"""Return the constrained optimum on a sphere, and the multiplier that gives it.

    For ``y = b0 + b'x + x'Bx`` restricted to ``||x|| = radius``, the Lagrange
    condition is ``(B - mu*I) x = -b/2``. Writing ``g = V'b`` for the linear
    coefficients in the eigenbasis of *B*:

    .. math::

        x(\mu) = -\tfrac{1}{2} (B - \mu I)^{-1} b,
        \qquad
        \lVert x(\mu) \rVert^2 = \sum_i \frac{g_i^2}{4 (\lambda_i - \mu)^2}.

    The Hessian of the Lagrangian is ``2 (B - mu*I)``, so the point is a
    constrained *maximum* exactly when ``mu > lambda_max``, and a constrained
    *minimum* when ``mu < lambda_min``. On either of those intervals every
    ``|lambda_i - mu|`` grows as ``mu`` moves away from the spectrum, so the
    radius is strictly monotone in ``mu`` and the multiplier for a given radius
    is unique. Finding it is therefore a one-dimensional root-find, not a
    search: this is Draper's ridge analysis, and equivalently the trust-region
    subproblem.

    The "hard case" is handled as well. When *b* has no component along the
    leading eigenspace the radius stays bounded as ``mu`` approaches that
    eigenvalue, so no interior multiplier exists; the optimum for any larger
    radius is the limiting point plus a step along the leading eigenvector.
    A model whose stationary point sits at the centre (``b = 0``) is the common
    instance, and its ridge runs straight along that eigenvector.

    Parameters
    ----------
    b : np.ndarray
        Linear coefficients, shape (k,).
    eigenvalues : np.ndarray
        Ascending eigenvalues of *B*, shape (k,), as ``numpy.linalg.eigh``
        returns them.
    eigenvectors : np.ndarray
        The matching eigenvectors in columns, shape (k, k).
    radius : float
        Radius of the sphere, in coded units. Must be positive.
    maximise : bool
        True traces the ridge of maxima, False the ridge of minima.

    Returns
    -------
    tuple[np.ndarray, float]
        The optimal coded point on that sphere, and the multiplier ``mu`` that
        produced it. ``mu`` is the boundary eigenvalue in the hard case.
    """
    g = eigenvectors.T @ b
    edge = float(eigenvalues[-1] if maximise else eigenvalues[0])
    sign = 1.0 if maximise else -1.0
    scale = max(1.0, float(np.abs(eigenvalues).max()))

    def point_at(mu: float) -> np.ndarray:
        return eigenvectors @ (-0.5 * g / (eigenvalues - mu))

    leading = np.isclose(eigenvalues, edge, rtol=1e-10, atol=1e-12)
    if np.all(np.abs(g[leading]) <= _RIDGE_HARD_CASE_TOL * max(1.0, float(np.linalg.norm(g)))):
        limit = np.zeros_like(g)
        limit[~leading] = -0.5 * g[~leading] / (eigenvalues[~leading] - edge)
        limit_radius = float(np.linalg.norm(limit))
        if limit_radius <= radius:
            step = np.sqrt(max(radius**2 - limit_radius**2, 0.0))
            limit[leading] = step / np.sqrt(float(leading.sum()))
            return eigenvectors @ limit, edge

    # Bracket the multiplier strictly outside the spectrum, then solve. Work on
    # 1 / ||x(mu)||, which is smooth and near-linear in mu; ||x(mu)|| itself
    # blows up at the inner end of the bracket.
    def residual(mu: float) -> float:
        return 1.0 / float(np.linalg.norm(point_at(mu))) - 1.0 / radius

    offset = _RIDGE_BRACKET_START * scale
    while residual(edge + sign * offset) > 0.0:
        offset /= 4.0
    inner = edge + sign * offset
    outer = edge + sign * offset
    while residual(outer) < 0.0 and abs(outer - edge) < _RIDGE_BRACKET_MAX:
        outer = edge + (outer - edge) * 4.0

    lo, hi = sorted((inner, outer))
    mu = float(optimize.brentq(residual, lo, hi, xtol=1e-14, rtol=8.9e-16))
    return point_at(mu), mu


def _ridge_analysis(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    *,
    direction: str = "maximize",
    n_radii: int = 10,
    max_radius: float = 1.0,
) -> dict[str, Any]:
    """Trace the constrained optimum along spheres of increasing radius.

    When a second-order model's stationary point falls outside the region the
    experiment covered, or is a saddle, it is not a usable recommendation. The
    ridge answers the question actually being asked: *given that I will move no
    further than r from the centre, where is the best point and what does the
    model predict there?* Tracing r upwards shows how fast the prediction
    improves, which factors have to move to get it, and where the returns
    flatten off.

    Parameters
    ----------
    coefficients : list[dict]
        Model coefficients, each with ``"term"`` and ``"coefficient"``.
    factor_names : list[str]
        Ordered factor names.
    direction : {"maximize", "minimize"}
        Which ridge to trace.
    n_radii : int
        How many radii to report, over and above the centre.
    max_radius : float
        The largest radius to trace, in coded units. ``optimize_responses``
        derives it from ``search_bounds``: the factorial cube's default of
        (-1, 1) traces out to 1, and a rotatable central composite design's
        ``(-1.41, 1.41)`` traces out to its axial distance.

    Returns
    -------
    dict
        ``direction``, ``eigenvalues``, ``max_radius``, ``stationary_point_radius``
        (the distance to the unconstrained stationary point, beyond which the
        ridge stops moving because the constraint no longer binds; ``None`` when
        *B* is singular), and ``path``: one entry per radius, each carrying
        ``radius``, ``mu``, ``coded`` and ``predicted_response``.

        Returns a dict with a single ``error`` key when the model has no
        quadratic or interaction terms: there is then no curvature to trace, and
        steepest ascent is the right tool.

    Raises
    ------
    ValueError
        If *direction* is not one of the two accepted values, *n_radii* < 1, or
        *max_radius* is not positive.

    References
    ----------
    Draper, N.R. (1963). Ridge analysis of response surfaces.
    *Technometrics*, 5(4), 469-479.
    """
    if direction not in ("maximize", "minimize"):
        msg = f"direction must be 'maximize' or 'minimize'; got {direction!r}."
        raise ValueError(msg)
    if n_radii < 1:
        msg = f"n_radii must be at least 1; got {n_radii}."
        raise ValueError(msg)
    if not max_radius > 0:
        msg = f"max_radius must be positive; got {max_radius}."
        raise ValueError(msg)

    b0, b, B = _extract_b_and_B(coefficients, factor_names)
    if np.allclose(B, 0):
        return {
            "error": (
                "Model has no quadratic or interaction terms - ridge analysis needs curvature to trace. "
                "Use 'steepest_ascent' or 'steepest_descent' for a first-order model."
            )
        }

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    path: list[dict[str, Any]] = []
    for radius in np.linspace(0.0, max_radius, n_radii + 1):
        if radius == 0.0:
            x, mu = np.zeros(len(factor_names)), float("nan")
        else:
            x, mu = _ridge_point(b, eigenvalues, eigenvectors, float(radius), maximise=direction == "maximize")
        entry: dict[str, Any] = {
            "radius": float(radius),
            "mu": mu,
            "coded": {n: float(x[i]) for i, n in enumerate(factor_names)},
            "predicted_response": float(b0 + b @ x + x @ B @ x),
        }
        path.append(entry)

    stationary_radius: float | None = None
    with contextlib.suppress(np.linalg.LinAlgError):
        stationary_radius = float(np.linalg.norm(np.linalg.solve(2.0 * B, -b)))

    return {
        "direction": direction,
        "eigenvalues": [float(e) for e in eigenvalues],
        "max_radius": float(max_radius),
        "stationary_point_radius": stationary_radius,
        "path": path,
    }


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

#: Weight on the augmentation term in the Chebyshev scalarisation. Small enough
#: not to distort the front, large enough to reject weakly-dominated points.
_PARETO_AUGMENT = 1e-4
#: Relative tolerance when deciding that one objective vector dominates another.
_PARETO_TOL = 1e-9
#: Below this, an objective is flat across the payoff table and is not rescaled.
_SPREAD_FLOOR = 1e-12
#: Tighter than SLSQP's default, since each evaluation is a polynomial.
_SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 500}


def _objective_senses(goals: list[dict[str, Any]]) -> list[tuple[str, float | None]]:
    """Map each goal onto ``(kind, target)``, where kind drives the scalarisation."""
    senses: list[tuple[str, float | None]] = []
    for goal in goals:
        kind = str(goal.get("goal", "maximize")).lower()
        if kind == "target":
            if goal.get("target") is None:
                msg = "A goal with goal='target' must supply 'target'."
                raise ValueError(msg)
            senses.append(("target", float(goal["target"])))
        elif kind in ("maximize", "minimize"):
            senses.append((kind, None))
        else:
            msg = f"Unknown goal {kind!r}; expected 'maximize', 'minimize' or 'target'."
            raise ValueError(msg)
    return senses


def _as_utilities(
    values: np.ndarray,
    senses: list[tuple[str, float | None]],
) -> np.ndarray:
    """Turn raw predicted responses into quantities where more is always better.

    A ``"target"`` goal becomes the negated *squared* deviation rather than the
    negated absolute one: both are maximised at the target and both order points
    identically, but the square is differentiable there, which matters because
    SLSQP has to work through this point.
    """
    out = np.empty_like(values)
    for i, (kind, target) in enumerate(senses):
        if kind == "maximize":
            out[i] = values[i]
        elif kind == "minimize":
            out[i] = -values[i]
        else:
            out[i] = -((values[i] - float(target)) ** 2)  # type: ignore[arg-type]
    return out


def _simplex_weights(n_objectives: int, n_points: int) -> np.ndarray:
    """Return weight vectors spread evenly over the unit simplex (Das-Dennis).

    For two objectives this is just ``n_points`` evenly spaced pairs. For more,
    it is the standard stars-and-bars lattice: every way of splitting *divisions*
    into *n_objectives* non-negative parts, where *divisions* is the smallest
    number giving at least *n_points* vectors.
    """
    if n_objectives == 1:
        return np.ones((1, 1))
    if n_objectives == 2:
        share = np.linspace(0.0, 1.0, max(n_points, 2))
        return np.column_stack([share, 1.0 - share])

    divisions = 1
    while math.comb(divisions + n_objectives - 1, n_objectives - 1) < n_points:
        divisions += 1

    rows = [
        np.diff((0, *cuts, divisions + n_objectives)) - 1
        for cuts in itertools.combinations(range(1, divisions + n_objectives), n_objectives - 1)
    ]
    return np.array(rows, dtype=float) / divisions


def _non_dominated(utilities: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the rows no other row dominates (more is better)."""
    keep = np.ones(len(utilities), dtype=bool)
    for i, row in enumerate(utilities):
        if not keep[i]:
            continue
        # Strictly better somewhere, and no worse anywhere: that dominates row i.
        scale = np.maximum(np.abs(row), 1.0)
        better_or_equal = (utilities >= row - _PARETO_TOL * scale).all(axis=1)
        strictly_better = (utilities > row + _PARETO_TOL * scale).any(axis=1)
        if (better_or_equal & strictly_better).any():
            keep[i] = False
    return keep


def _front_entries(
    points: np.ndarray,
    raw: Callable[[np.ndarray], np.ndarray],
    factor_names: list[str],
    names: list[str],
) -> list[dict[str, Any]]:
    """Describe each distinct front point, sorted by the first response.

    Neighbouring weight vectors often land on the same solution, and duplicates
    say nothing about the trade-off, so they are dropped here rather than
    reported as separate options.
    """
    entries: list[dict[str, Any]] = []
    seen: list[np.ndarray] = []
    for x in points:
        if any(np.allclose(x, other, atol=1e-6) for other in seen):
            continue
        seen.append(x)
        values = raw(x)
        entry: dict[str, Any] = {
            "coded": {n: float(x[i]) for i, n in enumerate(factor_names)},
            "responses": {name: float(values[i]) for i, name in enumerate(names)},
        }
        entries.append(entry)

    entries.sort(key=lambda entry: entry["responses"][names[0]])
    return entries


def _payoff_table(
    best_of: Callable[[Callable[[np.ndarray], float]], np.ndarray],
    raw: Callable[[np.ndarray], np.ndarray],
    utility: Callable[[np.ndarray], np.ndarray],
    senses: list[tuple[str, float | None]],
    n_objectives: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimise each objective alone and record what every response does there.

    The diagonal of the result is the ideal point; the worst entry in each column
    estimates the nadir. Together they set the scale each objective is normalised
    by, so responses in different units carry equal say in the scalarisation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The table in raw response units and in utility units, both (m, m).
    """

    def negated(x: np.ndarray, index: int) -> float:
        return float(-utility(x)[index])

    anchors = [best_of(functools.partial(negated, index=i)) for i in range(n_objectives)]
    payoff_raw = np.array([raw(x) for x in anchors])
    return payoff_raw, np.array([_as_utilities(row, senses) for row in payoff_raw])


def _pareto_front(  # noqa: PLR0913 - the models, the goals, the naming and the search region are four
    # separate inputs; bundling them into a config object would only move the width, as for _optimize_desirability
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    *,
    n_points: int = 21,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    random_state: int | np.random.Generator | None = 42,
) -> dict[str, Any]:
    """Compute the Pareto front of several fitted response-surface models.

    Where ``"desirability"`` collapses the responses into one number and returns
    the single point that maximises it, this returns the whole set of
    non-dominated compromises: every setting where no response can be improved
    without giving something up elsewhere. Choosing among them is a judgement
    about trade-offs, and it is better made on the trade-offs themselves than on
    a weight chosen in advance.

    Method: augmented weighted Chebyshev scalarisation, solved with SLSQP from
    several starts, over a Das-Dennis lattice of weights, then filtered to the
    non-dominated set. Each response is normalised by the range of its own
    payoff table, so responses in different units carry equal say.

    NSGA-II would be the usual choice, and is the wrong one here. It is built
    for expensive black-box objectives, where a population is the only way to
    make progress. These objectives are low-order polynomials over a box,
    evaluated in microseconds and differentiable everywhere, so a gradient
    solver reaches each front point to solver tolerance rather than to whatever
    a finite population converged to, with no random seed and no generation
    count to tune. Chebyshev scalarisation is used rather than a weighted sum
    because a weighted sum can only ever find points on the convex hull of the
    front, and quadratic models routinely produce non-convex fronts.

    Parameters
    ----------
    fitted_models : list[dict]
        Each with ``"coefficients"`` and ``"response_name"``.
    goals : list[dict]
        One per response, with ``"goal"`` in ``{"maximize", "minimize",
        "target"}`` and, for ``"target"``, a ``"target"`` value. Matched to
        *fitted_models* by response name when both sides name their responses,
        otherwise by position. The ``"low"`` / ``"high"`` desirability limits are
        not used: the front is computed on the predicted responses themselves.
    factor_names : list[str]
        Ordered factor names.
    n_points : int
        Target number of weight vectors. The front returned is usually smaller,
        because dominated and duplicate solutions are dropped.
    search_bounds : tuple, dict, or None
        The coded region to search. Defaults to the factorial cube, (-1, 1).
    random_state : int, Generator, or None
        Seed for the extra random starts. The default keeps the result
        reproducible.

    Returns
    -------
    dict
        ``objectives`` (response name and goal, in order), ``ideal`` and
        ``nadir`` (the payoff table's corners, in raw response units),
        ``n_weights``, and ``front``: the non-dominated points, each with
        ``coded`` and ``responses``, sorted by the first response.

    Raises
    ------
    ValueError
        If fewer than two models are supplied (a front needs a trade-off), or a
        goal is malformed.
    """
    if len(fitted_models) < 2:
        msg = (
            f"A Pareto front needs at least two responses to trade off; got {len(fitted_models)}. "
            "Use 'desirability' or 'stationary_point' for a single response."
        )
        raise ValueError(msg)

    goals = _align_goals_to_models(fitted_models, goals)
    senses = _objective_senses(goals)
    evaluators = [_build_model_evaluator(m["coefficients"], factor_names) for m in fitted_models]
    names = [str(m.get("response_name", f"response_{i + 1}")) for i, m in enumerate(fitted_models)]

    bounds = _resolve_search_bounds(search_bounds, factor_names)
    lows = np.array([low for low, _ in bounds])
    highs = np.array([high for _, high in bounds])
    centre = (lows + highs) / 2.0

    rng = check_random_state(random_state)
    starts = [centre, lows.copy(), highs.copy(), *(rng.uniform(lows, highs) for _ in range(5))]

    def raw(x: np.ndarray) -> np.ndarray:
        return np.array([evaluator(x) for evaluator in evaluators])

    def utility(x: np.ndarray) -> np.ndarray:
        return _as_utilities(raw(x), senses)

    def best_of(objective: Callable[[np.ndarray], float]) -> np.ndarray:
        # SLSQP's default ftol of 1e-6 leaves front points a visible distance
        # short of the true frontier. These objectives cost microseconds to
        # evaluate, so there is nothing to buy by stopping early.
        winner, best = centre, np.inf
        for x0 in starts:
            res = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, options=_SLSQP_OPTIONS)
            if res.fun < best:
                winner, best = res.x, res.fun
        return winner

    payoff_raw, payoff_utility = _payoff_table(best_of, raw, utility, senses, len(names))
    ideal_utility = payoff_utility.diagonal().copy()
    nadir_utility = payoff_utility.min(axis=0)
    spread = np.where(np.abs(ideal_utility - nadir_utility) > _SPREAD_FLOOR, ideal_utility - nadir_utility, 1.0)

    def chebyshev(x: np.ndarray, w: np.ndarray) -> float:
        gap = w * (ideal_utility - utility(x)) / spread
        return float(gap.max() + _PARETO_AUGMENT * gap.sum())

    weights = _simplex_weights(len(names), n_points)
    points = np.array([best_of(functools.partial(chebyshev, w=w)) for w in weights])
    keep = _non_dominated(np.array([utility(x) for x in points]))
    front = _front_entries(points[keep], raw, factor_names, names)

    return {
        "objectives": [{"response": name, "goal": senses[i][0]} for i, name in enumerate(names)],
        "ideal": {name: float(payoff_raw[j, j]) for j, name in enumerate(names)},
        "nadir": {name: float(payoff_raw[int(np.argmin(payoff_utility[:, j])), j]) for j, name in enumerate(names)},
        "n_weights": len(weights),
        "front": front,
    }


# ---------------------------------------------------------------------------
# Coded ↔ actual conversion helpers
# ---------------------------------------------------------------------------


def _add_actual_units(
    result: dict[str, Any],
    factor_ranges: dict[str, dict[str, float]] | None,
    key: str,
) -> dict[str, Any]:
    """Add an ``"actual"`` reading to every entry under *key*, in place.

    Coded-to-actual conversion is reporting, not optimisation, so the methods
    that build these lists do not carry *factor_ranges* through their own
    signatures just to decorate their output at the end.
    """
    if factor_ranges:
        for entry in result.get(key, []):
            entry["actual"] = _coded_to_actual(entry["coded"], factor_ranges)
    return result


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
    ridge_direction: str = "maximize",
    n_pareto_points: int = 21,
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
        ``"ridge_analysis"``, ``"pareto_front"``.
    factor_ranges : dict or None
        Maps factor name to ``{"low": float, "high": float}`` in actual
        units.  Used for coded ↔ actual conversion.
    step_size : float
        Step magnitude for steepest ascent/descent (coded units).
    n_steps : int
        Number of steps along a path: the steepest ascent/descent steps, or the
        radii reported by ridge analysis over and above the centre.
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
    ridge_direction : {"maximize", "minimize"}
        Which ridge ``method="ridge_analysis"`` traces.
    n_pareto_points : int
        Target number of weight vectors for ``method="pareto_front"``. The front
        returned is usually smaller, since dominated and duplicate solutions are
        dropped.

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
        region = _resolve_search_bounds(search_bounds, factor_names)
        result["ridge_analysis"] = _add_actual_units(
            _ridge_analysis(
                coefficients,
                factor_names,
                direction=ridge_direction,
                n_radii=n_steps,
                max_radius=max(max(abs(low), abs(high)) for low, high in region),
            ),
            factor_ranges,
            "path",
        )

    elif method == "pareto_front":
        if goals is None:
            msg = "Goals are required for Pareto front optimization."
            raise ValueError(msg)
        result["pareto_front"] = _add_actual_units(
            _pareto_front(
                fitted_models,
                goals,
                factor_names,
                n_points=n_pareto_points,
                search_bounds=search_bounds,
            ),
            factor_ranges,
            "front",
        )

    return result
