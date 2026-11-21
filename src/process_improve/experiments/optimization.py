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

:func:`feasible_region` answers the reverse question: at which settings every
response is predicted inside a specification window, and how surely a new run
there meets it.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
import math
import numbers
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from patsy import PatsyError
from scipy import optimize, special, stats

from process_improve._random import check_random_state
from process_improve._warnings import user_stacklevel
from process_improve.experiments._desirability import (
    SCORED_GOALS,
    check_goal,
    check_importances,
    individual_desirability,
    overall_desirability,
)
from process_improve.experiments._exact_optimum import StraightRegion, exact_optimum
from process_improve.experiments._uniform_sampling import UniformSampler, _affine_form

if TYPE_CHECKING:
    from process_improve.experiments._count_rules import UsedSets
    from process_improve.experiments.region import DesignRegion

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


_POWER = re.compile(r"I\((\w+)\s*\*\*\s*(\d+)\)|(?:np\.)?power\((\w+)\s*,\s*(\d+)\)")


def _parse_term(term: str) -> tuple[str, ...]:
    """Classify a coefficient term name into its components.

    Returns
    -------
    tuple[str, ...]
        Empty tuple for ``"Intercept"``, single-element for linear,
        ``("A", "B")`` for interaction ``"A:B"``, ``("A", "A")`` for
        quadratic ``"I(A ** 2)"``, and each factor of a product of powers
        repeated as often as the power multiplies it: ``("A", "A", "A")`` for
        ``"I(A ** 3)"``, ``("A", "B", "B")`` for ``"A:I(B ** 2)"``.
    """
    if term == "Intercept":
        return ()

    # A power is ``I(A ** 2)`` (older statsmodels) or ``np.power(A, 2)`` /
    # ``power(A, 2)`` (newer). Both spellings appear in the wild depending on
    # the installed statsmodels / patsy version. SEC-27 (#276): if either is
    # missed, the term silently falls through to the linear branch and the
    # downstream surface / optimisation produces wrong results.
    components: list[str] = []
    for part in term.split(":"):
        m = _POWER.fullmatch(part.strip())
        if m:
            components += [m.group(1) or m.group(3)] * int(m.group(2) or m.group(4))
        else:
            components.append(part)
    return tuple(components)


def _parsed_terms(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    *,
    max_order: int | None = None,
) -> list[tuple[tuple[str, ...], float]]:
    """Parse every coefficient's term, refusing terms the model evaluation cannot represent.

    Parameters
    ----------
    coefficients : list[dict]
        Each with ``"term"`` and ``"coefficient"``.
    factor_names : list[str]
        The factors a term may be built from.
    max_order : int or None
        The largest number of factors a term may multiply together; 2 for the
        second-order analyses, which work with ``b`` and ``B`` only. ``None``
        allows any product.

    Returns
    -------
    list[tuple[tuple[str, ...], float]]
        The parsed components of each term, with its coefficient.

    Raises
    ------
    ValueError
        If a term names something other than a product of whole powers of the
        factors (a categorical ``C(A)[T.1]``, a misspelt factor), or has more than
        *max_order* components (``A:B:C`` or ``I(A ** 3)`` in a second-order
        analysis, which would otherwise be dropped and the analysis run on a
        truncated model).
    """
    parsed = [(_parse_term(entry["term"]), float(entry["coefficient"])) for entry in coefficients]
    known = set(factor_names)
    unknown = [e["term"] for e, (parts, _) in zip(coefficients, parsed, strict=True) if not set(parts) <= known]
    if unknown:
        msg = (
            f"Cannot evaluate the model term(s) {unknown}: a term must be the intercept, a factor, or a product of "
            "factors and their whole powers such as 'A:B', 'I(A ** 2)' or 'A:I(B ** 2)', over the factors "
            f"{factor_names}."
        )
        raise ValueError(msg)
    if max_order is not None:
        too_high = [e["term"] for e, (parts, _) in zip(coefficients, parsed, strict=True) if len(parts) > max_order]
        if too_high:
            msg = (
                f"This analysis needs a second-order model, with linear, two-factor interaction and squared "
                f"terms only; the model also has {too_high}. Use method='desirability' or 'pareto_front', "
                "which evaluate any polynomial, or refit without those terms."
            )
            raise ValueError(msg)
    return parsed


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
    parsed = _parsed_terms(coefficients, factor_names)

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

    Raises
    ------
    ValueError
        If a term is not part of a second-order model in *factor_names*; see
        :func:`_parsed_terms`.
    """
    k = len(factor_names)
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    b0 = 0.0
    b = np.zeros(k)
    B = np.zeros((k, k))

    for components, coef in _parsed_terms(coefficients, factor_names, max_order=2):
        if len(components) == 0:
            b0 = coef
        elif len(components) == 1:
            b[name_to_idx[components[0]]] = coef
        else:  # two components: _parsed_terms(max_order=2) refuses anything longer
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


#: An eigenvalue of *B* this small, relative to the largest in magnitude, is
#: zero: the surface is flat along its eigenvector (a ridge), not curved. The
#: default of the ``flat_tolerance`` argument.
_FLAT_EIGENVALUE_TOL = 1e-8

#: An eigenvalue below this share of the largest in size, though above the flat
#: tolerance, gets a note: the surface is nearly flat along its eigenvector, and
#: the data may not tell it from a ridge (#789 B3).
_NEARLY_FLAT = 0.05


@dataclass
class _SurfaceShape:
    """The shape of a second-order surface ``b0 + b'x + x'Bx``, read off the eigenvalues of *B*."""

    classification: str
    flat: np.ndarray  # True for an eigenvalue that counts as zero
    drifts: bool  # b has a component along a flat direction, so there is no stationary point
    ridge_of: str | None = None  # "maxima" or "minima" for a ridge system, else None


def _surface_shape(
    b: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, flat_tolerance: float = _FLAT_EIGENVALUE_TOL
) -> _SurfaceShape:
    """Classify the surface, treating eigenvalues at most ``flat_tolerance`` of the largest in size as a ridge.

    All curved directions bending down gives a maximum, all bending up a
    minimum, and both a saddle. When some eigenvalue is zero the surface is a
    ridge system (Myers, Montgomery and Anderson-Cook, *Response Surface
    Methodology*, sec. 6.4): a *stationary ridge*, a line or plane of equal
    optima, when *b* has no component along the flat directions, and a *rising
    ridge*, with no stationary point, when it has one and the response keeps
    changing along them. The slope along the flat directions is judged as their
    eigenvalues are: at most ``flat_tolerance`` of the largest eigenvalue in size
    is none, so a ridge whose estimated slope is small is stationary.
    """
    scale = float(np.abs(eigenvalues).max())
    flat = np.abs(eigenvalues) <= flat_tolerance * scale
    slope = float(np.linalg.norm(eigenvectors[:, flat].T @ b))
    drifts = slope > max(_FLAT_EIGENVALUE_TOL * max(1.0, float(np.linalg.norm(b))), flat_tolerance * scale)
    curved = eigenvalues[~flat]
    if np.all(curved < 0):
        kind, ridge_of = "maximum", "maxima"
    elif np.all(curved > 0):
        kind, ridge_of = "minimum", "minima"
    else:
        return _SurfaceShape("saddle_point", flat, drifts)
    if not flat.any():
        return _SurfaceShape(kind, flat, drifts)
    return _SurfaceShape("rising_ridge" if drifts else "stationary_ridge", flat, drifts, ridge_of)


def _eigen_by_size(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the eigenvalues of *B*, largest in size first, and their eigenvectors as columns in the same order."""
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(-np.abs(eigenvalues), kind="stable")
    return eigenvalues[order], eigenvectors[:, order]


def _refuse_unused_factors(coefficients: list[dict[str, Any]], factor_names: list[str]) -> None:
    """Refuse a factor name that no model term uses: it would add a zero eigenvalue to *B* (#796 item 4).

    A row and column of zeros in *B* reads as a direction along which the surface is
    flat, so a maximum would be called a stationary ridge. Such a name is not a factor
    of the model: a block column, or a copy of a factor in other units.
    """
    used = {part for parts, _ in _parsed_terms(coefficients, factor_names) for part in parts}
    unused = [name for name in factor_names if name not in used]
    if unused:
        msg = (
            f"No term of the model uses {unused}, so the second-order analysis would read each as a direction "
            "along which the surface is flat, and could call a maximum or minimum a ridge. Leave them out of "
            "factor_names."
        )
        raise ValueError(msg)


def _nearly_flat_note(eigenvalues: np.ndarray, shape: _SurfaceShape) -> str | None:
    """Say when a curved direction has an eigenvalue below :data:`_NEARLY_FLAT` of the largest in size."""
    scale = float(np.abs(eigenvalues).max())
    small = [i for i, ev in enumerate(eigenvalues) if not shape.flat[i] and abs(ev) < _NEARLY_FLAT * scale]
    if not small:
        return None
    sizes = ", ".join(f"{eigenvalues[i]:.4g} ({100 * abs(eigenvalues[i]) / scale:.2g}% of the largest)" for i in small)
    return (
        f"The surface is nearly flat along the eigenvector of the eigenvalue {sizes}: the data may not tell this "
        f"{shape.classification.replace('_', ' ')} from a ridge. flat_tolerance={_NEARLY_FLAT} reads it as flat."
    )


def _find_stationary_point(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    flat_tolerance: float = _FLAT_EIGENVALUE_TOL,
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
    flat_tolerance : float
        An eigenvalue of ``B`` at most this share of the largest in size is
        read as zero, a direction along which the surface is flat.

    Returns
    -------
    dict
        ``stationary_point_coded``, ``predicted_response``, ``classification``
        (``"maximum"``, ``"minimum"``, ``"saddle_point"`` or
        ``"stationary_ridge"``), ``eigenvalues`` (list of floats, spectrum of
        the pure-quadratic matrix ``B``, largest in size first, as
        :func:`_canonical_analysis` gives them), and ``inside_design_space`` (bool,
        whether the stationary point falls inside ``search_bounds``). Also
        includes ``stationary_point_actual`` when ``factor_ranges`` is provided.
        For a stationary ridge, where an eigenvalue is zero and every point
        along its eigenvector is equally good, the point reported is the one
        nearest the design centre and ``ridge_of`` says whether the ridge is
        one of ``"maxima"`` or ``"minima"``. ``classification_note`` says when
        an eigenvalue is small next to the largest though not flat, so the
        surface may be a ridge.

        Returns a dict with an ``error`` key instead when the model has no
        quadratic or interaction terms, or when there is no stationary point
        (a rising ridge, or a saddle that drifts along a flat direction); the
        latter also carries ``classification`` and ``eigenvalues``.
    """
    b0, b, B = _extract_b_and_B(coefficients, factor_names)

    # Check that B has quadratic terms (not purely first-order)
    if np.allclose(B, 0):
        return {"error": "Model has no quadratic or interaction terms - cannot find stationary point."}
    _refuse_unused_factors(coefficients, factor_names)

    eigenvalues, eigenvectors = _eigen_by_size(B)
    shape = _surface_shape(b, eigenvalues, eigenvectors, flat_tolerance)
    if shape.drifts:
        return {
            "error": (
                "No stationary point: B has a zero eigenvalue and the linear terms keep the response changing along "
                "its eigenvector (a rising ridge). Use 'ridge_analysis' or steepest ascent/descent to follow it."
            ),
            "classification": shape.classification,
            "eigenvalues": [float(e) for e in eigenvalues],
        }

    # Solve 2*B*x_s = -b in the eigenbasis. Along a flat direction b has no
    # component, so any step there is equally stationary; take none, which gives
    # the stationary point nearest the centre.
    curved = ~shape.flat
    x_s = eigenvectors[:, curved] @ (-0.5 * (eigenvectors[:, curved].T @ b) / eigenvalues[curved])

    # Predicted response at stationary point
    y_s = float(b0 + b @ x_s + x_s @ B @ x_s)

    # Is the stationary point inside the region the experiment covered? The
    # default region is the factorial cube; a central composite design reaches
    # further, so its axial distance can be supplied via search_bounds.
    region = _resolve_search_bounds(search_bounds, factor_names)
    inside_design_space = bool(all(low <= value <= high for value, (low, high) in zip(x_s, region, strict=True)))

    result: dict[str, Any] = {
        "stationary_point_coded": {n: float(x_s[i]) for i, n in enumerate(factor_names)},
        "predicted_response": y_s,
        "classification": shape.classification,
        "eigenvalues": [float(e) for e in eigenvalues],
        "inside_design_space": inside_design_space,
    }
    if shape.ridge_of is not None:
        result["ridge_of"] = shape.ridge_of
    if note := _nearly_flat_note(eigenvalues, shape):
        result["classification_note"] = note

    if factor_ranges:
        result["stationary_point_actual"] = _coded_to_actual(result["stationary_point_coded"], factor_ranges)

    return result


# ---------------------------------------------------------------------------
# Canonical analysis
# ---------------------------------------------------------------------------


def _canonical_analysis(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    flat_tolerance: float = _FLAT_EIGENVALUE_TOL,
) -> dict[str, Any]:
    """Canonical analysis of a second-order response surface model.

    Computes eigenvalues and eigenvectors of the *B* matrix to determine
    the shape and orientation of the response surface, largest eigenvalue in
    size first. An eigenvalue at most ``flat_tolerance`` of the largest in size
    is read as zero.

    Returns
    -------
    dict
        ``eigenvalues``, ``eigenvectors``, ``classification`` (``"maximum"``,
        ``"minimum"``, ``"saddle_point"``, ``"stationary_ridge"`` or
        ``"rising_ridge"``), and ``canonical_form_description``, which labels
        each canonical axis concave, convex, or flat (a zero eigenvalue). A
        ridge also carries ``ridge_of``, ``"maxima"`` or ``"minima"``, and
        ``classification_note`` says when an eigenvalue is small next to the
        largest though not flat.

        ``ridge_form`` is the surface in the ridge form of Box and Draper (2007,
        Eq. 11.1.2), the form a ridge is read in: y = y0 + theta * X_k + sum of lambda_i X_i^2,
        with X_i = v_i'(x - x0). Its axis ``k`` is that of the smallest
        eigenvalue in size, the ridge's direction, and its ``origin`` x0 is the
        point nearest the design centre where the response stops changing
        along every other axis that is not flat. It holds the ``origin`` (coded), the
        ``origin_response`` y0 there, the ``linear`` coefficient theta along
        the ridge, v_k'b, and each axis's ``offset``, the constant c_i of the
        plane X_i = v_i'x + c_i = 0. The form is exact for any surface: a
        maximum, minimum or saddle is a ridge whose theta is zero and whose
        smallest eigenvalue is not. The one exception is a slope along a flat
        axis other than ``k``, which only a surface with several flat axes
        can have, and which the form leaves out (#971 K2). It is not Box and Draper's B canonical
        form, which is centred on the stationary point (below).

        ``rotated_form`` is Box and Draper's A canonical form (Eq. 10.3.9), the
        axes rotated about the design centre: y = b0 + sum of theta_i X_i + sum of
        lambda_i X_i^2, with X_i = v_i'x. It holds the ``linear`` slope theta_i =
        v_i'b along every axis, in the order of ``eigenvalues``, and the
        ``stationary_point``, X_iS = -theta_i / (2 lambda_i) on each axis, ``None``
        along a flat one. It exists with or without a stationary point.

        When the surface has a stationary point x_s, ``canonical_axes`` gives
        each canonical variable W_i = v_i'(x - x_s) as its ``constant``,
        -v_i'x_s, and ``coefficients``: Box and Draper's B canonical form is
        y = y_s + sum of lambda_i W_i^2.
    """
    b0, b, B = _extract_b_and_B(coefficients, factor_names)

    if np.allclose(B, 0):
        return {"error": "Model has no quadratic or interaction terms - canonical analysis not applicable."}
    _refuse_unused_factors(coefficients, factor_names)

    eigenvalues, eigenvectors = _eigen_by_size(B)
    shape = _surface_shape(b, eigenvalues, eigenvectors, flat_tolerance)

    desc_parts = []
    for i, (ev, is_flat) in enumerate(zip(eigenvalues, shape.flat, strict=True)):
        direction = "flat" if is_flat else ("concave" if ev < 0 else "convex")
        desc_parts.append(f"W{i + 1}: eigenvalue={ev:.4f} ({direction})")

    result: dict[str, Any] = {
        "eigenvalues": [float(e) for e in eigenvalues],
        "eigenvectors": [[float(v) for v in eigenvectors[:, i]] for i in range(len(eigenvalues))],
        "classification": shape.classification,
        "canonical_form_description": desc_parts,
        "factor_names": factor_names,
        "ridge_form": _ridge_form((b0, b, B), eigenvalues, eigenvectors, factor_names, shape.flat),
        "rotated_form": _rotated_form(b, eigenvalues, eigenvectors, shape.flat),
    }
    if not shape.drifts:  # a stationary point exists: the B canonical form's axes are centred on it
        curved = ~shape.flat
        x_s = eigenvectors[:, curved] @ (-0.5 * (eigenvectors[:, curved].T @ b) / eigenvalues[curved])
        result["canonical_axes"] = [
            {
                "constant": float(-eigenvectors[:, i] @ x_s),
                "coefficients": dict(zip(factor_names, map(float, eigenvectors[:, i]), strict=True)),
            }
            for i in range(len(eigenvalues))
        ]
    if shape.ridge_of is not None:
        result["ridge_of"] = shape.ridge_of
    if note := _nearly_flat_note(eigenvalues, shape):
        result["classification_note"] = note
    return result


def _rotated_form(b: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, flat: np.ndarray) -> dict[str, Any]:
    """Return Box and Draper's A canonical form (Eq. 10.3.9): the slope along each axis, and where it stops changing."""
    slopes = eigenvectors.T @ b
    return {
        "linear": [float(s) for s in slopes],
        "stationary_point": [
            None if is_flat else float(-s / (2 * ev)) for s, ev, is_flat in zip(slopes, eigenvalues, flat, strict=True)
        ],
    }


def _ridge_form(
    surface: tuple[float, np.ndarray, np.ndarray],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    factor_names: list[str],
    flat: np.ndarray,
) -> dict[str, Any]:
    """Return Box and Draper's ridge form (Eq. 11.1.2), taking the last axis (the smallest eigenvalue) as the ridge.

    The origin x0 = -1/2 sum over the other curved axes of (v_i'b / lambda_i) v_i is the point nearest
    the centre where the gradient has no component along them, so the form keeps one linear term, along
    the ridge. An axis whose eigenvalue is *flat* has no such point and is left out, as the last axis
    is (#965). *surface* is the fitted quadratic's intercept, linear vector b and matrix B.
    """
    b0, b, B = surface
    others = ~flat
    others[-1] = False
    origin = eigenvectors[:, others] @ (-0.5 * (eigenvectors[:, others].T @ b) / eigenvalues[others])
    return {
        "ridge_axis": len(eigenvalues) - 1,
        "origin": dict(zip(factor_names, map(float, origin), strict=True)),
        "origin_response": float(b0 + b @ origin + origin @ B @ origin),
        "linear": float(eigenvectors[:, -1] @ b),
        "offsets": [float(-eigenvectors[:, i] @ origin) for i in range(len(eigenvalues))],
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
    """Return the path of steepest ascent (or descent) from the design centre, for ``optimize_responses``.

    :func:`~process_improve.experiments.sequential.steepest_path` sets the path out; this steps
    ``step_size``, a distance in coded units, as ``optimize_responses`` does.

    Parameters
    ----------
    coefficients : list[dict]
        Model coefficients.
    factor_names : list[str]
        Ordered factor names.
    step_size : float
        Euclidean distance, in coded units, between successive points on the
        path (default 0.5). Each step moves every factor in proportion to its
        linear coefficient, so a factor moves by ``step_size * b_j / ||b||``.
        Must be positive.
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
        The path as :func:`~process_improve.experiments.sequential.steepest_path` returns it,
        or ``{"error": ...}`` when every linear coefficient is zero.

    Raises
    ------
    ValueError
        If *step_size* is not a positive finite number or *n_steps* < 1: a
        negative step would walk the path the wrong way.
    """
    from process_improve.experiments.sequential import NoSteepestDirectionError, steepest_path  # noqa: PLC0415

    if not (np.isfinite(step_size) and step_size > 0):
        msg = f"step_size must be a positive number of coded units; got {step_size}."
        raise ValueError(msg)
    if n_steps < 1:
        msg = f"n_steps must be at least 1; got {n_steps}."
        raise ValueError(msg)
    model = {"coefficients": coefficients, "factor_names": factor_names, "factor_coding": factor_ranges or {}}
    goal = "minimize" if direction == "descent" else "maximize"
    try:
        return steepest_path(model, step_size, n_steps, goal=goal)
    except NoSteepestDirectionError:
        return {"error": "All linear coefficients are zero - no steepest direction."}


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

    When every model names its response and every goal names one, the goals are
    reordered by name, and names that do not pair up one to one are an error:
    pairing them by position instead would optimise each response against another
    response's goal whenever the lists are in different orders. When either side
    leaves a name out, the goals are taken in list order, with a warning when there
    is more than one response, since position is then the only reading left.

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
        If the two lists differ in length, or if both sides name every response
        but the names do not correspond one to one (a typo, a case difference, or
        a duplicate).
    """
    if len(goals) != len(fitted_models):
        msg = f"Got {len(fitted_models)} fitted model(s) but {len(goals)} goal(s); they must correspond one to one."
        raise ValueError(msg)

    model_names = [m.get("response_name") for m in fitted_models]
    goal_names = [g.get("response") for g in goals]

    if any(n is None for n in model_names) or any(n is None for n in goal_names):
        if len(goals) > 1:
            warnings.warn(
                "Matching goals to fitted models by position: not every model has 'response_name' and not every "
                "goal has 'response'. Name both to have them matched by name instead.",
                UserWarning,
                stacklevel=user_stacklevel(),
            )
        return goals

    model_keys = [str(n) for n in model_names]
    by_name = {str(g["response"]): g for g in goals}
    if len(by_name) != len(goals) or len(set(model_keys)) != len(model_keys) or set(by_name) != set(model_keys):
        unmatched_goals = sorted(set(by_name) - set(model_keys))
        unmatched_models = sorted(set(model_keys) - set(by_name))
        msg = (
            f"The goals' 'response' names {[str(n) for n in goal_names]} do not correspond one to one with the "
            f"models' 'response_name' values {model_keys}: goal name(s) {unmatched_goals} match no model and "
            f"model name(s) {unmatched_models} match no goal, and each name must appear once on each side. "
            "Fix the names (they are case-sensitive), or leave 'response' out of every goal to pair them by position."
        )
        raise ValueError(msg)

    return [by_name[key] for key in model_keys]


#: How close to 0 (coded -1) a factor with a ``min_when_used`` is taken as left out.
_LEFT_OUT_TOL = 1e-9
#: SLSQP iterations spent on a neighbouring set of factors before it is searched in full (if it is better by then).
_SET_SCREEN_ITERATIONS = 8
#: A coefficient of a fitted straight constraint this small, beside its largest, is rounding and is taken as 0.
#: Left in a gradient (as 1e-15, say), it gave the limit of an ingredient that a search leaves out a slope on the
#: ingredients in use, and SLSQP then stopped at its start.
_FITTED_ZERO = 1e-12


@dataclass
class _SearchSpace:
    """Where SLSQP may look: bounds, scipy constraint dicts, and feasible starting points.

    ``gaps`` holds ``(column, zero, minimum)`` for each factor with a ``min_when_used``,
    in the search's units: ``zero`` is where the factor is left out (coded -1 for a
    process factor, 0 for a mixture component). A search cannot cross from there to
    the minimum, so each one keeps the set of factors its start uses (see
    :meth:`bounds_at`), and :func:`_best_set_of_factors` then tries the neighbouring sets.
    """

    bounds: list[tuple[float, float]]
    constraints: list[dict[str, Any]]
    starts: list[np.ndarray]
    gaps: list[tuple[int, float, float]] = field(default_factory=list)
    #: Whether a setting uses a number of factors that the region's count rules allow; None without such rules.
    allowed: Callable[[np.ndarray], bool] | None = None
    #: The same space as bounds and straight constraints, for the exact search; None when a constraint is curved.
    straight: StraightRegion | None = None

    def left_out(self, x: np.ndarray, column: int, zero: float) -> bool:
        """Whether ``x`` leaves the factor in ``column`` out."""
        return bool(x[column] <= zero + _LEFT_OUT_TOL)

    def bounds_at(self, start: np.ndarray) -> list[tuple[float, float]]:
        """Return the bounds of a search from ``start``.

        A factor with a ``min_when_used`` stays left out when ``start`` leaves it out (or
        when the search box ends below its minimum), and at or above its minimum otherwise.
        """
        bounds = list(self.bounds)
        for j, zero, minimum in self.gaps:
            out = self.left_out(start, j, zero) or bounds[j][1] < minimum
            bounds[j] = (zero, zero) if out else (minimum, bounds[j][1])
        return bounds

    def constraints_at(self, start: np.ndarray) -> list[dict[str, Any]]:
        """Return the constraints of a search from ``start``: those that a factor free to move enters.

        A straight constraint on factors that the search holds fixed (the limits of an
        ingredient that ``start`` leaves out) is a constant during it, so it cannot steer
        the search, and whether it holds is for the test of the result. Most of a
        formulation's ingredients are left out of any one blend, and so are most of
        these constraints. A constraint without a gradient is kept.
        """
        free = np.array([low < high for low, high in self.bounds_at(start)])
        return [c for c in self.constraints if "jac" not in c or np.any(c["jac"](start)[free] != 0.0)]


def _search_space(
    factor_names: list[str],
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None,
    region: DesignRegion | None,
    rng: np.random.Generator,
    n_starts: int,
    score: Callable[[np.ndarray], float] | None = None,
) -> _SearchSpace:
    """Translate a search box, and optionally a :class:`DesignRegion`, into SLSQP inputs.

    Without a region the search is the coded box, started from its centre and from
    random points in it. With a box region its constraints become inequality
    constraints on top of that box. A mixture region replaces the box: its component
    bounds become the bounds, ``sum(x) = 1`` an equality constraint, and its linear
    constraints inequalities. A mixture-process region does both: the components'
    part as a mixture region's, the process factors' as a box region's. The starts are
    then drawn uniformly from the feasible set itself, since a start outside it can
    leave SLSQP stranded at an infeasible point.

    A factor with a ``min_when_used`` is left out by about half of the starts (see
    :meth:`DesignRegion.sample`), and the space records its minimum, so each search stays
    on the set of factors its start uses (:meth:`_SearchSpace.bounds_at`). With ``score``,
    which is smaller for a better setting, the starts of such a region are the most
    promising of many (see :func:`_region_starts`).
    """
    bounds = _resolve_search_bounds(search_bounds, factor_names)
    lows, highs = np.array(bounds).T
    if region is None:
        starts = [(lows + highs) / 2.0, *(rng.uniform(lows, highs) for _ in range(n_starts - 1))]
        return _SearchSpace(bounds, [], starts, straight=StraightRegion(bounds))

    if region.rules is not None:
        example = ", ".join(f"{name}={levels[0]!r}" for name, levels in region.rules.levels.items())
        msg = (
            f"The region's constraints on the categorical factors {region.rules.factors} give one region of "
            f"settings per allowed combination of levels; search one of them, as region.given({example})."
        )
        raise ValueError(msg)
    if set(region.names) != set(factor_names):
        msg = f"The region's factors {region.names} do not match the model's factors {factor_names}."
        raise ValueError(msg)
    idx = [factor_names.index(n) for n in region.names]  # region column j is model column idx[j]
    gaps = [(idx[j], region.bounds[j][0], minimum) for j, minimum in region.gaps]
    sets = region._sets  # the sets of factors that count rules allow in use together, if any
    for j, zero, _minimum in gaps:  # an amount is not below 0, however wide the search box
        bounds[j] = (max(bounds[j][0], zero), bounds[j][1])
    lows, highs = np.array(bounds).T
    components = [factor_names.index(n) for n in region.mixture_names]
    for j in components:
        bounds[j] = region.bounds[region.names.index(factor_names[j])]
    limits = tuple(np.array(region.bounds, dtype=float).T)
    forms = [_straight_form(_affine_form(g, *limits), idx, len(factor_names)) for g in region.inequalities]
    constraints = [_slsqp_inequality(g, form, idx) for g, form in zip(region.inequalities, forms, strict=True)]
    straight = _straight_region(bounds, forms, components, gaps, sets)
    if components:
        ones = np.zeros(len(factor_names))
        ones[components] = 1.0
        constraints.append({"type": "eq", "fun": lambda x: float(x[components].sum() - 1.0), "jac": lambda _x: ones})

    def placed(sample: np.ndarray) -> np.ndarray:
        start = np.empty(len(factor_names))
        start[idx] = sample
        return start

    in_region = None if score is None else (lambda sample: score(placed(sample)))
    starts = [placed(sample) for sample in _region_starts(region, (lows[idx], highs[idx]), rng, n_starts, in_region)]
    allowed = None if sets is None else (lambda x: bool(sets.allows(x[idx][None, :])[0]))
    return _SearchSpace(bounds, constraints, starts, gaps, allowed, straight)


#: With a score, the starts of a region whose factors can be left out are the best of this many times as many settings.
_START_CANDIDATES = 30


def _region_starts(
    region: DesignRegion,
    box: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    n_starts: int,
    score: Callable[[np.ndarray], float] | None = None,
) -> np.ndarray:
    """Return up to ``n_starts`` settings of the region's columns to search from, uniform over it where it is sampled.

    ``box`` is the search box in the region's columns, which may reach beyond the coded cube.

    A search keeps the set of factors its start uses, so in a region whose factors can
    be left out the starts decide which sets are searched at all. With ``score`` they
    are chosen, not drawn: of the region's candidate blends, or of ``_START_CANDIDATES``
    times as many draws from it, the best by ``score`` of as many different sets.
    """
    sets = region._sets
    choose = bool(region.gaps) and score is not None
    if not n_starts:
        return np.empty((0, len(region.names)))
    if any(region.names[j] in region.mixture_names for j, _minimum in region.gaps):
        # A region with such a component is not sampled: the searches start from its vertices and edge midpoints.
        points = region.support_points()
        if not choose:
            points = points[rng.choice(len(points), size=n_starts, replace=len(points) < n_starts)]
    elif region.kind != "box":
        points = region.sample(n_starts, rng)
    else:  # uniform over the search box cut by the constraints
        points = UniformSampler(
            region.inequalities,
            box,
            lambda m: rng.uniform(*box, size=(m, len(box[0]))),
            seeds=region.seed_points,
            tolerances=region._widen(),  # so a start passes region.feasible()
            gaps=region.gaps,  # so the starts differ in which factors with a min_when_used they leave out
            allowed=None if sets is None else sets.allows,
        ).draw(n_starts * (_START_CANDIDATES if choose else 1), rng)
    if not choose:
        return points
    assert score is not None
    columns = [column for column, _minimum in region.gaps]
    used = points[:, columns] > np.array([region.bounds[column][0] for column in columns]) + _LEFT_OUT_TOL
    best_of_each_set: dict[bytes, int] = {}
    for i in np.argsort([score(point) for point in points], kind="stable"):
        best_of_each_set.setdefault(used[i].tobytes(), int(i))
    return points[list(best_of_each_set.values())[:n_starts]]


def _straight_form(
    form: tuple[np.ndarray, float] | None, idx: list[int], n_factors: int
) -> tuple[np.ndarray, float] | None:
    """Return a fitted straight constraint ``a @ x <= b`` over the model's factors, its rounding taken as 0.

    ``form`` is over the region's columns, which ``idx`` places among the model's
    factors; None, for a curved constraint, stays None.
    """
    if form is None:
        return None
    a, b = form
    row = np.zeros(n_factors)
    row[idx] = np.where(np.abs(a) > _FITTED_ZERO * np.abs(a).max(), a, 0.0)
    return row, b


def _straight_region(
    bounds: list[tuple[float, float]],
    forms: list[tuple[np.ndarray, float] | None],
    components: list[int],
    gaps: list[tuple[int, float, float]],
    sets: UsedSets | None,
) -> StraightRegion | None:
    """Return the search space as the exact search takes it, or None when that search does not apply.

    It does not when a constraint is curved. Nor when a factor that a count rule counts
    has no ``min_when_used``: it is in use at any amount above 0, however small, so the
    best setting may be one that is approached and never reached, and there is no best
    setting to prove.

    ``gaps`` is in the order of ``sets.optional``, so a count rule's members are positions in it.
    """
    rows = [form for form in forms if form is not None]
    if len(rows) < len(forms) or any(minimum <= zero for _column, zero, minimum in gaps):
        return None
    members = sets._members if sets is not None else []
    counts = [(list(counted), rule.at_least - always, rule.at_most - always) for counted, always, rule in members]
    return StraightRegion(bounds, rows, components, gaps, counts)


def _slsqp_inequality(
    g: Callable[[np.ndarray], np.ndarray], form: tuple[np.ndarray, float] | None, idx: list[int]
) -> dict[str, Any]:
    """Return a region's inequality ``g(x) <= 0`` as SLSQP's ``-g(x) >= 0``, with its gradient when ``g`` is straight.

    Without a gradient SLSQP estimates one by finite differences, a call of ``g`` per
    factor at every iteration. A formulation of 45 ingredients has a limit for each group
    total, and those estimates took nearly all of a search's time. ``form`` is the
    ``(a, b)`` of ``g(x) = a @ x - b`` over the model's factors (see :func:`_straight_form`),
    or None when ``g`` is curved, and ``idx`` places the region's columns among them.
    """
    constraint: dict[str, Any] = {"type": "ineq", "fun": lambda x: -float(g(x[idx][None, :])[0])}
    if form is not None:
        gradient = -form[0]
        constraint["jac"] = lambda _x: gradient
    return constraint


def _straight_models(
    fitted_models: list[dict[str, Any]], factor_names: list[str]
) -> list[tuple[np.ndarray, float]] | None:
    """Return each model's slopes and intercept, or None when a model has a term of more than one factor."""
    models = []
    for fitted in fitted_models:
        slopes, intercept = np.zeros(len(factor_names)), 0.0
        for parts, coefficient in _parsed_terms(fitted["coefficients"], factor_names):
            if len(parts) > 1:
                return None
            if parts:
                slopes[factor_names.index(parts[0])] += coefficient
            else:
                intercept += coefficient
        models.append((slopes, intercept))
    return models


def _region_for_method(region: DesignRegion | None, method: str) -> DesignRegion | None:
    """Return ``region`` for the methods that search inside one, warning and dropping it otherwise."""
    if region is not None and method not in {"desirability", "pareto_front"}:
        warnings.warn(
            f"region is honoured by 'desirability' and 'pareto_front' only; it is ignored by {method!r}, "
            "which works over search_bounds instead.",
            UserWarning,
            stacklevel=user_stacklevel(),
        )
        return None
    return region


def _multistart_results(
    objective: Callable[[np.ndarray], float],
    space: _SearchSpace,
    accept: Callable[[np.ndarray], bool],
    options: dict[str, Any] | None = None,
) -> list[optimize.OptimizeResult]:
    """Run SLSQP from every start in ``space``; return the accepted results, best first (ties in start order)."""
    results = []
    for x0 in space.starts:
        res = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=space.bounds_at(x0),
            constraints=space.constraints_at(x0),
            options=options,
        )
        if accept(res.x) and np.isfinite(res.fun):
            results.append(res)
    return sorted(results, key=lambda res: res.fun)


def _best_set_of_factors(
    objective: Callable[[np.ndarray], float],
    space: _SearchSpace,
    accept: Callable[[np.ndarray], bool],
    best: optimize.OptimizeResult,
    options: dict[str, Any] | None,
) -> optimize.OptimizeResult:
    """Change which factors with a ``min_when_used`` are in use, one or two at a time, while that improves ``best``.

    A search keeps the set of factors its start uses, and the starts cover only some of
    the sets. From the best result, the search is run again over each neighbouring set:
    one factor left out or brought in at its minimum, and then one factor in use
    exchanged for one that is left out, which a limit on a total can require (bringing
    the second in before the first is left out may not fit under the limit, or may score
    worse on the way). A better result is taken at once, and the passes end when no
    neighbouring set gives one. It is a local search over the sets, not a proof that no
    better set exists.
    """

    def search(start: np.ndarray) -> optimize.OptimizeResult | None:
        # A few iterations tell whether the set is better, since the start is next to its optimum; most are not.
        for limit in ({"maxiter": _SET_SCREEN_ITERATIONS}, {}):
            res = optimize.minimize(
                objective,
                start,
                method="SLSQP",
                bounds=space.bounds_at(start),
                constraints=space.constraints_at(start),
                options={**(options or {}), **limit},
            )
            better = np.isfinite(res.fun) and res.fun < best.fun - 1e-9 * max(1.0, abs(best.fun))
            if not (better and accept(res.x)):
                return None
            start = res.x
        return res

    def changed(columns: tuple[int, ...]) -> np.ndarray:
        start = best.x.copy()
        for j in columns:
            zero, minimum = ends[j]
            start[j] = minimum if space.left_out(start, j, zero) else zero
        return start

    ends = {j: (zero, minimum) for j, zero, minimum in space.gaps}
    improved = True
    while improved:
        improved = False
        left_out = [j for j, (zero, _minimum) in ends.items() if space.left_out(best.x, j, zero)]
        in_use = [j for j in ends if j not in left_out]
        moves = [(j,) for j in ends] + [(i, j) for i in in_use for j in left_out]
        for columns in moves:
            start = changed(columns)
            if space.allowed is not None and not space.allowed(start):
                continue  # the count rules do not allow this set of factors
            if (res := search(start)) is not None:
                best, improved = res, True
                break  # the sets next to the new best are not the ones listed above
    return best


#: How many times the search over the sets of factors may start again from a setting the models' tangents gave.
_TANGENT_STEPS = 10
#: A composite desirability within this of 1 is 1: the search has nothing left to find.
_FULL_DESIRABILITY = 1e-12
#: From how many of the best results, each of a different set of factors, the search over the sets is run.
_SET_SEARCHES = 3
#: The searches started in a region whose factors can be left out, each in the best setting of a different set.
#: Over ten components in three responses with blending terms, the set of the best setting was among the 14
#: most promising of 535 in every case tried, and first in most.
_STARTS_OVER_SETS = 30


def _of_different_sets(
    results: list[optimize.OptimizeResult], space: _SearchSpace, n: int
) -> list[optimize.OptimizeResult]:
    """Return the first ``n`` of ``results`` that each use a different set of the factors that can be left out."""
    kept: dict[tuple[bool, ...], optimize.OptimizeResult] = {}
    for res in results:
        kept.setdefault(tuple(space.left_out(res.x, j, zero) for j, zero, _minimum in space.gaps), res)
    return list(kept.values())[:n]


@dataclass
class _SetSearch:
    """The search for the best set of factors to use, for models that are not all straight lines.

    Two searches take turns until neither improves the setting. :func:`_best_set_of_factors`
    tries the sets next to the best one. The other replaces each model by its tangent at
    the best setting, a straight line, for which the best set of factors is found exactly
    (:func:`exact_optimum`); the search then runs on the models themselves from that
    setting. The first sees only the neighbouring sets, and the second any set, as far
    as the tangents describe the models there. Neither proves the result.
    """

    fitted_models: list[dict[str, Any]]
    factor_names: list[str]
    goals: list[dict[str, Any]]
    importances: list[float] | None

    def best(
        self,
        objective: Callable[[np.ndarray], float],
        space: _SearchSpace,
        accept: Callable[[np.ndarray], bool],
        best: optimize.OptimizeResult,
    ) -> optimize.OptimizeResult:
        """Return the best result the two searches reach from ``best``."""
        best = _best_set_of_factors(objective, space, accept, best, None)
        for _ in range(_TANGENT_STEPS if space.straight is not None else 0):
            start = self._from_tangents(space, best.x)
            if start is None:
                break
            found = optimize.minimize(
                objective, start, method="SLSQP", bounds=space.bounds_at(start), constraints=space.constraints_at(start)
            )
            if not (
                np.isfinite(found.fun) and found.fun < best.fun - 1e-9 * max(1.0, abs(best.fun)) and accept(found.x)
            ):
                break
            best = _best_set_of_factors(objective, space, accept, found, None)
        return best

    def _from_tangents(self, space: _SearchSpace, x: np.ndarray) -> np.ndarray | None:
        """Return the best setting for the models' tangents at ``x``, or None when none scores above 0 for them."""
        tangents = [_tangent(fitted["coefficients"], self.factor_names, x) for fitted in self.fitted_models]

        def desirability(point: np.ndarray) -> float:
            d_values = [
                individual_desirability(float(slopes @ point + intercept), goal)
                for (slopes, intercept), goal in zip(tangents, self.goals, strict=True)
            ]
            return overall_desirability(d_values, self.goals, self.importances)

        assert space.straight is not None
        found = exact_optimum(space.straight, tangents, self.goals, self.importances, desirability)
        return None if found is None else found.x


def _tangent(coefficients: list[dict[str, Any]], factor_names: list[str], x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the slopes and the intercept of the straight line that touches a polynomial model at ``x``."""
    position = {name: j for j, name in enumerate(factor_names)}
    value, slopes = 0.0, np.zeros(len(factor_names))
    for parts, coefficient in _parsed_terms(coefficients, factor_names):
        columns = [position[part] for part in parts]
        value += coefficient * float(np.prod(x[columns]))
        for p, column in enumerate(columns):  # the product rule: each factor of the term in turn
            slopes[column] += coefficient * float(np.prod(x[columns[:p] + columns[p + 1 :]]))
    return slopes, value - float(slopes @ x)


def _multistart_slsqp(
    objective: Callable[[np.ndarray], float],
    space: _SearchSpace,
    accept: Callable[[np.ndarray], bool],
    options: dict[str, Any] | None = None,
) -> optimize.OptimizeResult | None:
    """Run SLSQP from every start in ``space``; return the best accepted result, or ``None``."""
    results = _multistart_results(objective, space, accept, options)
    return results[0] if results else None


def _within_region(region: DesignRegion | None, factor_names: list[str], x: np.ndarray) -> bool | None:
    """Whether ``x`` (model column order) lies in ``region``; ``None`` without a region.

    A box region's own ``[-1, 1]`` cube is not imposed: the search box
    (``search_bounds``) bounds the search, and the region adds its constraints on top.
    Nor is it for the process factors of a mixture-process region, whose components
    must still sum to 1.
    """
    if region is None:
        return None
    point = x[[factor_names.index(n) for n in region.names]][None, :]
    if region.kind == "mixture":
        return bool(region.feasible(point, tol=1e-6)[0])
    inside = all(float(g(point)[0]) <= 1e-6 for g in region.inequalities)
    if region._sets is not None:  # the number of factors in use is one the count rules allow
        inside = inside and bool(region._sets.allows(point)[0])
    components = [region.names.index(n) for n in region.mixture_names]
    return inside and abs(float(point[0, components].sum()) - 1.0) <= 1e-6 if components else inside


def _ramp_values(y_values: Sequence[float], goals: list[dict[str, Any]]) -> np.ndarray:
    """Return the unclipped linear desirability ramps, all positive exactly where every d is.

    A maximise goal contributes ``(y - low) / (high - low)``, a minimise goal
    ``(high - y) / (high - low)``, a target goal both of its sides, and an in-range
    goal both of its margins, each over the width of the range. Unlike the
    desirabilities themselves, these keep a slope outside the acceptable limits,
    so they show which way to move from a setting where some response scores 0.
    """
    pieces: list[float] = []
    for y, goal in zip(y_values, goals, strict=True):
        low, high = float(goal["low"]), float(goal["high"])
        if goal["goal"] == "maximize":
            pieces.append((y - low) / (high - low))
        elif goal["goal"] == "minimize":
            pieces.append((high - y) / (high - low))
        elif goal["goal"] == "in_range":
            pieces.extend(((y - low) / (high - low), (high - y) / (high - low)))
        else:
            target = float(goal["target"])
            pieces.extend(((y - low) / (target - low), (high - y) / (high - target)))
    return np.array(pieces)


def _closest_to_specification(
    evaluators: list[Callable[[np.ndarray], float]],
    goals: list[dict[str, Any]],
    space: _SearchSpace,
    accept: Callable[[np.ndarray], bool],
) -> np.ndarray | None:
    """Return the setting that maximises the smallest unclipped ramp, a start with every d > 0 if one exists.

    A Derringer-Suich desirability is exactly 0, with zero gradient, wherever its
    response lies outside the acceptable limits, so a gradient search started there
    cannot leave. Tight limits can leave the region where the composite is positive
    small enough for every random start to miss it. Maximising ``t`` subject to
    every ramp being at least ``t`` (and ``t <= 1``, beyond which every
    desirability is already 1) is smooth everywhere and reaches that region
    whenever the search finds a setting with ``t > 0``; when none exists, the
    result is the setting that comes closest to meeting every limit.
    """
    n = len(space.bounds)

    def ramps(z: np.ndarray) -> np.ndarray:
        return _ramp_values([f(z[:n]) for f in evaluators], goals)

    def with_t(c: dict[str, Any]) -> dict[str, Any]:
        wider = {**c, "fun": lambda z: c["fun"](z[:n])}
        if "jac" in c:  # the constraint does not depend on t
            wider["jac"] = lambda z: np.append(c["jac"](z[:n]), 0.0)
        return wider

    every_ramp = {"type": "ineq", "fun": lambda z: ramps(z) - z[n]}
    best: tuple[float, np.ndarray] | None = None
    for x0 in space.starts:
        z0 = np.append(x0, min(1.0, float(ramps(np.append(x0, 0.0)).min())))
        res = optimize.minimize(
            lambda z: -z[n],
            z0,
            method="SLSQP",
            bounds=[*space.bounds_at(x0), (None, 1.0)],
            constraints=[every_ramp, *map(with_t, space.constraints_at(x0))],
        )
        x = res.x[:n]
        t = float(ramps(res.x).min())
        if np.all(np.isfinite(res.x)) and accept(x) and (best is None or t > best[0]):
            best = (t, x)
    return None if best is None else best[1]


#: How far, as a fraction of its range, an in-range response may sit outside its limits at an accepted point.
_LIMIT_TOL = 1e-6


def _response_limits(
    evaluators: list[Callable[[np.ndarray], float]], goals: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return SLSQP constraints that hold each ``"in_range"`` response within its limits.

    Each margin is divided by the width of the range, so the constraints of responses in
    different units are on one scale.
    """
    constraints: list[dict[str, Any]] = []
    for f, goal in zip(evaluators, goals, strict=True):
        if goal["goal"] != "in_range":
            continue
        low, high = float(goal["low"]), float(goal["high"])
        width = high - low
        constraints.append({"type": "ineq", "fun": lambda x, f=f, low=low, width=width: (f(x) - low) / width})
        constraints.append({"type": "ineq", "fun": lambda x, f=f, high=high, width=width: (high - f(x)) / width})
    return constraints


def _check_desirability_goals(goals: list[dict[str, Any]]) -> None:
    """Check every goal, and that at least one scores a response rather than only limiting it."""
    for goal in goals:
        check_goal(goal)
    if not any(goal["goal"] in SCORED_GOALS for goal in goals):
        msg = (
            "Every goal is 'in_range', which only limits a response: give at least one response a goal of "
            "'maximize', 'minimize' or 'target' to optimise."
        )
        raise ValueError(msg)


def _no_optimum(region: DesignRegion | None, limits: list[dict[str, Any]]) -> Exception:
    """Return the error for a search in which no start ended at an accepted point."""
    if limits:
        return ValueError(
            "No setting in the search region holds every 'in_range' response within its limits; widen "
            "search_bounds or the limits."
        )
    if region is None:
        return RuntimeError("optimization produced no result")
    # A counted factor with no minimum counts as used at any amount above 0, so under "at least n" the best setting
    # can hold a trace of it that the search approaches and never reaches (#1259).
    floored = {f.name for f in region.factors if f.min_when_used is not None}
    traces = [name for rule in region.counts for name in rule.names if name not in floored]
    if traces:
        return RuntimeError(
            f"no start converged inside the region. {sorted(set(traces))} are counted by a count_nonzero limit "
            "and have no min_when_used, so any amount above 0 counts as used, and the best setting can be one with "
            "a trace of one of them, which a search approaches and never reaches. Give each a min_when_used, the "
            "least amount at which it is used."
        )
    return RuntimeError("no start converged inside the region")


#: An alternative's composite desirability is at least this share of the best's.
_ALTERNATIVE_SHARE = 0.95

#: An alternative lies at least this far, in coded units, from the best and from every other alternative.
_ALTERNATIVE_DISTANCE = 0.5


@dataclass(frozen=True)
class _Scored:
    """What a desirability result reports at a setting: the models and their goals, and how settings are shown."""

    factor_names: list[str]
    names: list[str]
    evaluators: list[Callable[[np.ndarray], float]]
    goals: list[dict[str, Any]]
    importances: list[float] | None
    factor_ranges: dict[str, dict[str, float]] | None
    region: DesignRegion | None

    def at(self, x: np.ndarray) -> dict[str, Any]:
        """Return the settings, the predicted responses and their desirabilities at coded settings ``x``."""
        predictions = {name: float(f(x)) for name, f in zip(self.names, self.evaluators, strict=True)}
        each = [individual_desirability(predictions[name], g) for name, g in zip(self.names, self.goals, strict=True)]
        point: dict[str, Any] = {
            "optimal_coded": {n: float(x[i]) for i, n in enumerate(self.factor_names)},
            "predicted_responses": predictions,
            "individual_desirability": {name: float(d) for name, d in zip(self.names, each, strict=True)},
            "composite_desirability": float(overall_desirability(each, self.goals, self.importances)),
        }
        if self.region is not None:
            point["within_region"] = _within_region(self.region, self.factor_names, x)
        if self.factor_ranges:
            point["optimal_actual"] = _coded_to_actual(point["optimal_coded"], self.factor_ranges)
        return point


def _alternatives(results: list[optimize.OptimizeResult], n_alternatives: int) -> list[np.ndarray]:
    """Return up to ``n_alternatives`` other optima of a search: nearly as good as the best, and far from it.

    The results are taken best first. One is kept when its composite desirability is at least
    :data:`_ALTERNATIVE_SHARE` of the best's and it lies at least :data:`_ALTERNATIVE_DISTANCE`
    from the best and from every alternative kept before it, so each is a different choice.
    """
    if n_alternatives == 0 or not results or -results[0].fun <= 0.0:
        return []
    kept = [results[0].x]
    for res in results[1:]:
        if -res.fun < _ALTERNATIVE_SHARE * -results[0].fun or len(kept) > n_alternatives:
            break
        if min(float(np.linalg.norm(res.x - x)) for x in kept) >= _ALTERNATIVE_DISTANCE:
            kept.append(res.x)
    return kept[1:]


def _optimize_desirability(  # noqa: PLR0913
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    factor_ranges: dict[str, dict[str, float]] | None = None,
    importances: list[float] | None = None,
    random_state: int | np.random.Generator | None = 42,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    region: DesignRegion | None = None,
    *,
    align_goals: bool = True,
    n_alternatives: int = 0,
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
    region : DesignRegion or None
        Constraints the optimum must satisfy; see :func:`_search_space`.
    align_goals : bool
        False when the caller has already aligned *goals* with
        :func:`_align_goals_to_models`, so a positional pairing is not warned about twice.
    n_alternatives : int
        How many other optima to report (see :func:`_alternatives`). Asking for any searches
        from ten more starts for each.

    Returns
    -------
    dict
        Optimal settings, predicted responses, individual and composite
        desirability, ``within_region`` when a region is given,
        ``alternatives`` when *n_alternatives* is above 0, and ``proven_optimal``:
        whether no setting of the search region can score higher (see
        :func:`_exact_result`, and a composite desirability of 1).
    """
    if align_goals:
        goals = _align_goals_to_models(fitted_models, goals)
    _check_desirability_goals(goals)
    evaluators = [_build_model_evaluator(m["coefficients"], factor_names) for m in fitted_models]

    def neg_composite(x: np.ndarray) -> float:
        """Return the negated composite desirability at coded settings ``x``, for minimization."""
        d_vals = [individual_desirability(f(x), goal) for f, goal in zip(evaluators, goals, strict=True)]
        return -overall_desirability(d_vals, goals, importances)

    # Multi-start: try centre + random points.
    # SEC-33 (#282): the hard-coded ``42`` moved to the public signature
    # ``random_state=42`` (default preserves the previous deterministic
    # behaviour). Resolved via the ENG-08 helper.
    rng = check_random_state(random_state)

    limits = _response_limits(evaluators, goals)

    def accept(x: np.ndarray) -> bool:
        within = _within_region(region, factor_names, x) is not False
        return within and all(c["fun"](x) >= -_LIMIT_TOL for c in limits)

    # With straight models over straight constraints the best setting is found exactly, and proven. That
    # takes no starting points, which cost the most to find in a region whose factors can be left out.
    space = _search_space(factor_names, search_bounds, region, rng, n_starts=0)
    exact = _exact_result(fitted_models, factor_names, goals, importances, space, neg_composite, accept)
    if exact is not None and exact.found.proven and space.gaps and not n_alternatives:
        results = [exact]
    else:
        # Start from the centre of the searched region, then sample across it, so
        # that widening the bounds actually widens where the search looks.
        # Where factors can be left out a search keeps its start's set of them, so more sets are started from.
        n_starts = (_STARTS_OVER_SETS if space.gaps else 10) + 10 * n_alternatives
        space = _search_space(factor_names, search_bounds, region, rng, n_starts, score=neg_composite)
        space.constraints.extend(limits)
        if (first := _first_start(exact, evaluators, goals, space, accept)) is not None:
            space.starts.insert(0, first)
        results = _with_exact(_multistart_results(neg_composite, space, accept), exact, bool(space.gaps))
    if not results:
        raise _no_optimum(region, limits)
    proven = _is_proven(-results[0].fun, exact)
    if space.gaps and not proven:
        search = _SetSearch(fitted_models, factor_names, goals, importances)
        leaders = _of_different_sets(results, space, _SET_SEARCHES)
        results = sorted(
            [*(search.best(neg_composite, space, accept, res) for res in leaders), *results], key=lambda res: res.fun
        )

    best_result = results[0]
    composite_d = float(-best_result.fun)
    if composite_d <= 0.0:
        warnings.warn(
            "No setting in the search region was found where every response has a desirability above 0, so "
            "the composite desirability is 0. The setting reported is the one that came closest to meeting every "
            "goal's limits; widen search_bounds or relax the limits.",
            UserWarning,
            stacklevel=user_stacklevel(),
        )

    names = [m.get("response_name", "response") for m in fitted_models]
    scored = _Scored(factor_names, names, evaluators, goals, importances, factor_ranges, region)
    proven = proven or _is_proven(composite_d, exact)  # the search over the sets may have reached the bound
    result = scored.at(best_result.x) | {"optimizer_success": bool(best_result.success), "proven_optimal": proven}
    if n_alternatives:
        result["alternatives"] = [scored.at(x) for x in _alternatives(results, n_alternatives)]
    return result


def _is_proven(desirability: float, exact: optimize.OptimizeResult | None) -> bool:
    """Whether no setting of the search region scores above ``desirability``.

    None does when it is 1, or when it reaches the bound of the exact search, whichever
    search found the setting that scores it.
    """
    return desirability >= 1.0 - _FULL_DESIRABILITY or (exact is not None and exact.found.proves(desirability))


def _first_start(
    exact: optimize.OptimizeResult | None,
    evaluators: list[Callable[[np.ndarray], float]],
    goals: list[dict[str, Any]],
    space: _SearchSpace,
    accept: Callable[[np.ndarray], bool],
) -> np.ndarray | None:
    """Return the setting to search from first, or None when there is none.

    Where factors can be left out it is the exact setting, when there is one. Otherwise it
    is the setting closest to meeting every limit: from there the search can climb even
    when every other start sits where some d is 0, and on a tie at D = 0 it is the
    setting reported.
    """
    if exact is not None and space.gaps:
        return exact.x
    return _closest_to_specification(evaluators, goals, space, accept)


#: A searched setting whose composite desirability is within this share of the exact one's is as good.
_AS_GOOD = 1e-6


def _with_exact(
    searched: list[optimize.OptimizeResult], exact: optimize.OptimizeResult | None, left_out: bool
) -> list[optimize.OptimizeResult]:
    """Return the search's results, best first, with the exact one ahead of them where it is the one to report.

    Where no factor can be left out (``left_out`` is False) the search's best setting is
    reported as it was before there was an exact search, which then only says whether
    that setting is proven: a kept study opens with the settings it showed. Such a region
    is one piece, in which the search does not end at a poorer set of factors; it can
    stop a little short along a factor of small effect, and it leaves a factor without
    an effect where it began, in the middle of its range, where the exact search would
    put it at an end.

    Where factors can be left out the exact setting is reported, unless the search did
    as well.
    """
    if exact is None or (searched and not left_out):
        return searched
    if searched and searched[0].fun <= exact.fun * (1.0 - _AS_GOOD):
        return searched
    return [exact, *searched]


def _exact_result(
    fitted_models: list[dict[str, Any]],
    factor_names: list[str],
    goals: list[dict[str, Any]],
    importances: list[float] | None,
    space: _SearchSpace,
    neg_composite: Callable[[np.ndarray], float],
    accept: Callable[[np.ndarray], bool],
) -> optimize.OptimizeResult | None:
    """Return the exact optimum as a search result that holds what was ``found``, or None when it does not apply.

    It applies when every model is a straight line in the factors and every constraint of
    the space is one (see :mod:`~process_improve.experiments._exact_optimum`), and some
    setting scores above 0.
    """
    models = _straight_models(fitted_models, factor_names)
    if models is None or space.straight is None:
        return None
    found = exact_optimum(space.straight, models, goals, importances, lambda x: -neg_composite(x))
    if found is None or not accept(found.x):
        return None
    return optimize.OptimizeResult(x=found.x, fun=neg_composite(found.x), success=True, found=found)


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


def _ridge_within_bounds(
    coefficients: list[dict[str, Any]],
    factor_names: list[str],
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None,
    direction: str,
    n_radii: int,
) -> dict[str, Any]:
    """Trace the ridge out to the farthest bound, flagging the points that leave the search box.

    The ridge lives on spheres around the design centre, so it can follow a box
    only as far as its largest ``|bound|``. Bounds that differ between factors, or
    that are not symmetric about 0, are therefore not followed: each path entry
    carries ``inside_search_bounds`` instead, and a warning says so.
    """
    box = _resolve_search_bounds(search_bounds, factor_names)
    max_radius = max(max(abs(low), abs(high)) for low, high in box)
    ridge = _ridge_analysis(coefficients, factor_names, direction=direction, n_radii=n_radii, max_radius=max_radius)
    for entry in ridge.get("path", []):
        entry["inside_search_bounds"] = all(
            low - 1e-9 <= entry["coded"][name] <= high + 1e-9
            for name, (low, high) in zip(factor_names, box, strict=True)
        )
    if len(set(box)) > 1 or box[0][0] != -box[0][1]:
        warnings.warn(
            f"ridge_analysis traces spheres around the design centre out to radius {max_radius}, the "
            "largest |bound|; it cannot follow search_bounds that differ between factors or are not symmetric "
            "about 0. Path points outside the bounds have inside_search_bounds=False.",
            UserWarning,
            stacklevel=user_stacklevel(),
        )
    return ridge


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
            if kind == "in_range":
                msg += " To hold a response within limits, use method='desirability' with an 'in_range' goal."
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


def _pareto_front(
    fitted_models: list[dict[str, Any]],
    goals: list[dict[str, Any]],
    factor_names: list[str],
    *,
    n_points: int = 21,
    search_bounds: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    random_state: int | np.random.Generator | None = 42,
    region: DesignRegion | None = None,
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
    region : DesignRegion or None
        Constraints every front point must satisfy.

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

    rng = check_random_state(random_state)
    if region is None:
        bounds = _resolve_search_bounds(search_bounds, factor_names)
        lows, highs = np.array(bounds).T
        starts = [(lows + highs) / 2.0, lows.copy(), highs.copy(), *(rng.uniform(lows, highs) for _ in range(5))]
        space = _SearchSpace(bounds, [], starts)
    else:
        space = _search_space(factor_names, search_bounds, region, rng, n_starts=8)

    def raw(x: np.ndarray) -> np.ndarray:
        return np.array([evaluator(x) for evaluator in evaluators])

    def utility(x: np.ndarray) -> np.ndarray:
        return _as_utilities(raw(x), senses)

    def best_of(objective: Callable[[np.ndarray], float]) -> np.ndarray | None:
        # shortcut: with min_when_used factors each front point keeps the set of factors one of the 8 starts
        # uses, upgrade by running _best_set_of_factors on each point if fronts over many such factors are needed.
        # SLSQP's default ftol of 1e-6 leaves front points a visible distance
        # short of the true frontier. These objectives cost microseconds to
        # evaluate, so there is nothing to buy by stopping early.
        res = _multistart_slsqp(
            objective, space, lambda x: _within_region(region, factor_names, x) is not False, _SLSQP_OPTIONS
        )
        return None if res is None else res.x

    def anchor(objective: Callable[[np.ndarray], float]) -> np.ndarray:
        if (x := best_of(objective)) is None:
            raise RuntimeError("Pareto front: no start converged inside the region for a single objective.")
        return x

    payoff_raw, payoff_utility = _payoff_table(anchor, raw, utility, senses, len(names))
    ideal_utility = payoff_utility.diagonal().copy()
    nadir_utility = payoff_utility.min(axis=0)
    spread = np.where(np.abs(ideal_utility - nadir_utility) > _SPREAD_FLOOR, ideal_utility - nadir_utility, 1.0)

    def chebyshev(x: np.ndarray, w: np.ndarray) -> float:
        gap = w * (ideal_utility - utility(x)) / spread
        return float(gap.max() + _PARETO_AUGMENT * gap.sum())

    weights = _simplex_weights(len(names), n_points)
    found = [best_of(functools.partial(chebyshev, w=w)) for w in weights]
    points = np.array([x for x in found if x is not None])  # a weight with no feasible optimum adds no point
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
    """Convert coded factor settings to actual units, as plain Python floats."""
    actual = {}
    for name, coded_val in coded.items():
        if name in factor_ranges:
            lo = float(factor_ranges[name]["low"])
            hi = float(factor_ranges[name]["high"])
            actual[name] = (lo + hi) / 2.0 + float(coded_val) * (hi - lo) / 2.0
        else:
            actual[name] = float(coded_val)
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


#: Methods that analyse the surface of one fitted model.
_SINGLE_RESPONSE_METHODS = {
    "stationary_point",
    "canonical_analysis",
    "steepest_ascent",
    "steepest_descent",
    "ridge_analysis",
}


def _span(coding: dict[str, Any]) -> tuple[float, float]:
    return float(coding["low"]), float(coding["high"])


#: The range under which a coded setting and an actual one are the same number.
_AS_GIVEN = {"low": -1.0, "high": 1.0}


def _ranges_from_coding(
    fitted_models: list[dict[str, Any]],
    factor_ranges: dict[str, dict[str, float]] | None,
    components: Sequence[str] = (),
) -> dict[str, dict[str, float]] | None:
    """Check that each model is on the coded scale, and add the factor ranges its coding implies.

    ``analyze_experiment`` reports the scale of its coefficients as ``coding``, and how
    each factor maps to -1 and +1 as ``factor_coding``. The optimiser evaluates models at
    coded settings, so a model whose coefficients are in actual units would be searched
    over the wrong region (-1 to +1 degC, say). A coded model instead says which actual
    settings its coded optimum stands for. A model without these keys, such as one
    written out by hand, is taken to be coded, as before. Mixture ``components`` (those
    of the region searched) are proportions on either scale, so a model's coding of one
    is left out, and a range given for one in *factor_ranges* is replaced by the range
    that reports a setting as it is: the range given would read the proportion as a
    coded setting and report an amount that no run has.

    Raises
    ------
    ValueError
        If a model's coefficients are in actual units, if two models code a factor from
        different ranges, or if *factor_ranges* disagrees with a model's coding.
    """
    implied: dict[str, dict[str, float]] = {}
    for i, model in enumerate(fitted_models):
        coding = {f: how for f, how in (model.get("factor_coding") or {}).items() if f not in components}
        name = model.get("response_name", f"fitted_models[{i}]")
        if coding and model.get("coding") == "actual":
            spans = ", ".join(f"{factor} from {how['low']} to {how['high']}" for factor, how in coding.items())
            msg = (
                f"The coefficients of {name!r} are in actual units ({spans}), but optimize_responses evaluates "
                "models at coded settings, -1 to +1. Fit it with analyze_experiment(..., coding='coded'), or with "
                "coding set to the factor ranges, and pass that result. A split-plot fit is made on the runs as "
                "given: code its factors to -1 and +1 before fitting it."
            )
            raise ValueError(msg)
        for factor, how in coding.items():
            if not all(isinstance(how[end], numbers.Real) for end in ("low", "high")):
                continue  # a categorical factor's two levels: its setting stays coded
            known = implied.setdefault(factor, {"low": float(how["low"]), "high": float(how["high"])})
            if not np.allclose(_span(known), _span(how)):
                msg = (
                    f"The models code {factor} from different ranges ({known['low']} to {known['high']}, and "
                    f"{how['low']} to {how['high']} for {name!r}), so a coded setting does not mean the same in "
                    "each. Fit them all with the same ranges: analyze_experiment(..., coding={...})."
                )
                raise ValueError(msg)
    ranges = {factor: _AS_GIVEN if factor in components else given for factor, given in (factor_ranges or {}).items()}
    for factor, given in ranges.items():
        if factor in implied and not np.allclose(_span(given), _span(implied[factor])):
            msg = (
                f"factor_ranges gives {factor} from {given['low']} to {given['high']}, but the coefficients are coded "
                f"with {factor} from {implied[factor]['low']} to {implied[factor]['high']}. Leave factor_ranges out, "
                "or fit with analyze_experiment(..., coding=factor_ranges) so that the two agree."
            )
            raise ValueError(msg)
    return {**implied, **ranges} or None


def _check_fitted_models(fitted_models: list[dict[str, Any]], method: str) -> None:
    """Check that *fitted_models* carries what *method* needs, naming what is missing.

    Raises
    ------
    ValueError
        If the list is empty, a model lacks ``"coefficients"`` or
        ``"factor_names"``, or a single-response method gets more than one model
        (it would otherwise analyse the first and ignore the rest).
    """
    if not fitted_models:
        msg = "At least one fitted model is required."
        raise ValueError(msg)
    for i, model in enumerate(fitted_models):
        if not isinstance(model, dict):
            msg = f"fitted_models[{i}] must be a dict; got {type(model).__name__}."
            raise TypeError(msg)
        if "coefficients" not in model:
            msg = (
                f"fitted_models[{i}] has no 'coefficients'. Run analyze_experiment with 'coefficients' among its "
                "analysis_type values, and pass its result."
            )
            raise ValueError(msg)
        if "factor_names" not in model:
            msg = (
                f"fitted_models[{i}] has no 'factor_names', the ordered factors its coefficients refer to "
                "(analyze_experiment returns them alongside 'coefficients'). Add e.g. 'factor_names': ['A', 'B']."
            )
            raise ValueError(msg)
    if method in _SINGLE_RESPONSE_METHODS and len(fitted_models) > 1:
        names = [m.get("response_name", f"model {i}") for i, m in enumerate(fitted_models)]
        msg = (
            f"method={method!r} analyses one response; got {len(fitted_models)} models ({names}). "
            "Call it once per response, or use 'desirability' or 'pareto_front' to optimise them together."
        )
        raise ValueError(msg)


def _check_alternatives(n_alternatives: int) -> int:
    """Return ``n_alternatives``, refusing anything but a whole number of at least 0."""
    if isinstance(n_alternatives, bool) or not isinstance(n_alternatives, int) or n_alternatives < 0:
        msg = f"n_alternatives must be a whole number of at least 0; got {n_alternatives!r}."
        raise ValueError(msg)
    return n_alternatives


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
    region: DesignRegion | None = None,
    random_state: int | np.random.Generator | None = 42,
    n_alternatives: int = 0,
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
    for goal in aligned_goals:
        check_goal(goal)
    importances = response_importance
    if importances is None:
        importances = [g.get("importance", 1.0) for g in aligned_goals]
    check_importances(importances, len(fitted_models))

    desirability = _optimize_desirability(
        fitted_models,
        aligned_goals,
        factor_names,
        factor_ranges,
        importances,
        random_state=random_state,
        search_bounds=search_bounds,
        region=region,
        align_goals=False,
        n_alternatives=n_alternatives,
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


def optimize_responses(  # noqa: PLR0913
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
    ridge_direction: str = "maximize",
    n_pareto_points: int = 21,
    region: DesignRegion | None = None,
    random_state: int | np.random.Generator | None = 42,
    n_alternatives: int = 0,
    flat_tolerance: float = _FLAT_EIGENVALUE_TOL,
) -> dict[str, Any]:
    """Find optimal factor settings for one or multiple responses.

    Parameters
    ----------
    fitted_models : list[dict]
        One model for the single-response methods (``"stationary_point"``,
        ``"canonical_analysis"``, ``"steepest_ascent"``, ``"steepest_descent"``,
        ``"ridge_analysis"``), one or more for ``"desirability"`` and two or more
        for ``"pareto_front"``. The models are evaluated at coded settings, so
        their coefficients must be on the coded scale: pass the result of
        ``analyze_experiment(..., analysis_type="coefficients", coding="coded")``
        as it is. One fitted with ``coding="actual"`` on factors that are not
        already coded is refused, since its optimum would be searched for between
        -1 and +1 in actual units. Each dict describes a fitted model with keys:

        - ``"response_name"`` (str) - name of the response.
        - ``"coefficients"`` (list[dict]) - coefficient list, each with
          ``"term"`` and ``"coefficient"`` keys as returned by
          ``analyze_experiment(..., analysis_type="coefficients")``.
        - ``"factor_names"`` (list[str]) - ordered factor names.
        - ``"mse_residual"`` (float, optional) - mean squared error.
        - ``"r_squared"`` (float, optional) - model R-squared.
        - ``"coding"`` and ``"factor_coding"`` (optional) - the scale of the
          coefficients and each factor's coded range, as ``analyze_experiment``
          reports them. A model without them is taken to be coded.

    goals : list[dict] or None
        Per-response optimisation goals.  Each dict has keys:

        - ``"response"`` (str) - response name. Matched against each model's
          ``"response_name"``; when both sides name their responses the goals
          are reordered to match, so the two lists need not be in the same
          order, and names that do not pair up one to one raise a
          ``ValueError``. When either side omits a name, goals are taken in
          list order, with a warning.
        - ``"goal"`` (str) - ``"maximize"``, ``"minimize"``, ``"target"``
          or ``"in_range"``. An ``"in_range"`` goal (Design-Expert's "is in
          range") holds the predicted response between ``low`` and ``high``:
          the search keeps to settings where it does, its individual
          desirability is 1 there, and it is left out of the composite. At
          least one goal must be of another kind. ``"pareto_front"`` takes
          the other three only.
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
        units.  Used for coded ↔ actual conversion. A coded model from
        ``analyze_experiment`` supplies the ranges of its own factors, and
        a range given here for one of them must agree with it. A range
        for a mixture component of *region* is not used: a proportion is
        the same on either scale, and is reported as it is.
    step_size : float
        Step length for steepest ascent/descent: the Euclidean distance, in
        coded units, between successive points on the path. Must be positive.
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
        ``(-1, 1)`` on every factor. ``"ridge_analysis"`` traces spheres out to
        the largest ``|bound|`` and flags each path point with
        ``inside_search_bounds``, since a sphere cannot follow bounds that
        differ between factors or are asymmetric.

        That default suits a two-level design but understates a central
        composite design, whose axial runs sit at plus or minus alpha: leaving
        it at the cube would refuse to consider settings the experiment
        actually covered. Pass ``(-1.41, 1.41)`` for a two-factor rotatable
        central composite design, or a mapping such as
        ``{"T": (-1.41, 1.41)}`` to widen one factor only. Factors left out of
        a mapping keep the (-1, 1) default.
    ridge_direction : {"maximize", "minimize"}
        Which ridge ``method="ridge_analysis"`` traces.
    n_pareto_points : int
        Target number of weight vectors for ``method="pareto_front"``. The front
        returned is usually smaller, since dominated and duplicate solutions are
        dropped.
    region : DesignRegion or None
        The region the optimum must lie in, for ``"desirability"`` and
        ``"pareto_front"``: the same constraints the design was built under
        (``DesignRegion.from_dict(result.metadata["region"])``), so the
        recommended settings are ones the process can run. For a box region the
        constraints apply on top of *search_bounds*. For a mixture region the
        factors are the proportions the Scheffé model was fitted on, their bounds
        replace *search_bounds*, and the optimum sums to 1. The desirability
        result then carries ``within_region``. Other methods ignore it, with a
        warning.

        The desirability result also carries ``proven_optimal``: whether no
        setting of the region scores higher than the one reported, to one
        part in a million. When every model is a straight line in the factors
        (no interaction or squared term) and every constraint is one, the
        best setting and that bound are found by mixed-integer linear
        programming. Where factors can be left out
        (``Factor(min_when_used=...)``, ``count_nonzero`` limits) that
        setting is the one reported, and it is proven. Elsewhere the setting
        reported is the search's, as before, and is proven when it reaches
        the bound. With a curved model or constraint the optimum is searched
        for from starting points, and ``proven_optimal`` is False unless the
        composite desirability reaches 1.
    random_state : int, Generator, or None
        Seed for the random starting points of the multistart search used by
        ``"desirability"`` and ``"pareto_front"``; the other methods are
        deterministic. The default of 42 keeps results reproducible; try another
        seed to check that an optimum does not depend on where the search began.
    n_alternatives : int
        For ``"desirability"``: how many other settings that do almost as well to
        report beside the best, under ``"alternatives"``. Each is a local optimum of
        the multistart search whose composite desirability is at least 95% of the
        best's, at least 0.5 apart in coded units from the best and from every other
        alternative, best first. Asking for any searches from ten more starts for
        each. The default of 0 reports none and leaves the search as it was.
    flat_tolerance : float
        For ``"stationary_point"`` and ``"canonical_analysis"``: an eigenvalue of
        the pure-quadratic matrix ``B`` at most this share of the largest in size
        is read as zero, so the surface is classified as a ridge along its
        eigenvector, and so is the slope along that eigenvector: a ridge rises
        only when its slope is above this share of the largest eigenvalue. The
        default, 1e-8, reads only rounding as zero. An estimated eigenvalue is
        rarely exactly zero, and the sources read a small one as flat: Box,
        Hunter and Hunter's stationary ridge has eigenvalues -9.02 and 0.0017 and
        a slope of 0.005 along it, which ``flat_tolerance=0.05`` reads as a
        stationary ridge; their rising ridge, with a slope of 2.97, still rises.
        Below 5% of the largest, the result's ``classification_note`` says so.

    Returns
    -------
    dict[str, Any]
        Results keyed by method.  Always includes ``"method"`` and
        ``"factor_names"``.

    Raises
    ------
    ValueError
        If *method* is unknown, if *flat_tolerance* is not in [0, 1), if
        *fitted_models* is empty or a model lacks ``"coefficients"`` or
        ``"factor_names"``, if ``"stationary_point"`` or ``"canonical_analysis"``
        gets a factor name that no model term uses, if a single-response method
        (``"stationary_point"``, ``"canonical_analysis"``, ``"steepest_ascent"``,
        ``"steepest_descent"``, ``"ridge_analysis"``) gets more than one model, if
        a method that needs goals is called without them, or if a model's
        coefficients are in actual units, two models code a factor from
        different ranges, or *factor_ranges* disagrees with a model's coding.

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
    if not 0 <= flat_tolerance < 1:
        msg = f"flat_tolerance must be at least 0 and below 1, a share of the largest eigenvalue; got {flat_tolerance}."
        raise ValueError(msg)

    _check_fitted_models(fitted_models, method)
    factor_ranges = _ranges_from_coding(fitted_models, factor_ranges, region.mixture_names if region else ())

    # Use factor_names from the first model as the canonical ordering
    factor_names = fitted_models[0]["factor_names"]
    coefficients = fitted_models[0]["coefficients"]

    result: dict[str, Any] = {"method": method, "factor_names": factor_names}
    region = _region_for_method(region, method)

    if method == "stationary_point":
        result["stationary_point"] = _find_stationary_point(
            coefficients, factor_names, factor_ranges, search_bounds, flat_tolerance
        )

    elif method == "canonical_analysis":
        result["canonical_analysis"] = _canonical_analysis(coefficients, factor_names, flat_tolerance)
        # Also include the stationary point for context
        result["stationary_point"] = _find_stationary_point(
            coefficients, factor_names, factor_ranges, search_bounds, flat_tolerance
        )

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
            region=region,
            random_state=random_state,
            n_alternatives=_check_alternatives(n_alternatives),
        )

    elif method == "ridge_analysis":
        result["ridge_analysis"] = _add_actual_units(
            _ridge_within_bounds(coefficients, factor_names, search_bounds, ridge_direction, n_steps),
            factor_ranges,
            "path",
        )

    else:  # "pareto_front", the last of _METHODS, which every call was checked against
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
                random_state=random_state,
                region=region,
            ),
            factor_ranges,
            "front",
        )

    return result


# ---------------------------------------------------------------------------
# The settings that meet a specification
# ---------------------------------------------------------------------------


def _shares_below(
    predicted: np.ndarray, standard_error: np.ndarray, df: float, low: float | None, high: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the chance that one new run falls below ``low``, and below ``high``; an end of None is open.

    The new run is distributed about its prediction as Student's t with the model's residual
    degrees of freedom, scaled by the standard error of a new observation. The chance that it
    falls inside the window is the second less the first.
    """
    share = lambda end, open_end: (  # noqa: E731
        np.full(len(predicted), open_end) if end is None else stats.t.cdf((end - predicted) / standard_error, df)
    )
    return share(low, 0.0), share(high, 1.0)


#: How many settings of the earlier responses' errors each later response is averaged over, when the responses'
#: errors are correlated: 2 to this power. With 128 the chance of two windows is right to four decimals.
_JOINT_NODES_LOG2 = 7


def _joint_probability(below_low: np.ndarray, below_high: np.ndarray, correlations: np.ndarray) -> np.ndarray:
    """Return the chance that every response's new run is inside its window, their errors correlated.

    ``below_low`` and ``below_high`` hold a row for each setting and a column for each response: the chance
    that its new run falls below its window's low end, and below its high end (:func:`_shares_below`).
    Each response keeps that distribution of its own, and the responses are joined as normal variables
    with a correlation are: a Gaussian copula. The chance is worked by Genz's method (1992): the first
    response's error is drawn inside its window, the second's window is then judged given the first, and so
    on, and the product is averaged over a set of evenly spread draws. ``correlations`` holds a correlation
    for each draw, as :func:`_correlations` gives them: the same one each time when it is known, a draw of its
    own each time when it is estimated, so the average is over the doubt in it too.
    """
    k = below_low.shape[1]
    # Perfectly correlated responses are allowed: a hair off them keeps the factors below defined.
    factors = np.linalg.cholesky((1 - 1e-9) * correlations + 1e-9 * np.eye(k))
    steps = 2**_JOINT_NODES_LOG2
    # Evenly spread over the unit cube, a point for each draw; off its faces by half a step.
    nodes = (
        np.zeros((1, 0))
        if k == 1
        else stats.qmc.Sobol(k - 1, scramble=False).random_base2(_JOINT_NODES_LOG2) + 0.5 / steps
    )
    low, high = special.ndtri(below_low), special.ndtri(below_high)  # each end on the normal scale
    total = np.zeros(len(below_low))
    # One response has one node and no correlation to draw from: its first factor stands for all.
    for node, factor in zip(nodes, factors[: len(nodes)], strict=True):
        d, e = below_low[:, 0], below_high[:, 0]
        inside = e - d
        drawn: list[np.ndarray] = []
        for i in range(1, k):
            # The response before, drawn inside its window; off the very ends, where the normal scale is infinite.
            drawn.append(special.ndtri(np.clip(d + node[i - 1] * (e - d), 1e-12, 1 - 1e-12)))
            shift = factor[i, :i] @ np.array(drawn)
            d = special.ndtr((low[:, i] - shift) / factor[i, i])
            e = special.ndtr((high[:, i] - shift) / factor[i, i])
            inside = inside * (e - d)
        total += inside
    return total / len(nodes)


def _correlations(
    correlation: np.ndarray, degrees_of_freedom: float | None, random_state: int | np.random.Generator | None
) -> np.ndarray:
    """Return a correlation of the responses' errors for each draw of :func:`_joint_probability`.

    Without ``degrees_of_freedom`` the correlation is taken as known, the same for every draw. With them it is
    an estimate from residuals with that many degrees of freedom, and each draw takes a correlation of its own
    from what the residuals leave possible: the covariance of the errors is drawn from the inverse Wishart
    distribution with those degrees of freedom about the residuals' cross-products, the posterior of a
    multivariate regression under Jeffreys' prior, and scaled to a correlation. The correlation of such a draw
    does not depend on the scale of the cross-products, so the correlation alone is enough to draw from.
    """
    k, draws = len(correlation), 2**_JOINT_NODES_LOG2
    if degrees_of_freedom is None:
        return np.broadcast_to(correlation, (draws, k, k))
    # A perfect correlation is allowed, as in _joint_probability: a hair off it keeps the scale positive definite.
    scale = (1 - 1e-9) * correlation + 1e-9 * np.eye(k)
    covariances = stats.invwishart.rvs(
        df=degrees_of_freedom, scale=scale, size=draws, random_state=check_random_state(random_state)
    ).reshape(draws, k, k)
    spread = np.sqrt(np.einsum("nii->ni", covariances))
    return covariances / (spread[:, :, None] * spread[:, None, :])


def _checked_correlation(raw: object, k: int) -> np.ndarray:
    """Return the responses' error correlation as a ``k`` by ``k`` array, or say why it is not one."""
    matrix = np.asarray(raw, dtype=float)
    if (
        matrix.shape != (k, k)
        or not np.allclose(matrix, matrix.T)
        or not np.allclose(np.diag(matrix), 1.0)
        or float(np.linalg.eigvalsh(matrix).min()) < -1e-9
    ):
        msg = (
            f"residual_correlation must be the {k} by {k} correlation matrix of the responses' errors, in the "
            "order of the models: symmetric, with 1 on its diagonal, and with no negative eigenvalue."
        )
        raise ValueError(msg)
    return matrix


def _matrix_over(model: dict[str, Any], points: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return a model's matrix over some settings, a column for each of its terms, and its coefficients."""
    missing = [factor for factor in model["factor_names"] if factor not in points.columns]
    if missing:
        msg = f"The points have no column for the factor(s) {missing} of the model of {name!r}."
        raise ValueError(msg)
    parsed = _parsed_terms(model["coefficients"], model["factor_names"])
    columns = [np.prod([points[part].to_numpy(dtype=float) for part in parts], axis=0) for parts, _ in parsed]
    matrix = np.column_stack([np.broadcast_to(column, len(points)) for column in columns])
    return matrix, np.array([coefficient for _, coefficient in parsed])


def _new_run_scatter(model: dict[str, Any], matrix: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return the standard error of one new run at each row of a model's matrix, and the fit's degrees of freedom.

    None when the model does not hold its residual mean square, its residual degrees of freedom and its
    unscaled covariance, or has no degrees of freedom left to estimate the error from. A model that holds
    a ``"prior_sigma"`` is scaled by it instead: the error is then known, not estimated, so the degrees of
    freedom are infinite (the normal distribution) and a saturated fit has a scatter too.
    """
    # A result of analyze_experiment holds these in its model summary.
    summary = model.get("model_summary") or {}
    mse, df, prior = (model.get(key, summary.get(key)) for key in ("mse_residual", "df_residual", "prior_sigma"))
    covariance = model.get("unscaled_covariance")
    if prior is not None:
        mse, df = float(prior) ** 2, np.inf
    if mse is None or df is None or covariance is None or float(df) <= 0:
        return None
    leverage = np.einsum("ij,jk,ik->i", matrix, np.asarray(covariance, dtype=float), matrix)
    return np.sqrt(float(mse) * (1.0 + leverage)), float(df)


def feasible_region(
    fitted_models: list[dict[str, Any]],
    windows: Mapping[str, Sequence[float | None]],
    points: pd.DataFrame,
    residual_correlation: Sequence[Sequence[float]] | np.ndarray | None = None,
    *,
    correlation_degrees_of_freedom: float | None = None,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Say, at each of some settings, whether every response is predicted inside its window, and how surely.

    The best settings answer "where is the response best?". This answers the reverse:
    which settings give results inside a specification, such as a yield between 78 and
    82 with an impurity below 0.5? The settings where every prediction is inside its
    window are the region that meets the specification; those where a new run meets it
    with a high probability, 0.9 say, are the design space of quality by design.

    Parameters
    ----------
    fitted_models : list[dict]
        Each response's model, as :func:`optimize_responses` takes it: its
        ``"response_name"``, its ``"coefficients"`` (each a ``"term"`` and a
        ``"coefficient"``) and its ``"factor_names"``. For the probability a model also
        holds what the scatter of a new run about its prediction is worked from:

        - ``"mse_residual"`` (float) - the residual mean square of the fit.
        - ``"df_residual"`` (float) - its residual degrees of freedom.
        - ``"prior_sigma"`` (float, optional) - the standard deviation of the error,
          when it is known from before these runs. It replaces the two above, and the
          new run is then normal about its prediction.
        - ``"unscaled_covariance"`` (2-D array) - the covariance of the coefficients
          over the residual variance, the inverse of ``X'X``, its rows and columns in
          the order of ``"coefficients"``. A statsmodels fit holds it as
          ``normalized_cov_params``.

        A result of ``analyze_experiment(..., analysis_type="coefficients")`` holds all
        three, the first two in its ``"model_summary"``, so it is taken as it is.
    windows : Mapping[str, Sequence[float | None]]
        Each response's window by its name, ``(low, high)``, on the scale its model was
        fitted on. An end of ``None`` is open: ``(None, 0.5)`` is "0.5 at most".
    points : pd.DataFrame
        The settings to judge, a row each, in a column for each factor of the models,
        on the scale the models were fitted on (coded, for a model from
        ``analyze_experiment(..., coding="coded")``). A grid over two factors gives a
        picture of the region; settings drawn evenly from the ranges give its share of
        them, as the mean of ``feasible``.
    residual_correlation : array-like of shape (n_responses, n_responses), optional
        The correlation of the responses' errors on one run, in the order of
        *fitted_models*: the correlation of the models' residuals estimates it. None
        (the default) takes the errors as independent. It changes ``"probability"``
        alone.
    correlation_degrees_of_freedom : float, optional
        The residual degrees of freedom *residual_correlation* was estimated from: the
        runs every model used, less the terms of the largest model. With it the chance
        is averaged over the doubt in the correlation as well; without it (the default)
        the correlation is taken as known. It must be more than the number of responses
        less one.
    random_state : int, numpy.random.Generator or None, optional
        Seeds the draws of the correlation that *correlation_degrees_of_freedom* asks
        for; the rest is not random. See the reproducibility contract.

    Returns
    -------
    pd.DataFrame
        A row for each of *points*, under its index:

        - a column for each response, by its name: its prediction;
        - ``"feasible"`` (bool) - whether every prediction is inside its window, the
          ends counted as inside;
        - ``"probability"`` (float) - the chance that one new run at those settings
          meets every window, or NaN when a model does not hold what it is worked from.

    Raises
    ------
    ValueError
        If the windows and the models do not name the same responses, if a window's
        low end is above its high end, if *points* lacks a factor's column, if
        *residual_correlation* is not a correlation matrix of the responses, or if
        *correlation_degrees_of_freedom* is given without it or is too few.

    Notes
    -----
    A response's new run is distributed about its prediction as Student's t with the
    residual degrees of freedom of its model, scaled by the standard error of a new
    observation, ``sqrt(mse_residual * (1 + x' C x))``, where ``x`` is the settings'
    row of the model matrix and ``C`` the unscaled covariance. The probability for
    one response is the share of that distribution inside its window. A model with a
    ``"prior_sigma"`` takes ``prior_sigma ** 2`` for the mean square and the normal
    distribution for the t, since its error is not estimated from these runs. Without a
    *residual_correlation* the responses' errors are taken as independent, so the
    probability of meeting every window is the product over the responses. Responses
    measured on the same run are often correlated, and the product is then a guide:
    errors that move together make it likelier that both windows are met, or both
    missed. With a *residual_correlation* each response keeps its own distribution,
    and the responses are joined as normal variables with that correlation are (a
    Gaussian copula), which Genz's method (1992) integrates. A correlation estimated
    from few residual degrees of freedom is itself uncertain, and with one degree of
    freedom two models' residuals are always perfectly correlated: estimate it from
    fits that leave several, and give *correlation_degrees_of_freedom* so that each of
    the integration's draws takes a correlation of its own from the inverse Wishart
    posterior of the error covariance. Taking an estimate as known understates the
    chance a little, by up to a few points with three responses and five degrees of
    freedom. More runs shrink
    ``C`` and add degrees of freedom, so the region of a given probability grows
    towards the region where the predictions are inside their windows, and never
    beyond the window's own scatter: a window narrower than the noise is met with a
    low probability at every setting.

    Examples
    --------
    >>> import pandas as pd
    >>> model = {
    ...     "response_name": "y",
    ...     "coefficients": [{"term": "Intercept", "coefficient": 50.0}, {"term": "A", "coefficient": 10.0}],
    ...     "factor_names": ["A"],
    ... }
    >>> feasible_region([model], {"y": (45, 55)}, pd.DataFrame({"A": [-1.0, 0.0, 0.5, 1.0]}))["feasible"].tolist()
    [False, True, True, False]

    References
    ----------
    Genz, A. (1992). Numerical computation of multivariate normal probabilities.
    *Journal of Computational and Graphical Statistics*, 1(2), 141-149.
    """
    names = [model.get("response_name", f"Response {i + 1}") for i, model in enumerate(fitted_models)]
    if sorted(names) != sorted(windows) or len(set(names)) != len(names):
        msg = (
            f"The windows name the responses {sorted(windows)}, and the models {sorted(names)}: "
            "give one window for each model."
        )
        raise ValueError(msg)
    correlation = None if residual_correlation is None else _checked_correlation(residual_correlation, len(names))
    if correlation_degrees_of_freedom is not None and (
        correlation is None or not correlation_degrees_of_freedom > len(names) - 1
    ):
        msg = (
            "correlation_degrees_of_freedom goes with a residual_correlation, and must be more than the number of "
            f"responses less one ({len(names) - 1}); got {correlation_degrees_of_freedom!r}."
        )
        raise ValueError(msg)
    out = pd.DataFrame(index=points.index)
    feasible = np.ones(len(points), dtype=bool)
    shares: list[tuple[np.ndarray, np.ndarray]] | None = []  # each response's, or None once one has no scatter
    for name, model in zip(names, fitted_models, strict=True):
        low, high = windows[name]
        if low is not None and high is not None and low > high:
            msg = f"The window of {name!r} runs from {low} down to {high}: give its low end first."
            raise ValueError(msg)
        matrix, coefficients = _matrix_over(model, points, name)
        predicted = matrix @ coefficients
        out[name] = predicted
        feasible &= (low is None or predicted >= low) & (high is None or predicted <= high)
        scatter = _new_run_scatter(model, matrix)
        if scatter is None or shares is None:
            shares = None
        else:
            shares.append(_shares_below(predicted, *scatter, low, high))
    out["feasible"] = feasible
    if shares is None:
        out["probability"] = np.nan
    elif correlation is None:
        out["probability"] = np.prod([high - low for low, high in shares], axis=0)
    else:
        below_low, below_high = (np.column_stack(ends) for ends in zip(*shares, strict=True))
        drawn = _correlations(correlation, correlation_degrees_of_freedom, random_state)
        out["probability"] = _joint_probability(below_low, below_high, drawn)
    return out
