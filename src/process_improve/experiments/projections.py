# (c) Kevin Dunn, 2010-2026. MIT License.
r"""What a design does once most of its factors turn out not to matter: its projections.

A screening design is often judged on its projections. Once the analysis shows that only a few
factors matter, the runs already made are a design in those few, and the question is what that
smaller design can estimate and how well. :func:`evaluate_projections` evaluates a design on every
projection onto a given number of its factors (or on a given number from each group of factors,
such as three quantitative and one categorical) and returns each projection's metrics, how many of
the projections estimate the model, and the metrics' averages: the projection estimation capacity
and the average efficiencies of Loeppky, Sitter and Tang (2007) and of Núñez Ares and Goos (2020).

Examples
--------
>>> from process_improve.experiments import Factor, generate_design
>>> from process_improve.experiments.projections import evaluate_projections
>>> factors = [Factor(name=n, low=-1, high=1) for n in "ABCDEF"]
>>> dsd = generate_design(factors, design_type="dsd", n_center_points=3)
>>> found = evaluate_projections(dsd, 3, model="quadratic")
>>> found["n_estimable"], found["n_projections"]
(20, 20)
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from process_improve.experiments.designs_utils import RESERVED_COLUMN_NAMES
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import DesignResult

#: The models a projection is evaluated under, each including the ones before it.
PROJECTION_MODELS = ("main_effects", "interactions", "quadratic")

#: The most projections evaluated in one call; past it a ValueError says how many there would be.
MOST_PROJECTIONS = 50_000


def _factor_frame(design_matrix: pd.DataFrame | DesignResult) -> pd.DataFrame:
    """Return the design's factor columns: a DesignResult's factors, or a frame without its run columns."""
    if isinstance(design_matrix, DesignResult):
        return pd.DataFrame(design_matrix.design)[list(design_matrix.factor_names)]
    frame = pd.DataFrame(design_matrix)
    return frame[[c for c in frame.columns if c not in RESERVED_COLUMN_NAMES]]


def projection_formula(frame: pd.DataFrame, factors: list[str], model: str) -> str:
    """Return the model in a projection's factors as a formula's right-hand side.

    ``"main_effects"``; ``"interactions"``, main effects and every two-factor interaction; or
    ``"quadratic"``, those and the square of every numeric factor the projection runs at three
    or more levels (a two-level factor has no square to estimate, nor a categorical one).

    Parameters
    ----------
    frame : pandas.DataFrame
        The design's factor columns, in coded units.
    factors : list[str]
        The projection's factors.
    model : {"main_effects", "interactions", "quadratic"}
        The model.

    Returns
    -------
    str
        The right-hand side, such as ``"(A + B) ** 2 + I(A ** 2) + I(B ** 2)"``.
    """
    if model not in PROJECTION_MODELS:
        raise ValueError(f"model must be one of {PROJECTION_MODELS}; got model={model!r}.")
    joined = " + ".join(factors)
    if model == "main_effects":
        return joined
    rhs = f"({joined}) ** 2" if len(factors) > 1 else joined
    if model == "quadratic":
        curved = [f for f in factors if pd.api.types.is_numeric_dtype(frame[f]) and frame[f].round(9).nunique() >= 3]
        rhs += "".join(f" + I({f} ** 2)" for f in curved)
    return rhs


def _numeric_model_matrix(frame: pd.DataFrame, factors: list[str], model: str) -> np.ndarray | None:
    """Return a numeric projection's model matrix as :func:`projection_formula` builds it; ``None`` for labels."""
    columns = frame[factors]
    if not all(pd.api.types.is_numeric_dtype(columns[f]) for f in factors):
        return None
    values = columns.to_numpy(dtype=float)
    parts = [np.ones((len(values), 1)), values]
    if model != "main_effects":
        parts += [values[:, [a]] * values[:, [b]] for a, b in itertools.combinations(range(len(factors)), 2)]
    if model == "quadratic":
        parts += [values[:, [j]] ** 2 for j, f in enumerate(factors) if columns[f].round(9).nunique() >= 3]
    return np.hstack(parts)


def _estimates(frame: pd.DataFrame, factors: list[str], model: str) -> bool:
    """Say whether a projection estimates the model: its model matrix has full column rank."""
    matrix = _numeric_model_matrix(frame, factors, model)
    if matrix is None:
        info = evaluate_design(
            frame[factors], model=projection_formula(frame, factors, model), metric="information_matrix"
        )
        return bool(info["information_matrix"]["rank"] == info["information_matrix"]["n_parameters"])
    lengths = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(lengths > 0, lengths, 1.0)
    return bool(matrix.shape[0] >= matrix.shape[1] and np.linalg.matrix_rank(scaled) == matrix.shape[1])


def _subsets(names: list[str], size: int | dict[str, int], groups: dict[str, list[str]] | None) -> list[list[str]]:
    """Return every projection's factors: any ``size`` of them, or that many from each group."""
    if isinstance(size, int):
        pool = names if groups is None else [n for members in groups.values() for n in members]
        return [list(s) for s in itertools.combinations(pool, size)]
    if groups is None:
        raise ValueError("A size per group needs the groups: groups={'name': [factors, ...], ...}.")
    unknown = set(size) - set(groups)
    if unknown:
        raise ValueError(f"The sizes name groups {sorted(unknown)} that are not in groups.")
    per_group = [list(itertools.combinations(groups[g], n)) for g, n in size.items()]
    return [[f for part in combo for f in part] for combo in itertools.product(*per_group)]


def evaluate_projections(
    design_matrix: pd.DataFrame | DesignResult,
    size: int | dict[str, int],
    model: str = "quadratic",
    metric: str | list[str] | None = None,
    groups: dict[str, list[str]] | None = None,
    **options: Any,  # noqa: ANN401 - passed on to evaluate_design
) -> dict[str, Any]:
    """Evaluate a design on every projection onto ``size`` of its factors.

    Each projection keeps every run and only the chosen factors' columns, and is evaluated under
    the model in its own factors (see :func:`projection_formula`). A projection estimates the
    model when its model matrix has full column rank. The share of projections that do is the
    projection estimation capacity; each metric is then averaged over those projections.

    Parameters
    ----------
    design_matrix : pandas.DataFrame or DesignResult
        The design, in coded units.
    size : int or dict[str, int]
        How many factors each projection holds; or, with ``groups``, how many from each group
        (``{"quantitative": 3, "categorical": 1}``).
    model : {"main_effects", "interactions", "quadratic"}
        The model in the projection's factors. Default ``"quadratic"``, the full second-order model.
    metric : str, list[str] or None
        :func:`~process_improve.experiments.evaluate_design` metrics to work out on every
        projection that estimates the model, such as ``"d_efficiency"`` or ``"g_efficiency"``.
        ``None`` (default) only counts the projections that estimate the model.
    groups : dict[str, list[str]] or None
        Named groups of factors, for a ``size`` per group. With an integer ``size`` the
        projections are drawn from the grouped factors only.
    **options
        Passed on to :func:`~process_improve.experiments.evaluate_design` (``integration``,
        ``max_search``, ``n_samples``, ``random_state``, ...).

    Returns
    -------
    dict[str, Any]
        ``n_projections``, ``n_estimable`` and ``fraction_estimable``; ``projections``, one dict
        per projection with its ``factors``, whether it is ``estimable``, and its metrics'
        results under ``metrics``; and ``averages``, each number a metric returns (such as
        ``"d_efficiency"`` and ``"max_prediction_variance"`` for ``"g_efficiency"``) averaged
        over the projections that estimate the model.

    Raises
    ------
    ValueError
        For an unknown model, a group size without groups, or more than ``MOST_PROJECTIONS`` projections.
    """
    frame = _factor_frame(design_matrix)
    subsets = _subsets(list(frame.columns), size, groups)
    if len(subsets) > MOST_PROJECTIONS:
        raise ValueError(f"{len(subsets)} projections is more than {MOST_PROJECTIONS}; ask for fewer factors.")
    metrics = [] if metric is None else [metric] if isinstance(metric, str) else list(metric)
    projections: list[dict[str, Any]] = []
    for factors in subsets:
        estimable = _estimates(frame, factors, model)
        found: dict[str, Any] = {}
        if estimable and metrics:
            found = evaluate_design(
                frame[factors], model=projection_formula(frame, factors, model), metric=metrics, **options
            )
            found.pop("notes", None)
        projections.append({"factors": factors, "estimable": estimable, "metrics": found})
    usable = [p for p in projections if p["estimable"]]
    numbers: dict[str, list[float]] = {}
    for p in usable:
        for name, value in p["metrics"].items():
            entries = value.items() if isinstance(value, dict) else [(name, value)]
            for key, number in entries:
                if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
                    numbers.setdefault(key, []).append(float(number))
    return {
        "n_projections": len(projections),
        "n_estimable": len(usable),
        "fraction_estimable": len(usable) / len(projections) if projections else float("nan"),
        "projections": projections,
        "averages": {key: float(np.mean(values)) for key, values in numbers.items()},
    }
