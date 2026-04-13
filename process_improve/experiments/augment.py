# (c) Kevin Dunn, 2010-2026. MIT License.

"""Design augmentation: extend or modify an existing experimental design.

Provides :func:`augment_design`, which takes an existing design matrix and
augments it by adding runs (foldover, semifold, center points, axial points,
D-optimal runs), upgrading to a response surface design, adding blocks, or
replicating.

Example
-------
>>> import pandas as pd
>>> from process_improve.experiments.augment import augment_design
>>> design = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
>>> result = augment_design(design, augmentation_type="add_center_points", n_additional_runs=3)
>>> result["augmented_design"].shape
(7, 2)
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrix
from pyDOE3 import fullfact

from process_improve.experiments.evaluate import (
    _defining_relation_from_generators,
    _word_to_str,
    evaluate_design,
)

# ---------------------------------------------------------------------------
# Internal context shared across augmentation handlers
# ---------------------------------------------------------------------------


@dataclass
class _AugmentContext:
    """Input context for all augmentation handlers."""

    existing_design: pd.DataFrame
    factor_names: list[str]
    augmentation_type: str
    target_model: str | None
    n_additional_runs: int | None
    fold_on: str | None
    alpha: str | float | None
    generators: list[str] | None


# ---------------------------------------------------------------------------
# "What changed" explainer
# ---------------------------------------------------------------------------


def _safe_evaluate(design: pd.DataFrame, generators: list[str] | None, model: str | None = None) -> dict[str, Any]:
    """Evaluate design metrics, returning empty dict on failure."""
    try:
        metrics = ["d_efficiency", "degrees_of_freedom"]
        if generators:
            metrics.extend(["alias_structure", "resolution"])
        return evaluate_design(design, model=model, metric=metrics)
    except Exception:  # noqa: BLE001
        return {}


def _explain_changes(  # noqa: PLR0913
    before: pd.DataFrame,
    after: pd.DataFrame,
    factor_names: list[str],
    augmentation_type: str,
    generators_before: list[str] | None = None,
    generators_after: list[str] | None = None,
    extra_notes: list[str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Generate before/after comparison narrative.

    Returns
    -------
    tuple[str, dict, dict]
        (explanation_text, before_metrics, after_metrics)
    """
    before_metrics = _safe_evaluate(before[factor_names], generators_before)
    after_metrics = _safe_evaluate(after[factor_names], generators_after)

    lines: list[str] = []

    # Run count
    n_before = len(before)
    n_after = len(after)
    lines.append(f"Design grew from {n_before} to {n_after} runs (+{n_after - n_before} added).")

    # D-efficiency
    d_before = before_metrics.get("d_efficiency")
    d_after = after_metrics.get("d_efficiency")
    if d_before is not None and d_after is not None:
        lines.append(f"D-efficiency: {d_before:.1f}% -> {d_after:.1f}%.")

    # Resolution
    res_before = before_metrics.get("resolution")
    res_after = after_metrics.get("resolution")
    if res_before is not None and res_after is not None:
        if res_after > res_before:
            lines.append(f"Resolution improved from {res_before} to {res_after}.")
        elif res_after == res_before:
            lines.append(f"Resolution unchanged at {res_before}.")

    # Degrees of freedom
    dof_before = before_metrics.get("degrees_of_freedom", {})
    dof_after = after_metrics.get("degrees_of_freedom", {})
    if "residual" in dof_before and "residual" in dof_after:
        lines.append(
            f"Residual degrees of freedom: {dof_before['residual']} -> {dof_after['residual']}."
        )

    # Alias diff
    aliases_before = set(before_metrics.get("alias_structure", []))
    aliases_after = set(after_metrics.get("alias_structure", []))
    removed = aliases_before - aliases_after
    if removed:
        lines.append("De-aliased effects:")
        for chain in sorted(removed):
            effect = chain.split(" = ")[0].strip()
            lines.append(f"  {effect} is now independently estimable.")

    # Extra notes from the handler
    if extra_notes:
        lines.extend(extra_notes)

    explanation = " ".join(lines) if not removed and not extra_notes else "\n".join(lines)
    return explanation, before_metrics, after_metrics


# ---------------------------------------------------------------------------
# Augmentation handlers
# ---------------------------------------------------------------------------


def _augment_foldover(ctx: _AugmentContext) -> dict[str, Any]:
    """Full foldover: negate all factor signs and append."""
    df = ctx.existing_design[ctx.factor_names].copy()
    folded = -df
    augmented = pd.concat([df, folded], ignore_index=True)

    # Compute new defining relation after foldover
    notes: list[str] = []
    generators_after = None
    if ctx.generators:
        words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
        # After full foldover, odd-length words are eliminated (confounded with
        # the block indicator).  Even-length words survive.
        surviving = [w for w in words if len(w) % 2 == 0]
        if surviving:
            generators_after = [f"I={_word_to_str(w, ctx.factor_names)}" for w in surviving]
            notes.append(f"New defining relation: {', '.join(generators_after)}.")
        else:
            notes.append("All defining words eliminated — design is now full resolution.")

        eliminated = [w for w in words if len(w) % 2 != 0]
        if eliminated:
            eliminated_strs = [_word_to_str(w, ctx.factor_names) for w in eliminated]
            notes.append(f"Eliminated defining words: {', '.join(eliminated_strs)}.")

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, generators_after, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": folded.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "defining_relation": generators_after,
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_semifold(ctx: _AugmentContext) -> dict[str, Any]:
    """Semifold: negate one factor in a selected half of runs."""
    df = ctx.existing_design[ctx.factor_names].copy()

    # Determine which factor to fold on
    fold_factor = ctx.fold_on
    if fold_factor is not None and fold_factor not in ctx.factor_names:
        raise ValueError(
            f"fold_on={fold_factor!r} not in factor names: {ctx.factor_names}"
        )

    if fold_factor is None:
        fold_factor = _auto_select_fold_factor(ctx)

    fold_idx = ctx.factor_names.index(fold_factor)

    # Select runs where fold factor is -1, negate the fold factor
    mask = df[fold_factor] == -1
    fold_half = df[mask].copy()
    fold_half[fold_factor] = -fold_half[fold_factor]  # negate fold factor
    augmented = pd.concat([df, fold_half], ignore_index=True)

    # Compute new defining relation after semifold
    notes: list[str] = [f"Folded on factor: {fold_factor}."]
    generators_after = None
    if ctx.generators:
        words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
        # Semifold on factor F eliminates words that contain F
        surviving = [w for w in words if fold_idx not in w]
        eliminated = [w for w in words if fold_idx in w]
        if surviving:
            generators_after = [f"I={_word_to_str(w, ctx.factor_names)}" for w in surviving]
            notes.append(f"Surviving defining words: {', '.join(generators_after)}.")
        else:
            notes.append("All defining words eliminated — design is now full resolution.")
        if eliminated:
            eliminated_strs = [_word_to_str(w, ctx.factor_names) for w in eliminated]
            notes.append(f"De-aliased by removing words: {', '.join(eliminated_strs)}.")

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, generators_after, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": fold_half.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "fold_on": fold_factor,
        "defining_relation": generators_after,
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _auto_select_fold_factor(ctx: _AugmentContext) -> str:
    """Pick the factor whose semifold de-aliases the most short defining words.

    For each candidate factor, count how many minimum-length words in the
    defining relation contain that factor.  The factor that eliminates the
    most short words is the best choice.
    """
    if not ctx.generators:
        # No generators — just pick the first factor
        return ctx.factor_names[0]

    words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
    if not words:
        return ctx.factor_names[0]

    min_len = min(len(w) for w in words)
    short_words = [w for w in words if len(w) == min_len]

    best_factor = ctx.factor_names[0]
    best_count = 0
    for i, name in enumerate(ctx.factor_names):
        count = sum(1 for w in short_words if i in w)
        if count > best_count:
            best_count = count
            best_factor = name

    return best_factor


def _augment_add_center_points(ctx: _AugmentContext) -> dict[str, Any]:
    """Append center point rows (all zeros in coded units)."""
    df = ctx.existing_design[ctx.factor_names].copy()
    n_center = ctx.n_additional_runs if ctx.n_additional_runs is not None else 3

    center_rows = pd.DataFrame(
        np.zeros((n_center, len(ctx.factor_names))),
        columns=ctx.factor_names,
    )
    augmented = pd.concat([df, center_rows], ignore_index=True)

    notes = [
        f"Added {n_center} center point(s) at the midpoint of all factors.",
        "Center points enable testing for curvature (quadratic effects).",
    ]
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, ctx.generators, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": center_rows.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_replicate(ctx: _AugmentContext) -> dict[str, Any]:
    """Append one or more complete copies of the existing design."""
    df = ctx.existing_design[ctx.factor_names].copy()
    n_copies = ctx.n_additional_runs if ctx.n_additional_runs is not None else 1

    replicated = pd.concat([df] * n_copies, ignore_index=True)
    augmented = pd.concat([df, replicated], ignore_index=True)

    notes = [
        f"Added {n_copies} complete replicate(s) of the original {len(df)}-run design.",
        "Replication provides pure error degrees of freedom for lack-of-fit testing.",
    ]
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, ctx.generators, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": replicated.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_add_axial_points(ctx: _AugmentContext) -> dict[str, Any]:
    """Add 2k axial (star) points to create a CCD structure."""
    df = ctx.existing_design[ctx.factor_names].copy()
    k = len(ctx.factor_names)

    # Determine alpha value
    alpha_value = _compute_alpha(df, ctx.factor_names, ctx.alpha)

    # Generate 2k axial points
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = alpha_value
        axial[2 * i + 1, i] = -alpha_value
    axial_df = pd.DataFrame(axial, columns=ctx.factor_names)

    augmented = pd.concat([df, axial_df], ignore_index=True)

    notes = [
        f"Added {2 * k} axial (star) points with alpha = {alpha_value:.4f}.",
        "The design now supports estimation of quadratic (second-order) effects.",
        "Consider adding center points if not already present.",
    ]
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, ctx.generators, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": axial_df.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "alpha": float(alpha_value),
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _compute_alpha(
    design: pd.DataFrame,
    factor_names: list[str],
    alpha: str | float | None,
) -> float:
    """Compute the axial distance alpha.

    Parameters
    ----------
    design : DataFrame
        Existing design matrix.
    factor_names : list[str]
        Factor column names.
    alpha : str, float, or None
        ``"rotatable"``, ``"face_centered"``, ``"orthogonal"``, or numeric.
    """
    if isinstance(alpha, (int, float)):
        return float(alpha)

    # Count factorial points (non-center rows)
    center_mask = (design[factor_names].abs() < 1e-10).all(axis=1)
    n_factorial = int((~center_mask).sum())
    k = len(factor_names)

    if alpha is None or alpha == "rotatable":
        # Rotatable: alpha = n_factorial^(1/4)
        return float(n_factorial ** 0.25)
    elif alpha == "face_centered":
        return 1.0
    elif alpha == "orthogonal":
        # Orthogonal block alpha for CCD
        n_axial = 2 * k
        n_total = n_factorial + n_axial
        return float(np.sqrt(k * (np.sqrt(n_total) - np.sqrt(n_factorial)) / 2))
    else:
        raise ValueError(f"Unknown alpha type: {alpha!r}. Use 'rotatable', 'face_centered', 'orthogonal', or numeric.")


def _build_model_rhs(factor_names: list[str], model: str) -> str:
    """Build a patsy right-hand-side formula string for the given model type."""
    joined = " + ".join(factor_names)
    if model == "main_effects":
        return joined
    if model == "interactions":
        return f"({joined}) ** 2"
    if model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in factor_names)
        return f"({joined}) ** 2 + {squared}"
    return model


def _greedy_d_optimal_select(
    current: pd.DataFrame,
    candidates: pd.DataFrame,
    n_to_add: int,
    factor_names: list[str],
    model: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Greedily select *n_to_add* D-optimal points from *candidates*."""
    rhs = _build_model_rhs(factor_names, model)
    new_rows: list[pd.DataFrame] = []

    for _ in range(min(n_to_add, len(candidates))):
        best_idx = -1
        best_det = -np.inf

        for idx in candidates.index:
            trial = pd.concat([current, candidates.iloc[[idx]]], ignore_index=True)
            X_trial = np.asarray(dmatrix(rhs, trial, return_type="dataframe"), dtype=float)
            sign, logdet = np.linalg.slogdet(X_trial.T @ X_trial)
            det_val = logdet if sign > 0 else -np.inf
            if det_val > best_det:
                best_det = det_val
                best_idx = idx

        if best_idx < 0:
            break

        new_row = candidates.loc[[best_idx]]
        current = pd.concat([current, new_row], ignore_index=True)
        new_rows.append(new_row)
        candidates = candidates.drop(best_idx).reset_index(drop=True)

    return current, new_rows


def _augment_add_runs_optimal(ctx: _AugmentContext) -> dict[str, Any]:
    """Add D-optimal runs to the existing design."""
    if ctx.n_additional_runs is None:
        raise ValueError("n_additional_runs is required for add_runs_optimal.")

    df = ctx.existing_design[ctx.factor_names].copy()
    k = len(ctx.factor_names)
    model = ctx.target_model or "interactions"

    # Generate candidate set: 3-level full factorial (-1, 0, +1)
    candidates_raw = fullfact([3] * k) - 1.0
    candidates = pd.DataFrame(candidates_raw, columns=ctx.factor_names)

    # Remove candidates that are already in the design (within tolerance)
    existing_tuples = set(map(tuple, np.round(df.values, 8)))
    mask = ~candidates.apply(lambda row: tuple(np.round(row.values, 8)) in existing_tuples, axis=1)
    candidates = candidates[mask].reset_index(drop=True)

    if len(candidates) == 0:
        raise ValueError("No candidate points available after filtering existing design points.")

    augmented, new_rows = _greedy_d_optimal_select(
        df, candidates, ctx.n_additional_runs, ctx.factor_names, model,
    )
    new_runs_df = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame(columns=ctx.factor_names)
    notes = [
        f"Added {len(new_rows)} D-optimal run(s) to maximize information for the {model} model.",
        "Existing runs were preserved; only new runs were optimized.",
    ]
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, ctx.generators, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": new_runs_df.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_upgrade_to_rsm(ctx: _AugmentContext) -> dict[str, Any]:
    """Upgrade a screening/factorial design to an RSM (CCD) design."""
    df = ctx.existing_design[ctx.factor_names].copy()
    k = len(ctx.factor_names)

    # Detect existing center points
    center_mask = (df.abs() < 1e-10).all(axis=1)
    n_existing_centers = int(center_mask.sum())

    # Add axial points
    alpha_val = ctx.alpha if ctx.alpha is not None else "rotatable"
    alpha_numeric = _compute_alpha(df, ctx.factor_names, alpha_val)

    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = alpha_numeric
        axial[2 * i + 1, i] = -alpha_numeric
    axial_df = pd.DataFrame(axial, columns=ctx.factor_names)

    # Add center points if needed (target 3-5 total)
    n_target_centers = max(3, 5 - n_existing_centers)
    n_new_centers = max(0, n_target_centers - n_existing_centers)
    center_df = pd.DataFrame(
        np.zeros((n_new_centers, k)), columns=ctx.factor_names,
    )

    new_runs = pd.concat([axial_df, center_df], ignore_index=True)
    augmented = pd.concat([df, new_runs], ignore_index=True)

    notes = [
        f"Upgraded to Central Composite Design (CCD) with alpha = {alpha_numeric:.4f}.",
        f"Added {2 * k} axial points and {n_new_centers} center point(s).",
        "The design now supports estimation of a full quadratic (second-order) model.",
    ]
    if n_existing_centers > 0:
        notes.append(f"Existing {n_existing_centers} center point(s) were preserved.")

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names,
        ctx.augmentation_type, ctx.generators, ctx.generators, notes,
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": new_runs.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "alpha": float(alpha_numeric),
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_add_blocks(ctx: _AugmentContext) -> dict[str, Any]:
    """Retroactively assign blocks by confounding with high-order interactions."""
    df = ctx.existing_design[ctx.factor_names].copy()
    n_blocks = ctx.n_additional_runs if ctx.n_additional_runs is not None else 2
    k = len(ctx.factor_names)

    if n_blocks < 2:
        raise ValueError("Number of blocks must be at least 2.")

    # Determine how many confounding columns needed: 2^b blocks requires b columns
    b = int(np.ceil(np.log2(n_blocks)))
    n_blocks_actual = 2 ** b  # round up to power of 2

    # Choose the highest-order interactions for confounding
    # Generate all interactions from order k down to order 2
    confounding_columns: list[tuple[str, np.ndarray]] = []
    for order in range(k, 1, -1):
        for combo in itertools.combinations(range(k), order):
            if len(confounding_columns) >= b:
                break
            col_product = np.ones(len(df))
            for idx in combo:
                col_product *= df.iloc[:, idx].values
            word = "".join(ctx.factor_names[i] for i in combo)
            confounding_columns.append((word, col_product))
        if len(confounding_columns) >= b:
            break

    if len(confounding_columns) < b:
        raise ValueError(
            f"Cannot create {n_blocks_actual} blocks with {k} factors. "
            f"Need {b} confounding columns but only found {len(confounding_columns)}."
        )

    # Assign blocks using signs of confounding columns
    block_assignment = np.zeros(len(df), dtype=int)
    for i, (_word, col) in enumerate(confounding_columns[:b]):
        block_assignment += (col > 0).astype(int) * (2 ** i)
    block_assignment += 1  # 1-based

    augmented = df.copy()
    augmented["Block"] = block_assignment.tolist()

    confounded_words = [word for word, _ in confounding_columns[:b]]
    notes = [
        f"Assigned {n_blocks_actual} blocks by confounding with: {', '.join(confounded_words)}.",
        f"Block effect is aliased with the {', '.join(confounded_words)} interaction(s).",
        "Block effects should be orthogonal to main effects and low-order interactions.",
    ]
    explanation = "\n".join(notes)

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": [],
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "n_blocks": n_blocks_actual,
        "confounded_with": confounded_words,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Augmentation dispatch registry
# ---------------------------------------------------------------------------

_AUGMENT_REGISTRY: dict[str, Callable[[_AugmentContext], dict[str, Any]]] = {
    "foldover": _augment_foldover,
    "semifold": _augment_semifold,
    "add_center_points": _augment_add_center_points,
    "add_axial_points": _augment_add_axial_points,
    "add_runs_optimal": _augment_add_runs_optimal,
    "upgrade_to_rsm": _augment_upgrade_to_rsm,
    "add_blocks": _augment_add_blocks,
    "replicate": _augment_replicate,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_design(  # noqa: PLR0913
    existing_design: pd.DataFrame,
    augmentation_type: str,
    target_model: str | None = None,
    n_additional_runs: int | None = None,
    fold_on: str | None = None,
    alpha: str | float | None = None,
    generators: list[str] | None = None,
) -> dict[str, Any]:
    """Extend or modify an existing experimental design.

    Parameters
    ----------
    existing_design : DataFrame
        The current design matrix with factor columns in coded units (-1/+1).
    augmentation_type : str
        One of ``"foldover"``, ``"semifold"``, ``"add_center_points"``,
        ``"add_axial_points"``, ``"add_runs_optimal"``, ``"upgrade_to_rsm"``,
        ``"add_blocks"``, ``"replicate"``.
    target_model : str or None
        Desired model after augmentation: ``"main_effects"``,
        ``"interactions"``, ``"quadratic"``.  Used by ``"add_runs_optimal"``
        and ``"upgrade_to_rsm"``.
    n_additional_runs : int or None
        Budget for additional runs.  Interpretation depends on the
        augmentation type (number of center points, number of D-optimal
        runs, number of replicates, or number of blocks).
    fold_on : str or None
        For ``"semifold"`` only: which factor to fold on.  If ``None``,
        the best factor is auto-selected.
    alpha : str, float, or None
        Axial distance for ``"add_axial_points"`` and ``"upgrade_to_rsm"``.
        String values: ``"rotatable"``, ``"face_centered"``,
        ``"orthogonal"``.  Or a numeric value.
    generators : list[str] or None
        Generator strings from the original design (e.g. ``["D=ABC"]``).
        Needed for meaningful alias analysis in foldover/semifold.

    Returns
    -------
    dict[str, Any]
        Keys include ``"augmented_design"`` (list of dicts),
        ``"new_runs"`` (list of dicts), ``"n_runs_before"``,
        ``"n_runs_after"``, ``"explanation"`` (narrative),
        ``"before_metrics"``, ``"after_metrics"``, and
        augmentation-specific keys.

    Raises
    ------
    ValueError
        If *augmentation_type* is unknown, or if required parameters
        are missing for the requested augmentation.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.experiments.augment import augment_design
    >>> design = pd.DataFrame({
    ...     "A": [-1, 1, -1, 1, -1, 1, -1, 1],
    ...     "B": [-1, -1, 1, 1, -1, -1, 1, 1],
    ...     "C": [-1, -1, -1, -1, 1, 1, 1, 1],
    ... })
    >>> result = augment_design(design, "add_center_points", n_additional_runs=3)
    >>> result["n_runs_after"]
    11
    """
    if augmentation_type not in _AUGMENT_REGISTRY:
        available = sorted(_AUGMENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown augmentation_type={augmentation_type!r}. "
            f"Choose from: {', '.join(available)}."
        )

    factor_names = [c for c in existing_design.columns if c not in ("RunOrder", "Block")]

    ctx = _AugmentContext(
        existing_design=existing_design,
        factor_names=factor_names,
        augmentation_type=augmentation_type,
        target_model=target_model,
        n_additional_runs=n_additional_runs,
        fold_on=fold_on,
        alpha=alpha,
        generators=generators,
    )

    handler = _AUGMENT_REGISTRY[augmentation_type]
    return handler(ctx)
