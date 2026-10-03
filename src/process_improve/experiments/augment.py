# (c) Kevin Dunn, 2010-2026. MIT License.

"""Design augmentation: extend or modify an existing experimental design.

Provides :func:`augment_design`, which takes an existing design matrix and
augments it by adding runs (foldover, semifold, center points, axial points,
D-optimal runs), upgrading to a response surface design, adding n_blocks, or
replicating.

Example
-------
>>> import pandas as pd
>>> from process_improve.experiments.augment import augment_design
>>> design = pd.DataFrame({"A": [-1, 1, -1, 1], "B": [-1, -1, 1, 1]})
>>> result = augment_design(design, augmentation_type="add_center_points", n_additional_runs=3)
>>> pd.DataFrame(result["augmented_design"]).shape
(7, 2)
"""

from __future__ import annotations

import itertools
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrix

from process_improve._random import check_random_state
from process_improve.experiments._blocking import confounding_blocks, exchange_blocks, is_regular_two_level
from process_improve.experiments.designs_response_surface import orthogonal_alpha
from process_improve.experiments.evaluate import (
    _defining_relation_from_generators,
    _roman,
    _word_to_str,
    evaluate_design,
)
from process_improve.experiments.models import validate_formula_is_safe, validate_identifier_is_safe

logger = logging.getLogger(__name__)

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
    random_state: int | np.random.Generator | None = 42


# ---------------------------------------------------------------------------
# "What changed" explainer
# ---------------------------------------------------------------------------


def _safe_evaluate(design: pd.DataFrame, model: str | None = None) -> dict[str, Any]:
    """Evaluate D-efficiency and degrees of freedom, returning an empty dict (with a warning) on failure."""
    try:
        return evaluate_design(design, model=model, metric=["d_efficiency", "degrees_of_freedom"])
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        # Evaluation may not apply to every design; return no metrics, say so, and let
        # unexpected error types propagate.
        warnings.warn(f"Design metrics were not computed: {exc}", UserWarning, stacklevel=5)
        return {}


#: Below this, two effect columns count as uncorrelated; above 1 minus it, as fully aliased.
_ALIAS_TOL = 1e-9


def _effect_columns(design: pd.DataFrame, factor_names: list[str]) -> dict[str, np.ndarray]:
    """Centred, unit-length columns of every main effect and two-factor interaction.

    A column that is constant (an effect aliased with the intercept) is left out,
    since it has no correlation to report.
    """
    x = design[factor_names].to_numpy(dtype=float)
    raw = {name: x[:, i] for i, name in enumerate(factor_names)}
    for (i, a), (j, b) in itertools.combinations(enumerate(factor_names), 2):
        raw[f"{a}:{b}"] = x[:, i] * x[:, j]
    columns = {}
    for name, column in raw.items():
        centred = column - column.mean()
        norm = float(np.linalg.norm(centred))
        if norm > _ALIAS_TOL:
            columns[name] = centred / norm
    return columns


def _alias_changes(before: pd.DataFrame, after: pd.DataFrame, factor_names: list[str]) -> list[str]:
    """Describe how the aliasing among main effects and two-factor interactions changed.

    Pairs of effects whose columns are identical (up to sign) in the existing design
    are aliased; each pair is then looked up in the augmented design. A pair is only
    reported as separated when the two columns are uncorrelated there: a pair that
    is merely correlated (a semifold leaves ``|r| = 1/3``) still cannot be estimated
    independently, and a pair the new runs do not touch stays fully aliased.
    Interactions of three or more factors are not considered.
    """
    cols_before = _effect_columns(before, factor_names)
    cols_after = _effect_columns(after, factor_names)
    names = [n for n in cols_before if n in cols_after]
    pairs = [
        (a, b)
        for a, b in itertools.combinations(names, 2)
        if abs(float(cols_before[a] @ cols_before[b])) > 1 - _ALIAS_TOL
    ]
    if not pairs:
        return []

    r_after = {(a, b): abs(float(cols_after[a] @ cols_after[b])) for a, b in pairs}
    still = [f"{a} = {b}" for (a, b), r in r_after.items() if r > 1 - _ALIAS_TOL]
    partial = [f"{a} with {b} (|r| = {r:.2f})" for (a, b), r in r_after.items() if _ALIAS_TOL < r <= 1 - _ALIAS_TOL]
    involved = {name for pair in pairs for name in pair}
    cleared = [n for n in names if n in involved and all(r <= _ALIAS_TOL for pair, r in r_after.items() if n in pair)]

    lines = ["Aliasing among main effects and two-factor interactions (higher-order interactions not considered):"]
    if cleared:
        lines.append(f"  Now uncorrelated with every effect they were aliased with: {', '.join(cleared)}.")
    if partial:
        lines.append(
            "  Partially de-aliased, still correlated and so not estimable independently of each other: "
            f"{'; '.join(partial)}."
        )
    if still:
        lines.append(f"  Still fully aliased: {'; '.join(still)}.")
    return lines


def _resolution_lines(resolution_before: int | None, resolution_after: int | str | None) -> list[str]:
    """Report the resolution change; *resolution_after* is an int, ``"full"``, or None when not regular."""
    if resolution_before is None:
        return []
    before = _roman(resolution_before)
    if resolution_after == "full":
        return [f"Resolution {before} -> no defining words remain: the augmented design is a full factorial."]
    if not isinstance(resolution_after, int):
        return []
    if resolution_after == resolution_before:
        return [f"Resolution unchanged at {before}."]
    return [f"Resolution changed from {before} to {_roman(resolution_after)}."]


def _explain_changes(  # noqa: PLR0913
    before: pd.DataFrame,
    after: pd.DataFrame,
    factor_names: list[str],
    generators: list[str] | None = None,
    extra_notes: list[str] | None = None,
    resolution_after: int | str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Generate before/after comparison narrative.

    Parameters
    ----------
    before, after : DataFrame
        The existing and the augmented design.
    factor_names : list[str]
        Factor columns.
    generators : list[str] or None
        The existing design's generators, from which its resolution is read.
    extra_notes : list[str] or None
        Handler-specific lines.
    resolution_after : int, "full" or None
        The augmented design's resolution, ``"full"`` for a full factorial, or
        None when it is not a regular fraction (or not known).

    Returns
    -------
    tuple[str, dict, dict]
        (explanation_text, before_metrics, after_metrics)
    """
    before_metrics = _safe_evaluate(before[factor_names])
    after_metrics = _safe_evaluate(after[factor_names])
    resolution_before = _resolution(generators, factor_names)
    if resolution_before is not None:
        before_metrics["resolution"] = resolution_before
        after_metrics["resolution"] = resolution_after if isinstance(resolution_after, int) else None

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

    lines.extend(_resolution_lines(resolution_before, resolution_after))

    # Degrees of freedom
    dof_before = before_metrics.get("degrees_of_freedom", {})
    dof_after = after_metrics.get("degrees_of_freedom", {})
    if "residual" in dof_before and "residual" in dof_after:
        lines.append(f"Residual degrees of freedom: {dof_before['residual']} -> {dof_after['residual']}.")

    alias_lines = _alias_changes(before, after, factor_names)
    lines.extend(alias_lines)

    # Extra notes from the handler
    if extra_notes:
        lines.extend(extra_notes)

    explanation = " ".join(lines) if not alias_lines and not extra_notes else "\n".join(lines)
    return explanation, before_metrics, after_metrics


def _resolution(generators: list[str] | None, factor_names: list[str]) -> int | None:
    """Resolution of the regular fraction *generators* define: its shortest defining word."""
    if not generators:
        return None
    words = _defining_relation_from_generators(generators, factor_names)
    return min((len(w) for w in words), default=None)


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
    resolution_after: int | str | None = None
    if ctx.generators:
        words = _defining_relation_from_generators(ctx.generators, ctx.factor_names)
        # Negating every factor flips the sign of every odd-length word in the second
        # half, so the odd words no longer hold in the combined design (each is now
        # confounded with the contrast between the two halves). Even-length words hold
        # in both halves and survive: the 2FI chains they create are still aliased.
        surviving = [w for w in words if len(w) % 2 == 0]
        if surviving:
            generators_after = [f"I={_word_to_str(w, ctx.factor_names)}" for w in surviving]
            notes.append(f"New defining relation: {', '.join(generators_after)}.")
            resolution_after = min(len(w) for w in surviving)
        else:
            resolution_after = "full"

        eliminated = [w for w in words if len(w) % 2 != 0]
        if eliminated:
            eliminated_strs = [_word_to_str(w, ctx.factor_names) for w in eliminated]
            notes.append(f"Eliminated defining words: {', '.join(eliminated_strs)}.")

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names, ctx.generators, notes, resolution_after
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
        raise ValueError(f"fold_on={fold_factor!r} not in factor names: {ctx.factor_names}")

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
        # The added half repeats half of the runs with the fold factor's sign switched.
        # A word without it holds in every run and survives. A word containing it holds
        # in the original runs and is reversed in the added ones, so it no longer holds
        # anywhere: effects aliased through it become partially aliased (Mee and Peralta,
        # 2000), correlated rather than independent, and the combined design is not a
        # regular fraction.
        surviving = [w for w in words if fold_idx not in w]
        broken = [w for w in words if fold_idx in w]
        if surviving:
            generators_after = [f"I={_word_to_str(w, ctx.factor_names)}" for w in surviving]
            notes.append(f"Defining words that still hold in every run: {', '.join(generators_after)}.")
        if broken:
            broken_strs = [_word_to_str(w, ctx.factor_names) for w in broken]
            notes.append(
                f"Words containing {fold_factor} ({', '.join(broken_strs)}) now hold in only part of the runs: "
                "effects aliased through them are partially de-aliased, not independent. The combined design is "
                "not a regular fraction, so it has no resolution in the usual sense."
            )

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names, ctx.generators, notes
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
        # No generators - just pick the first factor
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
        ctx.existing_design,
        augmented,
        ctx.factor_names,
        ctx.generators,
        notes,
        _resolution(ctx.generators, ctx.factor_names),
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
    """Append one or more complete copies of the existing design.

    The number of copies is ``ctx.n_additional_runs`` (default 1 when
    ``ctx.n_additional_runs`` is ``None``); :func:`augment_design` refuses 0.
    """
    df = ctx.existing_design[ctx.factor_names].copy()
    n_copies = ctx.n_additional_runs if ctx.n_additional_runs is not None else 1

    replicated = pd.concat([df] * n_copies, ignore_index=True)
    augmented = pd.concat([df, replicated], ignore_index=True)

    notes = [
        f"Added {n_copies} complete replicate(s) of the original {len(df)}-run design.",
        "Replication provides pure error degrees of freedom for lack-of-fit testing.",
    ]
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design,
        augmented,
        ctx.factor_names,
        ctx.generators,
        notes,
        _resolution(ctx.generators, ctx.factor_names),
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
        ctx.existing_design, augmented, ctx.factor_names, ctx.generators, notes
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
    n_new_centers: int = 0,
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
    n_new_centers : int
        Centre runs added alongside the axial runs, which the orthogonal
        distance has to count.
    """
    if isinstance(alpha, (int, float)):
        return float(alpha)

    # Count factorial points (non-center rows)
    center_mask = (design[factor_names].abs() < 1e-10).all(axis=1)
    n_factorial = int((~center_mask).sum())
    k = len(factor_names)

    if alpha is None or alpha == "rotatable":
        # Rotatable: alpha = n_factorial^(1/4)
        return float(n_factorial**0.25)
    elif alpha == "face_centered":
        return 1.0
    elif alpha == "orthogonal":
        # Quadratic columns mutually orthogonal, counting every existing run and the 2k new axial runs.
        return orthogonal_alpha(n_factorial, len(design) + 2 * k + n_new_centers)
    else:
        raise ValueError(f"Unknown alpha type: {alpha!r}. Use 'rotatable', 'face_centered', 'orthogonal', or numeric.")


def _build_model_rhs(factor_names: list[str], model: str) -> str:
    """Build a patsy right-hand-side formula string for the given model type.

    The returned RHS is validated before it can reach ``patsy.dmatrix`` so a
    custom ``model`` string cannot smuggle in arbitrary Python (SEC-14).
    """
    for name in factor_names:
        validate_identifier_is_safe(name)

    joined = " + ".join(factor_names)
    if model == "main_effects":
        rhs = joined
    elif model == "interactions":
        rhs = f"({joined}) ** 2"
    elif model == "quadratic":
        squared = " + ".join(f"I({f} ** 2)" for f in factor_names)
        rhs = f"({joined}) ** 2 + {squared}"
    else:
        rhs = model

    validate_formula_is_safe(rhs, factor_names, allow_transforms=True)
    return rhs


def _model_rows(rhs: str, factor_names: list[str], existing: pd.DataFrame, candidates: np.ndarray) -> tuple:
    """Model-matrix rows for the existing runs and the candidates, from one patsy build.

    Building both from one frame keeps any stateful transform in a custom formula
    (``center(A)``, say) on a single scale.
    """
    stacked = pd.concat([existing, pd.DataFrame(candidates, columns=factor_names)], ignore_index=True)
    rows = np.asarray(dmatrix(rhs, stacked, return_type="dataframe"), dtype=float)
    return rows[: len(existing)], rows[len(existing) :]


def _augment_add_runs_optimal(ctx: _AugmentContext) -> dict[str, Any]:
    """Add D-optimal runs to the existing design, which stays fixed.

    The runs come from a Fedorov exchange over a candidate grid in coded units, with
    the existing runs as fixed rows. Unlike a one-run-at-a-time greedy search, the
    exchange can start from a design that cannot yet estimate the target model, as
    every screening design aimed at a quadratic model is. When the requested runs
    cannot supply the rank the existing runs lack, more are added, and the
    explanation says so. Candidates may repeat, which replicates a run where the
    criterion wants it.
    """
    from process_improve.experiments.designs_constrained import (  # noqa: PLC0415
        Criterion,
        _budget_for_fixed_runs,
        _Region,
        build_candidates,
        fedorov_exchange,
    )
    from process_improve.experiments.factor import Factor  # noqa: PLC0415

    if ctx.n_additional_runs is None:
        raise ValueError("n_additional_runs is required for add_runs_optimal.")

    df = ctx.existing_design[ctx.factor_names].astype(float)
    model = ctx.target_model or "interactions"
    rhs = _build_model_rhs(ctx.factor_names, model)  # validated before patsy sees it (SEC-14)

    region = _Region([Factor(name=n, low=-1, high=1) for n in ctx.factor_names], [], [])
    grid_model = model if model in ("main_effects", "interactions") else "quadratic"
    coded, _cats, _counts = build_candidates(region, None, grid_model)
    f_fixed, f_cand = _model_rows(rhs, ctx.factor_names, df, coded)
    n_parameters = f_cand.shape[1]
    if np.linalg.matrix_rank(np.vstack([f_fixed, f_cand])) < n_parameters:
        raise ValueError(f"The candidate grid cannot support the {model!r} model; use a simpler target_model.")

    total = _budget_for_fixed_runs(f_fixed, n_parameters, len(df) + ctx.n_additional_runs)
    n_new = total - len(df)
    rows, _value = fedorov_exchange(f_cand, n_new, f_fixed, check_random_state(ctx.random_state), Criterion.d())
    new_runs_df = pd.DataFrame(coded[rows], columns=ctx.factor_names)
    augmented = pd.concat([df, new_runs_df], ignore_index=True)

    notes = [
        f"Added {n_new} D-optimal run(s) to maximize information for the {model} model.",
        "Existing runs were preserved; only new runs were optimized.",
    ]
    if n_new > ctx.n_additional_runs:
        notes.append(
            f"The {len(df)} existing run(s) span only {int(np.linalg.matrix_rank(f_fixed))} of the "
            f"{n_parameters} coefficients of the {model} model, so {n_new} runs were needed rather than "
            f"the {ctx.n_additional_runs} requested."
        )
    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names, ctx.generators, notes
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


#: upgrade_to_rsm tops the centre runs up to this many, the usual three to five for a CCD.
_RSM_CENTER_RUNS = 5


def _model_support(design: pd.DataFrame, factor_names: list[str], model: str) -> tuple[int, int, list[str]]:
    """Return the model's coefficient count, the rank the design gives it, and the terms that cannot all be estimated.

    The terms are those with weight in the null space of the model matrix: some
    combination of their columns is zero on every run, so they are aliased.
    """
    rhs = _build_model_rhs(factor_names, model)
    frame = dmatrix(rhs, design[factor_names].astype(float), return_type="dataframe")
    x = frame.to_numpy(dtype=float)
    _u, singular, vt = np.linalg.svd(x, full_matrices=True)
    rank = int((singular > singular.max() * max(x.shape) * np.finfo(float).eps).sum())
    null_space = vt[rank:]
    aliased = [
        str(term)
        for term, w in zip(frame.columns, np.abs(null_space).max(axis=0, initial=0.0), strict=True)
        if w > 1e-8
    ]
    return x.shape[1], rank, aliased


def _augment_upgrade_to_rsm(ctx: _AugmentContext) -> dict[str, Any]:
    """Upgrade a screening/factorial design to an RSM (CCD) design.

    Adds 2k axial runs and tops the centre runs up to five, then checks that the
    target model (default quadratic) can be estimated. It cannot when the cube is
    a resolution III or IV fraction, whose aliased two-factor interactions are zero
    on every axial and centre run; that is reported, with a warning, rather than
    claimed away.
    """
    df = ctx.existing_design[ctx.factor_names].copy()
    k = len(ctx.factor_names)
    model = ctx.target_model or "quadratic"

    # Detect existing center points, and top them up to a fixed total
    center_mask = (df.abs() < 1e-10).all(axis=1)
    n_existing_centers = int(center_mask.sum())
    n_new_centers = max(0, _RSM_CENTER_RUNS - n_existing_centers)

    # Add axial points
    alpha_val = ctx.alpha if ctx.alpha is not None else "rotatable"
    alpha_numeric = _compute_alpha(df, ctx.factor_names, alpha_val, n_new_centers)

    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i] = alpha_numeric
        axial[2 * i + 1, i] = -alpha_numeric
    axial_df = pd.DataFrame(axial, columns=ctx.factor_names)
    center_df = pd.DataFrame(np.zeros((n_new_centers, k)), columns=ctx.factor_names)

    new_runs = pd.concat([axial_df, center_df], ignore_index=True)
    augmented = pd.concat([df, new_runs], ignore_index=True)

    notes = [
        f"Upgraded to Central Composite Design (CCD) with alpha = {alpha_numeric:.4f}.",
        f"Added {2 * k} axial points and {n_new_centers} center point(s).",
    ]
    if n_existing_centers > 0:
        notes.append(f"Existing {n_existing_centers} center point(s) were preserved.")
    n_coefficients, rank, aliased = _model_support(augmented, ctx.factor_names, model)
    if rank == n_coefficients:
        notes.append(f"The design now supports the {model} model: all {n_coefficients} coefficients are estimable.")
    else:
        message = (
            f"The {model} model has {n_coefficients} coefficients but the upgraded design supports only {rank}: "
            f"{', '.join(aliased)} cannot all be estimated. Axial and centre runs are zero on every interaction "
            "column, so interactions aliased in the cube stay aliased; a central composite design needs a "
            "resolution V cube. Add runs that separate those interactions first (for a half fraction, its "
            "other half), or use 'add_runs_optimal' with this target_model."
        )
        notes.append(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    explanation, before_m, after_m = _explain_changes(
        ctx.existing_design, augmented, ctx.factor_names, ctx.generators, notes
    )

    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": new_runs.to_dict(orient="records"),
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "alpha": float(alpha_numeric),
        "n_estimable": rank,
        "n_coefficients": n_coefficients,
        "explanation": explanation,
        "before_metrics": before_m,
        "after_metrics": after_m,
    }


def _augment_add_blocks(ctx: _AugmentContext) -> dict[str, Any]:
    """Assign existing runs to blocks: by confounding interaction words in a regular two-level design, else by exchange.

    See :mod:`process_improve.experiments._blocking`. In a two-level factorial the block
    contrasts are chosen so that no main effect is confounded with blocks (and as few
    two-factor interactions as possible); every block contrast, including the products
    of the generators, is reported.
    """
    df = ctx.existing_design[ctx.factor_names].copy()
    n_blocks = ctx.n_additional_runs if ctx.n_additional_runs is not None else 2
    if n_blocks < 2:
        raise ValueError("Number of blocks must be at least 2.")
    numeric = df.to_numpy(dtype=float)
    if is_regular_two_level(numeric):
        blocking = confounding_blocks(numeric, n_blocks, ctx.factor_names)
        notes = [
            f"Assigned {n_blocks} blocks with generators {', '.join(blocking.generators)}.",
            f"Confounded with blocks: {', '.join(blocking.confounded_with)} (no main effect).",
        ]
    else:
        blocking = exchange_blocks(numeric, n_blocks, check_random_state(ctx.random_state))
        notes = [
            (
                f"Assigned {n_blocks} blocks by exchange, keeping the {blocking.model} model estimable with a "
                "separate mean in every block."
            ),
        ]
    augmented = df.copy()
    augmented["Block"] = blocking.labels.tolist()
    return {
        "augmented_design": augmented.to_dict(orient="records"),
        "new_runs": [],
        "n_runs_before": len(ctx.existing_design),
        "n_runs_after": len(augmented),
        "n_blocks": n_blocks,
        "confounded_with": blocking.confounded_with,
        "explanation": "\n".join(notes),
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


def _resolve_factor_names(design: pd.DataFrame, factor_names: list[str] | None) -> list[str]:
    """Return the factor columns of *design*, refusing columns that are not coded factors.

    Raises
    ------
    ValueError
        If *factor_names* names a missing column or repeats one, or, when it is
        None, if a candidate column is non-numeric, or lies wholly on one side
        of 0 while reaching beyond [-1, 1]. A coded factor is either spread
        across 0 or held at a level within [-1, 1]; a measured response such as
        a yield of 10 to 16 is neither.
    """
    if factor_names is not None:
        missing = [n for n in factor_names if n not in design.columns]
        if missing or len(set(factor_names)) != len(factor_names) or not factor_names:
            msg = f"factor_names must name distinct columns of the design; got {factor_names}, missing {missing}."
            raise ValueError(msg)
        return list(factor_names)

    names = [c for c in design.columns if c not in ("RunOrder", "Block")]

    def looks_coded(column: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(column):
            return False
        low, high = float(column.min()), float(column.max())
        return low < 0 < high or max(abs(low), abs(high)) <= 1.0

    not_coded = [c for c in names if not looks_coded(design[c])]
    if not_coded:
        msg = (
            f"Column(s) {not_coded} do not look like factors in coded units (numeric, and either spread across 0 "
            "or within [-1, 1]); a response column would be augmented as if it were a factor. Pass "
            "factor_names=[...] naming the factor columns, or drop the other columns."
        )
        raise ValueError(msg)
    return names


def augment_design(  # noqa: PLR0913
    existing_design: pd.DataFrame,
    augmentation_type: str,
    target_model: str | None = None,
    n_additional_runs: int | None = None,
    fold_on: str | None = None,
    alpha: str | float | None = None,
    generators: list[str] | None = None,
    random_state: int | np.random.Generator | None = 42,
    factor_names: list[str] | None = None,
) -> dict[str, Any]:
    """Extend or modify an existing experimental design.

    Parameters
    ----------
    existing_design : DataFrame
        The current design matrix with factor columns in coded units (-1/+1).
        Only the factor columns are augmented and returned; see *factor_names*.
    augmentation_type : str
        One of ``"foldover"``, ``"semifold"``, ``"add_center_points"``,
        ``"add_axial_points"``, ``"add_runs_optimal"``, ``"upgrade_to_rsm"``,
        ``"add_blocks"``, ``"replicate"``.
    target_model : str or None
        Desired model after augmentation: ``"main_effects"``,
        ``"interactions"``, ``"quadratic"``. ``"add_runs_optimal"`` chooses
        runs for it (default ``"interactions"``); ``"upgrade_to_rsm"`` checks
        that the upgraded design can estimate it (default ``"quadratic"``),
        warning and listing the aliased terms when it cannot.
    n_additional_runs : int or None
        Budget for additional runs, a positive whole number. Interpretation
        depends on the augmentation type (number of center points, number of
        D-optimal runs, number of blocks, ...). A regular two-level design
        splits into 2, 4, 8, ... blocks; another count raises. For ``"replicate"``, this is the
        number of complete copies of the existing design that are appended
        (each copy adds ``len(existing_design)`` runs); the default of
        ``None`` becomes 1 complete copy.
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
    random_state : int, numpy.random.Generator or None, default 42
        Seeds the exchange's random starts for ``"add_runs_optimal"``; the default
        keeps the result reproducible, and ``None`` draws fresh starts.
    factor_names : list[str] or None
        The factor columns. ``None`` takes every column except ``RunOrder`` and
        ``Block``, and then refuses a column that does not look like a coded
        factor (non-numeric, or all on one side of 0 and beyond [-1, 1]), such
        as a response measured on the runs: augmenting it as a factor would add
        axial runs on it or negate it in a foldover. Name the factors to keep
        other columns out.

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
        If *augmentation_type* is unknown, if required parameters are missing
        for the requested augmentation, or if a column would be treated as a
        factor without looking like one (see *factor_names*).

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
        raise ValueError(f"Unknown augmentation_type={augmentation_type!r}. Choose from: {', '.join(available)}.")

    if n_additional_runs is not None and (
        isinstance(n_additional_runs, bool) or int(n_additional_runs) != n_additional_runs or n_additional_runs < 1
    ):
        msg = (
            f"n_additional_runs must be a positive whole number (runs, centre points, copies or blocks); "
            f"got {n_additional_runs!r}."
        )
        raise ValueError(msg)
    factor_names = _resolve_factor_names(existing_design, factor_names)

    ctx = _AugmentContext(
        existing_design=existing_design,
        factor_names=factor_names,
        augmentation_type=augmentation_type,
        target_model=target_model,
        n_additional_runs=n_additional_runs,
        fold_on=fold_on,
        alpha=alpha,
        generators=generators,
        random_state=random_state,
    )

    handler = _AUGMENT_REGISTRY[augmentation_type]
    return handler(ctx)
