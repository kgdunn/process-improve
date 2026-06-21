"""Tests for analyze_omars(): staged analysis of OMARS-design data."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import OmarsResult, analyze_omars

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fccd() -> pd.DataFrame:
    """Face-centered central composite design for three factors (levels in {-1, 0, 1}).

    Eight factorial corners, six axial points on the faces, and six centre
    runs: 20 runs in total, with mutually orthogonal main effects.
    """
    factorial = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    axial = np.array(
        [[s if k == ax else 0 for k in range(3)] for ax in range(3) for s in (-1, 1)],
        dtype=float,
    )
    centre = np.zeros((6, 3))
    return pd.DataFrame(np.vstack([factorial, axial, centre]), columns=list("ABC"))


def _response(design: pd.DataFrame, *, seed: int = 3, noise: float = 0.2) -> np.ndarray:
    """Response with active main effects A, B, an A:B interaction and an A^2 term."""
    x = design.to_numpy()
    rng = np.random.default_rng(seed)
    noise_vec = rng.normal(0, noise, size=x.shape[0])
    return 6 * x[:, 0] + 5 * x[:, 1] + 4 * (x[:, 0] * x[:, 1]) + 4 * (x[:, 0] ** 2) + noise_vec


def _orthogonal_three_level() -> pd.DataFrame:
    """Full 3-level factorial for three factors: 27 runs, levels in {-1, 0, 1}.

    On the full factorial grid every second-order column (each quadratic and
    each two-factor interaction) is mutually orthogonal: the off-diagonal of
    the second-order Gram matrix is exactly zero.  This is the zero-aliasing
    limit of the minimal aliasing that OMARS designs are built to achieve, so
    it is the cleanest available stand-in for a true OMARS design until a
    catalogued OMARS generator is available.  Because there is no aliasing,
    the staged analysis can recover the *exact* set of active terms, with no
    spurious ones.
    """
    grid = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)
    return pd.DataFrame(grid, columns=list("ABC"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_exposed_in_experiments_namespace() -> None:
    from process_improve.experiments import analyze_omars as exported

    assert exported is analyze_omars
    assert isinstance(analyze_omars(_fccd(), _response(_fccd())), OmarsResult)


# ---------------------------------------------------------------------------
# Staged mechanics
# ---------------------------------------------------------------------------


def test_main_effects_recovered() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design))

    assert result.success is True
    assert result.active_main_effects == ["A", "B"]
    assert result.initial_error_df == 10
    # The inactive main effect C is pooled back, gaining one error degree of freedom.
    assert result.updated_error_df == 11
    # Inactive factor C has a clearly non-significant main effect.
    assert result.main_effect_p_values["C"] > 0.1
    assert result.main_effect_p_values["A"] < 0.01


def test_staged_variance_estimates_are_stable() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design))

    # Initial and pooled RMSE track the true noise scale (0.2) closely.
    assert result.initial_rmse == pytest.approx(0.227, abs=0.01)
    assert result.updated_rmse == pytest.approx(0.229, abs=0.01)
    # Strong curvature: the overall second-order gate opens decisively.
    assert result.second_order_overall_p_value < 0.001


def test_true_second_order_terms_are_detected() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design))

    # The genuinely active second-order terms must be among those flagged.
    assert "A:B" in result.active_interactions
    assert "A^2" in result.active_quadratics


def test_strong_heredity_gives_clean_recovery() -> None:
    design = _fccd()
    result = analyze_omars(
        design,
        _response(design),
        interaction_heredity="strong",
        quadratic_heredity="strong",
    )

    # With only A and B active, strong heredity limits candidates to
    # {A^2, B^2, A:B}. The interaction is recovered exactly (the spurious A:C
    # that appears under no-heredity is correctly excluded). The true A^2 is
    # found; B^2 may also appear because a CCD aliases its quadratic terms,
    # which is the design weakness an OMARS design is built to avoid.
    assert result.active_interactions == ["A:B"]
    assert "A^2" in result.active_quadratics
    assert set(result.active_quadratics) <= {"A^2", "B^2"}


# ---------------------------------------------------------------------------
# Exact recovery on a fully orthogonal three-level design
#
# Unlike the CCD above (whose quadratics are aliased), this design has
# mutually orthogonal second-order terms, so the staged analysis must recover
# the active terms exactly, with no spurious extras.
# ---------------------------------------------------------------------------


def test_orthogonal_design_has_zero_second_order_aliasing() -> None:
    from process_improve.experiments.omars import _full_second_order, _quadratic_columns

    coded = _orthogonal_three_level().to_numpy()
    second_order = _full_second_order(coded, _quadratic_columns(coded))
    gram = second_order.T @ second_order
    off_diagonal = np.abs(gram - np.diag(np.diag(gram)))
    assert off_diagonal.max() == pytest.approx(0.0, abs=1e-9)


def test_exact_recovery_on_orthogonal_design() -> None:
    design = _orthogonal_three_level()
    x = design.to_numpy()
    rng = np.random.default_rng(0)
    # Active: main effects A and B, the A:B interaction, and the A^2 quadratic.
    y = 6 * x[:, 0] + 5 * x[:, 1] + 4 * (x[:, 0] * x[:, 1]) + 4 * (x[:, 0] ** 2) + rng.normal(0, 0.3, x.shape[0])

    result = analyze_omars(design, y)

    # With no aliasing, recovery is exact: precisely the true terms, nothing more.
    assert result.active_main_effects == ["A", "B"]
    assert result.active_interactions == ["A:B"]
    assert result.active_quadratics == ["A^2"]


def test_exact_recovery_of_pure_quadratic_without_parent_main_effect() -> None:
    design = _orthogonal_three_level()
    x = design.to_numpy()
    rng = np.random.default_rng(1)
    # Active: main effect A, and a pure C^2 curvature whose factor C has no
    # linear main effect. Under no heredity the quadratic is still recovered.
    y = 6 * x[:, 0] + 5 * (x[:, 2] ** 2) + rng.normal(0, 0.3, x.shape[0])

    result = analyze_omars(design, y)

    assert result.active_main_effects == ["A"]
    assert result.active_interactions == []
    assert result.active_quadratics == ["C^2"]


def test_gate_stays_shut_without_curvature() -> None:
    design = _fccd()
    x = design.to_numpy()
    rng = np.random.default_rng(11)
    y = 6 * x[:, 0] + 5 * x[:, 1] + rng.normal(0, 0.2, size=x.shape[0])

    result = analyze_omars(design, y, alpha_second_overall=0.05)

    assert result.active_main_effects == ["A", "B"]
    assert result.active_interactions == []
    assert result.active_quadratics == []
    assert result.second_order_overall_p_value > 0.05


def test_main_effects_only_mode() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design), second_order=False)

    assert result.active_main_effects == ["A", "B"]
    assert result.active_interactions == []
    assert result.active_quadratics == []
    # No second-order work was done.
    assert np.isnan(result.second_order_overall_p_value)
    assert result.updated_error_df == result.initial_error_df
    assert result.updated_rmse == result.initial_rmse


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_effects_to_drop_excludes_term() -> None:
    design = _fccd()
    baseline = analyze_omars(design, _response(design))
    assert "A:C" in baseline.active_interactions  # CCD aliasing flags A:C by default

    dropped = analyze_omars(design, _response(design), effects_to_drop=["A:C"])
    assert "A:C" not in dropped.active_interactions
    # The true terms survive the exclusion.
    assert "A:B" in dropped.active_interactions
    assert "A^2" in dropped.active_quadratics


def test_force_main_effect_into_model() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design), second_order=False, force_main_effects=["C"])

    assert "C" in result.active_main_effects
    assert result.forced_main_effects == ["C"]


# ---------------------------------------------------------------------------
# Degenerate designs
# ---------------------------------------------------------------------------


def test_saturated_design_reports_failure() -> None:
    # A 10-run, three-factor, three-level design whose full second-order model
    # is saturated (rank 10 == 10 runs), leaving no error degrees of freedom.
    saturated = pd.DataFrame(
        [
            [0, 1, -1],
            [1, 0, -1],
            [0, 0, -1],
            [0, -1, 0],
            [-1, 0, 1],
            [-1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, 0, 0],
            [-1, -1, 0],
        ],
        columns=list("ABC"),
        dtype=float,
    )
    rng = np.random.default_rng(0)
    result = analyze_omars(saturated, rng.normal(size=10))

    assert result.success is False
    assert "error degrees of freedom" in result.details["reason"]


def test_perfect_fit_reports_failure() -> None:
    design = _fccd()
    x = design.to_numpy()
    # A response lying exactly in the model's column space: zero residual variance.
    result = analyze_omars(design, 6 * x[:, 0] + 5 * x[:, 1])

    assert result.success is False
    assert "zero" in result.details["reason"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_unknown_heredity_option_raises() -> None:
    design = _fccd()
    with pytest.raises(ValueError, match="quadratic_heredity"):
        analyze_omars(design, _response(design), quadratic_heredity="weak")
    with pytest.raises(ValueError, match="interaction_heredity"):
        analyze_omars(design, _response(design), interaction_heredity="maybe")


def test_unknown_forced_factor_raises() -> None:
    design = _fccd()
    with pytest.raises(ValueError, match="unknown factor"):
        analyze_omars(design, _response(design), force_main_effects=["Z"])


def test_unknown_dropped_effect_raises() -> None:
    design = _fccd()
    with pytest.raises(ValueError, match="unknown second-order"):
        analyze_omars(design, _response(design), effects_to_drop=["Q^2"])


def test_length_mismatch_raises() -> None:
    design = _fccd()
    with pytest.raises(ValueError, match="rows but response"):
        analyze_omars(design, np.zeros(design.shape[0] + 1))


def test_missing_values_raise() -> None:
    design = _fccd()
    y = _response(design)
    y[0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        analyze_omars(design, y)


def test_constant_factor_raises() -> None:
    design = pd.DataFrame({"A": [1, 1, 1, 1], "B": [-1, 1, -1, 1]})
    with pytest.raises(ValueError, match="constant"):
        analyze_omars(design, [1.0, 2.0, 3.0, 4.0])


# ---------------------------------------------------------------------------
# Heredity variants and subset-size limit
# ---------------------------------------------------------------------------


def test_weak_interaction_heredity_runs() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design), interaction_heredity="weak")

    # Weak heredity keeps interactions with at least one active parent (A or B),
    # so the true A:B interaction is still recovered.
    assert "A:B" in result.active_interactions
    assert "A^2" in result.active_quadratics


def test_user_subset_limit_is_respected() -> None:
    design = _fccd()
    result = analyze_omars(design, _response(design), max_subset_terms=2)

    assert result.subset_limit == 2
    assert "user specified" in result.subset_limit_reason
    # No more than the cap of second-order terms may be declared active.
    assert len(result.active_interactions) + len(result.active_quadratics) <= 2


# ---------------------------------------------------------------------------
# Designs without a full second-order structure
# ---------------------------------------------------------------------------


def test_two_level_design_has_no_quadratics() -> None:
    # A replicated 2^3 factorial: two levels only, so no quadratic terms exist;
    # only interactions populate the second-order space.
    corners = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
    design = pd.DataFrame(np.vstack([corners, corners]), columns=list("ABC"))
    x = design.to_numpy()
    rng = np.random.default_rng(5)
    y = 5 * x[:, 0] + 4 * (x[:, 1] * x[:, 2]) + rng.normal(0, 0.2, size=x.shape[0])

    result = analyze_omars(design, y)

    assert result.active_quadratics == []
    assert "B:C" in result.active_interactions


def test_single_factor_has_no_second_order_space() -> None:
    # One factor: the second-order space is empty (rank 0), so the gate cannot open.
    design = pd.DataFrame({"A": [-1, 1, -1, 1, -1, 1]})
    result = analyze_omars(design, [1.0, 5.0, 2.0, 6.0, 1.0, 5.0])

    assert result.success is True
    assert result.details["full_second_order_rank"] == 0
    assert np.isnan(result.second_order_overall_p_value)
    assert result.active_interactions == []
    assert result.active_quadratics == []
