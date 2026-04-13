"""Tests for the augment_design() design augmentation API."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.augment import (
    _AUGMENT_REGISTRY,
    _auto_select_fold_factor,
    _AugmentContext,
    _compute_alpha,
    augment_design,
)
from process_improve.experiments.designs import generate_design
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Factor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_factorial_df(k: int, names: str = "ABCDEFGHIJ") -> pd.DataFrame:
    """Create a 2^k full factorial design as a DataFrame of -1/+1 coded values."""
    n = 2**k
    cols = {}
    for i in range(k):
        cols[names[i]] = [1 if (j >> (k - 1 - i)) & 1 else -1 for j in range(n)]
    return pd.DataFrame(cols)


def _fractional_factorial_df(
    generators: list[str], names: str = "ABCDEFGHIJ",
) -> pd.DataFrame:
    """Create a fractional factorial design from generator strings.

    Example: generators=["D=ABC"] with names "ABCD" gives a 2^(4-1) design.
    """
    factors = [Factor(name=names[i], low=0, high=10) for i in range(len(names))]
    result = generate_design(
        factors[:len(names)],
        design_type="fractional_factorial",
        generators=generators,
        center_points=0,
    )
    # Extract coded design without RunOrder
    df = pd.DataFrame(result.design)
    return df.drop(columns=["RunOrder"], errors="ignore")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    """Test the augment_design dispatcher."""

    def test_unknown_type_raises(self) -> None:
        """Unknown augmentation_type raises ValueError."""
        df = _full_factorial_df(2)
        with pytest.raises(ValueError, match="Unknown augmentation_type"):
            augment_design(df, augmentation_type="nonexistent")

    def test_all_types_in_registry(self) -> None:
        """All 8 augmentation types are registered."""
        expected = {
            "foldover", "semifold", "add_center_points", "add_axial_points",
            "add_runs_optimal", "upgrade_to_rsm", "add_blocks", "replicate",
        }
        assert set(_AUGMENT_REGISTRY.keys()) == expected

    def test_drops_runorder_column(self) -> None:
        """RunOrder column should be ignored as a factor."""
        df = _full_factorial_df(2)
        df.insert(0, "RunOrder", range(1, len(df) + 1))
        result = augment_design(df, "add_center_points", n_additional_runs=1)
        aug = pd.DataFrame(result["augmented_design"])
        assert "RunOrder" not in aug.columns
        assert set(aug.columns) == {"A", "B"}


# ---------------------------------------------------------------------------
# Add center points
# ---------------------------------------------------------------------------


class TestAddCenterPoints:
    """Test add_center_points augmentation."""

    def test_adds_correct_count(self) -> None:
        """n_additional_runs=5 adds exactly 5 rows."""
        df = _full_factorial_df(3)
        result = augment_design(df, "add_center_points", n_additional_runs=5)
        assert result["n_runs_after"] == 8 + 5

    def test_center_points_are_zeros(self) -> None:
        """New center point rows are all zeros in coded units."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_center_points", n_additional_runs=3)
        new_runs = pd.DataFrame(result["new_runs"])
        assert (new_runs == 0).all().all()

    def test_default_count(self) -> None:
        """Without n_additional_runs, defaults to 3 center points."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_center_points")
        assert result["n_runs_after"] == 4 + 3

    def test_explanation_mentions_curvature(self) -> None:
        """Explanation should mention curvature."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_center_points")
        assert "curvature" in result["explanation"].lower()

    def test_original_rows_preserved(self) -> None:
        """Original design rows should be unchanged."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_center_points", n_additional_runs=2)
        aug = pd.DataFrame(result["augmented_design"])
        original_part = aug.iloc[:4]
        pd.testing.assert_frame_equal(
            original_part.reset_index(drop=True),
            df.reset_index(drop=True).astype(float),
        )


# ---------------------------------------------------------------------------
# Replicate
# ---------------------------------------------------------------------------


class TestReplicate:
    """Test replicate augmentation."""

    def test_doubles_run_count(self) -> None:
        """Default replication (1 copy) doubles the run count."""
        df = _full_factorial_df(2)
        result = augment_design(df, "replicate")
        assert result["n_runs_after"] == 8

    def test_custom_replicate_count(self) -> None:
        """n_additional_runs=2 triples the design."""
        df = _full_factorial_df(2)
        result = augment_design(df, "replicate", n_additional_runs=2)
        assert result["n_runs_after"] == 12

    def test_rows_match_originals(self) -> None:
        """Replicated rows should match the original design."""
        df = _full_factorial_df(2)
        result = augment_design(df, "replicate")
        aug = pd.DataFrame(result["augmented_design"])
        first_half = aug.iloc[:4].reset_index(drop=True)
        second_half = aug.iloc[4:].reset_index(drop=True)
        pd.testing.assert_frame_equal(first_half, second_half)

    def test_explanation_mentions_replicate(self) -> None:
        """Explanation should mention replication."""
        df = _full_factorial_df(2)
        result = augment_design(df, "replicate")
        assert "replicate" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# Foldover
# ---------------------------------------------------------------------------


class TestFoldover:
    """Test foldover augmentation."""

    def test_doubles_run_count(self) -> None:
        """Foldover doubles the number of runs."""
        df = _full_factorial_df(3)
        result = augment_design(df, "foldover")
        assert result["n_runs_after"] == 16

    def test_negated_signs(self) -> None:
        """Folded half has all signs negated."""
        df = _full_factorial_df(2)
        result = augment_design(df, "foldover")
        aug = pd.DataFrame(result["augmented_design"])
        original = aug.iloc[:4].values
        folded = aug.iloc[4:].values
        np.testing.assert_array_equal(folded, -original)

    def test_with_generators_updates_defining_relation(self) -> None:
        """Foldover with generators should update the defining relation."""
        # 2^(4-1) with D=ABC: I=ABCD (length 4, even)
        # After foldover, even-length words survive
        df = _fractional_factorial_df(["D=ABC"], names="ABCD")
        result = augment_design(df, "foldover", generators=["D=ABC"])
        # ABCD has length 4 (even) — it should survive
        assert result["defining_relation"] is not None
        assert any("ABCD" in w for w in result["defining_relation"])

    def test_res_iii_foldover_clears_odd_words(self) -> None:
        """Resolution III foldover should eliminate odd-length defining words."""
        # 2^(3-1) with C=AB: I=ABC (length 3, odd)
        df = _fractional_factorial_df(["C=AB"], names="ABC")
        result = augment_design(df, "foldover", generators=["C=AB"])
        # ABC has length 3 (odd) — should be eliminated
        explanation = result["explanation"]
        assert "eliminated" in explanation.lower() or "de-alias" in explanation.lower()

    def test_without_generators_still_works(self) -> None:
        """Foldover without generators should still produce valid augmentation."""
        df = _full_factorial_df(3)
        result = augment_design(df, "foldover")
        assert result["n_runs_after"] == 16
        assert result["defining_relation"] is None
        assert "explanation" in result

    def test_explanation_nonempty(self) -> None:
        """Explanation should never be empty."""
        df = _full_factorial_df(2)
        result = augment_design(df, "foldover")
        assert len(result["explanation"]) > 0


# ---------------------------------------------------------------------------
# Semifold
# ---------------------------------------------------------------------------


class TestSemifold:
    """Test semifold augmentation."""

    def test_adds_half_runs(self) -> None:
        """Semifold adds N/2 new runs."""
        df = _full_factorial_df(3)
        result = augment_design(df, "semifold")
        # Half of 8 rows where fold factor = -1 = 4 rows added
        assert result["n_runs_after"] == 12

    def test_explicit_fold_on(self) -> None:
        """Explicit fold_on uses specified factor."""
        df = _full_factorial_df(3)
        result = augment_design(df, "semifold", fold_on="C")
        assert result["fold_on"] == "C"

    def test_auto_selects_fold_factor(self) -> None:
        """Without fold_on, a factor is auto-selected."""
        df = _fractional_factorial_df(["C=AB"], names="ABC")
        result = augment_design(df, "semifold", generators=["C=AB"])
        assert result["fold_on"] in ["A", "B", "C"]

    def test_invalid_fold_factor_raises(self) -> None:
        """fold_on with nonexistent factor raises ValueError."""
        df = _full_factorial_df(2)
        with pytest.raises(ValueError, match="fold_on"):
            augment_design(df, "semifold", fold_on="Z")

    def test_with_generators_reports_dealiasing(self) -> None:
        """Semifold with generators should report de-aliasing."""
        df = _fractional_factorial_df(["D=ABC"], names="ABCD")
        result = augment_design(df, "semifold", generators=["D=ABC"], fold_on="A")
        assert "explanation" in result
        # Should mention fold factor
        assert "A" in result["explanation"]


# ---------------------------------------------------------------------------
# Add axial points
# ---------------------------------------------------------------------------


class TestAddAxialPoints:
    """Test add_axial_points augmentation."""

    def test_adds_2k_points(self) -> None:
        """k factors -> 2k axial points added."""
        df = _full_factorial_df(3)
        result = augment_design(df, "add_axial_points", alpha="face_centered")
        assert result["n_runs_after"] == 8 + 6  # 3 factors * 2

    def test_face_centered_alpha(self) -> None:
        """Face-centered alpha should be 1.0."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_axial_points", alpha="face_centered")
        assert result["alpha"] == pytest.approx(1.0)

    def test_rotatable_alpha(self) -> None:
        """Rotatable alpha should be n_factorial^(1/4)."""
        df = _full_factorial_df(3)  # 8 factorial runs
        result = augment_design(df, "add_axial_points", alpha="rotatable")
        expected = 8 ** 0.25  # ~1.6818
        assert result["alpha"] == pytest.approx(expected, abs=0.01)

    def test_numeric_alpha(self) -> None:
        """Numeric alpha is used directly."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_axial_points", alpha=1.5)
        assert result["alpha"] == pytest.approx(1.5)

    def test_axial_point_structure(self) -> None:
        """Each axial point should have exactly one non-zero factor at +/-alpha."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_axial_points", alpha="face_centered")
        new_runs = pd.DataFrame(result["new_runs"])
        for _, row in new_runs.iterrows():
            nonzero = (row.abs() > 1e-10).sum()
            assert nonzero == 1, f"Axial point should have exactly 1 nonzero: {row.to_dict()}"

    def test_explanation_mentions_quadratic(self) -> None:
        """Explanation should mention quadratic effects."""
        df = _full_factorial_df(2)
        result = augment_design(df, "add_axial_points", alpha="face_centered")
        assert "quadratic" in result["explanation"].lower()


# ---------------------------------------------------------------------------
# Upgrade to RSM
# ---------------------------------------------------------------------------


class TestUpgradeToRSM:
    """Test upgrade_to_rsm augmentation."""

    def test_factorial_to_ccd(self) -> None:
        """2^3 factorial should gain axial + center points."""
        df = _full_factorial_df(3)
        result = augment_design(df, "upgrade_to_rsm")
        # 8 factorial + 6 axial + some center points
        assert result["n_runs_after"] > 14

    def test_quadratic_model_estimable(self) -> None:
        """After RSM upgrade, quadratic model should be estimable."""
        df = _full_factorial_df(3)
        result = augment_design(df, "upgrade_to_rsm", alpha="face_centered")
        aug = pd.DataFrame(result["augmented_design"])
        # Should be able to evaluate with quadratic model without singularity
        metrics = evaluate_design(aug, model="quadratic", metric="d_efficiency")
        assert metrics["d_efficiency"] is not None
        assert metrics["d_efficiency"] > 0

    def test_alpha_parameter_passed(self) -> None:
        """Alpha parameter should be reflected in result."""
        df = _full_factorial_df(2)
        result = augment_design(df, "upgrade_to_rsm", alpha="face_centered")
        assert result["alpha"] == pytest.approx(1.0)

    def test_preserves_existing_center_points(self) -> None:
        """Existing center points should be preserved, not duplicated excessively."""
        df = _full_factorial_df(2)
        # Add 3 center points first
        center = pd.DataFrame({"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]})
        df_with_centers = pd.concat([df, center], ignore_index=True)
        result = augment_design(df_with_centers, "upgrade_to_rsm", alpha="face_centered")
        aug = pd.DataFrame(result["augmented_design"])
        # Count center points in augmented design
        center_mask = (aug.abs() < 1e-10).all(axis=1)
        n_centers = center_mask.sum()
        # Should have original 3 + some new, but not an unreasonable number
        assert n_centers >= 3
        assert n_centers <= 8

    def test_explanation_mentions_ccd(self) -> None:
        """Explanation should mention CCD or Central Composite."""
        df = _full_factorial_df(2)
        result = augment_design(df, "upgrade_to_rsm")
        assert "central composite" in result["explanation"].lower() or "ccd" in result["explanation"].lower()
