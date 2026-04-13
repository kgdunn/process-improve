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
