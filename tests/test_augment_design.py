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
