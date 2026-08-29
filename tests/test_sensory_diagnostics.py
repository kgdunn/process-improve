"""Tests for the panel diagnostics that decide whether an attribute is modellable.

Also pins the A5 empty-panel guard: ``mixed_assessor_model`` used to return
column-less frames, so an over-filtered panel surfaced as a ``KeyError`` in the
caller rather than as a diagnosable error at the source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.sensory import (
    assessor_variance_equality,
    boundary_occupancy,
    detection_rate,
    mixed_assessor_model,
)

PANEL_COLUMNS = ["panelist_id", "session", "product", "attribute", "replicate", "score"]


def _panel(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["session"] = frame.get("session", 1)
    frame["replicate"] = frame.get("replicate", 1)
    return frame[PANEL_COLUMNS]


class TestEmptyPanelGuard:
    """A5: an empty panel must not become a KeyError somewhere downstream."""

    def test_empty_panel_raises_value_error(self) -> None:
        empty = pd.DataFrame(columns=PANEL_COLUMNS)
        with pytest.raises(ValueError, match="no rows"):
            mixed_assessor_model(empty)

    def test_error_names_the_condition_and_the_likely_cause(self) -> None:
        empty = pd.DataFrame(columns=PANEL_COLUMNS)
        with pytest.raises(ValueError, match="no rows") as excinfo:
            mixed_assessor_model(empty)
        message = str(excinfo.value)
        assert "mixed_assessor_model" in message
        assert "filter" in message

    def test_missing_columns_raise_before_the_key_error(self) -> None:
        frame = pd.DataFrame({"panelist_id": ["a"], "product": ["p"], "score": [1.0]})
        with pytest.raises(ValueError, match="attribute"):
            mixed_assessor_model(frame)

    def test_panel_with_rows_still_works(self) -> None:
        rng = np.random.default_rng(0)
        rows = [
            {
                "panelist_id": f"p{pid}",
                "product": product,
                "attribute": "sweetness",
                "score": 5.0 + offset + rng.normal(scale=0.4),
            }
            for pid in range(6)
            for product, offset in (("A", 0.0), ("B", 2.0), ("C", -1.5))
        ]
        result = mixed_assessor_model(_panel(rows))
        assert "f_product_mam" in result.ftests.columns
        assert len(result.ftests) == 1


class TestBoundaryOccupancy:
    def test_floor_ceiling_and_exact_zero_are_counted_separately(self) -> None:
        rows = [
            # 'pinned' sits hard against the floor, half of it as exact zeros.
            {"panelist_id": "p1", "product": "A", "attribute": "pinned", "score": 0.0},
            {"panelist_id": "p2", "product": "A", "attribute": "pinned", "score": 0.0},
            {"panelist_id": "p3", "product": "A", "attribute": "pinned", "score": 0.4},
            {"panelist_id": "p4", "product": "A", "attribute": "pinned", "score": 0.5},
            # 'mid' uses the middle of the scale and touches neither bound.
            {"panelist_id": "p1", "product": "A", "attribute": "mid", "score": 4.0},
            {"panelist_id": "p2", "product": "A", "attribute": "mid", "score": 5.0},
            {"panelist_id": "p3", "product": "A", "attribute": "mid", "score": 6.0},
            {"panelist_id": "p4", "product": "A", "attribute": "mid", "score": 7.0},
            # 'saturated' is pinned at the top.
            {"panelist_id": "p1", "product": "A", "attribute": "saturated", "score": 10.0},
            {"panelist_id": "p2", "product": "A", "attribute": "saturated", "score": 9.6},
            {"panelist_id": "p3", "product": "A", "attribute": "saturated", "score": 9.8},
            {"panelist_id": "p4", "product": "A", "attribute": "saturated", "score": 5.0},
        ]
        table = boundary_occupancy(_panel(rows)).set_index("attribute")

        assert table.loc["pinned", "n"] == 4
        assert table.loc["pinned", "at_floor"] == 4
        assert table.loc["pinned", "exact_zero"] == 2
        assert table.loc["pinned", "at_ceiling"] == 0

        assert table.loc["mid", "at_floor"] == 0
        assert table.loc["mid", "at_ceiling"] == 0
        assert table.loc["mid", "exact_zero"] == 0

        assert table.loc["saturated", "at_ceiling"] == 3
        assert table.loc["saturated", "at_floor"] == 0

    def test_not_perceived_recorded_as_small_positive_is_not_an_exact_zero(self) -> None:
        """A panel that records 'not perceived' as 0.2 looks floor-pinned but is not zero."""
        rows = [{"panelist_id": f"p{i}", "product": "A", "attribute": "trace", "score": 0.2} for i in range(5)]
        table = boundary_occupancy(_panel(rows)).set_index("attribute")
        assert table.loc["trace", "at_floor"] == 5
        assert table.loc["trace", "exact_zero"] == 0

    def test_fractions_are_reported_alongside_the_counts(self) -> None:
        rows = [
            {"panelist_id": "p1", "product": "A", "attribute": "x", "score": 0.0},
            {"panelist_id": "p2", "product": "A", "attribute": "x", "score": 5.0},
            {"panelist_id": "p3", "product": "A", "attribute": "x", "score": 5.0},
            {"panelist_id": "p4", "product": "A", "attribute": "x", "score": 5.0},
        ]
        table = boundary_occupancy(_panel(rows)).set_index("attribute")
        assert table.loc["x", "frac_floor"] == pytest.approx(0.25)
        assert table.loc["x", "frac_exact_zero"] == pytest.approx(0.25)

    def test_custom_scale_bounds(self) -> None:
        rows = [
            {"panelist_id": "p1", "product": "A", "attribute": "x", "score": 100.0},
            {"panelist_id": "p2", "product": "A", "attribute": "x", "score": 50.0},
        ]
        table = boundary_occupancy(_panel(rows), lo=0.0, hi=100.0).set_index("attribute")
        assert table.loc["x", "at_ceiling"] == 1

    def test_missing_scores_are_excluded_from_n(self) -> None:
        rows = [
            {"panelist_id": "p1", "product": "A", "attribute": "x", "score": 0.0},
            {"panelist_id": "p2", "product": "A", "attribute": "x", "score": np.nan},
        ]
        table = boundary_occupancy(_panel(rows)).set_index("attribute")
        assert table.loc["x", "n"] == 1
        assert table.loc["x", "frac_floor"] == pytest.approx(1.0)

    def test_empty_panel_raises(self) -> None:
        with pytest.raises(ValueError, match="no rows"):
            boundary_occupancy(pd.DataFrame(columns=PANEL_COLUMNS))

    def test_band_must_be_a_fraction_of_the_scale(self) -> None:
        rows = [{"panelist_id": "p1", "product": "A", "attribute": "x", "score": 1.0}]
        with pytest.raises(ValueError, match="band"):
            boundary_occupancy(_panel(rows), band=0.9)

    def test_lo_must_be_below_hi(self) -> None:
        rows = [{"panelist_id": "p1", "product": "A", "attribute": "x", "score": 1.0}]
        with pytest.raises(ValueError, match="lo"):
            boundary_occupancy(_panel(rows), lo=10.0, hi=0.0)


class TestDetectionRate:
    def test_product_by_attribute_probabilities(self) -> None:
        rows = [
            # Product A: everyone detects 'note'.
            *[{"panelist_id": f"p{i}", "product": "A", "attribute": "note", "score": 4.0} for i in range(4)],
            # Product B: nobody does.
            *[{"panelist_id": f"p{i}", "product": "B", "attribute": "note", "score": 0.0} for i in range(4)],
            # Product C: half do.
            *[{"panelist_id": f"p{i}", "product": "C", "attribute": "note", "score": 4.0} for i in range(2)],
            *[{"panelist_id": f"p{i}", "product": "C", "attribute": "note", "score": 0.0} for i in range(2, 4)],
        ]
        table = detection_rate(_panel(rows))
        assert table.loc["A", "note"] == pytest.approx(1.0)
        assert table.loc["B", "note"] == pytest.approx(0.0)
        assert table.loc["C", "note"] == pytest.approx(0.5)

    def test_scores_inside_the_floor_band_do_not_count_as_detected(self) -> None:
        rows = [
            {"panelist_id": "p1", "product": "A", "attribute": "note", "score": 0.5},
            {"panelist_id": "p2", "product": "A", "attribute": "note", "score": 1.5},
        ]
        table = detection_rate(_panel(rows))
        assert table.loc["A", "note"] == pytest.approx(0.5)

    def test_products_and_attributes_form_the_full_grid(self) -> None:
        rows = [
            {"panelist_id": "p1", "product": "A", "attribute": "one", "score": 5.0},
            {"panelist_id": "p1", "product": "B", "attribute": "two", "score": 5.0},
        ]
        table = detection_rate(_panel(rows))
        assert list(table.index) == ["A", "B"]
        assert list(table.columns) == ["one", "two"]
        # A cell nobody assessed is missing, not zero: "never detected" and
        # "never asked" are different answers.
        assert np.isnan(table.loc["A", "two"])

    def test_empty_panel_raises(self) -> None:
        with pytest.raises(ValueError, match="no rows"):
            detection_rate(pd.DataFrame(columns=PANEL_COLUMNS))


class TestAssessorVarianceEquality:
    @staticmethod
    def _panel_with_one_noisy_assessor(seed: int = 0) -> pd.DataFrame:
        """Two attributes with identical product effects; one has a noisy assessor."""
        rng = np.random.default_rng(seed)
        product_effect = {"A": 0.0, "B": 2.0, "C": -1.0, "D": 1.0}
        rows = []
        for attribute, noisy_spreads in (("equal", {}), ("unequal", {"p0": 6.0})):
            for pid in range(6):
                name = f"p{pid}"
                spread = noisy_spreads.get(name, 0.5)
                rows.extend(
                    {
                        "panelist_id": name,
                        "product": product,
                        "attribute": attribute,
                        "replicate": replicate,
                        "score": 5.0 + effect + rng.normal(scale=spread),
                    }
                    for product, effect in product_effect.items()
                    for replicate in range(4)
                )
        frame = pd.DataFrame(rows)
        frame["session"] = 1
        return frame[PANEL_COLUMNS]

    def test_fires_for_the_unequal_attribute_and_stays_quiet_for_the_equal_one(self) -> None:
        table = assessor_variance_equality(self._panel_with_one_noisy_assessor()).set_index("attribute")
        assert table.loc["unequal", "p_equal_variance"] < 0.01
        assert table.loc["equal", "p_equal_variance"] > 0.05

    def test_spread_ratio_tracks_the_planted_inequality(self) -> None:
        table = assessor_variance_equality(self._panel_with_one_noisy_assessor()).set_index("attribute")
        assert table.loc["unequal", "spread_ratio_max_min"] > 4.0
        assert table.loc["equal", "spread_ratio_max_min"] < 4.0

    def test_reports_the_assessor_count(self) -> None:
        table = assessor_variance_equality(self._panel_with_one_noisy_assessor()).set_index("attribute")
        assert (table["n_assessors"] == 6).all()

    def test_columns_are_as_documented(self) -> None:
        table = assessor_variance_equality(self._panel_with_one_noisy_assessor())
        assert list(table.columns) == [
            "attribute",
            "levene_stat",
            "p_equal_variance",
            "spread_ratio_max_min",
            "n_assessors",
        ]

    def test_product_effects_are_removed_before_testing(self) -> None:
        """Assessors who agree perfectly must not be flagged by a large product effect."""
        rows = [
            {
                "panelist_id": f"p{pid}",
                "product": product,
                "attribute": "x",
                "replicate": replicate,
                "score": effect + (0.1 if replicate else -0.1),
            }
            for pid in range(5)
            for product, effect in (("A", 0.0), ("B", 9.0), ("C", 4.0))
            for replicate in range(2)
        ]
        frame = pd.DataFrame(rows)
        frame["session"] = 1
        table = assessor_variance_equality(frame[PANEL_COLUMNS]).set_index("attribute")
        assert table.loc["x", "spread_ratio_max_min"] == pytest.approx(1.0)

    def test_single_assessor_gives_nan_rather_than_an_error(self) -> None:
        rows = [
            {"panelist_id": "p0", "product": product, "attribute": "x", "replicate": r, "score": float(r)}
            for product in ("A", "B")
            for r in range(3)
        ]
        frame = pd.DataFrame(rows)
        frame["session"] = 1
        table = assessor_variance_equality(frame[PANEL_COLUMNS]).set_index("attribute")
        assert np.isnan(table.loc["x", "p_equal_variance"])
        assert table.loc["x", "n_assessors"] == 1

    def test_empty_panel_raises(self) -> None:
        with pytest.raises(ValueError, match="no rows"):
            assessor_variance_equality(pd.DataFrame(columns=PANEL_COLUMNS))
