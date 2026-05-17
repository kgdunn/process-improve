"""Tests for the DOE Column structure: coding conversions and helpers."""

from __future__ import annotations

import numpy as np

from process_improve.experiments.structures import c, create_names


class TestCreateNames:
    """Factor-name generation."""

    def test_letters_skip_the_letter_i(self) -> None:
        """The ambiguous letter 'I' is skipped in favour of the next letter."""
        names = create_names(9)
        assert "I" not in names
        assert len(names) == 9
        assert names[:8] == ["A", "B", "C", "D", "E", "F", "G", "H"]

    def test_numeric_names_unpadded(self) -> None:
        assert create_names(3, letters=False, padded=False) == ["X1", "X2", "X3"]


class TestColumnCoding:
    """Round-trips between real-world and coded units."""

    def test_to_coded_maps_range_to_plus_minus_one(self) -> None:
        col = c(4, 5, 6, 4, 6, range=(4, 6))
        coded = col.to_coded()
        assert coded.pi_is_coded is True
        assert np.allclose(coded.values, [-1.0, 0.0, 1.0, -1.0, 1.0])

    def test_to_coded_is_idempotent_when_already_coded(self) -> None:
        col = c(4, 5, 6, 4, 6, range=(4, 6))
        coded_once = col.to_coded()
        coded_twice = coded_once.to_coded()
        assert np.allclose(coded_once.values, coded_twice.values)

    def test_to_realworld_round_trips(self) -> None:
        col = c(4, 5, 6, 4, 6, range=(4, 6))
        restored = col.to_coded().to_realworld()
        assert restored.pi_is_coded is False
        assert np.allclose(restored.values, [4, 5, 6, 4, 6])

    def test_to_realworld_is_idempotent_when_not_coded(self) -> None:
        col = c(4, 5, 6, 4, 6, range=(4, 6))
        assert np.allclose(col.to_realworld().values, col.values)

    def test_extend_appends_values_and_keeps_name(self) -> None:
        col = c(-1, 0, 1, name="Temp")
        extended = col.extend([1, -1])
        assert len(extended) == 5
        assert "Temp" in extended.name
        assert list(extended.values) == [-1, 0, 1, 1, -1]
