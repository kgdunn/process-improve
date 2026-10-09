"""Tests for the DOE Column structure: coding conversions and helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments.structures import c, create_names, gather


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


class TestColumnConstruction:
    """The c() factory across input shapes."""

    def test_accepts_a_numpy_array(self) -> None:
        col = c(np.array([1.0, 2.0, 3.0, 4.0]))
        assert len(col) == 4
        assert list(col.values) == [1.0, 2.0, 3.0, 4.0]

    def test_categorical_values_without_explicit_levels(self) -> None:
        """Non-numeric values infer their levels from the unique entries."""
        col = c(["low", "high", "low", "high"])
        assert set(col.pi_levels[col.pi_name]) == {"low", "high"}

    def test_non_iterable_range_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="iterable"):
            c(1, 2, 3, 4, range=99)

    def test_a_series_keeps_its_own_index(self) -> None:
        """A pandas Series brings its index along unless ``index=`` overrides it."""
        col = c(pd.Series([1, 2], index=[10, 11]))
        assert list(col.index) == [10, 11]
        assert list(col.values) == [1.0, 2.0]

    @pytest.mark.parametrize(
        ("args", "kwargs", "error", "message"),
        [
            pytest.param(
                (1, 2),
                {"range": (0, 1, 2)},
                ValueError,
                "The `range` variable must be a tuple with 2 values; got 3 value(s).",
                id="range-of-three-values",
            ),
            pytest.param(
                ("a", "b"),
                {"levels": 5},
                TypeError,
                "Levels must be list or tuple of the unique level names.",
                id="levels-not-iterable",
            ),
        ],
    )
    def test_malformed_metadata_is_rejected(self, args: tuple, kwargs: dict, error: type, message: str) -> None:
        """A ``range`` or ``levels`` that cannot describe the column is refused with a message saying why."""
        with pytest.raises(error, match=f"^{re.escape(message)}$"):
            c(*args, **kwargs)


class TestColumnMetadata:
    """Coding conversions and extension at the edges of the metadata."""

    def test_an_unnamed_column_keeps_its_empty_name_when_coded(self) -> None:
        """Only a named column gets the ' [coded]' suffix."""
        coded = c(4, 5, 6, lo=4, hi=6, name="").to_coded()
        assert list(coded.values) == [-1.0, 0.0, 1.0]
        assert coded.name == ""

    def test_a_categorical_column_cannot_be_converted_to_real_world_units(self) -> None:
        """A ``levels=`` column has no center or range, so the affine map is undefined."""
        col = c(["Dry", "Wet", "Dry"], levels=("Dry", "Wet"))
        with pytest.raises(ValueError, match=r"^Cannot convert between coded and real-world units: no center/range"):
            col.to_realworld()

    def test_extend_requires_a_list(self) -> None:
        with pytest.raises(TypeError, match=r"^'values' must be a list; got tuple\.$"):
            c(-1, 0, 1, name="Temp").extend((1, 2))


class TestGather:
    """Collecting columns into an ``Expt``."""

    def test_a_plain_list_becomes_a_column(self) -> None:
        expt = gather(y=[1.0, 2.0, 3.0])
        assert list(expt.columns) == ["y"]
        assert list(expt["y"]) == [1.0, 2.0, 3.0]

    def test_a_name_given_twice_is_refused(self) -> None:
        """A positional column named 'A' and a keyword 'A' would overwrite each other."""
        a = c(1, 2, 3, name="A")
        with pytest.raises(ValueError, match=r"^Duplicate column name 'A' in gather\(\)\.$"):
            gather(a, A=a)
