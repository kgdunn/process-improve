"""Tests for the DOE Column structure: coding conversions and helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import lm
from process_improve.experiments.structures import Expt, c, create_names, gather


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

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            (([1, 2], [3, 4]), [1.0, 2.0, 3.0, 4.0]),
            ((1, [2, 3]), [1.0, 2.0, 3.0]),
            (((1, 2), np.array([[3.0], [4.0]])), [1.0, 2.0, 3.0, 4.0]),
        ],
    )
    def test_concatenates_every_argument(self, args: tuple, expected: list) -> None:
        """Like R's c(), each argument adds its entries; none replaces the earlier ones (#513)."""
        assert list(c(*args).values) == expected

    def test_strings_are_entries_not_iterables(self) -> None:
        """Each string is one entry, with or without levels (it used to give an empty column)."""
        assert list(c("Dry", "Wet", "Dry").values) == ["Dry", "Wet", "Dry"]
        moisture = c("Dry", "Wet", "Dry", levels=("Dry", "Wet"))
        assert list(moisture.values) == ["Dry", "Wet", "Dry"]
        assert moisture.pi_levels[moisture.pi_name] == ["Dry", "Wet"]

    def test_a_single_series_keeps_its_index(self) -> None:
        col = c(pd.Series([5, 6], index=["a", "b"]))
        assert list(col.index) == ["a", "b"]
        assert list(col.values) == [5.0, 6.0]

    def test_a_missing_value_keeps_the_column_numeric(self) -> None:
        """``None`` is a missing value, as NaN is."""
        col = c(1, None, 3)
        assert col.pi_numeric
        np.testing.assert_array_equal(col.values, [1.0, np.nan, 3.0])

    def test_numbers_and_text_make_a_categorical_column(self) -> None:
        col = c(0, 1, "green")
        assert not col.pi_numeric
        assert col.pi_levels[col.pi_name] == [0, 1, "green"]

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


class TestMetadataDefaults:
    """``pi_*`` metadata on objects that no factory set up (#513)."""

    def test_a_directly_built_expt_has_no_title(self) -> None:
        """It used to raise AttributeError from repr() and get_title()."""
        expt = Expt({"A": [1, 2]})
        assert expt.get_title() == ""
        assert "Size: 2 experiments" in repr(expt)

    def test_a_model_of_a_directly_built_expt(self) -> None:
        expt = Expt({"A": [-1, 1, -1, 1, 0], "B": [-1, -1, 1, 1, 0], "y": [1.0, 3.0, 2.0, 5.0, 2.6]})
        model = lm("y ~ A*B", expt)
        assert model.get_title() == ""
        model.summary()

    def test_concat_keeps_the_metadata_its_inputs_share(self) -> None:
        first, second = (gather(A=c(-1, 1, name="A"), y=c(5, 6, name="y"), title=title) for title in ("Run", "Run"))
        combined = pd.concat([first, second])
        assert isinstance(combined, Expt)
        assert combined.get_title() == "Run"

    def test_concat_drops_the_metadata_its_inputs_disagree_on(self) -> None:
        first, second = (gather(A=c(-1, 1, name="A"), y=c(5, 6, name="y"), title=title) for title in ("1", "2"))
        assert pd.concat([first, second]).get_title() == ""

    def test_concatenated_columns_keep_their_shared_metadata(self) -> None:
        temperature = pd.concat([c(4, 6, lo=4, hi=6, name="T"), c(5, 6, lo=4, hi=6, name="T")])
        assert (temperature.pi_name, temperature.pi_lo, temperature.pi_hi) == ("T", 4, 6)

    def test_concat_drops_metadata_it_cannot_compare(self) -> None:
        """An array's ``==`` has no single truth value, so concatenating must not raise on it."""
        first, second = c(1, 2, name="T"), c(3, 4, name="T")
        first.pi_range = second.pi_range = np.array([1, 4])
        temperature = pd.concat([first, second])
        assert temperature.pi_range is None
        assert temperature.pi_name == "T"

    def test_levels_survive_slicing(self) -> None:
        """``pi_levels`` is in ``Column._metadata`` now, so a slice keeps it."""
        moisture = c("Dry", "Wet", "Dry")
        assert moisture[:2].pi_levels == moisture.pi_levels


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


def _names_an_argument(message: str, arguments: dict[str, object]) -> bool:
    """Return True if a rejection message identifies one of the arguments, by position, keyword or value."""
    for key, value in arguments.items():
        by_key = re.search(rf"\b(argument|position)\s+{re.escape(key)}\b|'{re.escape(key)}'", message, re.IGNORECASE)
        by_value = re.search(rf"(?<!\w){re.escape(repr(value))}(?!\w)", message)
        if by_key or by_value:
            return True
    return False


def _result_or_type_error(call: Callable[[], Any]) -> tuple[Any, str | None]:
    """Run ``call``; return its result, or ``None`` and the message of the TypeError it raised."""
    try:
        return call(), None
    except TypeError as err:
        return None, str(err)


@pytest.mark.xfail(strict=True, reason="#677: gather() silently drops array and tuple inputs")
@pytest.mark.parametrize(
    "values",
    [pytest.param(np.array([4.0, 5.0, 6.0]), id="numpy-array"), pytest.param((4.0, 5.0, 6.0), id="tuple")],
)
def test_gather_keeps_every_input_or_rejects_it_by_name(values: object) -> None:
    """An array or tuple passed to gather() is a column of the result, or a TypeError names it."""
    a = c(1, 2, 3, name="A")
    expt, rejection = _result_or_type_error(lambda: gather(A=a, x=values))
    if rejection is not None:
        assert _names_an_argument(rejection, {"x": values}), rejection
    else:
        assert list(expt.columns) == ["A", "x"]
        assert list(expt["x"]) == [4.0, 5.0, 6.0]
