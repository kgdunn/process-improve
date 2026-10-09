"""``DataFrameDict``: its construction guards, item access, and equality semantics.

``DataFrameDict`` subclasses ``dict`` but stores all of its data in the
``self.datadict`` instance attribute, leaving the inherited ``dict`` base
empty. Before #343 it inherited ``dict.__eq__`` / ``dict.__ne__``, which
compared the (always-empty) base and therefore reported *every* instance as
equal regardless of the data it held. The equality tests pin the value-based
behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import DataFrameDict


def _make(value: float, n: int = 4) -> DataFrameDict:
    """Build a small DataFrameDict whose F block carries the given value."""
    rng = np.random.default_rng(0)
    return DataFrameDict(
        {
            "F": {"main": pd.DataFrame({"f1": [value] * n, "f2": rng.standard_normal(n)})},
            "Z": {"conds": pd.DataFrame({"z1": [value] * n})},
            "Y": {"out": pd.DataFrame({"y1": [value] * n})},
        }
    )


class TestDataFrameDictEquality:
    def test_equal_data_compares_equal(self) -> None:
        # Exercise both operators explicitly: __ne__ is a separate method.
        assert (_make(1.0) == _make(1.0)) is True
        assert (_make(1.0) != _make(1.0)) is False

    def test_differing_data_compares_unequal(self) -> None:
        a, b = _make(1.0), _make(999.0)
        # The bug in #343: these used to compare *equal* (empty dict bases).
        assert (a != b) is True
        assert (a == b) is False

    def test_differing_block_structure_compares_unequal(self) -> None:
        full = _make(1.0)
        # Same F/Y data but an extra Z group -> different structure.
        extra = DataFrameDict(
            {
                "F": {"main": pd.DataFrame({"f1": [1.0] * 4, "f2": np.random.default_rng(0).standard_normal(4)})},
                "Z": {"conds": pd.DataFrame({"z1": [1.0] * 4}), "extra": pd.DataFrame({"z2": [2.0] * 4})},
                "Y": {"out": pd.DataFrame({"y1": [1.0] * 4})},
            }
        )
        assert full != extra

    def test_identity_is_equal(self) -> None:
        a = _make(1.0)
        same = a  # alias the same object to hit the `self is other` fast path
        assert (a == same) is True

    def test_unrelated_type_is_not_equal(self) -> None:
        a = _make(1.0)
        assert (a != "not a DataFrameDict") is True
        assert (a != {"F": {}, "Z": {}, "Y": {}}) is True
        assert (a == 42) is False

    def test_remains_unhashable(self) -> None:
        # Defining __eq__ must not accidentally make instances hashable; like
        # the dict base they must stay unhashable so they cannot be silently
        # used as set members or dict keys. Assert the contract at the class
        # level (__hash__ is None) rather than calling hash() on a known-
        # unhashable instance, which CodeQL flags as py/hash-unhashable-value.
        assert DataFrameDict.__hash__ is None


class TestDataFrameDictConstruction:
    @pytest.mark.parametrize(
        ("blocks", "error", "message"),
        [
            (
                {"F": {"g": np.ones((4, 2))}},
                TypeError,
                r"^Expected a DataFrame for block F, group 'g'; got ndarray\.$",
            ),
            (
                {"F": {"g": pd.DataFrame(np.ones((4, 2)))}, "Y": {"y": pd.DataFrame(np.ones((3, 1)))}},
                ValueError,
                r"^DataFrames in block Y must have the same number of rows \(4\)\. Group y has 3 rows\.$",
            ),
        ],
        ids=["group-not-a-dataframe", "group-with-other-row-count"],
    )
    def test_a_malformed_group_is_refused(self, blocks: dict, error: type[Exception], message: str) -> None:
        """Every group must be a DataFrame with as many rows as the first F group."""
        with pytest.raises(error, match=message):
            DataFrameDict(blocks)

    def test_repr_lists_the_groups_of_each_block(self) -> None:
        """The repr counts the samples and names the groups in every block."""
        assert repr(_make(1.0)) == (
            "DataFrameDict with 4 samples and 3 blocks: ['Z', 'F', 'Y']\n"
            "  F groups: ['main']\n"
            "  Z groups: ['conds']\n"
            "  Y groups: ['out']"
        )


class TestDataFrameDictItemAccess:
    @pytest.mark.parametrize(
        ("key", "value", "error", "message"),
        [
            (
                "Q",
                pd.DataFrame({"q": [1.0] * 4}),
                KeyError,
                r"Key Q is not a valid partitionable block\. Valid keys are: \['Z', 'F', 'Y'\]",
            ),
            ("Z", "x", TypeError, r"^Expected a DataFrame for key Z, got <class 'str'>\.$"),
            (
                "Z",
                pd.DataFrame({"z1": [1.0] * 3}),
                ValueError,
                r"^DataFrames in block Z must have the same number of rows \(4\)\. Provided DataFrame has 3 rows\.$",
            ),
        ],
        ids=["unknown-block", "not-a-dataframe", "other-row-count"],
    )
    def test_setting_a_block_refuses_what_does_not_fit(
        self, key: str, value: object, error: type[Exception], message: str
    ) -> None:
        """Only Z, F and Y can be set, and only to a DataFrame with the same number of rows."""
        with pytest.raises(error, match=message):
            _make(1.0)[key] = value

    def test_setting_a_valid_block_stores_the_frame(self) -> None:
        """A frame that fits replaces the block as given, and equality sees the change."""
        frame = pd.DataFrame({"z9": [5.0] * 4})
        changed = _make(1.0)
        changed["Z"] = frame
        assert changed["Z"] is frame
        # A bare frame is a different kind of value from a dict of frames, so never equal.
        assert changed != _make(1.0)

    @pytest.mark.parametrize(
        ("lookup", "rows"),
        [(0, [0]), (np.int64(2), [2]), ([0, 2], [0, 2]), (np.array([3, 1]), [3, 1]), (([1, 3], ...), [1, 3])],
        ids=["int", "numpy-int", "list", "array", "tuple-with-ellipsis"],
    )
    def test_a_row_lookup_selects_the_same_rows_in_every_group(self, lookup: object, rows: list[int]) -> None:
        """Integer-like lookups select rows by position, in every group of every block."""
        source = _make(1.0)
        subset = source[lookup]
        assert isinstance(subset, DataFrameDict)
        assert len(subset) == len(rows)
        for block in ("F", "Z", "Y"):
            for group, frame in source[block].items():
                pd.testing.assert_frame_equal(subset[block][group], frame.iloc[rows])

    @pytest.mark.parametrize(
        ("lookup", "message"),
        [
            (([0, 1], 5), r"^Invalid tuple structure for lookup: \(\[0, 1\], 5\)$"),
            (1.5, r"^Lookup must be an int, list of ints, or a string\. Got 1\.5; <class 'float'>$"),
        ],
        ids=["tuple-without-ellipsis", "float"],
    )
    def test_an_unsupported_lookup_is_refused(self, lookup: object, message: str) -> None:
        """A tuple lookup must end in an Ellipsis, and a scalar lookup must be an integer."""
        with pytest.raises(TypeError, match=message):
            _make(1.0)[lookup]
