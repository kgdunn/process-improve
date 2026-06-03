"""Equality semantics for ``DataFrameDict`` (regression test for #343).

``DataFrameDict`` subclasses ``dict`` but stores all of its data in the
``self.datadict`` instance attribute, leaving the inherited ``dict`` base
empty. Before #343 it inherited ``dict.__eq__`` / ``dict.__ne__``, which
compared the (always-empty) base and therefore reported *every* instance as
equal regardless of the data it held. These tests pin the value-based
behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
