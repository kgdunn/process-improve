"""The stored Taguchi arrays reproduce pyDOE3's element for element, so designs do not change.

The equivalence tests need pyDOE3; the structural checks do not.
"""

from __future__ import annotations

import re
from types import ModuleType

import numpy as np
import pytest

from process_improve.experiments import _orthogonal_arrays
from process_improve.experiments.designs_screening import _is_strength_two

NAMES = _orthogonal_arrays.ORTHOGONAL_ARRAY_NAMES


@pytest.fixture(scope="module")
def pydoe3() -> ModuleType:
    return pytest.importorskip("pyDOE3")


@pytest.mark.parametrize("name", NAMES)
def test_array_matches_pydoe3(pydoe3: ModuleType, name: str) -> None:
    np.testing.assert_array_equal(_orthogonal_arrays.orthogonal_array(name), pydoe3.get_orthogonal_array(name))


def test_names_match_pydoe3_in_order_except_the_unbalanced_l64(pydoe3: ModuleType) -> None:
    assert not _is_strength_two(pydoe3.get_orthogonal_array("L64(2^31)"))
    assert [n for n in pydoe3.list_orthogonal_arrays() if n != "L64(2^31)"] == list(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_array_is_balanced_in_every_pair_of_columns(name: str) -> None:
    assert _is_strength_two(_orthogonal_arrays.orthogonal_array(name))


@pytest.mark.parametrize("name", NAMES)
def test_rows_match_the_run_count_and_levels_count_up_from_zero(name: str) -> None:
    array = _orthogonal_arrays.orthogonal_array(name)
    n_runs = int(re.match(r"L(\d+)", name).group(1))  # type: ignore[union-attr]
    assert array.shape[0] == n_runs
    for column in array.T:
        assert set(column) == set(range(column.max() + 1))


def test_names_are_in_order_of_run_count() -> None:
    runs = [_orthogonal_arrays.orthogonal_array(name).shape[0] for name in NAMES]
    assert runs == sorted(runs)


def test_array_is_read_only() -> None:
    array = _orthogonal_arrays.orthogonal_array("L8(2^7)")
    with pytest.raises(ValueError, match="read-only"):
        array[0, 0] = 1


def test_unknown_name_is_refused() -> None:
    with pytest.raises(ValueError, match=r"Unknown orthogonal array 'L64\(2\^31\)'"):
        _orthogonal_arrays.orthogonal_array("L64(2^31)")
