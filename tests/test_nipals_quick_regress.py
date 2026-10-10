"""The complete-data fast path of ``quick_regress`` must equal the masked loop exactly.

``_quick_regress_complete`` exists only for speed. Any difference from
``_quick_regress_masked`` on complete data, even in the last bit, would change
fitted models, so the comparison is ``np.array_equal``, never ``allclose``.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.multivariate._common import epsqrt
from process_improve.multivariate._nipals import (
    _quick_regress_complete,
    _quick_regress_masked,
    quick_regress,
    ssq,
)


def _floor(x: np.ndarray) -> float:
    return epsqrt * float(ssq(x))


@pytest.mark.parametrize("seed", range(40))
def test_the_fast_path_equals_the_masked_loop_bit_for_bit(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = rng.integers(1, 300), rng.integers(1, 30)
    # Magnitudes from 1e-6 to 1e6, so rounding differences would have room to show.
    y = rng.normal(size=(n_rows, n_cols)) * 10.0 ** rng.uniform(-6, 6)

    x_a = rng.normal(size=(n_rows, 1))  # Case A: one coefficient per column
    fast_a = _quick_regress_complete(y, x_a, _floor(x_a))
    assert np.array_equal(fast_a, _quick_regress_masked(y, x_a, _floor(x_a)))
    assert fast_a.shape == (n_cols, 1)

    x_b = rng.normal(size=(n_cols, 1))  # Case B: one coefficient per row
    fast_b = _quick_regress_complete(y, x_b, _floor(x_b))
    assert np.array_equal(fast_b, _quick_regress_masked(y, x_b, _floor(x_b)))
    assert fast_b.shape == (n_rows, 1)


def test_a_non_contiguous_view_gives_the_same_answer() -> None:
    """NIPALS hands in slices; the fast path must not depend on memory layout."""
    rng = np.random.default_rng(1)
    block = rng.normal(size=(40, 12))
    view = block[::2, ::3]  # neither C- nor F-contiguous
    x = rng.normal(size=(view.shape[0], 1))
    assert np.array_equal(
        _quick_regress_complete(view, x, _floor(x)),
        _quick_regress_masked(np.ascontiguousarray(view), x, _floor(x)),
    )


def test_an_all_zero_x_gives_zero_coefficients_on_both_paths() -> None:
    y = np.arange(12.0).reshape(4, 3)
    x = np.zeros((4, 1))
    assert np.array_equal(quick_regress(y, x), np.zeros((3, 1)))
    assert np.array_equal(_quick_regress_masked(y, x, _floor(x)), np.zeros((3, 1)))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_missing_or_infinite_cell_takes_the_masked_path(bad: float) -> None:
    """The fast path assumes every cell is finite; anything else must use the loop."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=(10, 4))
    y[3, 1] = bad
    x = rng.normal(size=(10, 1))
    assert np.array_equal(quick_regress(y, x), _quick_regress_masked(y, x, _floor(x)), equal_nan=True)


def test_incompatible_shapes_raise_on_both_paths() -> None:
    y = np.ones((5, 3))
    x = np.ones((4, 1))
    with pytest.raises(ValueError, match="not compatible"):
        _quick_regress_complete(y, x, _floor(x))
    with pytest.raises(ValueError, match="not compatible"):
        _quick_regress_masked(y, x, _floor(x))
