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
from process_improve.multivariate.methods import PCA, PLS


def _floor(x: np.ndarray) -> float:
    return epsqrt * float(ssq(x))


@pytest.mark.parametrize("seed", range(40))
def test_the_fast_path_equals_the_masked_loop_bit_for_bit(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = rng.integers(1, 300), rng.integers(1, 30)
    # Magnitudes from 1e-6 to 1e6, so rounding differences would have room to show.
    y = rng.normal(size=(n_rows, n_cols)) * 10.0 ** rng.uniform(-6, 6)

    x_a = rng.normal(size=(n_rows, 1))  # Case A: one coefficient per column
    fast_a = _quick_regress_complete(y, x_a, _floor(x_a), "column")
    assert np.array_equal(fast_a, _quick_regress_masked(y, x_a, _floor(x_a), "column"))
    assert fast_a.shape == (n_cols, 1)

    x_b = rng.normal(size=(n_cols, 1))  # Case B: one coefficient per row
    fast_b = _quick_regress_complete(y, x_b, _floor(x_b), "row")
    assert np.array_equal(fast_b, _quick_regress_masked(y, x_b, _floor(x_b), "row"))
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


def test_a_square_block_is_regressed_on_the_axis_asked_for() -> None:
    """A square Y fits both cases: ``per`` decides, on both paths, with and without a missing cell."""
    rng = np.random.default_rng(3)
    y = rng.normal(size=(6, 6))
    x = rng.normal(size=(6, 1))
    for block in (y, np.where(np.eye(6, dtype=bool)[::-1], np.nan, y)):
        observed = ~np.isnan(block)
        columns = (np.nan_to_num(block).T @ x) / (observed.T @ x**2)
        rows = (np.nan_to_num(block) @ x) / (observed @ x**2)
        assert np.allclose(quick_regress(block, x, per="column"), columns)
        assert np.allclose(quick_regress(block, x, per="row"), rows)


def test_a_square_block_without_per_is_refused() -> None:
    """The shapes cannot tell the two cases apart, so the call must say which it means."""
    with pytest.raises(ValueError, match="square"):
        quick_regress(np.ones((5, 5)), np.ones((5, 1)))
    with pytest.raises(ValueError, match="not compatible"):
        quick_regress(np.ones((5, 3)), np.ones((3, 1)), per="column")


@pytest.mark.parametrize(("n_rows", "n_x", "n_y"), [(14, 4, 14), (10, 10, 3)])
def test_pls_is_right_when_a_block_has_as_many_rows_as_columns(n_rows: int, n_x: int, n_y: int) -> None:
    """With N equal to K or M, NIPALS took a score as Case A: the first weight was wrong."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(n_rows, n_x))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    y = x @ rng.normal(size=(n_x, n_y)) + rng.normal(size=(n_rows, n_y))
    y = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    w = PLS(n_components=1, scale=False).fit(x, y).x_weights_.to_numpy()[:, 0]
    exact = np.linalg.eigh(x.T @ y @ y.T @ x)[1][:, -1]  # the dominant eigenvector of X'YY'X
    assert abs(w @ exact) == pytest.approx(1.0, abs=1e-8)


def test_nipals_pca_is_right_when_the_block_is_square() -> None:
    """With N equal to K, NIPALS PCA took a score as Case A and its loading was wrong."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 20)) @ rng.normal(size=(20, 20))
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    p = PCA(n_components=1, algorithm="nipals").fit(x).loadings_.to_numpy()[:, 0]
    assert abs(p @ np.linalg.svd(x)[2][0]) == pytest.approx(1.0, abs=1e-8)
