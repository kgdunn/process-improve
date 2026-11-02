"""Direct unit tests for ``align_with_path`` DTW averaging (issue #197).

When several samples of a batch map to the same reference index (a compression
in the warping path), the synced value for that index must be the average of
*those batch samples only*. The previous implementation seeded the running
average from an ``initial_row`` argument (a reference row in one caller, an
out-of-space batch row in the other), so the row-0 average was contaminated by
an unrelated row. These tests pin the corrected averaging.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from process_improve.batch.preprocessing import align_with_path


def test_compression_at_first_index_averages_only_batch_rows() -> None:
    batch = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [1.0, 3.0, 5.0]})
    # Reference index 0 receives batch samples 0 and 1 (a compression); reference
    # index 1 receives batch sample 2. Columns are [reference_index, test_index].
    md_path = np.array([[0, 0], [0, 1], [1, 2]])

    synced = align_with_path(md_path=md_path, batch=batch)

    assert synced.shape == (2, 2)
    # Row 0 = mean of batch rows 0 and 1; previously this leaked the seed row in.
    np.testing.assert_allclose(synced.iloc[0].to_numpy(), [15.0, 2.0])
    np.testing.assert_allclose(synced.iloc[1].to_numpy(), [30.0, 5.0])


def test_one_to_one_path_is_a_passthrough() -> None:
    batch = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
    md_path = np.array([[0, 0], [1, 1], [2, 2]])

    synced = align_with_path(md_path=md_path, batch=batch)

    np.testing.assert_allclose(synced["a"].to_numpy(), [10.0, 20.0, 30.0])


def _reference_align_with_path(md_path: np.ndarray, batch: pd.DataFrame) -> pd.DataFrame:
    """Align the way ``align_with_path`` used to, one pandas row at a time; kept as an oracle."""
    numeric = batch.select_dtypes(include="number")
    passthrough = batch.columns.difference(numeric.columns, sort=False)
    row = 0
    nr = md_path[:, 0].max() + 1
    synced = pd.DataFrame(np.zeros((nr, numeric.shape[1])), columns=numeric.columns)
    synced.iloc[row, :] = numeric.iloc[md_path[0, 1], :]
    temp = numeric.iloc[md_path[0, 1], :]
    for idx in np.arange(1, md_path.shape[0]):
        if md_path[idx, 0] != md_path[idx - 1, 0]:
            row += 1
            synced.iloc[row, :] = temp = numeric.iloc[md_path[idx, 1], :]
        else:
            temp = np.vstack((temp, numeric.iloc[md_path[idx, 1], :]))
            synced.iloc[row, :] = np.nanmean(temp, axis=0)
    for column in passthrough:
        synced[column] = batch[column].iloc[0]
    return pd.DataFrame(synced[batch.columns])


def _random_dtw_path(rng: np.random.Generator, n_ref: int, n_test: int) -> np.ndarray:
    """Return a valid warping path from (0, 0) to the far corner in unit steps."""
    steps = [(0, 0)]
    r = t = 0
    while (r, t) != (n_ref - 1, n_test - 1):
        moves = [m for m in ((1, 0), (0, 1), (1, 1)) if r + m[0] < n_ref and t + m[1] < n_test]
        dr, dt = moves[rng.integers(len(moves))]
        r, t = r + dr, t + dt
        steps.append((r, t))
    return np.array(steps)


@pytest.mark.parametrize("seed", range(25))
def test_the_vectorised_alignment_equals_the_row_at_a_time_one_exactly(seed: int) -> None:
    """Same numbers to the last bit, NaN cells, integer columns and an identifier column included."""
    rng = np.random.default_rng(seed)
    n_ref, n_test = int(rng.integers(2, 80)), int(rng.integers(2, 80))
    batch = pd.DataFrame(rng.normal(size=(n_test, 3)) * 10.0 ** rng.uniform(-4, 4), columns=["a", "b", "c"])
    batch.loc[rng.random(n_test) < 0.1, "b"] = np.nan
    batch["counts"] = rng.integers(0, 100, size=n_test)
    batch["batch_id"] = "B7"
    md_path = _random_dtw_path(rng, n_ref, n_test)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean of an all-NaN group, on both sides
        expected = _reference_align_with_path(md_path, batch)
        actual = align_with_path(md_path=md_path, batch=batch)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
