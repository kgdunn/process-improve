"""Direct unit tests for ``align_with_path`` DTW averaging (issue #197).

When several samples of a batch map to the same reference index (a compression
in the warping path), the synced value for that index must be the average of
*those batch samples only*. The previous implementation seeded the running
average from an ``initial_row`` argument (a reference row in one caller, an
out-of-space batch row in the other), so the row-0 average was contaminated by
an unrelated row. These tests pin the corrected averaging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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
