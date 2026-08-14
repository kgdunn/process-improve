"""Regression tests for the 2026-08 repo-wide correctness audit: regression + batch.

Each test pins a specific defect found and fixed in the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.regression._robust_regression import robust_regression


class TestRobustRegressionDegenerateX:
    def test_constant_x_gives_finite_uniform_leverage(self) -> None:
        """The degenerate-x branch previously divided ~0 by ~0 for leverage."""
        x = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        y = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
        out = robust_regression(x, y)
        leverage = np.asarray(out["leverage"], dtype=float)
        assert np.isfinite(leverage).all()
        np.testing.assert_allclose(leverage, 1.0 / len(x))


class TestDTWDistance:
    def test_distance_is_the_accumulated_cost(self) -> None:
        """The DTW distance must equal D[-1, -1], not a sum of prefix sums."""
        numba = pytest.importorskip("numba")  # noqa: F841
        from process_improve.batch.alignment_helpers import backtrack_optimal_path, distance_matrix

        rng = np.random.default_rng(0)
        test = rng.standard_normal((20, 2)).cumsum(axis=0)
        ref = rng.standard_normal((25, 2)).cumsum(axis=0)
        weights = np.eye(2)
        D = distance_matrix(test, ref, weights)
        _path, distance = backtrack_optimal_path(D)
        assert distance == pytest.approx(float(D[-1, -1]))

    def test_identical_batches_have_near_zero_distance(self) -> None:
        """Aligning a batch against itself must report ~0 distance.

        Pre-fix, the 'distance' summed the cumulative-cost matrix entries
        along the diagonal path, which is 0 only in exact arithmetic but grows
        with path length as soon as any cost is non-zero; more importantly the
        reported value was not comparable across batches of different length.
        """
        pytest.importorskip("numba")
        from process_improve.batch.preprocessing import dtw_core

        rng = np.random.default_rng(1)
        batch = pd.DataFrame(rng.standard_normal((30, 3)).cumsum(axis=0))
        result = dtw_core(batch, batch, np.eye(3))
        assert result.distance == pytest.approx(0.0, abs=1e-10)


class TestKassidasWeights:
    def test_well_aligned_variable_gets_the_largest_weight(self) -> None:
        """A variable with near-zero deviation SSQ must be up-weighted.

        The previous guard substituted 10000 for a near-zero SSQ, giving the
        BEST-aligned variable a weight of ~1e-4 (the exact opposite of the
        Kassidas weighting).
        """
        pytest.importorskip("numba")
        from process_improve.batch.preprocessing import batch_dtw

        rng = np.random.default_rng(2)
        n_time = 40
        t = np.linspace(0, 1, n_time)
        batches = {}
        for i in range(4):
            frame = pd.DataFrame(
                {
                    # 'clean' has an identical trajectory in every batch.
                    "clean": np.sin(2 * np.pi * t) * 10,
                    # 'noisy' differs per batch.
                    "noisy": np.cos(2 * np.pi * t) * 10 + rng.standard_normal(n_time) * 3,
                }
            )
            batches[f"b{i}"] = frame
        result = batch_dtw(
            batches,
            columns_to_align=["clean", "noisy"],
            reference_batch="b0",
            settings={"show_progress": False, "maximum_iterations": 3},
        )
        weights = result["weight_history"].iloc[-1]
        assert weights["clean"] > weights["noisy"]
