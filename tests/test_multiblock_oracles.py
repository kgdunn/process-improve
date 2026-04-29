"""
Self-consistency tests for the multi-block reference oracles.

These tests prove that the two independent reference implementations
(``merged-then-recover`` and ``full multiblock NIPALS``) produce numerically
equivalent super-scores, block-scores, weights and loadings on a small
synthetic problem.

If both reference paths agree, we have a trustworthy Python oracle that the
production :class:`MBPCA` and :class:`MBPLS` classes can be validated against
when they land in PR3 / PR6 - without ever needing to run MATLAB.

The MATLAB ``unit_tests.m`` MBPCA_tests / MBPLS_tests blocks structure their
self-consistency checks the same way; this module is the Python equivalent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._multiblock_oracles import (
    mbpca_full_multiblock,
    mbpca_merged_then_recover,
    mbpls_full_multiblock,
    mbpls_merged_then_recover,
)


def _preprocess(matrix: np.ndarray) -> np.ndarray:
    """Mean-centre and unit-variance scale (ddof=1) - matches MCUVScaler."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, ddof=1, keepdims=True)
    scale[scale == 0] = 1.0
    return centred / scale


@pytest.fixture
def two_block_synthetic() -> tuple[list[np.ndarray], np.ndarray]:
    """Two correlated X-blocks plus a Y-block driven by a shared latent factor."""
    rng = np.random.default_rng(42)
    n_rows = 50
    latent = rng.standard_normal((n_rows, 2))

    block_a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((n_rows, 6))
    block_b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((n_rows, 4))
    y_block = latent @ rng.standard_normal((2, 2)) + 0.05 * rng.standard_normal((n_rows, 2))

    return [_preprocess(block_a), _preprocess(block_b)], _preprocess(y_block)


# ---------------------------------------------------------------------------
# MBPCA self-consistency
# ---------------------------------------------------------------------------


class TestMBPCAOracleSelfConsistency:
    """Both reference paths must agree on every quantity (up to a global sign)."""

    @pytest.fixture
    def fits(self, two_block_synthetic):
        blocks, _ = two_block_synthetic
        n_components = 2
        merged = mbpca_merged_then_recover(blocks, n_components)
        full = mbpca_full_multiblock(blocks, n_components)
        return merged, full

    def test_super_scores_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(np.abs(merged.super_scores), np.abs(full.super_scores), decimal=6)

    def test_super_loadings_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(np.abs(merged.super_loadings), np.abs(full.super_loadings), decimal=6)

    def test_block_scores_agree(self, fits) -> None:
        merged, full = fits
        for b in range(len(merged.block_scores)):
            np.testing.assert_array_almost_equal(
                np.abs(merged.block_scores[b]), np.abs(full.block_scores[b]), decimal=6
            )

    def test_block_loadings_agree(self, fits) -> None:
        merged, full = fits
        for b in range(len(merged.block_loadings)):
            np.testing.assert_array_almost_equal(
                np.abs(merged.block_loadings[b]), np.abs(full.block_loadings[b]), decimal=6
            )

    def test_block_score_summary_agrees(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(
            np.abs(merged.block_score_summary), np.abs(full.block_score_summary), decimal=6
        )


# ---------------------------------------------------------------------------
# MBPLS self-consistency
# ---------------------------------------------------------------------------


class TestMBPLSOracleSelfConsistency:
    """Both reference paths must agree on every quantity (up to a global sign).

    Y predictions are sign-invariant (they involve the product of two
    sign-flipped quantities) so the comparison there is direct, not absolute.
    """

    @pytest.fixture
    def fits(self, two_block_synthetic):
        blocks, y_block = two_block_synthetic
        n_components = 2
        merged = mbpls_merged_then_recover(blocks, y_block, n_components)
        full = mbpls_full_multiblock(blocks, y_block, n_components)
        return merged, full

    def test_super_scores_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(np.abs(merged.super_scores), np.abs(full.super_scores), decimal=6)

    def test_super_y_scores_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(
            np.abs(merged.super_y_scores), np.abs(full.super_y_scores), decimal=6
        )

    def test_super_weights_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(np.abs(merged.super_weights), np.abs(full.super_weights), decimal=6)

    def test_super_y_loadings_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(
            np.abs(merged.super_y_loadings), np.abs(full.super_y_loadings), decimal=6
        )

    def test_block_weights_agree(self, fits) -> None:
        merged, full = fits
        for b in range(len(merged.block_weights)):
            np.testing.assert_array_almost_equal(
                np.abs(merged.block_weights[b]), np.abs(full.block_weights[b]), decimal=6
            )

    def test_block_loadings_agree(self, fits) -> None:
        merged, full = fits
        for b in range(len(merged.block_loadings)):
            np.testing.assert_array_almost_equal(
                np.abs(merged.block_loadings[b]), np.abs(full.block_loadings[b]), decimal=6
            )

    def test_y_predictions_agree(self, fits) -> None:
        merged, full = fits
        np.testing.assert_array_almost_equal(merged.y_predictions, full.y_predictions, decimal=6)


# ---------------------------------------------------------------------------
# Sanity: super-weights are unit norm
# ---------------------------------------------------------------------------


class TestOracleSanity:
    """Light sanity checks that hold by construction."""

    def test_mbpca_super_loadings_unit_norm(self, two_block_synthetic) -> None:
        blocks, _ = two_block_synthetic
        result = mbpca_full_multiblock(blocks, n_components=2)
        norms = np.linalg.norm(result.super_loadings, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)

    def test_mbpls_super_weights_unit_norm(self, two_block_synthetic) -> None:
        blocks, y_block = two_block_synthetic
        result = mbpls_full_multiblock(blocks, y_block, n_components=2)
        norms = np.linalg.norm(result.super_weights, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)

    def test_mbpls_block_weights_unit_norm(self, two_block_synthetic) -> None:
        blocks, y_block = two_block_synthetic
        result = mbpls_full_multiblock(blocks, y_block, n_components=2)
        for w in result.block_weights:
            norms = np.linalg.norm(w, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)
