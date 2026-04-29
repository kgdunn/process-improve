"""
Reference (oracle) implementations of multi-block PCA and PLS in pure numpy.

These are *not* the production implementations. They are independent reference
implementations used by the test suite to validate the future :class:`MBPCA`
and :class:`MBPLS` classes.

The two reference algorithms
----------------------------
For both MBPCA and MBPLS, the reference computes the model two ways:

1. **Merged-then-recover.** Concatenate all blocks column-wise after dividing
   each block by ``sqrt(K_b)`` (equal block weighting). Run an ordinary
   single-block PCA / PLS on the concatenated matrix. Then back-extract the
   per-block scores, weights and loadings analytically.

2. **Full multiblock NIPALS.** The hierarchical algorithm of Wold/Westerhuis:
   alternately compute block scores, then a super-score, then deflate.

If the multi-block PR ports are correct, they must agree with both reference
paths to numerical precision.

Notation
--------
- ``B`` blocks, each of shape ``(N, K_b)``. All blocks share ``N``.
- ``A`` is the number of latent variables.
- Block scaling factor: ``sqrt(K_b)`` (Westerhuis & Smilde convention).
- Sign convention: largest-magnitude element of every loading vector is
  positive (Wold, Esbensen & Geladi 1987).

References
----------
- Westerhuis, J. A., Kourti, T. & MacGregor, J. F. "Analysis of multiblock and
  hierarchical PCA and PLS models." Journal of Chemometrics, 12 (1998),
  301-321. https://doi.org/10.1002/(SICI)1099-128X(199809/10)12:5<301::AID-CEM515>3.0.CO;2-S
- Westerhuis, J. A. & Smilde, A. K. "Deflation in multiblock PLS." Journal of
  Chemometrics, 15 (2001), 485-493. https://doi.org/10.1002/cem.652
- Wold, S., Esbensen, K. & Geladi, P. "Principal Component Analysis."
  Chemometrics and Intelligent Laboratory Systems, 2 (1987), 37-52.

Ported from the legacy ConnectMV ``unit_tests.m`` MBPCA_tests / MBPLS_tests
self-consistency blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flip_sign_largest_magnitude_positive(loading: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign-flip a (loading, score) pair so the largest-|loading| element is positive."""
    idx = int(np.argmax(np.abs(loading)))
    if loading[idx] < 0:
        return -loading, -score
    return loading, score


@dataclass
class MBPCAReference:
    """Container for reference MBPCA results."""

    super_scores: np.ndarray  # (N, A)
    super_loadings: np.ndarray  # (B, A)
    block_scores: list[np.ndarray]  # B entries, each (N, A)
    block_loadings: list[np.ndarray]  # B entries, each (K_b, A)
    block_score_summary: np.ndarray  # (N, B, A) - block scores stacked into the super-block matrix


@dataclass
class MBPLSReference:
    """Container for reference MBPLS results."""

    super_scores: np.ndarray  # (N, A)
    super_y_scores: np.ndarray  # (N, A)
    super_weights: np.ndarray  # (B, A)
    super_y_loadings: np.ndarray  # (M, A)  Y-block loadings (paper calls them c_a)
    block_scores: list[np.ndarray]  # B entries, each (N, A)
    block_weights: list[np.ndarray]  # B entries, each (K_b, A)
    block_loadings: list[np.ndarray]  # B entries, each (K_b, A)
    block_score_summary: np.ndarray  # (N, B, A)
    y_predictions: np.ndarray  # (N, M)
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MBPCA path 1: merged-then-recover
# ---------------------------------------------------------------------------


def mbpca_merged_then_recover(blocks: list[np.ndarray], n_components: int) -> MBPCAReference:
    """Compute MBPCA by running single-block PCA on the column-stacked,
    block-weighted matrix, then back-extracting per-block quantities.

    Each block must already be preprocessed (mean-centred, unit-variance).

    For each latent variable, the per-block recovery operates on the
    *currently-deflated* merged matrix, not on the original. This matches the
    legacy MATLAB ``MBPCA_tests`` self-consistency reference.
    """
    n_blocks = len(blocks)
    n_rows = blocks[0].shape[0]
    block_widths = [int(b.shape[1]) for b in blocks]
    assert all(b.shape[0] == n_rows for b in blocks), "All blocks must share the same row count."

    # Column-concatenate blocks after dividing by sqrt(K_b)
    weighted = [b / np.sqrt(block_widths[i]) for i, b in enumerate(blocks)]
    x_def = np.concatenate(weighted, axis=1)

    super_scores = np.zeros((n_rows, n_components))
    super_loadings = np.zeros((n_blocks, n_components))
    block_scores: list[np.ndarray] = [np.zeros((n_rows, n_components)) for _ in range(n_blocks)]
    block_loadings: list[np.ndarray] = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_score_summary = np.zeros((n_rows, n_blocks, n_components))

    rng = np.random.default_rng(0)
    for a in range(n_components):
        # NIPALS on the (current) merged matrix
        t_a = rng.standard_normal(n_rows)
        prev = t_a * 2
        while np.linalg.norm(prev - t_a) > np.finfo(float).eps ** (2 / 3):
            prev = t_a
            p_a = x_def.T @ t_a / (t_a @ t_a)
            p_a = p_a / np.linalg.norm(p_a)
            t_a = x_def @ p_a / (p_a @ p_a)
        super_scores[:, a] = t_a

        # Recover per-block scores and loadings from the SAME deflated matrix
        start = 0
        for b in range(n_blocks):
            stop = start + block_widths[b]
            # Multiply by sqrt(K_b) so X_portion looks like the original block
            x_portion = x_def[:, start:stop] * np.sqrt(block_widths[b])
            p_b = x_portion.T @ t_a / (t_a @ t_a)
            p_b = p_b / np.linalg.norm(p_b)
            t_b = x_portion @ p_b / (p_b @ p_b) / np.sqrt(block_widths[b])

            p_b, t_b = _flip_sign_largest_magnitude_positive(p_b, t_b)
            block_loadings[b][:, a] = p_b
            block_scores[b][:, a] = t_b
            block_score_summary[:, b, a] = t_b
            start = stop

        # Super-loading: regress columns of the block-score summary onto t_a
        super_loadings[:, a] = block_score_summary[:, :, a].T @ t_a / (t_a @ t_a)

        # Deflate the merged matrix for the next component
        x_def = x_def - np.outer(t_a, p_a)

    return MBPCAReference(
        super_scores=super_scores,
        super_loadings=super_loadings,
        block_scores=block_scores,
        block_loadings=block_loadings,
        block_score_summary=block_score_summary,
    )


# ---------------------------------------------------------------------------
# MBPCA path 2: full multiblock NIPALS
# ---------------------------------------------------------------------------


def mbpca_full_multiblock(blocks: list[np.ndarray], n_components: int) -> MBPCAReference:
    """Compute MBPCA using the hierarchical NIPALS loop (Westerhuis 1998)."""
    n_blocks = len(blocks)
    n_rows = blocks[0].shape[0]
    block_widths = [int(b.shape[1]) for b in blocks]
    sqrt_kb = [np.sqrt(k) for k in block_widths]
    assert all(b.shape[0] == n_rows for b in blocks)

    x_def: list[np.ndarray] = [b.copy() for b in blocks]

    super_scores = np.zeros((n_rows, n_components))
    super_loadings = np.zeros((n_blocks, n_components))
    block_scores = [np.zeros((n_rows, n_components)) for _ in range(n_blocks)]
    block_loadings = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_score_summary = np.zeros((n_rows, n_blocks, n_components))

    rng = np.random.default_rng(0)
    for a in range(n_components):
        t_super = rng.standard_normal(n_rows)
        prev = t_super * 2
        # Tightened tolerance per the legacy MATLAB MBPCA self-consistency block.
        while np.linalg.norm(prev - t_super) > np.finfo(float).eps ** (9 / 10):
            prev = t_super
            t_b_summary = np.zeros((n_rows, n_blocks))
            local_loadings: list[np.ndarray] = []
            local_scores: list[np.ndarray] = []
            for b in range(n_blocks):
                p_b = x_def[b].T @ t_super / (t_super @ t_super)
                p_b = p_b / np.linalg.norm(p_b)
                t_b = x_def[b] @ p_b / (p_b @ p_b) / sqrt_kb[b]
                local_loadings.append(p_b)
                local_scores.append(t_b)
                t_b_summary[:, b] = t_b

            p_s = t_b_summary.T @ t_super / (t_super @ t_super)
            p_s = p_s / np.linalg.norm(p_s)
            t_super = t_b_summary @ p_s / (p_s @ p_s)

        # Deflate each block using the super-score and the block loading
        for b in range(n_blocks):
            p_deflate = local_loadings[b] * p_s[b] * sqrt_kb[b]
            x_def[b] = x_def[b] - np.outer(t_super, p_deflate)
            block_loadings[b][:, a], block_scores[b][:, a] = _flip_sign_largest_magnitude_positive(
                local_loadings[b], local_scores[b]
            )
            block_score_summary[:, b, a] = block_scores[b][:, a]

        super_scores[:, a] = t_super
        super_loadings[:, a] = p_s

    return MBPCAReference(
        super_scores=super_scores,
        super_loadings=super_loadings,
        block_scores=block_scores,
        block_loadings=block_loadings,
        block_score_summary=block_score_summary,
    )


# ---------------------------------------------------------------------------
# MBPLS path 1: merged-then-recover
# ---------------------------------------------------------------------------


def mbpls_merged_then_recover(  # noqa: PLR0915
    blocks: list[np.ndarray], y_block: np.ndarray, n_components: int
) -> MBPLSReference:
    """Compute MBPLS by running single-block PLS on the column-stacked,
    block-weighted matrix, then back-extracting per-block quantities.

    Both ``blocks`` and ``y_block`` must already be preprocessed.
    """
    n_blocks = len(blocks)
    n_rows = blocks[0].shape[0]
    block_widths = [int(b.shape[1]) for b in blocks]
    sqrt_kb = [np.sqrt(k) for k in block_widths]
    n_y_cols = y_block.shape[1]

    weighted = [b / sqrt_kb[i] for i, b in enumerate(blocks)]
    x_def = np.concatenate(weighted, axis=1)
    y_def = y_block.copy()

    super_scores = np.zeros((n_rows, n_components))
    super_y_scores = np.zeros((n_rows, n_components))
    super_y_loadings = np.zeros((n_y_cols, n_components))
    super_weights = np.zeros((n_blocks, n_components))
    p_merged = np.zeros((x_def.shape[1], n_components))
    w_merged = np.zeros((x_def.shape[1], n_components))

    block_scores = [np.zeros((n_rows, n_components)) for _ in range(n_blocks)]
    block_weights = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_loadings = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_score_summary = np.zeros((n_rows, n_blocks, n_components))

    rng = np.random.default_rng(0)
    for a in range(n_components):
        # NIPALS PLS on the (currently-deflated) merged X
        u_a = rng.standard_normal(n_rows)
        prev = u_a * 2
        while np.linalg.norm(prev - u_a) > np.finfo(float).eps ** (6 / 7):
            prev = u_a
            w_a = x_def.T @ u_a / (u_a @ u_a)
            w_a = w_a / np.linalg.norm(w_a)
            t_a = x_def @ w_a / (w_a @ w_a)
            c_a = y_def.T @ t_a / (t_a @ t_a)
            u_a = y_def @ c_a / (c_a @ c_a)
        p_a = x_def.T @ t_a / (t_a @ t_a)

        # Recover per-block quantities from the SAME deflated matrix BEFORE
        # the next deflation step, matching legacy MATLAB MBPLS_tests.
        start = 0
        for b in range(n_blocks):
            stop = start + block_widths[b]
            x_portion = x_def[:, start:stop] * sqrt_kb[b]
            w_b = x_portion.T @ u_a / (u_a @ u_a)
            w_b = w_b / np.linalg.norm(w_b)
            t_b = x_portion @ w_b / (w_b @ w_b) / sqrt_kb[b]
            p_b = x_portion.T @ t_a / (t_a @ t_a)

            block_weights[b][:, a] = w_b
            block_scores[b][:, a] = t_b
            block_loadings[b][:, a] = p_b
            block_score_summary[:, b, a] = t_b
            start = stop

        w_s = block_score_summary[:, :, a].T @ u_a / (u_a @ u_a)
        super_weights[:, a] = w_s / np.linalg.norm(w_s)

        super_scores[:, a] = t_a
        super_y_scores[:, a] = u_a
        super_y_loadings[:, a] = c_a
        p_merged[:, a] = p_a
        w_merged[:, a] = w_a

        # Deflate AFTER recovery
        x_def = x_def - np.outer(t_a, p_a)
        y_def = y_def - np.outer(t_a, c_a)

    y_predictions = super_scores @ super_y_loadings.T

    return MBPLSReference(
        super_scores=super_scores,
        super_y_scores=super_y_scores,
        super_weights=super_weights,
        super_y_loadings=super_y_loadings,
        block_scores=block_scores,
        block_weights=block_weights,
        block_loadings=block_loadings,
        block_score_summary=block_score_summary,
        y_predictions=y_predictions,
        extras={"big_x_loadings": p_merged, "big_x_weights": w_merged},
    )


# ---------------------------------------------------------------------------
# MBPLS path 2: full multiblock NIPALS
# ---------------------------------------------------------------------------


def mbpls_full_multiblock(
    blocks: list[np.ndarray], y_block: np.ndarray, n_components: int
) -> MBPLSReference:
    """Compute MBPLS using the hierarchical NIPALS loop (Westerhuis & Smilde 2001)."""
    n_blocks = len(blocks)
    n_rows = blocks[0].shape[0]
    block_widths = [int(b.shape[1]) for b in blocks]
    sqrt_kb = [np.sqrt(k) for k in block_widths]
    n_y_cols = y_block.shape[1]

    x_def = [b.copy() for b in blocks]
    y_def = y_block.copy()

    super_scores = np.zeros((n_rows, n_components))
    super_y_scores = np.zeros((n_rows, n_components))
    super_weights = np.zeros((n_blocks, n_components))
    super_y_loadings = np.zeros((n_y_cols, n_components))
    block_scores = [np.zeros((n_rows, n_components)) for _ in range(n_blocks)]
    block_weights = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_loadings = [np.zeros((block_widths[i], n_components)) for i in range(n_blocks)]
    block_score_summary = np.zeros((n_rows, n_blocks, n_components))

    rng = np.random.default_rng(0)
    for a in range(n_components):
        u_a = rng.standard_normal(n_rows)
        prev = u_a * 2
        local_w: list[np.ndarray] = []
        local_t: list[np.ndarray] = []
        t_b_summary = np.zeros((n_rows, n_blocks))
        while np.linalg.norm(prev - u_a) > np.finfo(float).eps ** (6 / 7):
            prev = u_a
            local_w = []
            local_t = []
            for b in range(n_blocks):
                w_b = x_def[b].T @ u_a / (u_a @ u_a)
                w_b = w_b / np.linalg.norm(w_b)
                t_b = x_def[b] @ w_b / (w_b @ w_b) / sqrt_kb[b]
                local_w.append(w_b)
                local_t.append(t_b)
                t_b_summary[:, b] = t_b

            w_s = t_b_summary.T @ u_a / (u_a @ u_a)
            w_s = w_s / np.linalg.norm(w_s)
            t_super = t_b_summary @ w_s / (w_s @ w_s)
            c_a = y_def.T @ t_super / (t_super @ t_super)
            u_a = y_def @ c_a / (c_a @ c_a)

        # Deflate using the super-score
        for b in range(n_blocks):
            p_b = x_def[b].T @ t_super / (t_super @ t_super)
            x_def[b] = x_def[b] - np.outer(t_super, p_b)
            block_loadings[b][:, a] = p_b
            block_weights[b][:, a] = local_w[b]
            block_scores[b][:, a] = local_t[b]
            block_score_summary[:, b, a] = local_t[b]
        y_def = y_def - np.outer(t_super, c_a)

        super_scores[:, a] = t_super
        super_y_scores[:, a] = u_a
        super_weights[:, a] = w_s
        super_y_loadings[:, a] = c_a

    y_predictions = super_scores @ super_y_loadings.T

    return MBPLSReference(
        super_scores=super_scores,
        super_y_scores=super_y_scores,
        super_weights=super_weights,
        super_y_loadings=super_y_loadings,
        block_scores=block_scores,
        block_weights=block_weights,
        block_loadings=block_loadings,
        block_score_summary=block_score_summary,
        y_predictions=y_predictions,
    )
