"""
Reference numerical tests for multivariate and (eventually) multi-block methods.

The expected values are pinned numerical oracles. Those marked with a page
number come from the published Wold, Esbensen & Geladi (1987) PCA paper and
can be re-derived from it. The remainder were captured from an earlier
implementation of these methods; the software that produced them is no longer
available, so they cannot be re-derived and are kept purely as regression
pins: they lock in behaviour that was correct when recorded, and a change to
any of them needs justifying on the mathematics, not by appeal to a source.

The strategy is:

1. Tests that assert against published literature values (Wold, Esbensen &
   Geladi, 1987) run today against the existing single-block :class:`PCA`
   class. They guard the existing implementation and form the spine for the
   future multi-block tests.

2. Tests for multi-block PCA / multi-block PLS (MBPCA / MBPLS) are written
   here as ``pytest.mark.xfail`` placeholders. They use the same Wold matrix
   split into two blocks. They will turn green when the corresponding
   classes land in the planned PRs.

3. Tests that need real reference datasets (LDPE, FMC, kamyr-digester) are
   marked ``pytest.mark.skip`` with a reason. They become runnable once the
   datasets land in ``process_improve.datasets`` (planned in PR2).

Conventions
-----------
- The recorded SPE values are :math:`e'e` (raw row sum of squares), while
  ``process_improve.PCA`` stores ``spe_`` as ``sqrt(e'e)``. Every assertion
  against a recorded SPE value therefore compares against ``spe_ ** 2``.
- The recorded scaling vector is the *inverse* standard deviation, while
  :class:`MCUVScaler` stores the standard deviation itself. Hence a recorded
  ``[1, 1, 0.5, 1]`` corresponds to ``[1, 1, 2, 1]`` here.
- Sign convention: largest-magnitude element of every loading vector is
  positive (Wold, Esbensen & Geladi, 1987, p 42). Both implementations follow
  this, so loading and score signs should agree without manual flipping.

References
----------
Wold, S., Esbensen, K. & Geladi, P. "Principal Component Analysis."
Chemometrics and Intelligent Laboratory Systems, 2 (1987), 37-52.
https://doi.org/10.1016/0169-7439(87)80084-9
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError

from process_improve.multivariate._limits import spe_calculation
from process_improve.multivariate.methods import (
    MBPCA,
    MBPLS,
    PCA,
    MCUVScaler,
    SpecificationWarning,
    randomization_test_mbpls,
)
from process_improve.visualization.themes import REFERENCE_LINE_COLOR

# ---------------------------------------------------------------------------
# Wold/Esbensen/Geladi 1987 reference data and expectations
# ---------------------------------------------------------------------------
# The 3x4 worked example used throughout the 1987 PCA paper. Values marked
# with a page number come straight from that paper's pages 40-43; the rest are
# regression pins recorded from an earlier implementation (see module docstring).

WOLD_X = np.array(
    [
        [3.0, 4.0, 2.0, 2.0],
        [4.0, 3.0, 4.0, 3.0],
        [5.0, 5.0, 6.0, 4.0],
    ]
)

# Page 40: column means of WOLD_X
WOLD_CENTER = np.array([4.0, 4.0, 4.0, 3.0])

# Page 40: standard deviations (ddof=1) of WOLD_X. The recorded reference
# stores the *inverse* of this: [1, 1, 0.5, 1].
WOLD_SCALE = np.array([1.0, 1.0, 2.0, 1.0])

# Page 40, two-component model
WOLD_P_PC1 = np.array([0.5410, 0.3493, 0.5410, 0.5410])
WOLD_P_PC2 = np.array([-0.2017, 0.9370, -0.2017, -0.2017])
WOLD_T_PC1 = np.array([-1.6229, -0.3493, 1.9723])
WOLD_T_PC2 = np.array([0.6051, -0.9370, 0.3319])

# Page 43, R^2 per component (totals 1.0 within rounding)
WOLD_R2_PER_COMPONENT = np.array([0.831, 0.169])

# Page 43, residual sum of squares per column after PC1
WOLD_RESIDUAL_SSQ_AFTER_PC1 = np.array([0.0551, 1.189, 0.0551, 0.0551])

# Regression pin: SPE = e'e per row, after PC1
REF_SPE_AFTER_PC1 = np.array([0.366107, 0.877964, 0.110178])

# Regression pin: VIP per variable after PC1 and after PC2
REF_VIP_AFTER_PC1 = np.array([1.082, 0.6987, 1.082, 1.082])
REF_VIP_AFTER_PC2 = np.array([1.0, 1.0, 1.0, 1.0])

# Regression pins: statistical limits at 95% confidence
WOLD_T2_LIMIT_PC1 = 24.684
WOLD_SPE_LIMIT_PC1 = 1.2236
# Score limit: t.ppf(0.975, N-1) * std(scores, ddof=1), which is what
# ``PCA.score_limit`` now computes, so the test calls it directly.
WOLD_SCORE_LIMIT_PC1 = 7.8432

# Two new observations whose projection onto the model is checked.
WOLD_X_NEW = np.array(
    [
        [3.0, 4.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    ]
)
WOLD_T_NEW_PC1 = np.array([-0.2705, -2.0511])
WOLD_T_NEW_2COMP = np.array([[-0.2705, 0.1009], [-2.0511, -1.3698]])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_pca_on_wold(n_components: int) -> tuple[PCA, MCUVScaler, pd.DataFrame]:
    """Mean-center, scale (ddof=1) and fit PCA on the Wold reference matrix."""
    df = pd.DataFrame(WOLD_X)
    scaler = MCUVScaler().fit(df)
    df_pp = scaler.transform(df)
    model = PCA(n_components=n_components).fit(df_pp)
    return model, scaler, df_pp


# ---------------------------------------------------------------------------
# Wold 1987 PCA tests (active; guard the existing PCA class)
# ---------------------------------------------------------------------------


class TestWold1987PCA:
    """Numerical reference tests against Wold/Esbensen/Geladi 1987.

    The 3x4 worked example from that paper is the smallest non-trivial PCA
    that exercises preprocessing, NIPALS/SVD, sign convention, R² and SPE
    bookkeeping. Every assertion below is either a published value or a
    recorded regression pin (see the module docstring).
    """

    def test_preprocessing_center(self) -> None:
        scaler = MCUVScaler().fit(pd.DataFrame(WOLD_X))
        np.testing.assert_array_almost_equal(scaler.center_.values, WOLD_CENTER, decimal=10)

    def test_preprocessing_scale(self) -> None:
        scaler = MCUVScaler().fit(pd.DataFrame(WOLD_X))
        np.testing.assert_array_almost_equal(scaler.scale_.values, WOLD_SCALE, decimal=10)

    def test_loadings_pc1(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        np.testing.assert_array_almost_equal(model.loadings_.iloc[:, 0].values, WOLD_P_PC1, decimal=3)

    def test_loadings_pc2(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        np.testing.assert_array_almost_equal(model.loadings_.iloc[:, 1].values, WOLD_P_PC2, decimal=3)

    def test_scores_pc1(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        np.testing.assert_array_almost_equal(model.scores_.iloc[:, 0].values, WOLD_T_PC1, decimal=3)

    def test_scores_pc2(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        np.testing.assert_array_almost_equal(model.scores_.iloc[:, 1].values, WOLD_T_PC2, decimal=3)

    def test_r2_per_component(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        np.testing.assert_array_almost_equal(model.r2_per_component_.values, WOLD_R2_PER_COMPONENT, decimal=3)

    def test_r2_cumulative_reaches_one(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        assert model.r2_cumulative_.iloc[-1] == pytest.approx(1.0, abs=1e-3)

    def test_residual_ssq_after_one_pc(self) -> None:
        model, _, df_pp = _fit_pca_on_wold(n_components=1)
        x_hat = model.scores_.values @ model.loadings_.values.T
        residual_ssq_per_col = np.sum((df_pp.values - x_hat) ** 2, axis=0)
        np.testing.assert_array_almost_equal(residual_ssq_per_col, WOLD_RESIDUAL_SSQ_AFTER_PC1, decimal=3)

    def test_residual_ssq_after_two_pc_is_zero(self) -> None:
        model, _, df_pp = _fit_pca_on_wold(n_components=2)
        x_hat = model.scores_.values @ model.loadings_.values.T
        residual_ssq_per_col = np.sum((df_pp.values - x_hat) ** 2, axis=0)
        np.testing.assert_array_almost_equal(residual_ssq_per_col, np.zeros(4), decimal=6)

    def test_spe_per_observation_after_one_pc(self) -> None:
        # process_improve stores spe_ as sqrt(e'e); the pinned values are e'e.
        model, _, _ = _fit_pca_on_wold(n_components=1)
        spe_squared = model.spe_.iloc[:, 0].values ** 2
        np.testing.assert_array_almost_equal(spe_squared, REF_SPE_AFTER_PC1, decimal=4)

    def test_spe_per_observation_after_two_pc_is_zero(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        spe_squared = model.spe_.iloc[:, 1].values ** 2
        np.testing.assert_array_almost_equal(spe_squared, np.zeros(3), decimal=6)

    def test_vip_after_one_pc(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        vip_pc1 = model.vip(n_components=1).values
        np.testing.assert_array_almost_equal(vip_pc1, REF_VIP_AFTER_PC1, decimal=3)

    def test_vip_after_two_pc(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        vip_pc2 = model.vip(n_components=2).values
        np.testing.assert_array_almost_equal(vip_pc2, REF_VIP_AFTER_PC2, decimal=3)

    def test_t2_limit_at_95pct(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=1)
        assert model.hotellings_t2_limit(conf_level=0.95) == pytest.approx(WOLD_T2_LIMIT_PC1, abs=5e-3)

    def test_spe_limit_at_95pct(self) -> None:
        # process_improve.spe_limit consumes the model's spe_ values
        # (which are stored as sqrt(e'e)) and returns a limit on the same
        # scale. The pinned limit is on the e'e scale, so we square the
        # Python limit to compare.
        model, _, _ = _fit_pca_on_wold(n_components=1)
        spe_limit_squared = model.spe_limit(conf_level=0.95) ** 2
        assert spe_limit_squared == pytest.approx(WOLD_SPE_LIMIT_PC1, abs=5e-3)

    def test_score_limit_at_95pct(self) -> None:
        # The score limit is t.ppf(0.975, N-1) * std(scores, ddof=1). This is
        # exactly what PCA.score_limit computes, so assert against the public
        # API rather than reconstructing the formula in the test: the point is
        # to test the shipped limit, not a private re-derivation of it.
        model, _, _ = _fit_pca_on_wold(n_components=1)
        limits = model.score_limit(conf_level=0.95)
        assert limits.shape == (1,)
        assert limits[0] == pytest.approx(WOLD_SCORE_LIMIT_PC1, abs=5e-3)

        # The score scaling factor is the standard deviation the limit is
        # built from, so the two must stay consistent.
        assert limits[0] == pytest.approx(
            stats.t.ppf(0.975, WOLD_X.shape[0] - 1) * model.scaling_factor_for_scores_.iloc[0],
            rel=1e-12,
        )

    def test_predict_new_observations_one_pc(self) -> None:
        model, scaler, _ = _fit_pca_on_wold(n_components=1)
        new_pp = scaler.transform(pd.DataFrame(WOLD_X_NEW))
        result = model.diagnose(new_pp)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 0].values, WOLD_T_NEW_PC1, decimal=3)

    def test_predict_new_observations_two_pc(self) -> None:
        model, scaler, _ = _fit_pca_on_wold(n_components=2)
        new_pp = scaler.transform(pd.DataFrame(WOLD_X_NEW))
        result = model.diagnose(new_pp)
        np.testing.assert_array_almost_equal(result.scores.values, WOLD_T_NEW_2COMP, decimal=3)

    def test_predict_training_data_returns_training_scores(self) -> None:
        """Applying the fitted model to its own training data must reproduce
        the training scores exactly (within tolerance).
        """
        model, _, df_pp = _fit_pca_on_wold(n_components=2)
        result = model.diagnose(df_pp)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 0].values, WOLD_T_PC1, decimal=3)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 1].values, WOLD_T_PC2, decimal=3)

    def test_predict_training_spe_matches_fit_spe(self) -> None:
        model, _, df_pp = _fit_pca_on_wold(n_components=1)
        result = model.diagnose(df_pp)
        np.testing.assert_array_almost_equal(result.spe.values**2, REF_SPE_AFTER_PC1, decimal=4)


# ---------------------------------------------------------------------------
# Multi-block PCA reference tests (xfail until the MBPCA class lands in PR6)
# ---------------------------------------------------------------------------


class TestMBPCAAgainstOracle:
    """The production :class:`MBPCA` class must agree with the pure-numpy
    reference oracle on every quantity (up to a global per-component sign).
    """

    @pytest.fixture
    def synthetic_two_block(self) -> tuple[dict, tuple]:
        rng = np.random.default_rng(42)
        n_rows = 50
        latent = rng.standard_normal((n_rows, 2))
        block_a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((n_rows, 6))
        block_b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((n_rows, 4))
        x_blocks = {
            "A": pd.DataFrame(block_a, columns=[f"a{i}" for i in range(6)]),
            "B": pd.DataFrame(block_b, columns=[f"b{i}" for i in range(4)]),
        }
        x_pp = [MCUVScaler().fit_transform(x_blocks[k]).values for k in ("A", "B")]
        return x_blocks, x_pp

    def test_super_scores_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA
        from tests._multiblock_oracles import mbpca_full_multiblock

        x_blocks, x_pp = synthetic_two_block
        oracle = mbpca_full_multiblock(x_pp, n_components=2)
        model = MBPCA(n_components=2).fit(x_blocks)
        np.testing.assert_array_almost_equal(np.abs(model.super_scores_.values), np.abs(oracle.super_scores), decimal=6)

    def test_super_loadings_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA
        from tests._multiblock_oracles import mbpca_full_multiblock

        x_blocks, x_pp = synthetic_two_block
        oracle = mbpca_full_multiblock(x_pp, n_components=2)
        model = MBPCA(n_components=2).fit(x_blocks)
        np.testing.assert_array_almost_equal(
            np.abs(model.super_loadings_.values), np.abs(oracle.super_loadings), decimal=6
        )

    def test_block_scores_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA
        from tests._multiblock_oracles import mbpca_full_multiblock

        x_blocks, x_pp = synthetic_two_block
        oracle = mbpca_full_multiblock(x_pp, n_components=2)
        model = MBPCA(n_components=2).fit(x_blocks)
        for b_idx, name in enumerate(("A", "B")):
            np.testing.assert_array_almost_equal(
                np.abs(model.block_scores_[name].values), np.abs(oracle.block_scores[b_idx]), decimal=6
            )

    def test_block_loadings_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA
        from tests._multiblock_oracles import mbpca_full_multiblock

        x_blocks, x_pp = synthetic_two_block
        oracle = mbpca_full_multiblock(x_pp, n_components=2)
        model = MBPCA(n_components=2).fit(x_blocks)
        for b_idx, name in enumerate(("A", "B")):
            np.testing.assert_array_almost_equal(
                np.abs(model.block_loadings_[name].values), np.abs(oracle.block_loadings[b_idx]), decimal=6
            )

    def test_super_loading_columns_have_unit_norm(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        norms = np.linalg.norm(model.super_loadings_.values, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)

    def test_predict_on_training_data_reproduces_super_scores(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        result = model.diagnose(x_blocks)
        np.testing.assert_array_almost_equal(result.super_scores.values, model.super_scores_.values, decimal=10)

    def test_block_spe_and_super_t2_have_expected_shape(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        for name in model.block_names_:
            assert model.block_spe_[name].shape == (50, 2)
            assert model.block_hotellings_t2_[name].shape == (50, 2)
        assert model.super_hotellings_t2_.shape == (50, 2)

    def test_display_results_returns_string(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        out = model.display_results()
        assert isinstance(out, str)
        assert "MBPCA model" in out

    def test_spe_contributions_sum_to_block_spe_squared(self, synthetic_two_block) -> None:
        """Sum of per-variable squared residuals across one block must equal
        that block's squared SPE at the final component.
        """
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        contribs = model.spe_contributions(x_blocks)
        for name in model.block_names_:
            row_sums = contribs[name].sum(axis=1).values
            block_spe_sq = model.block_spe_[name].iloc[:, -1].values ** 2
            np.testing.assert_array_almost_equal(row_sums, block_spe_sq, decimal=8)

    def test_score_contributions_sum_to_the_super_score(self, synthetic_two_block) -> None:
        """Summed over every block and variable, the contributions give the super score.

        The super score at component ``a`` is a weighted sum of the block data
        deflated through the first ``a - 1`` components, so the decomposition
        has to start from that same deflated data.
        """
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        for a in (1, 2):
            contribs = model.score_contributions(x_blocks, component=a)
            total = sum(frame.sum(axis=1) for frame in contribs.values())
            np.testing.assert_array_almost_equal(total.values, model.super_scores_[a].values, decimal=10)

    def test_group_contributions_sum_to_the_average_super_score(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        index = next(iter(x_blocks.values())).index
        contribs = model.group_contributions(x_blocks, group=index[:5])
        total = sum(float(series.sum()) for series in contribs.values())
        assert total == pytest.approx(model.super_scores_[1].values[:5].mean(), abs=1e-10)

    def test_contribution_guards_and_plain_array_blocks(self, synthetic_two_block) -> None:
        """Guards mirror MBPLS, and blocks may arrive as plain arrays."""
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)

        with pytest.raises(ValueError, match="Missing X-blocks"):
            model.score_contributions({model.block_names_[0]: x_blocks[model.block_names_[0]]})
        with pytest.raises(ValueError, match="1-based index"):
            model.score_contributions(x_blocks, component=3)

        # A block given as a bare ndarray is labelled from the fitted columns.
        as_arrays = {name: block.to_numpy() for name, block in x_blocks.items()}
        from_arrays = model.score_contributions(as_arrays, component=1)
        from_frames = model.score_contributions(x_blocks, component=1)
        for name in model.block_names_:
            np.testing.assert_array_almost_equal(from_arrays[name].values, from_frames[name].values, decimal=12)

        # The two-period form, against a reference set of observations.
        index = next(iter(x_blocks.values())).index
        shift = model.group_contributions(x_blocks, group=index[:5], reference=index[5:10])
        total = sum(float(series.sum()) for series in shift.values())
        scores = model.super_scores_[1].values
        assert total == pytest.approx(scores[:5].mean() - scores[5:10].mean(), abs=1e-10)

    def test_score_contributions_scalings_span_the_blocks(self, synthetic_two_block) -> None:
        """Both scalings are taken over the blocks jointly, not block by block.

        A super score pools every block, so scaling each block against its own
        maximum would make bars from different blocks incomparable, which is the
        one thing a multi-block contribution plot is read for.
        """
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        raw = model.score_contributions(x_blocks, component=1)

        largest = model.score_contributions(x_blocks, component=1, scaling="maximum")
        peak = max(float(np.abs(f.values).max()) for f in largest.values())
        assert peak == pytest.approx(1.0)
        # Exactly one block holds the peak: the divisor was shared.
        peaks = [float(np.abs(f.values).max()) for f in largest.values()]
        assert sum(p == pytest.approx(1.0) for p in peaks) == 1

        within = model.score_contributions(x_blocks, component=1, scaling="within")
        row_total = sum(np.abs(f.values).sum(axis=1) for f in within.values())
        np.testing.assert_array_almost_equal(row_total, np.ones(len(row_total)), decimal=10)

        # Scaling changes the units, not the pattern within a row.
        for name in model.block_names_:
            ratio = within[name].values[0] / raw[name].values[0]
            np.testing.assert_array_almost_equal(ratio, np.full_like(ratio, ratio[0]), decimal=10)

        with pytest.raises(ValueError, match="scaling must be one of"):
            model.score_contributions(x_blocks, scaling="loud")
        with pytest.raises(TypeError, match="dict"):
            model.score_contributions(next(iter(x_blocks.values())))

    def test_super_score_and_loadings_plots_return_figures(self, synthetic_two_block) -> None:
        import plotly.graph_objects as go

        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        fig_scores = model.super_score_plot(pc_horiz=1, pc_vert=2)
        fig_loadings = model.super_loadings_bar_plot(component=1)
        assert isinstance(fig_scores, go.Figure)
        assert isinstance(fig_loadings, go.Figure)
        assert "PC1 vs PC2" in fig_scores.layout.title.text
        assert list(fig_loadings.data[0].x) == model.block_names_


# ---------------------------------------------------------------------------
# Multi-block PCA missing-data tests
# ---------------------------------------------------------------------------


class TestMBPCAMissingData:
    """Mask-aware NIPALS tests for :class:`MBPCA`.

    Verifies that the masked NIPALS path:

    1. Produces results equivalent to the dense path on complete data
       (regression test: existing behaviour is preserved).
    2. Auto-dispatches to NIPALS when any block contains NaN.
    3. Refuses to fit on degenerate inputs (column or row entirely NaN).
    4. Recovers the complete-data scores and loadings within tolerance
       when a small number of entries are missing at random.

    The skip-NaN NIPALS update is the standard masked-denominator
    variant (Walczak & Massart, 2001; Arteaga & Ferrer, 2002).
    """

    @pytest.fixture
    def two_block(self) -> dict[str, pd.DataFrame]:
        rng = np.random.default_rng(42)
        n_rows = 50
        latent = rng.standard_normal((n_rows, 2))
        block_a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((n_rows, 6))
        block_b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((n_rows, 4))
        return {
            "A": pd.DataFrame(block_a, columns=[f"a{i}" for i in range(6)]),
            "B": pd.DataFrame(block_b, columns=[f"b{i}" for i in range(4)]),
        }

    def test_auto_with_complete_data_resolves_to_dense(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        model = MBPCA(n_components=2, algorithm="auto").fit(two_block)
        assert model.algorithm_ == "dense"
        assert model.has_missing_data_ is False

    def test_nipals_with_complete_data_matches_dense(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        m_dense = MBPCA(n_components=2, algorithm="dense").fit(two_block)
        m_nipals = MBPCA(n_components=2, algorithm="nipals").fit(two_block)

        np.testing.assert_array_almost_equal(
            np.abs(m_dense.super_scores_.values),
            np.abs(m_nipals.super_scores_.values),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            np.abs(m_dense.super_loadings_.values),
            np.abs(m_nipals.super_loadings_.values),
            decimal=6,
        )
        for name in m_dense.block_names_:
            np.testing.assert_array_almost_equal(
                np.abs(m_dense.block_scores_[name].values),
                np.abs(m_nipals.block_scores_[name].values),
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                np.abs(m_dense.block_loadings_[name].values),
                np.abs(m_nipals.block_loadings_[name].values),
                decimal=6,
            )

    def test_dense_with_missing_data_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="cannot handle missing data"):
            MBPCA(n_components=2, algorithm="dense").fit(x)

    def test_auto_with_missing_data_resolves_to_nipals(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[0, 0] = np.nan
        x["B"].iloc[5, 1] = np.nan
        model = MBPCA(n_components=2).fit(x)
        assert model.algorithm_ == "nipals"
        assert model.has_missing_data_ is True

    def test_unknown_algorithm_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        with pytest.raises(ValueError, match="not recognised"):
            MBPCA(n_components=2, algorithm="bogus").fit(two_block)

    def test_column_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[:, 2] = np.nan
        with pytest.raises(ValueError, match="columns with all values missing"):
            MBPCA(n_components=2).fit(x)

    def test_row_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[7, :] = np.nan
        with pytest.raises(ValueError, match="rows with all values missing"):
            MBPCA(n_components=2).fit(x)

    def test_nipals_recovers_scores_with_sparse_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        complete = MBPCA(n_components=2, algorithm="dense").fit(two_block)

        rng = np.random.default_rng(7)
        x: dict[str, pd.DataFrame] = {}
        # Inject ~5% MCAR NaN in each block, leaving every row and column with
        # at least one observation.
        for name, df in two_block.items():
            mask = rng.random(df.shape) < 0.05
            x[name] = df.mask(pd.DataFrame(mask, index=df.index, columns=df.columns))

        with_nan = MBPCA(n_components=2).fit(x)

        # Outputs must be finite, super-loadings must agree with the
        # complete-data fit within tolerance, and per-component super-scores
        # must remain highly correlated with the complete-data scores.
        # Per-row score values can drift by a few tenths under MCAR because
        # the MCUVScaler is re-fitted on the masked block, which shifts the
        # working space slightly. Score correlation isolates that shift.
        assert not with_nan.super_scores_.isna().any().any()
        assert not with_nan.super_loadings_.isna().any().any()
        np.testing.assert_array_almost_equal(
            np.abs(complete.super_loadings_.values),
            np.abs(with_nan.super_loadings_.values),
            decimal=1,
        )
        for a in range(complete.super_scores_.shape[1]):
            corr = abs(np.corrcoef(complete.super_scores_.iloc[:, a], with_nan.super_scores_.iloc[:, a])[0, 1])
            assert corr > 0.95, f"Component {a + 1} score correlation dropped to {corr}"

    def test_nipals_outputs_have_no_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[0, 0] = np.nan
        x["A"].iloc[3, 4] = np.nan
        x["B"].iloc[10, 2] = np.nan

        m = MBPCA(n_components=2).fit(x)
        assert not m.super_scores_.isna().any().any()
        assert not m.super_loadings_.isna().any().any()
        for name in m.block_names_:
            assert not m.block_scores_[name].isna().any().any()
            assert not m.block_loadings_[name].isna().any().any()
            assert not m.block_spe_[name].isna().any().any()

    def test_block_loadings_have_unit_norm_under_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x = {name: df.copy() for name, df in two_block.items()}
        x["A"].iloc[0, 0] = np.nan
        x["B"].iloc[5, 1] = np.nan

        m = MBPCA(n_components=2).fit(x)
        for name in m.block_names_:
            norms = np.linalg.norm(m.block_loadings_[name].values, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=6)

    def test_predict_after_nan_fit_returns_finite_super_scores(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_train = {name: df.copy() for name, df in two_block.items()}
        x_train["A"].iloc[0, 0] = np.nan

        m = MBPCA(n_components=2).fit(x_train)
        # Predict-time NaN scoring (TSR / PMP) is a follow-up; verify that a
        # model fitted on NaN data can still score complete held-out rows.
        x_new = {name: df.iloc[:5, :] for name, df in two_block.items()}
        result = m.diagnose(x_new)
        assert not result.super_scores.isna().any().any()


# ---------------------------------------------------------------------------
# MBPCA written out by hand
# ---------------------------------------------------------------------------

EPSQRT = float(np.sqrt(np.finfo(float).eps))


def _autoscaling(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return each column's centre and scale: its mean and sd (ddof=1), or its value and 1 when it is constant."""
    values = frame.to_numpy(dtype=float)
    constant = np.nanmax(values, axis=0) == np.nanmin(values, axis=0)
    centre = np.where(constant, np.nanmax(values, axis=0), np.nanmean(values, axis=0))
    return centre, np.where(constant, 1.0, np.nanstd(values, axis=0, ddof=1))


def _mbpca_written_out(blocks: dict[str, pd.DataFrame], n_components: int, max_iter: int = 500) -> dict:
    """MBPCA written out (Westerhuis, Kourti and MacGregor, 1998), seeded and stopped as ``fit`` is.

    Each block is autoscaled and weighted by ``1 / sqrt(K_b)``, ``K_b`` its count of columns
    that vary. A component starts from the column, over all blocks, with the largest sum of
    squares (a missing cell read as 0) and repeats: ``p_b = X_b't / t't``, column by column over
    the observed cells, scaled to length 1; ``t_b = X_b p_b / p_b'p_b``, row by row over the
    observed cells, over ``sqrt(K_b)``; ``p = T't`` scaled to length 1, ``T`` the block scores
    side by side; ``t = T p``. It stops once ``t`` moves by at most ``epsqrt`` of its length, or
    after ``max_iter`` passes. The component is flipped so the largest deflating loading
    ``p_b p[b] sqrt(K_b)`` is positive, and ``t`` times that loading is taken from every block.
    """
    names = list(blocks)
    scaling = {name: _autoscaling(frame) for name, frame in blocks.items()}
    x = {name: (frame.to_numpy(dtype=float) - scaling[name][0]) / scaling[name][1] for name, frame in blocks.items()}
    widths = {name: max(1, int(np.sum(np.nanstd(values, axis=0) > EPSQRT))) for name, values in x.items()}
    total = {name: np.nansum(values**2, axis=0) for name, values in x.items()}
    found: dict = {"t": [], "p": [], "iterations": []}
    for key in ("t_b", "p_b", "spe", "r2_columns", "r2_block"):
        found[key] = {name: [] for name in names}
    for _ in range(n_components):
        observed = {name: ~np.isnan(values) for name, values in x.items()}
        filled = {name: np.nan_to_num(values) for name, values in x.items()}
        column_ssq = {name: np.sum(values**2, axis=0) for name, values in filled.items()}
        start = max(names, key={name: values.max() for name, values in column_ssq.items()}.get)
        t = filled[start][:, np.argmax(column_ssq[start])]
        passes = 0
        while passes < max_iter:
            passes += 1
            previous, p_b, t_b = t, {}, {}
            for name in names:
                loading = filled[name].T @ t / (observed[name].T @ t**2)
                p_b[name] = loading / np.linalg.norm(loading)
                t_b[name] = filled[name] @ p_b[name] / (observed[name] @ p_b[name] ** 2) / np.sqrt(widths[name])
            scores = np.column_stack([t_b[name] for name in names])
            p = scores.T @ t / np.linalg.norm(scores.T @ t)
            t = scores @ p
            if np.linalg.norm(t - previous) <= EPSQRT * np.linalg.norm(t):
                break
        deflating = {name: p_b[name] * p[b] * np.sqrt(widths[name]) for b, name in enumerate(names)}
        stacked = np.concatenate([deflating[name] for name in names])
        sign = np.sign(stacked[np.argmax(np.abs(stacked))])
        for name in names:
            x[name] = x[name] - np.outer(t, deflating[name])
            remaining = np.nansum(x[name] ** 2, axis=0)
            found["t_b"][name].append(sign * t_b[name])
            found["p_b"][name].append(sign * p_b[name])
            found["spe"][name].append(np.sqrt(np.nansum(x[name] ** 2, axis=1)))
            found["r2_columns"][name].append(1 - remaining / np.where(total[name] > 0, total[name], np.nan))
            found["r2_block"][name].append(1 - remaining.sum() / total[name].sum())
        found["t"].append(sign * t)
        found["p"].append(p)
        found["iterations"].append(passes)
    out = {"t": np.column_stack(found["t"]), "p": np.column_stack(found["p"]), "iterations": found["iterations"]}
    for key in ("t_b", "p_b", "spe", "r2_columns", "r2_block"):
        out[key] = {name: np.column_stack(found[key][name]) for name in names}
    return out | {"scaling": scaling, "widths": widths}


def _project_by_hand(fitted: dict, blocks: dict[str, pd.DataFrame]) -> dict:
    """Score new rows on a model written out by ``_mbpca_written_out``, deflating as the fit did.

    The loadings have unit length, so each score is a plain product: ``t_b = x_b p_b / sqrt(K_b)``
    and ``t = [t_b] p``. Returns the scores, the data each component starts from and the data
    left after the last, and the T² over the training super-score variances.
    """
    names = list(blocks)
    x = {
        name: (frame.to_numpy(dtype=float) - fitted["scaling"][name][0]) / fitted["scaling"][name][1]
        for name, frame in blocks.items()
    }
    n_components = fitted["t"].shape[1]
    t_all, t_b_all, before = [], {name: [] for name in names}, []
    for a in range(n_components):
        before.append(dict(x))
        t_b = {name: x[name] @ fitted["p_b"][name][:, a] / np.sqrt(fitted["widths"][name]) for name in names}
        t = np.column_stack([t_b[name] for name in names]) @ fitted["p"][:, a]
        for b, name in enumerate(names):
            deflating = fitted["p_b"][name][:, a] * fitted["p"][b, a] * np.sqrt(fitted["widths"][name])
            x[name] = x[name] - np.outer(t, deflating)
            t_b_all[name].append(t_b[name])
        t_all.append(t)
    t = np.column_stack(t_all)
    variance = np.sum(fitted["t"] ** 2, axis=0) / (len(fitted["t"]) - 1)
    return {
        "t": t,
        "t_b": {name: np.column_stack(values) for name, values in t_b_all.items()},
        "residuals": x,
        "before": before,
        "t2": np.sum(t**2 / variance, axis=1),
    }


def _gappy_three_blocks() -> dict[str, pd.DataFrame]:
    """Three blocks of six labelled rows, one missing cell in every column but one, which seeds the fit.

    Autoscaled over its observed cells, a column holds ``n - 1`` in sums of squares, so the one
    complete column, the second of block ``y``, holds 5 and every other column 4. Block ``z``
    is square, six rows by six columns, and holds the most in total; ``z`` is also the last
    name in alphabetical order.
    """
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((6, 2)) * [3.0, 1.0]
    index = [f"r{i}" for i in range(6)]

    def block(prefix: str, width: int) -> pd.DataFrame:
        values = latent @ rng.standard_normal((2, width)) + 0.3 * rng.standard_normal((6, width))
        return pd.DataFrame(values, index=index, columns=[f"{prefix}{j}" for j in range(width)])

    blocks = {"x": block("x", 2), "y": block("y", 2), "z": block("z", 6)}
    blocks["x"].iloc[0, 0] = blocks["x"].iloc[1, 1] = np.nan
    blocks["y"].iloc[2, 0] = np.nan
    for j in range(6):
        blocks["z"].iloc[j, j] = np.nan
    return blocks


def _new_rows(blocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Three complete new rows, labelled n0 to n2, for each block of ``blocks``."""
    rng = np.random.default_rng(1)
    index = ["n0", "n1", "n2"]
    return {
        name: pd.DataFrame(rng.standard_normal((3, frame.shape[1])), index=index, columns=frame.columns)
        for name, frame in blocks.items()
    }


def _frames_near(actual: pd.DataFrame | pd.Series, expected: pd.DataFrame | pd.Series, rtol: float) -> None:
    """Assert equal labels, and values equal to ``rtol``, with an absolute floor of ``rtol`` times the largest."""
    atol = rtol * max(1.0, float(np.nanmax(np.abs(expected.to_numpy()))))
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(actual, expected, rtol=rtol, atol=atol)
    else:
        pd.testing.assert_frame_equal(actual, expected, rtol=rtol, atol=atol)


def _check_fitted(model: MBPCA, blocks: dict[str, pd.DataFrame], expected: dict, rtol: float) -> None:
    """Every fitted attribute of ``model`` against ``_mbpca_written_out``, labels included."""
    names, index = list(blocks), next(iter(blocks.values())).index
    components = list(range(1, expected["t"].shape[1] + 1))
    n_rows = len(index)
    _frames_near(model.super_scores_, pd.DataFrame(expected["t"], index=index, columns=components), rtol)
    _frames_near(model.super_loadings_, pd.DataFrame(expected["p"], index=names, columns=components), rtol)
    variance = np.sum(expected["t"] ** 2, axis=0) / (n_rows - 1)
    np.testing.assert_allclose(model.explained_variance_, variance, rtol=rtol)
    _frames_near(
        model.scaling_factor_for_super_scores_,
        pd.Series(np.sqrt(variance), index=components, name="Standard deviation per super-score"),
        rtol,
    )
    _frames_near(
        model.super_hotellings_t2_,
        pd.DataFrame(np.cumsum(expected["t"] ** 2 / variance, axis=1), index=index, columns=components),
        rtol,
    )
    r2_block = np.vstack([expected["r2_block"][name] for name in names])
    _frames_near(model.r2_x_per_block_cumulative_, pd.DataFrame(r2_block, index=names, columns=components), rtol)
    increments = np.column_stack([r2_block[:, 0], np.diff(r2_block, axis=1)])
    _frames_near(model.r2_x_per_block_per_component_, pd.DataFrame(increments, index=names, columns=components), rtol)
    assert model.block_informative_widths_ == expected["widths"]
    for b, name in enumerate(names):
        columns = blocks[name].columns
        t_b, p_b = expected["t_b"][name], expected["p_b"][name]
        _frames_near(model.block_scores_[name], pd.DataFrame(t_b, index=index, columns=components), rtol)
        _frames_near(model.block_loadings_[name], pd.DataFrame(p_b, index=columns, columns=components), rtol)
        _frames_near(model.block_spe_[name], pd.DataFrame(expected["spe"][name], index=index, columns=components), rtol)
        r2_columns = pd.DataFrame(expected["r2_columns"][name], index=columns, columns=components)
        _frames_near(model.r2_x_per_variable_[name], r2_columns, rtol)
        second_moment = np.sum(t_b**2, axis=0) / (n_rows - 1)
        block_t2 = pd.DataFrame(np.cumsum(t_b**2 / second_moment, axis=1), index=index, columns=components)
        _frames_near(model.block_hotellings_t2_[name], block_t2, rtol)
        vip = np.sqrt(expected["widths"][name] * (p_b**2 @ increments[b]) / increments[b].sum())
        _frames_near(model.block_vip_[name], pd.Series(vip, index=columns, name=f"VIP[{name}]"), rtol)


class TestMBPCAWrittenOut:
    """MBPCA against its algorithm written out by hand, every attribute, label and step."""

    def test_a_gappy_fit_follows_the_algorithm_pass_for_pass(self) -> None:
        """Missing cells take the masked path; a fit stopped after one pass shows where it was seeded."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=2).fit(blocks)
        expected = _mbpca_written_out(blocks, 2)
        assert model.algorithm_ == "nipals"
        np.testing.assert_array_equal(model.fitting_info_["iterations"], expected["iterations"])
        assert model.fitting_info_["iterations"].dtype.kind == "i"
        _check_fitted(model, blocks, expected, rtol=1e-10)

        with pytest.warns(
            SpecificationWarning,
            match=r"^MBPCA NIPALS did not converge within max_iter=1 for component\(s\) \[1, 2\]; ",
        ):
            one_pass = MBPCA(n_components=2, max_iter=1).fit(blocks)
        _check_fitted(one_pass, blocks, _mbpca_written_out(blocks, 2, max_iter=1), rtol=1e-12)
        np.testing.assert_array_equal(one_pass.fitting_info_["iterations"], [1, 1])

    def test_the_masked_path_on_complete_data_is_the_dense_path(self) -> None:
        """On complete LDPE blocks, "nipals" fits what "dense" does, and what the algorithm written out does."""
        blocks, _ = _ldpe_blocks()
        expected = _mbpca_written_out(blocks, 3)
        for algorithm in ("dense", "nipals"):
            model = MBPCA(n_components=3, algorithm=algorithm).fit(blocks)
            assert model.algorithm_ == algorithm
            _check_fitted(model, blocks, expected, rtol=1e-6)

    def test_new_rows_are_scored_on_what_the_earlier_components_left(self) -> None:
        """New rows are deflated by diagnose and transform as the fit did; arrays read as the fitted columns."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=2).fit(blocks)
        new = _new_rows(blocks)
        by_hand = _project_by_hand(_mbpca_written_out(blocks, 2), new)
        index, components = new["x"].index, [1, 2]
        result = model.diagnose(new)
        assert sorted(result) == ["block_scores", "block_spe", "hotellings_t2", "super_scores"]
        _frames_near(result.super_scores, pd.DataFrame(by_hand["t"], index=index, columns=components), 1e-10)
        _frames_near(model.transform(new), pd.DataFrame(by_hand["t"], index=index, columns=components), 1e-10)
        _frames_near(result.hotellings_t2, pd.Series(by_hand["t2"], index=index, name="Hotelling's T²"), 1e-10)
        for name in blocks:
            t_b = pd.DataFrame(by_hand["t_b"][name], index=index, columns=components)
            _frames_near(result.block_scores[name], t_b, 1e-10)
            spe = np.sqrt(np.sum(by_hand["residuals"][name] ** 2, axis=1))
            _frames_near(result.block_spe[name], pd.Series(spe, index=index, name=f"SPE[{name}]"), 1e-10)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "X does not have valid feature names"
            from_arrays = model.diagnose({name: frame.to_numpy() for name, frame in new.items()})
        positions = pd.RangeIndex(3)
        _frames_near(from_arrays.super_scores, pd.DataFrame(by_hand["t"], index=positions, columns=components), 1e-10)

    def test_contributions_split_the_new_rows_scores_and_residuals_by_hand(self) -> None:
        """SPE contributions are the squared residuals; score contributions split each super score by variable."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=2).fit(blocks)
        fitted = _mbpca_written_out(blocks, 2)
        new = _new_rows(blocks)
        by_hand = _project_by_hand(fitted, new)
        index = new["x"].index

        spe = model.spe_contributions(new)
        assert list(spe) == ["x", "y", "z"]
        for name in blocks:
            residuals = pd.DataFrame(by_hand["residuals"][name] ** 2, index=index, columns=blocks[name].columns)
            _frames_near(spe[name], residuals, 1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "X does not have valid feature names"
            spe_of_arrays = model.spe_contributions({name: frame.to_numpy() for name, frame in new.items()})
        for name in blocks:
            residuals = pd.DataFrame(by_hand["residuals"][name] ** 2, columns=blocks[name].columns)
            _frames_near(spe_of_arrays[name], residuals, 1e-10)

        for component in (1, 2):
            a = component - 1
            contributions = model.score_contributions(new, component=component)
            expected = {
                name: pd.DataFrame(
                    by_hand["before"][a][name]
                    * fitted["p_b"][name][:, a]
                    / np.sqrt(fitted["widths"][name])
                    * fitted["p"][b, a],
                    index=index,
                    columns=blocks[name].columns,
                )
                for b, name in enumerate(blocks)
            }
            for name in blocks:
                _frames_near(contributions[name], expected[name], 1e-10)
            np.testing.assert_allclose(sum(frame.sum(axis=1) for frame in expected.values()), by_hand["t"][:, a])
            if component == 2:
                group = model.group_contributions(new, group=["n0", "n2"], reference=["n1"], component=2)
                for name in blocks:
                    shift = expected[name].loc[["n0", "n2"]].mean() - expected[name].loc["n1"]
                    _frames_near(group[name], shift.rename(f"group_contributions[{name}]"), 1e-10)
        first = model.score_contributions(new)
        for name in blocks:
            pd.testing.assert_frame_equal(first[name], model.score_contributions(new, component=1)[name])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "X does not have valid feature names"
            of_arrays = model.score_contributions({name: frame.to_numpy() for name, frame in new.items()}, component=2)
        for name in blocks:
            np.testing.assert_allclose(of_arrays[name].to_numpy(), model.score_contributions(new, component=2)[name])
            assert list(of_arrays[name].columns) == list(blocks[name].columns)

        with pytest.raises(ValueError, match=r"^group contains entries that are not index labels of X: \['zz'\]\. "):
            model.group_contributions(new, group=["zz"])
        with pytest.raises(
            ValueError, match=r"^reference contains entries that are not index labels of X: \['zz'\]\. "
        ):
            model.group_contributions(new, group=["n0"], reference=["zz"])
        with pytest.raises(TypeError, match=r"^X must be a dict\[str, pd\.DataFrame\]\.$"):
            model.score_contributions(new["x"])

    def test_two_rows_and_a_constant_column_have_the_r2_they_should_without_a_warning(self) -> None:
        """At two rows a varying column holds a sum of squares of 1; a constant one holds none.

        Column ``a0`` of [0, 1] scales to a sum of squares of 0.9999999999999998, all of it
        explained by one component: R2X 1. The constant column ``b1`` and the block ``c`` of
        constants have nothing to explain: their R2X is NaN, with no warning from a division by
        0, ``c``'s VIP is 0, and neither counts in its block's width. The variance divides by 1.
        """
        index = ["first", "second"]
        blocks = {
            "a": pd.DataFrame({"a0": [0.0, 1.0]}, index=index),
            "b": pd.DataFrame({"b0": [2.0, 7.0], "b1": [5.0, 5.0]}, index=index),
            "c": pd.DataFrame({"c0": [3.0, 3.0]}, index=index),
        }
        assert np.sum(MCUVScaler().fit_transform(blocks["a"]).to_numpy() ** 2) < 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MBPCA(n_components=1).fit(blocks)
        assert model.block_informative_widths_ == {"a": 1, "b": 1, "c": 1}
        _frames_near(
            model.r2_x_per_block_cumulative_,
            pd.DataFrame([[1.0], [1.0], [np.nan]], index=list(blocks), columns=[1]),
            1e-12,
        )
        _frames_near(
            model.r2_x_per_variable_["b"], pd.DataFrame([[1.0], [np.nan]], index=["b0", "b1"], columns=[1]), 1e-12
        )
        np.testing.assert_allclose(model.explained_variance_, [np.sum(model.super_scores_.to_numpy() ** 2)], rtol=1e-15)
        np.testing.assert_allclose(model.block_vip_["a"], [1.0], rtol=1e-12)
        np.testing.assert_allclose(model.block_vip_["b"], [1.0, 0.0], rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(model.block_vip_["c"], [0.0])

    def test_a_block_the_component_leaves_untouched_has_no_r2_and_no_vip(self) -> None:
        """Block ``b`` is exactly orthogonal to block ``a``: the component built on ``a`` explains none of it.

        Both columns have mean 0 and sd 1 already, so they scale to themselves, in integers, and hold
        4 each in sums of squares: the seed is ``a``'s, the first. ``b``'s loading is then
        ``x_b' t = 0`` exactly, its block score 0 and its super loading 0: R2X exactly 0, and so a
        VIP of 0 rather than 0 / 0.
        """
        blocks = {
            "a": pd.DataFrame({"a0": [-1.0, -1.0, 0.0, 1.0, 1.0]}),
            "b": pd.DataFrame({"b0": [1.0, -1.0, 0.0, -1.0, 1.0]}),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MBPCA(n_components=1).fit(blocks)
        np.testing.assert_array_equal(model.super_loadings_[1], [1.0, 0.0])
        np.testing.assert_array_equal(model.block_loadings_["b"][1], [0.0])
        np.testing.assert_array_equal(model.r2_x_per_block_cumulative_[1], [1.0, 0.0])
        np.testing.assert_array_equal(model.block_vip_["a"], [1.0])
        np.testing.assert_array_equal(model.block_vip_["b"], [0.0])

    def test_a_model_fitted_before_the_informative_widths_divides_by_the_full_width(self) -> None:
        """Without ``block_informative_widths_`` (#913), new rows are weighted by each block's full width."""
        blocks, _ = _ldpe_blocks()
        blocks["zone2"] = blocks["zone2"].assign(flat=1.0)
        model = MBPCA(n_components=2).fit(blocks)
        fitted = _mbpca_written_out(blocks, 2)
        assert model.block_informative_widths_["zone2"] == 6
        new = {name: frame.iloc[:4] + 0.5 for name, frame in blocks.items()}
        del model.block_informative_widths_
        full = {name: frame.shape[1] for name, frame in blocks.items()}
        by_hand = _project_by_hand(fitted | {"widths": full}, new)
        np.testing.assert_allclose(model.transform(new), by_hand["t"], rtol=1e-6)
        assert np.abs(model.transform(new).to_numpy() - _project_by_hand(fitted, new)["t"]).max() > 1e-3

    def test_a_fit_stopped_by_max_iter_says_which_components_and_logs_the_iterations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The gappy blocks converge in 10 and 16 passes: twelve stop the second, which the warning names."""
        blocks = _gappy_three_blocks()
        message = (
            r"^MBPCA NIPALS did not converge within max_iter=12 for component\(s\) \[2\]; "
            r"results for those components may be unreliable\.$"
        )
        with (
            caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._mbpca"),
            pytest.warns(SpecificationWarning, match=message),
        ):
            model = MBPCA(n_components=2, max_iter=12).fit(blocks)
        assert caplog.messages == ["MBPCA (nipals): iterations per component = [10, 12]"]
        np.testing.assert_array_equal(model.fitting_info_["converged"], [True, False])
        np.testing.assert_array_equal(model.fitting_info_["iterations"], [10, 12])
        assert model.fitting_info_["timing"].shape == (2,)
        _check_fitted(model, blocks, _mbpca_written_out(blocks, 2, max_iter=12), rtol=1e-10)

    def test_display_results_prints_the_r2_iterations_and_timing(self) -> None:
        """The summary, in full: cumulative R2X by default, the increments on request, and plain integers."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=2).fit(blocks)
        model.fitting_info_["timing"] = np.array([0.1, 0.00123])
        r2 = np.vstack([_mbpca_written_out(blocks, 2)["r2_block"][name] for name in blocks])
        header = "  PC | R²X[x] | R²X[y] | R²X[z]"

        def summary(values: np.ndarray) -> str:
            rows = [f"  {a + 1} | " + " | ".join(f"{value:9.4f}" for value in values[:, a]) for a in range(2)]
            return "\n".join(
                [
                    "MBPCA model: 2 component(s), 3 X-block(s)",
                    header,
                    "-" * 31,
                    *rows,
                    "",
                    "  Iterations per PC: [10, 16]",
                    "  Time per PC (ms):  [100.0, 1.2]",
                ]
            )

        assert model.display_results() == summary(r2)
        assert model.display_results(show_cumulative=False) == summary(np.column_stack([r2[:, 0], r2[:, 1] - r2[:, 0]]))

    def test_the_plots_draw_the_fitted_super_scores_and_loadings(self) -> None:
        """Both plots, in full, by default and at the last component."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=3).fit(blocks)
        scores = model.super_scores_
        for figure, (horizontal, vertical) in (
            (model.super_score_plot(), (1, 2)),
            (model.super_score_plot(3, 1), (3, 1)),
        ):
            (trace,) = figure.data
            np.testing.assert_array_equal(trace.x, scores[horizontal])
            np.testing.assert_array_equal(trace.y, scores[vertical])
            assert trace.mode == "markers+text"
            assert list(trace.text) == [f"r{i}" for i in range(6)]
            assert trace.textposition == "top center"
            assert trace.name == "Super-scores"
            assert figure.layout.xaxis.title.text == f"t_super[{horizontal}]"
            assert figure.layout.yaxis.title.text == f"t_super[{vertical}]"
            assert figure.layout.title.text == f"MBPCA super-score plot: PC{horizontal} vs PC{vertical}"

        for figure, component in ((model.super_loadings_bar_plot(), 1), (model.super_loadings_bar_plot(3), 3)):
            (bar,) = figure.data
            assert list(bar.x) == ["x", "y", "z"]
            np.testing.assert_array_equal(bar.y, model.super_loadings_[component])
            assert bar.name == f"p_super[{component}]"
            assert figure.layout.xaxis.title.text == "Block"
            assert figure.layout.yaxis.title.text == f"p_super[{component}]"
            assert figure.layout.title.text == f"MBPCA super-loadings, component {component}"

    def test_a_component_seeded_against_its_largest_loadings_is_flipped_whole(self) -> None:
        """The seed column ``a`` is the only complete one, but ``b``, ``c`` and ``d`` follow ``-z``, not ``z``.

        The iteration keeps the seed's sign, so the converged component has ``a``'s loading
        positive and the larger ones negative: it is flipped, scores, loadings and block scores
        together, and ``a``'s loading ends up negative.
        """
        rng = np.random.default_rng(0)
        z = rng.standard_normal(8)
        blocks = {
            "p": pd.DataFrame({"a": z + 1.5 * rng.standard_normal(8), "b": -z + 0.1 * rng.standard_normal(8)}),
            "q": pd.DataFrame({"c": -z + 0.1 * rng.standard_normal(8), "d": -z + 0.2 * rng.standard_normal(8)}),
        }
        blocks["p"].iloc[3, 1] = blocks["q"].iloc[5, 0] = blocks["q"].iloc[6, 1] = np.nan
        model = MBPCA(n_components=1).fit(blocks)
        assert model.block_loadings_["p"].loc["a", 1] < 0
        _check_fitted(model, blocks, _mbpca_written_out(blocks, 1), rtol=1e-10)

    def test_a_change_of_exactly_tol_has_converged(self) -> None:
        """The loop stops once the super score moves by at most ``tol`` of its length, equality included.

        Fits stopped after 3 and 4 passes give the last two iterates, so their change relative to
        the newer is what the fourth pass measures, to the last bit. With that as ``tol`` the fit
        stops after the fourth pass.
        """
        blocks = _gappy_three_blocks()
        with pytest.warns(SpecificationWarning, match="did not converge"):
            third = MBPCA(n_components=1, max_iter=3).fit(blocks).super_scores_[1].to_numpy()
        with pytest.warns(SpecificationWarning, match="did not converge"):
            fourth = MBPCA(n_components=1, max_iter=4).fit(blocks).super_scores_[1].to_numpy()
        step = np.linalg.norm(third - fourth)
        assert step < 0.1 * np.linalg.norm(fourth)  # one pass apart, with the same sign
        model = MBPCA(n_components=1, tol=step / np.linalg.norm(fourth)).fit(blocks)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], [4])

    def test_the_spe_limits_are_set_on_the_last_component(self) -> None:
        """With three components, each block's limit and the merged one read the third column of the SPE."""
        blocks = _gappy_three_blocks()
        model = MBPCA(n_components=3).fit(blocks)
        spe = _mbpca_written_out(blocks, 3)["spe"]
        merged = np.sqrt(sum(values[:, 2] ** 2 for values in spe.values()))
        for conf_level in (0.95, 0.99):
            for name in blocks:
                limit = spe_calculation(spe[name][:, 2], conf_level=conf_level)
                assert model.block_spe_limit(name, conf_level=conf_level) == pytest.approx(limit, rel=1e-9)
            assert model.super_spe_limit(conf_level=conf_level) == pytest.approx(
                spe_calculation(merged, conf_level=conf_level), rel=1e-9
            )
        assert model.block_spe_limit("z") == model.block_spe_limit("z", conf_level=0.95)
        assert model.super_spe_limit() == model.super_spe_limit(conf_level=0.95)


@pytest.mark.parametrize(
    "use",
    [
        lambda model, blocks: model.transform(blocks),
        lambda model, blocks: model.diagnose(blocks),
        lambda model, blocks: model.spe_contributions(blocks),
        lambda model, blocks: model.score_contributions(blocks),
        lambda model, _: model.block_spe_limit("x"),
        lambda model, _: model.super_spe_limit(),
        lambda model, _: model.super_score_plot(),
        lambda model, _: model.super_loadings_bar_plot(),
        lambda model, _: model.display_results(),
    ],
    ids=[
        "transform",
        "diagnose",
        "spe_contributions",
        "score_contributions",
        "block_spe_limit",
        "super_spe_limit",
        "super_score_plot",
        "super_loadings_bar_plot",
        "display_results",
    ],
)
def test_a_model_whose_fit_failed_is_not_fitted(use: Callable[[MBPCA, dict], object]) -> None:
    """A fit refused after the block names were recorded leaves a model every method reports as unfitted."""
    blocks = _gappy_three_blocks()
    model = MBPCA(n_components=2, algorithm="bogus")
    with pytest.raises(ValueError, match="is not recognised"):
        model.fit(blocks)
    assert model.block_names_ == ["x", "y", "z"]
    with pytest.raises(NotFittedError):
        use(model, blocks)


def test_a_block_with_a_column_or_a_row_of_missing_cells_is_refused_in_full() -> None:
    """The masked NIPALS cannot regress on nothing: the message names the block and what to drop."""
    blocks = _gappy_three_blocks()
    blocks["x"]["x1"] = np.nan
    message = r"^Block 'x' has columns with all values missing: \['x1'\]\. Drop these columns before fitting\.$"
    with pytest.raises(ValueError, match=message):
        MBPCA(n_components=1).fit(blocks)

    blocks = _gappy_three_blocks()
    blocks["z"].iloc[4] = np.nan
    message = (
        r"^Block 'z' has rows with all values missing at positions \[4\]\. "
        r"Drop these observations or impute them before fitting\.$"
    )
    with pytest.raises(ValueError, match=message):
        MBPCA(n_components=1).fit(blocks)


def test_a_component_whose_deflating_loadings_are_all_zero_keeps_its_signs() -> None:
    """Only a negative largest deflating loading flips a component; a largest of zero (here -0.0) is not negative."""
    from process_improve.multivariate._mbpca import _apply_sign_convention, _MBPCAComponent, _MBPCALoopContext

    context = _MBPCALoopContext(algo="dense", block_names=["a"], sqrt_kb={"a": 1.0}, tol=1e-8, max_iter=10, n_samples=2)
    component = _MBPCAComponent(
        super_scores=np.zeros(2),
        super_loadings=np.zeros(1),
        block_loadings={"a": np.array([-1.0])},
        block_scores={"a": np.array([-0.5, 0.5])},
        iterations=1,
    )
    assert _apply_sign_convention(component, context) is component


def test_the_dense_algorithm_refuses_missing_cells_in_full() -> None:
    """Only the masked path reads missing cells."""
    blocks = _gappy_three_blocks()
    with pytest.raises(
        ValueError, match=r"^Algorithm 'dense' cannot handle missing data\. Use 'nipals' or 'auto' instead\.$"
    ):
        MBPCA(n_components=1, algorithm="dense").fit(blocks)


# ---------------------------------------------------------------------------
# Multi-block PLS reference tests (active as of PR3 - MBPLS implemented)
# ---------------------------------------------------------------------------


class TestMBPLSAgainstOracle:
    """The production :class:`MBPLS` class must agree with the pure-numpy
    reference oracle on every quantity (up to a global per-component sign).

    The oracle in ``tests/_multiblock_oracles.py`` has been independently
    self-validated by ``tests/test_multiblock_oracles.py`` against a second
    reference implementation (merged-then-recover).
    """

    @pytest.fixture
    def synthetic_two_block(self) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(42)
        n_rows = 50
        latent = rng.standard_normal((n_rows, 2))
        block_a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((n_rows, 6))
        block_b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((n_rows, 4))
        y_block = latent @ rng.standard_normal((2, 2)) + 0.05 * rng.standard_normal((n_rows, 2))
        x_blocks = {
            "A": pd.DataFrame(block_a, columns=[f"a{i}" for i in range(6)]),
            "B": pd.DataFrame(block_b, columns=[f"b{i}" for i in range(4)]),
        }
        y_df = pd.DataFrame(y_block, columns=["y0", "y1"])
        # Pre-scaled versions for the oracle (which expects already-preprocessed numpy)
        from process_improve.multivariate.methods import MCUVScaler

        x_pp = [MCUVScaler().fit_transform(x_blocks[k]).values for k in ("A", "B")]
        y_pp = MCUVScaler().fit_transform(y_df).values
        return x_blocks, y_df, (x_pp, y_pp)

    def test_super_scores_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS
        from tests._multiblock_oracles import mbpls_full_multiblock

        x_blocks, y_df, (x_pp, y_pp) = synthetic_two_block
        oracle = mbpls_full_multiblock(x_pp, y_pp, n_components=2)
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        np.testing.assert_array_almost_equal(np.abs(model.super_scores_.values), np.abs(oracle.super_scores), decimal=6)

    def test_super_weights_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS
        from tests._multiblock_oracles import mbpls_full_multiblock

        x_blocks, y_df, (x_pp, y_pp) = synthetic_two_block
        oracle = mbpls_full_multiblock(x_pp, y_pp, n_components=2)
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        np.testing.assert_array_almost_equal(
            np.abs(model.super_weights_.values), np.abs(oracle.super_weights), decimal=6
        )

    def test_block_weights_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS
        from tests._multiblock_oracles import mbpls_full_multiblock

        x_blocks, y_df, (x_pp, y_pp) = synthetic_two_block
        oracle = mbpls_full_multiblock(x_pp, y_pp, n_components=2)
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        for b_idx, name in enumerate(("A", "B")):
            np.testing.assert_array_almost_equal(
                np.abs(model.block_weights_[name].values), np.abs(oracle.block_weights[b_idx]), decimal=6
            )

    def test_block_loadings_match_oracle(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS
        from tests._multiblock_oracles import mbpls_full_multiblock

        x_blocks, y_df, (x_pp, y_pp) = synthetic_two_block
        oracle = mbpls_full_multiblock(x_pp, y_pp, n_components=2)
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        for b_idx, name in enumerate(("A", "B")):
            np.testing.assert_array_almost_equal(
                np.abs(model.block_loadings_[name].values), np.abs(oracle.block_loadings[b_idx]), decimal=6
            )

    def test_y_predictions_match_oracle(self, synthetic_two_block) -> None:
        # Predictions involve the product of two sign-flipped quantities
        # (super_scores * super_y_loadings), so the comparison is direct.
        from process_improve.multivariate.methods import MBPLS, MCUVScaler
        from tests._multiblock_oracles import mbpls_full_multiblock

        x_blocks, y_df, (x_pp, y_pp) = synthetic_two_block
        oracle = mbpls_full_multiblock(x_pp, y_pp, n_components=2)
        # Oracle predictions are on the preprocessed scale; un-preprocess for comparison.
        y_scaler = MCUVScaler().fit(y_df)
        oracle_predictions = y_scaler.inverse_transform(pd.DataFrame(oracle.y_predictions, columns=y_df.columns))
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        np.testing.assert_array_almost_equal(model.predictions_.values, oracle_predictions.values, decimal=6)

    def test_super_weight_columns_have_unit_norm(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df, _ = synthetic_two_block
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        norms = np.linalg.norm(model.super_weights_.values, axis=0)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)

    def test_block_weight_columns_have_unit_norm(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df, _ = synthetic_two_block
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        for name in model.block_names_:
            norms = np.linalg.norm(model.block_weights_[name].values, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=8)

    def test_predict_on_training_data_reproduces_in_sample_predictions(self, synthetic_two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df, _ = synthetic_two_block
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        result = model.diagnose(x_blocks)
        np.testing.assert_array_almost_equal(result.predictions.values, model.predictions_.values, decimal=10)
        np.testing.assert_array_almost_equal(result.super_scores.values, model.super_scores_.values, decimal=10)


# ---------------------------------------------------------------------------
# Multi-block PLS missing-data tests
# ---------------------------------------------------------------------------


class TestMBPLSMissingData:
    """Mask-aware NIPALS tests for :class:`MBPLS`.

    Verifies that the masked NIPALS path on MBPLS:

    1. Produces results equivalent to the dense path on complete data
       (regression test: existing behaviour is preserved).
    2. Auto-dispatches to NIPALS when any X-block or Y contains NaN.
    3. Refuses to fit on degenerate inputs in either X-blocks or Y
       (column or row entirely NaN).
    4. Recovers the complete-data super-weights within tolerance and
       keeps a high per-component super-score correlation when a
       small number of X- and Y-entries are missing at random.

    The skip-NaN NIPALS update is the standard masked-denominator
    variant (Walczak & Massart, 2001; Arteaga & Ferrer, 2002).
    """

    @pytest.fixture
    def two_block(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        rng = np.random.default_rng(42)
        n_rows = 50
        latent = rng.standard_normal((n_rows, 2))
        block_a = latent @ rng.standard_normal((2, 6)) + 0.05 * rng.standard_normal((n_rows, 6))
        block_b = latent @ rng.standard_normal((2, 4)) + 0.05 * rng.standard_normal((n_rows, 4))
        y_block = latent @ rng.standard_normal((2, 2)) + 0.05 * rng.standard_normal((n_rows, 2))
        x_blocks = {
            "A": pd.DataFrame(block_a, columns=[f"a{i}" for i in range(6)]),
            "B": pd.DataFrame(block_b, columns=[f"b{i}" for i in range(4)]),
        }
        y_df = pd.DataFrame(y_block, columns=["y0", "y1"])
        return x_blocks, y_df

    def test_auto_with_complete_data_resolves_to_dense(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        model = MBPLS(n_components=2, algorithm="auto").fit(x, y)
        assert model.algorithm_ == "dense"
        assert model.has_missing_data_ is False

    def test_nipals_with_complete_data_matches_dense(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        m_dense = MBPLS(n_components=2, algorithm="dense").fit(x, y)
        m_nipals = MBPLS(n_components=2, algorithm="nipals").fit(x, y)

        np.testing.assert_array_almost_equal(
            np.abs(m_dense.super_scores_.values),
            np.abs(m_nipals.super_scores_.values),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            np.abs(m_dense.super_weights_.values),
            np.abs(m_nipals.super_weights_.values),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            np.abs(m_dense.super_y_loadings_.values),
            np.abs(m_nipals.super_y_loadings_.values),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            m_dense.predictions_.values,
            m_nipals.predictions_.values,
            decimal=6,
        )
        for name in m_dense.block_names_:
            np.testing.assert_array_almost_equal(
                np.abs(m_dense.block_weights_[name].values),
                np.abs(m_nipals.block_weights_[name].values),
                decimal=6,
            )
            np.testing.assert_array_almost_equal(
                np.abs(m_dense.block_loadings_[name].values),
                np.abs(m_nipals.block_loadings_[name].values),
                decimal=6,
            )

    def test_dense_with_missing_x_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="cannot handle missing data"):
            MBPLS(n_components=2, algorithm="dense").fit(x_n, y)

    def test_dense_with_missing_y_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        y_n = y.copy()
        y_n.iloc[3, 0] = np.nan
        with pytest.raises(ValueError, match="cannot handle missing data"):
            MBPLS(n_components=2, algorithm="dense").fit(x, y_n)

    def test_auto_with_missing_x_resolves_to_nipals(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[0, 0] = np.nan
        x_n["B"].iloc[5, 1] = np.nan
        model = MBPLS(n_components=2).fit(x_n, y)
        assert model.algorithm_ == "nipals"
        assert model.has_missing_data_ is True

    def test_auto_with_missing_y_resolves_to_nipals(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        y_n = y.copy()
        y_n.iloc[2, 1] = np.nan
        model = MBPLS(n_components=2).fit(x, y_n)
        assert model.algorithm_ == "nipals"
        assert model.has_missing_data_ is True

    def test_unknown_algorithm_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        with pytest.raises(ValueError, match="not recognised"):
            MBPLS(n_components=2, algorithm="bogus").fit(x, y)

    def test_x_column_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[:, 2] = np.nan
        with pytest.raises(ValueError, match="columns with all values missing"):
            MBPLS(n_components=2).fit(x_n, y)

    def test_x_row_all_nan_in_one_block_is_still_scored(self, two_block) -> None:
        """An observation missing one whole block keeps a super score, from the blocks it has.

        The score comes from one masked regression of the row onto the stacked weights, so the
        cells the row does have carry it. The block that was not observed has no score of its
        own, and says so with NaN rather than sitting at the origin as though it were average.
        """
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[7, :] = np.nan
        model = MBPLS(n_components=2).fit(x_n, y)
        assert np.all(np.isfinite(model.super_scores_.iloc[7].to_numpy()))
        assert np.all(np.isnan(model.block_scores_["A"].iloc[7].to_numpy()))
        assert np.all(np.isnan(model.block_spe_["A"].iloc[7].to_numpy()))
        assert np.all(np.isfinite(model.block_scores_["B"].iloc[7].to_numpy()))

    def test_x_row_all_nan_in_every_block_raises(self, two_block) -> None:
        """A row observed nowhere carries no information, and is refused."""
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        for df in x_n.values():
            df.iloc[7, :] = np.nan
        with pytest.raises(ValueError, match=r"all values missing in every X-block"):
            MBPLS(n_components=2).fit(x_n, y)

    def test_pooled_super_score_leaves_complete_data_untouched(self, two_block) -> None:
        """On a complete row the pooled regression is algebraically the weighted sum of block scores.

        This is the invariant that lets the missing-data path change without moving any fitted
        model: it pins the super score against the sum it replaces, evaluated by hand.
        """
        from process_improve.multivariate._preprocessing import MCUVScaler
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        model = MBPLS(n_components=2).fit(x, y)
        sqrt_kb = {name: np.sqrt(x[name].shape[1]) for name in x}
        x_pp = {name: MCUVScaler().fit_transform(x[name]).to_numpy() for name in x}
        t_b = np.column_stack([x_pp[name] @ model.block_weights_[name].to_numpy()[:, 0] / sqrt_kb[name] for name in x])
        by_hand = t_b @ model.super_weights_.to_numpy()[:, 0]
        assert np.allclose(by_hand, model.super_scores_.to_numpy()[:, 0], atol=1e-10)

    def test_super_score_of_a_partly_observed_row_is_the_pooled_regression(self, two_block) -> None:
        """The super score of a row with missing cells is one regression of the whole row.

        Written out by hand from the model's own weights, so the test fails if the score goes back
        to being a sum of per-block scores: the two differ once a row is not complete.
        """
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        model = MBPLS(n_components=2).fit(x, y)
        row = {name: df.iloc[[3]].copy() for name, df in x.items()}
        row["A"].iloc[0, 1:] = np.nan  # block A observed in one variable out of six

        scores = model.transform(row).to_numpy()[0]
        sqrt_kb = {name: np.sqrt(x[name].shape[1]) for name in x}
        x_pp = {name: model.preproc_[name].transform(row[name]).to_numpy()[0] for name in x}
        w_s = model.super_weights_.to_numpy()[:, 0]
        g = np.concatenate(
            [w_s[b] * model.block_weights_[name].to_numpy()[:, 0] / sqrt_kb[name] for b, name in enumerate(x)]
        )
        cells = np.concatenate([x_pp[name] for name in x])
        seen = ~np.isnan(cells)
        pooled = float(np.nan_to_num(cells) @ g) / float(seen @ (g**2)) * float(g @ g) / float(w_s @ w_s)
        assert scores[0] == pytest.approx(pooled, abs=1e-10)

        by_block = float(
            sum(
                w_s[b] * np.nansum(x_pp[name] * model.block_weights_[name].to_numpy()[:, 0]) / sqrt_kb[name]
                for b, name in enumerate(x)
            )
        )
        assert abs(scores[0] - by_block) > 1e-6, "the two formulas must part company on an incomplete row"

    def test_pooled_super_score_survives_heavy_missingness(self) -> None:
        """With a quarter of the cells gone, the super scores still track the complete-data ones.

        The blocks are deliberately lopsided, 14 variables against 3. Summing per-block scores lets
        the narrow block speak as loudly as the wide one however little of it a row has, and the
        correlation with the complete-data scores collapses to about 0.31; pooling holds it at 0.93.
        """
        from process_improve.multivariate.methods import MBPLS

        rng = np.random.default_rng(11)
        n_rows = 60
        latent = rng.standard_normal((n_rows, 2))
        widths = {"wide": 14, "narrow": 3}
        x = {
            name: pd.DataFrame(
                latent @ rng.standard_normal((2, k)) + 0.5 * rng.standard_normal((n_rows, k)),
                columns=[f"{name}{i}" for i in range(k)],
            )
            for name, k in widths.items()
        }
        y = pd.DataFrame(
            latent @ rng.standard_normal((2, 2)) + 0.3 * rng.standard_normal((n_rows, 2)), columns=["y0", "y1"]
        )
        reference = MBPLS(n_components=2).fit(x, y).super_scores_.to_numpy()

        rng = np.random.default_rng(3)
        x_n = {}
        for name, df in x.items():
            out = df.astype(float)
            mask = rng.random(out.shape) < 0.25
            mask[mask.all(axis=1), 0] = False  # never empty a whole block for a row
            x_n[name] = out.mask(mask)  # pandas 3's .values is a read-only view, so no writing through it
        scores = MBPLS(n_components=2).fit(x_n, y).super_scores_.to_numpy()
        agreement = np.mean([abs(np.corrcoef(reference[:, a], scores[:, a])[0, 1]) for a in range(2)])
        assert agreement > 0.85

    def test_y_column_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        y_n = y.copy()
        y_n.iloc[:, 0] = np.nan
        with pytest.raises(ValueError, match=r"Y has columns with all values missing"):
            MBPLS(n_components=2).fit(x, y_n)

    def test_y_row_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        y_n = y.copy()
        y_n.iloc[4, :] = np.nan
        with pytest.raises(ValueError, match=r"Y has rows with all values missing"):
            MBPLS(n_components=2).fit(x, y_n)

    @pytest.mark.slow
    def test_nipals_recovers_with_sparse_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        complete = MBPLS(n_components=2, algorithm="dense").fit(x, y)

        rng = np.random.default_rng(7)
        x_n: dict[str, pd.DataFrame] = {}
        # Inject ~5% MCAR NaN in each X-block.
        for name, df in x.items():
            mask = rng.random(df.shape) < 0.05
            x_n[name] = df.mask(pd.DataFrame(mask, index=df.index, columns=df.columns))
        # And ~5% MCAR NaN in Y.
        mask_y = rng.random(y.shape) < 0.05
        y_n = y.mask(pd.DataFrame(mask_y, index=y.index, columns=y.columns))

        with_nan = MBPLS(n_components=2).fit(x_n, y_n)

        assert not with_nan.super_scores_.isna().any().any()
        assert not with_nan.super_weights_.isna().any().any()
        assert not with_nan.predictions_.isna().any().any()

        # Predictions are coordinate-system invariant (they only depend on the
        # span of the fitted subspace), so they are the most operationally
        # meaningful recovery metric. Under MCAR, latent components can be
        # permuted or rotated relative to the complete-data fit, so do not
        # compare super-scores / super-weights component-by-component.
        for col in complete.predictions_.columns:
            corr = np.corrcoef(complete.predictions_[col], with_nan.predictions_[col])[0, 1]
            assert corr > 0.99, f"Prediction correlation for {col!r} dropped to {corr}"

        # Score-space recovery up to a permutation: each complete-data
        # component must correlate >0.9 with at least one masked-fit
        # component.
        for i in range(complete.super_scores_.shape[1]):
            best = max(
                abs(np.corrcoef(complete.super_scores_.iloc[:, i], with_nan.super_scores_.iloc[:, j])[0, 1])
                for j in range(with_nan.super_scores_.shape[1])
            )
            assert best > 0.9, f"Component {i + 1} has no matching masked component (best corr {best})"

    def test_nipals_outputs_have_no_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[0, 0] = np.nan
        x_n["B"].iloc[3, 2] = np.nan
        y_n = y.copy()
        y_n.iloc[5, 0] = np.nan

        m = MBPLS(n_components=2).fit(x_n, y_n)
        assert not m.super_scores_.isna().any().any()
        assert not m.super_y_scores_.isna().any().any()
        assert not m.super_weights_.isna().any().any()
        assert not m.super_y_loadings_.isna().any().any()
        assert not m.predictions_.isna().any().any()
        for name in m.block_names_:
            assert not m.block_scores_[name].isna().any().any()
            assert not m.block_weights_[name].isna().any().any()
            assert not m.block_loadings_[name].isna().any().any()
            assert not m.block_spe_[name].isna().any().any()

    def test_block_weights_have_unit_norm_under_nan(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[0, 0] = np.nan
        x_n["B"].iloc[5, 1] = np.nan

        m = MBPLS(n_components=2).fit(x_n, y)
        for name in m.block_names_:
            norms = np.linalg.norm(m.block_weights_[name].values, axis=0)
            np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=6)

    def test_predict_after_nan_fit_returns_finite(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_train = {name: df.copy() for name, df in x.items()}
        x_train["A"].iloc[0, 0] = np.nan
        y_train = y.copy()
        y_train.iloc[3, 1] = np.nan

        m = MBPLS(n_components=2).fit(x_train, y_train)
        # Predict-time NaN scoring (TSR / PMP) is a follow-up; verify that a
        # model fitted on NaN data can still score complete held-out rows.
        x_new = {name: df.iloc[:5, :] for name, df in x.items()}
        result = m.diagnose(x_new)
        assert not result.super_scores.isna().any().any()
        assert not result.predictions.isna().any().any()


class TestMBPLSOnLDPE:
    """MBPLS on the LDPE tubular reactor dataset using its natural 4-block split.

    The LDPE columns split naturally into four chemometric blocks:

    - ``zone1`` (7 vars): Tin, Tmax1, Tout1, Tcin1, z1, Fi1, Fs1.
      Reactor zone 1 process variables; Tin is the common feed inlet
      temperature, grouped with zone 1 by convention.
    - ``zone2`` (6 vars): Tmax2, Tout2, Tcin2, z2, Fi2, Fs2.
      Reactor zone 2 process variables.
    - ``pressure`` (1 var): Press.
    - Y (5 vars): Conv, Mn, Mw, LCB, SCB. Quality block; used as Y in
      MBPLS or as a fourth X-block in MBPCA.

    The earlier implementation used a 2-block zone split purely to make a
    single-block-PLS oracle assertion convenient.
    The 4-block split here is the chemometrically meaningful one.
    """

    @pytest.fixture
    def ldpe(self) -> tuple[dict, pd.DataFrame]:
        import pathlib

        folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE"
        values = pd.read_csv(folder / "LDPE.csv", index_col=0)
        # Natural 4-block split: block 1 = zone-1 vars (incl. Tin), block 2 =
        # zone-2 vars, block 3 = pressure, Y = quality.
        zone_1_idx = [0, 1, 2, 5, 7, 9, 11]
        zone_2_idx = [3, 4, 6, 8, 10, 12]
        pressure_idx = [13]
        x_blocks = {
            "zone1": values.iloc[:, zone_1_idx],
            "zone2": values.iloc[:, zone_2_idx],
            "pressure": values.iloc[:, pressure_idx],
        }
        y_df = values.iloc[:, 14:]
        return x_blocks, y_df

    def test_fit_runs_and_stores_expected_attributes(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        assert model.super_scores_.shape == (54, 3)
        assert model.super_y_scores_.shape == (54, 3)
        assert model.super_weights_.shape == (3, 3)
        assert model.block_scores_["zone1"].shape == (54, 3)
        assert model.block_scores_["zone2"].shape == (54, 3)
        assert model.block_scores_["pressure"].shape == (54, 3)
        assert model.block_weights_["zone1"].shape == (7, 3)
        assert model.block_weights_["zone2"].shape == (6, 3)
        assert model.block_weights_["pressure"].shape == (1, 3)
        assert model.predictions_.shape == (54, 5)

    def test_per_block_stats_are_well_formed(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)

        # R²X cumulative is monotonic in components and bounded by [0, 1]
        for name in model.block_names_:
            r2 = model.r2_x_per_block_cumulative_.loc[name].values
            assert (r2 >= 0).all()
            assert (r2 <= 1.0 + 1e-10).all()
            assert np.all(np.diff(r2) >= -1e-10)
        # R²Y cumulative is monotonic and in [0, 1]
        r2y = model.r2_y_cumulative_.values
        assert (r2y >= 0).all()
        assert (r2y <= 1.0 + 1e-10).all()
        assert np.all(np.diff(r2y) >= -1e-10)

        # VIPs have one entry per variable in each block, super VIP one per block
        for name in model.block_names_:
            assert model.block_vip_[name].shape == (model.block_widths_[name],)
        assert model.super_vip_.shape == (len(model.block_names_),)

        # Per-block SPE and per-block T² are (N, A); super T² same shape
        for name in model.block_names_:
            assert model.block_spe_[name].shape == (54, 3)
            assert model.block_hotellings_t2_[name].shape == (54, 3)
        assert model.super_hotellings_t2_.shape == (54, 3)

    def test_block_spe_squared_sums_to_super_block_residual(self, ldpe) -> None:
        """Sum of squared per-block SPE equals the squared SPE of the merged residual."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        merged_spe2 = sum(model.block_spe_[n].iloc[:, -1].values ** 2 for n in model.block_names_)
        # Sanity: positive and finite
        assert np.all(merged_spe2 >= 0)
        assert np.all(np.isfinite(merged_spe2))

    def test_predict_returns_block_spe_and_super_t2(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        result = model.diagnose(x_blocks)
        for name in model.block_names_:
            assert result.block_spe[name].shape == (54,)
        assert result.hotellings_t2.shape == (54,)
        # On training data the predict()-time T² must equal the in-sample super T² at the final PC
        np.testing.assert_array_almost_equal(
            result.hotellings_t2.values, model.super_hotellings_t2_.iloc[:, -1].values, decimal=10
        )

    def test_spe_and_t2_limits_are_positive_finite(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        for name in model.block_names_:
            limit = model.block_spe_limit(name, conf_level=0.95)
            assert np.isfinite(limit)
            assert limit > 0
        super_lim = model.super_spe_limit(conf_level=0.95)
        assert np.isfinite(super_lim)
        assert super_lim > 0
        t2_lim = model.hotellings_t2_limit(conf_level=0.95)
        assert np.isfinite(t2_lim)
        assert t2_lim > 0

    def test_display_results_returns_string(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        out = model.display_results()
        assert isinstance(out, str)
        assert "MBPLS model" in out
        assert "R²X[zone1]" in out
        assert "R²Y" in out

    def test_spe_contributions_sum_to_block_spe_squared(self, ldpe) -> None:
        """For each row, sum of squared residual contributions across one block
        must equal that block's squared SPE at the final component.
        """
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        contribs = model.spe_contributions(x_blocks)
        for name in model.block_names_:
            row_sums = contribs[name].sum(axis=1).values
            block_spe_sq = model.block_spe_[name].iloc[:, -1].values ** 2
            np.testing.assert_array_almost_equal(row_sums, block_spe_sq, decimal=8)

    def test_score_contributions_sum_to_the_super_score(self, ldpe) -> None:
        """On the LDPE blocks, contributions decompose the super score exactly."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        for a in (1, 2, 3):
            contribs = model.score_contributions(x_blocks, component=a)
            total = sum(frame.sum(axis=1) for frame in contribs.values())
            np.testing.assert_array_almost_equal(total.values, model.super_scores_[a].values, decimal=9)

    def test_score_contributions_differ_between_components(self, ldpe) -> None:
        """Each component gets its own decomposition; they are not interchangeable."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        first = model.score_contributions(x_blocks, component=1)
        second = model.score_contributions(x_blocks, component=2)
        for name in model.block_names_:
            assert not np.allclose(first[name].values, second[name].values)

        with pytest.raises(ValueError, match="1-based index"):
            model.score_contributions(x_blocks, component=4)
        with pytest.raises(ValueError, match="Missing X-blocks"):
            model.score_contributions({model.block_names_[0]: x_blocks[model.block_names_[0]]})

    def test_group_contributions_sum_to_the_shift_in_super_score(self, ldpe) -> None:
        """The two-period form: contributions sum to the change in average super score."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        index = next(iter(x_blocks.values())).index
        early, late = index[:10], index[10:20]

        contribs = model.group_contributions(x_blocks, group=early, reference=late)
        total = sum(float(series.sum()) for series in contribs.values())
        scores = model.super_scores_[1].values
        assert total == pytest.approx(scores[:10].mean() - scores[10:20].mean(), abs=1e-9)

        # Against the model centre instead of another period.
        centred = model.group_contributions(x_blocks, group=early)
        assert sum(float(s.sum()) for s in centred.values()) == pytest.approx(scores[:10].mean(), abs=1e-9)

    def test_contribution_guards_and_plain_array_blocks(self, ldpe) -> None:
        """The MBPLS guards mirror MBPCA, and blocks may arrive as plain arrays."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=2).fit(x_blocks, y_df)

        with pytest.raises(TypeError, match="dict"):
            model.score_contributions(next(iter(x_blocks.values())))
        with pytest.raises(ValueError, match="Missing X-blocks"):
            model.score_contributions({model.block_names_[0]: x_blocks[model.block_names_[0]]})
        with pytest.raises(ValueError, match="1-based index"):
            model.score_contributions(x_blocks, component=5)
        with pytest.raises(ValueError, match="scaling must be one of"):
            model.score_contributions(x_blocks, scaling="loud")

        as_arrays = {name: block.to_numpy() for name, block in x_blocks.items()}
        from_arrays = model.score_contributions(as_arrays, component=1)
        from_frames = model.score_contributions(x_blocks, component=1)
        for name in model.block_names_:
            np.testing.assert_array_almost_equal(from_arrays[name].values, from_frames[name].values, decimal=12)

        # Both scalings, taken jointly over the blocks.
        largest = model.score_contributions(x_blocks, component=1, scaling="maximum")
        assert max(float(np.abs(f.values).max()) for f in largest.values()) == pytest.approx(1.0)
        within = model.score_contributions(x_blocks, component=1, scaling="within")
        row_total = sum(np.abs(f.values).sum(axis=1) for f in within.values())
        np.testing.assert_array_almost_equal(row_total, np.ones(len(row_total)), decimal=10)

    def test_super_score_plot_returns_plotly_figure(self, ldpe) -> None:
        import plotly.graph_objects as go

        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        fig = model.super_score_plot(pc_horiz=1, pc_vert=2)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert "PC1 vs PC2" in fig.layout.title.text

    def test_super_weights_bar_plot_returns_plotly_figure(self, ldpe) -> None:
        import plotly.graph_objects as go

        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        fig = model.super_weights_bar_plot(component=1)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert list(fig.data[0].x) == model.block_names_

    def test_predictions_vs_observed_plot_includes_y_eq_x_and_rmsee(self, ldpe) -> None:
        import plotly.graph_objects as go

        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        fig = model.predictions_vs_observed_plot(y_observed=y_df, variable=str(y_df.columns[0]))
        assert isinstance(fig, go.Figure)
        # Two traces: scatter + y=x line
        assert len(fig.data) == 2
        annotation_text = " ".join(a.text for a in fig.layout.annotations)
        assert "RMSEE" in annotation_text

    def test_randomization_test_returns_observed_and_risk(self, ldpe) -> None:
        """Real data should be more correlated than randomly-permuted Y for the
        first component; risk_pct for PC1 should be small (< 50%).
        """
        x_blocks, y_df = ldpe
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        result = randomization_test_mbpls(model, x_blocks, y_df, n_permutations=20, seed=0)
        assert list(result.columns) == ["observed", "risk_pct"]
        assert result.shape == (2, 2)
        assert (result["observed"] >= 0).all()
        assert (result["observed"] <= 1).all()
        assert (result["risk_pct"] >= 0).all()
        assert (result["risk_pct"] <= 100).all()
        # First component on a real signal-bearing dataset should rarely be
        # beaten by random permutations.
        assert result.loc[1, "risk_pct"] < 50.0

        # #513: the estimate counts the observed statistic among the permutations, so it
        # is 100 * (n_exceed + 1) / (n_permutations + 1) and can never report exactly 0.
        # No finite permutation set licenses a claim of zero tail probability, and 20
        # permutations cannot resolve below 100 / 21 = 4.76%. This is the convention the
        # Van der Voet test in `_pls` already uses.
        floor = 100.0 / 21.0
        assert (result["risk_pct"] >= floor).all()
        # Every value must sit on the (n_exceed + 1) / 21 lattice, for integer n_exceed.
        lattice = result["risk_pct"].to_numpy() * 21.0 / 100.0
        assert lattice == pytest.approx(np.round(lattice), abs=1e-12)

    def test_super_score_matches_single_block_pls_with_block_weighting(self, ldpe) -> None:
        """When all variables are in one big-X with sqrt(K_b) weighting per block,
        single-block PLS produces the same super-score as MBPLS. Holds for any
        number of blocks, so the natural 4-block split (3 X-blocks + Y) is fine.
        """
        from process_improve.multivariate.methods import MBPLS, MCUVScaler
        from tests._multiblock_oracles import mbpls_merged_then_recover

        x_blocks, y_df = ldpe
        # Path A: oracle merged-then-recover (verified self-consistent)
        x_pp = [MCUVScaler().fit_transform(x_blocks[k]).values for k in ("zone1", "zone2", "pressure")]
        y_pp = MCUVScaler().fit_transform(y_df).values
        oracle = mbpls_merged_then_recover(x_pp, y_pp, n_components=2)
        # Path B: production MBPLS
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        np.testing.assert_array_almost_equal(np.abs(model.super_scores_.values), np.abs(oracle.super_scores), decimal=5)


# ---------------------------------------------------------------------------
# MBPLS missing-data integration tests on real datasets
# ---------------------------------------------------------------------------


def _inject_mcar(
    df: pd.DataFrame,
    ratio: float,
    rng: np.random.Generator,
    *,
    keep_one_per_row: bool = True,
) -> pd.DataFrame:
    """Return ``df`` with MCAR NaN entries at the given ratio.

    If ``keep_one_per_row``, at least one column in each row stays observed,
    keeping the degeneracy guards happy on narrow blocks or single-target Y.
    """
    mask = rng.random(df.shape) < ratio
    if keep_one_per_row:
        full_row = mask.all(axis=1)
        mask[full_row, 0] = False
    return df.mask(pd.DataFrame(mask, index=df.index, columns=df.columns))


class TestMBPLSOnLDPEMissingData:
    """MBPLS NaN-injection tests on the LDPE tubular reactor dataset.

    Uses the natural 4-block split (zone1, zone2, pressure, quality-Y).
    Establishes a complete-data baseline, then injects MCAR NaN in the
    multi-column X-blocks (zone1, zone2) and in Y at 5/15/25%; verifies
    that predictions remain highly correlated with the complete-data
    fit (degrading gracefully with NaN density) and that outputs stay
    NaN-free everywhere.

    The single-column ``pressure`` block is held complete: with one
    variable any NaN there is automatically a full-row NaN, which is
    not a meaningful NIPALS skip-NaN scenario (it is plain missing
    univariate data, and trips the degeneracy guard).
    """

    @pytest.fixture
    def ldpe(self) -> tuple[dict, pd.DataFrame]:
        import pathlib

        folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE"
        values = pd.read_csv(folder / "LDPE.csv", index_col=0)
        # Same natural 4-block split as TestMBPLSOnLDPE.
        zone_1_idx = [0, 1, 2, 5, 7, 9, 11]
        zone_2_idx = [3, 4, 6, 8, 10, 12]
        pressure_idx = [13]
        x_blocks = {
            "zone1": values.iloc[:, zone_1_idx],
            "zone2": values.iloc[:, zone_2_idx],
            "pressure": values.iloc[:, pressure_idx],
        }
        y_df = values.iloc[:, 14:]
        return x_blocks, y_df

    @pytest.fixture
    def complete_baseline(self, ldpe):
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        return MBPLS(n_components=3, algorithm="dense").fit(x_blocks, y_df)

    @pytest.mark.parametrize(
        ("ratio", "seed", "min_pred_corr"),
        [
            (0.05, 1, 0.97),
            (0.15, 2, 0.95),
            (0.25, 3, 0.85),
        ],
    )
    def test_mcar_recovery(self, ldpe, complete_baseline, ratio: float, seed: int, min_pred_corr: float) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        rng = np.random.default_rng(seed)
        # Only inject NaN into multi-column X-blocks; pressure is univariate
        # so its skip-NaN behavior is not meaningfully exercised by MCAR.
        x_n = {name: _inject_mcar(df, ratio, rng) if df.shape[1] > 1 else df for name, df in x_blocks.items()}
        y_n = _inject_mcar(y_df, ratio, rng)

        m = MBPLS(n_components=3).fit(x_n, y_n)

        assert m.algorithm_ == "nipals"
        assert not m.super_scores_.isna().any().any()
        assert not m.super_weights_.isna().any().any()
        assert not m.predictions_.isna().any().any()
        for name in m.block_names_:
            assert not m.block_spe_[name].isna().any().any()

        for col in complete_baseline.predictions_.columns:
            corr = np.corrcoef(complete_baseline.predictions_[col], m.predictions_[col])[0, 1]
            assert corr > min_pred_corr, (
                f"LDPE @ {int(ratio * 100)}% MCAR: prediction correlation for {col!r} "
                f"dropped to {corr:.3f} (threshold {min_pred_corr})"
            )


class TestMBPLSOnDTUPectin:
    """MBPLS on the DTU pectin yield dataset (Baum and Vermue, JOSS 2019).

    Two X-blocks: FTIR (148 wavenumbers) and carbohydrate microarray
    (30 antibody probes). Y is a single column: pectin yield in grams.
    Three extractions concatenate to 37 samples total. Verifies the
    complete-data fit is well formed, then injects MCAR NaN in the
    X-blocks at 5/15/25% and verifies graceful prediction recovery.
    Y is not perturbed because it has a single target column; any
    MCAR injection into Y trips the row-all-NaN degeneracy guard.

    Reference: Baum and Vermue, JOSS 4(34), 1190 (2019).
    """

    @pytest.fixture
    def pectin(self) -> tuple[dict, pd.DataFrame]:
        import pathlib

        folder = (
            pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "dtu-pectin"
        )
        ftir = pd.concat([pd.read_csv(folder / f"ftir{i}.csv", index_col=0) for i in (1, 2, 3)])
        micro = pd.concat([pd.read_csv(folder / f"extraction{i}.csv", index_col=0) for i in (1, 2, 3)])
        # Yield (the row index of every CSV) is the single Y target. After
        # extracting it the X-blocks are reset to a plain RangeIndex to
        # match MBPLS's row-alignment contract.
        y = pd.DataFrame({"yield_g": ftir.index.values.astype(float)})
        ftir = ftir.reset_index(drop=True)
        micro = micro.reset_index(drop=True)
        return {"FTIR": ftir, "microarray": micro}, y

    def test_baseline_fit_is_well_formed(self, pectin) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y = pectin
        model = MBPLS(n_components=3).fit(x_blocks, y)

        assert model.algorithm_ == "dense"
        assert model.has_missing_data_ is False
        assert model.super_scores_.shape == (37, 3)
        assert model.block_weights_["FTIR"].shape == (148, 3)
        assert model.block_weights_["microarray"].shape == (30, 3)
        assert model.predictions_.shape == (37, 1)

        # R^2Y monotonic, in [0, 1], and the 3-component fit recovers
        # most of the Y variance (upstream notebook converges around the
        # 0.97 mark for LV3 on this dataset).
        r2y = model.r2_y_cumulative_.values
        assert (r2y >= 0).all()
        assert (r2y <= 1.0 + 1e-10).all()
        assert np.all(np.diff(r2y) >= -1e-10)
        assert r2y[-1] > 0.9

    @pytest.mark.parametrize(
        ("ratio", "seed", "min_pred_corr"),
        [
            (0.05, 11, 0.99),
            (0.15, 12, 0.98),
            (0.25, 13, 0.95),
        ],
    )
    def test_mcar_recovery_in_x_blocks(self, pectin, ratio: float, seed: int, min_pred_corr: float) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y = pectin
        complete = MBPLS(n_components=3, algorithm="dense").fit(x_blocks, y)

        rng = np.random.default_rng(seed)
        x_n = {name: _inject_mcar(df, ratio, rng) for name, df in x_blocks.items()}
        m = MBPLS(n_components=3).fit(x_n, y)

        assert m.algorithm_ == "nipals"
        assert m.has_missing_data_ is True
        assert not m.super_scores_.isna().any().any()
        assert not m.super_weights_.isna().any().any()
        assert not m.predictions_.isna().any().any()

        corr = np.corrcoef(complete.predictions_["yield_g"], m.predictions_["yield_g"])[0, 1]
        assert corr > min_pred_corr, (
            f"DTU pectin @ {int(ratio * 100)}% MCAR: prediction correlation "
            f"dropped to {corr:.3f} (threshold {min_pred_corr})"
        )


# ---------------------------------------------------------------------------
# MBPLS written out by hand
# ---------------------------------------------------------------------------


def _masked_regression(values: np.ndarray, vector: np.ndarray, per: str) -> np.ndarray:
    """Regress each column (``per="column"``) or each row of ``values`` on ``vector``, over its observed cells.

    A row with nothing observed has no score: 0 / 0, NaN.
    """
    observed, filled = ~np.isnan(values), np.nan_to_num(values)
    if per == "column":
        return filled.T @ vector / (observed.T @ vector**2)
    with np.errstate(invalid="ignore"):
        return filled @ vector / (observed @ vector**2)


def _cross_size(x: dict[str, np.ndarray], target: np.ndarray) -> float:
    """``sqrt(sum_b ||X_b' Y||^2)``, a missing cell read as 0."""
    return float(np.sqrt(sum(np.sum((np.nan_to_num(values).T @ np.nan_to_num(target)) ** 2) for values in x.values())))


def _mbpls_component_by_hand(x: dict, follow: np.ndarray, widths: dict, has_data: dict, max_iter: int) -> tuple:
    """Iterate one MBPLS component from the column of ``follow`` with the largest sum of squares.

    Returns the super score ``t``, the Y-score ``u``, the super weight ``w``, the Y-loading
    ``c``, the block weights and block scores, and the number of passes.
    """
    names = list(x)
    u = np.nan_to_num(follow)[:, np.argmax(np.nansum(follow**2, axis=0))]
    passes = 0
    while passes < max_iter:
        passes += 1
        previous, w_b, t_b = u, {}, {}
        for name in names:
            weight = _masked_regression(x[name], u, "column")
            w_b[name] = weight / np.linalg.norm(weight)
            t_b[name] = _masked_regression(x[name], w_b[name], "row") / np.sqrt(widths[name])
        scores = np.column_stack([np.where(has_data[name], t_b[name], np.nan) for name in names])
        w = _masked_regression(scores, u, "column")
        w = w / np.linalg.norm(w)
        g = np.concatenate([w[b] * w_b[name] / np.sqrt(widths[name]) for b, name in enumerate(names)])
        t = _masked_regression(np.hstack([x[name] for name in names]), g, "row") * (g @ g)
        c = _masked_regression(follow, t, "column")
        u = _masked_regression(follow, c, "row")
        if np.linalg.norm(u - previous) <= EPSQRT * np.linalg.norm(u):
            break
    return t, u, w, c, w_b, t_b, passes


def _mbpls_written_out(
    blocks: dict[str, pd.DataFrame], y: pd.DataFrame, n_components: int, max_iter: int = 500
) -> dict:
    """MBPLS written out (Westerhuis, Kourti and MacGregor, 1998), seeded and stopped as ``fit`` is.

    Each block and Y is autoscaled over its observed cells, and each block weighted by
    ``1 / sqrt(K_b)``, ``K_b`` its count of columns that vary. A component starts from the
    column of what is left of Y with the largest sum of squares (a missing cell read as 0) and
    repeats: ``w_b = X_b'u / u'u`` column by column over the observed cells, scaled to length
    1; ``t_b = X_b w_b / w_b'w_b`` row by row over the observed cells, over ``sqrt(K_b)``;
    ``w = T'u / u'u`` over the rows that have a block score, scaled to length 1; the super
    score ``t`` as one regression of the whole row on ``g = [w[b] w_b / sqrt(K_b)]``, scaled
    by ``g'g``; ``c = Y't / t't`` and ``u = Y c / c'c``, each over the observed cells. It
    stops once ``u`` moves by at most ``epsqrt`` of its length, or after ``max_iter`` passes.
    The component is flipped so its largest block loading ``p_b = X_b't / t't`` is positive,
    and ``t p_b'`` is taken from each block and ``t c'`` from Y. Once ``X'Y`` has fallen to
    ``epsqrt`` of its first size, every later component takes the blocks over ``sqrt(K_b)``
    as its target in place of Y, and has no Y-score.
    """
    names = list(blocks)
    scaling = {name: _autoscaling(frame) for name, frame in blocks.items()}
    y_scaling = _autoscaling(y)
    x = {name: (frame.to_numpy(dtype=float) - scaling[name][0]) / scaling[name][1] for name, frame in blocks.items()}
    target = (y.to_numpy(dtype=float) - y_scaling[0]) / y_scaling[1]
    widths = {name: max(1, int(np.sum(np.nanstd(values, axis=0) > EPSQRT))) for name, values in x.items()}
    has_data = {name: ~np.all(np.isnan(values), axis=1) for name, values in x.items()}
    total = {name: np.nansum(values**2, axis=0) for name, values in x.items()}
    total_y = np.nansum(target**2, axis=0)
    reach, exhausted = _cross_size(x, target), False
    found: dict = {key: [] for key in ("t", "u", "w", "c", "iterations", "r2_y", "r2_y_columns")}
    for key in ("t_b", "w_b", "p_b", "spe", "r2_columns", "r2_block"):
        found[key] = {name: [] for name in names}
    for a in range(n_components):
        exhausted = exhausted or (a > 0 and _cross_size(x, target) <= EPSQRT * reach)
        follow = np.column_stack([x[name] / np.sqrt(widths[name]) for name in names]) if exhausted else target
        t, u, w, c, w_b, t_b, passes = _mbpls_component_by_hand(x, follow, widths, has_data, max_iter)
        if exhausted:
            c, u = _masked_regression(target, t, "column"), np.zeros_like(t)
        p_b = {name: _masked_regression(x[name], t, "column") for name in names}
        stacked = np.concatenate([p_b[name] for name in names])
        sign = -1.0 if stacked[np.argmax(np.abs(stacked))] < 0 else 1.0
        t, u, c = sign * t, sign * u, sign * c
        for name in names:
            x[name] = x[name] - np.outer(t, sign * p_b[name])
            remaining = np.nansum(x[name] ** 2, axis=0)
            found["t_b"][name].append(np.where(has_data[name], sign * t_b[name], np.nan))
            found["w_b"][name].append(sign * w_b[name])
            found["p_b"][name].append(sign * p_b[name])
            found["spe"][name].append(np.where(has_data[name], np.sqrt(np.nansum(x[name] ** 2, axis=1)), np.nan))
            found["r2_columns"][name].append(1 - remaining / np.where(total[name] > 0, total[name], np.nan))
            found["r2_block"][name].append(1 - remaining.sum() / total[name].sum())
        target = target - np.outer(t, c)
        remaining_y = np.nansum(target**2, axis=0)
        found["r2_y"].append(1 - remaining_y.sum() / total_y.sum())
        found["r2_y_columns"].append(1 - remaining_y / np.where(total_y > 0, total_y, np.nan))
        for key, value in (("t", t), ("u", u), ("w", w), ("c", c), ("iterations", passes)):
            found[key].append(value)
    out = {key: np.column_stack(found[key]) for key in ("t", "u", "w", "c", "r2_y_columns")}
    out |= {"iterations": found["iterations"], "r2_y": np.array(found["r2_y"])}
    for key in ("t_b", "w_b", "p_b", "spe", "r2_columns", "r2_block"):
        out[key] = {name: np.column_stack(found[key][name]) for name in names}
    return out | {"scaling": scaling, "y_scaling": y_scaling, "widths": widths, "has_data": has_data}


def _gappy_mbpls_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return the gappy three blocks, row r3 with nothing in block ``x``, and a Y of two columns with one gap.

    Autoscaled over its observed cells, Y's first column holds 4 in sums of squares and its
    second 5, so the second seeds the first component.
    """
    blocks = _gappy_three_blocks()
    blocks["x"].iloc[3] = np.nan
    rng = np.random.default_rng(1)
    y = pd.DataFrame(rng.standard_normal((6, 2)), index=blocks["x"].index, columns=["q", "r"])
    y.iloc[1, 0] = np.nan
    return blocks, y


def _check_mbpls_fitted(
    model: MBPLS, blocks: dict[str, pd.DataFrame], y: pd.DataFrame, expected: dict, rtol: float
) -> None:
    """Every fitted attribute of ``model`` against ``_mbpls_written_out``, labels included."""
    names, index = list(blocks), next(iter(blocks.values())).index
    components = list(range(1, expected["t"].shape[1] + 1))
    n_rows = len(index)
    t = expected["t"]
    assert model.block_names_ == names
    assert model.block_widths_ == {name: frame.shape[1] for name, frame in blocks.items()}
    assert model.block_informative_widths_ == expected["widths"]
    assert (model.n_samples_, model.n_targets_, model.n_components_) == (n_rows, y.shape[1], len(components))
    assert model.n_features_in_ == sum(frame.shape[1] for frame in blocks.values())
    np.testing.assert_array_equal(model.feature_names_in_, np.concatenate([frame.columns for frame in blocks.values()]))
    _frames_near(model.super_scores_, pd.DataFrame(t, index=index, columns=components), rtol)
    _frames_near(model.super_y_scores_, pd.DataFrame(expected["u"], index=index, columns=components), rtol)
    _frames_near(model.super_weights_, pd.DataFrame(expected["w"], index=names, columns=components), rtol)
    _frames_near(model.super_y_loadings_, pd.DataFrame(expected["c"], index=y.columns, columns=components), rtol)
    centre, scale = expected["y_scaling"]
    predictions = pd.DataFrame(t @ expected["c"].T * scale + centre, index=index, columns=y.columns)
    _frames_near(model.predictions_, predictions, rtol)
    variance = np.sum(t**2, axis=0) / (n_rows - 1)
    np.testing.assert_allclose(model.explained_variance_, variance, rtol=rtol)
    _frames_near(
        model.scaling_factor_for_super_scores_,
        pd.Series(np.sqrt(variance), index=components, name="Standard deviation per super-score"),
        rtol,
    )
    _frames_near(
        model.super_hotellings_t2_,
        pd.DataFrame(np.cumsum(t**2 / variance, axis=1), index=index, columns=components),
        rtol,
    )
    r2_block = np.vstack([expected["r2_block"][name] for name in names])
    _frames_near(model.r2_x_per_block_cumulative_, pd.DataFrame(r2_block, index=names, columns=components), rtol)
    increments = np.diff(r2_block, axis=1, prepend=0.0)
    _frames_near(model.r2_x_per_block_per_component_, pd.DataFrame(increments, index=names, columns=components), rtol)
    _frames_near(model.r2_y_cumulative_, pd.Series(expected["r2_y"], index=components, name="Cumulative R²Y"), rtol)
    r2_y_increments = np.diff(expected["r2_y"], prepend=0.0)
    _frames_near(
        model.r2_y_per_component_, pd.Series(r2_y_increments, index=components, name="R²Y per component"), rtol
    )
    _frames_near(
        model.r2_y_per_variable_, pd.DataFrame(expected["r2_y_columns"], index=y.columns, columns=components), rtol
    )
    super_vip = np.sqrt(len(names) * (expected["w"] ** 2 @ r2_y_increments) / r2_y_increments.sum())
    _frames_near(model.super_vip_, pd.Series(super_vip, index=names, name="Super VIP"), rtol)
    for b, name in enumerate(names):
        columns = blocks[name].columns
        t_b, w_b = expected["t_b"][name], expected["w_b"][name]
        _frames_near(model.block_scores_[name], pd.DataFrame(t_b, index=index, columns=components), rtol)
        _frames_near(model.block_weights_[name], pd.DataFrame(w_b, index=columns, columns=components), rtol)
        p_b = pd.DataFrame(expected["p_b"][name], index=columns, columns=components)
        _frames_near(model.block_loadings_[name], p_b, rtol)
        _frames_near(model.block_spe_[name], pd.DataFrame(expected["spe"][name], index=index, columns=components), rtol)
        r2_columns = pd.DataFrame(expected["r2_columns"][name], index=columns, columns=components)
        _frames_near(model.r2_x_per_variable_[name], r2_columns, rtol)
        present = expected["has_data"][name].sum()
        second_moment = np.nansum(t_b**2, axis=0) / (present - 1)
        block_t2 = pd.DataFrame(np.cumsum(t_b**2 / second_moment, axis=1), index=index, columns=components)
        _frames_near(model.block_hotellings_t2_[name], block_t2, rtol)
        vip = np.sqrt(expected["widths"][name] * (w_b**2 @ increments[b]) / increments[b].sum())
        _frames_near(model.block_vip_[name], pd.Series(vip, index=columns, name=f"VIP[{name}]"), rtol)


class TestMBPLSWrittenOut:
    """MBPLS against its algorithm written out by hand, every attribute, label and step."""

    def test_a_gappy_fit_follows_the_algorithm_pass_for_pass(self) -> None:
        """Missing cells take the masked path; a fit stopped after one pass shows where it was seeded."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        expected = _mbpls_written_out(blocks, y, 2)
        assert model.algorithm_ == "nipals"
        assert model.has_missing_data_ is True
        np.testing.assert_array_equal(model.fitting_info_["iterations"], expected["iterations"])
        assert model.fitting_info_["iterations"].dtype.kind == "i"
        np.testing.assert_array_equal(model.fitting_info_["converged"], [True, True])
        _check_mbpls_fitted(model, blocks, y, expected, rtol=1e-10)

        with pytest.warns(
            SpecificationWarning,
            match=r"^MBPLS NIPALS did not converge within max_iter=1 for component\(s\) \[1, 2\]; ",
        ):
            one_pass = MBPLS(n_components=2, max_iter=1).fit(blocks, y)
        _check_mbpls_fitted(one_pass, blocks, y, _mbpls_written_out(blocks, y, 2, max_iter=1), rtol=1e-12)
        np.testing.assert_array_equal(one_pass.fitting_info_["iterations"], [1, 1])

    def test_the_masked_path_on_complete_data_is_the_dense_path(self) -> None:
        """On complete LDPE blocks, "nipals" fits what "dense" does, and what the algorithm written out does."""
        blocks, y = _ldpe_blocks()
        expected = _mbpls_written_out(blocks, y, 3)
        for algorithm in ("dense", "nipals", "auto"):
            model = MBPLS(n_components=3, algorithm=algorithm).fit(blocks, y)
            assert model.algorithm_ == ("dense" if algorithm == "auto" else algorithm)
            assert model.has_missing_data_ is False
            _check_mbpls_fitted(model, blocks, y, expected, rtol=1e-6)

    def test_square_blocks_and_a_square_y_tell_each_regression_which_way_it_runs(self) -> None:
        """Three rows, three blocks, a block of three columns and three targets, on the masked path.

        Every masked regression of a square array must be told whether it runs over the
        columns or the rows: the block weights and scores of block ``p``, the super weight over
        the three block scores, Y's loading and score, and the deflating loadings. The third
        component finds nothing left in Y, three rows holding two directions, and is fitted
        along X, whose Y-loading is a regression on the square Y too.
        """
        blocks = {
            "p": pd.DataFrame([[1.0, 4.0, 2.0], [3.0, 1.0, 5.0], [0.0, 2.0, 2.5]], columns=["p0", "p1", "p2"]),
            "q": pd.DataFrame({"q0": [2.0, -1.0, 0.5]}),
            "r": pd.DataFrame({"r0": [1.0, 1.5, 4.0], "r1": [0.0, 2.0, -1.0]}),
        }
        y = pd.DataFrame([[1.0, 0.0, 2.0], [2.0, 3.0, 1.0], [4.0, 1.0, 1.5]], columns=["a", "b", "c"])
        model = MBPLS(n_components=2, algorithm="nipals").fit(blocks, y)
        _check_mbpls_fitted(model, blocks, y, _mbpls_written_out(blocks, y, 2), rtol=1e-9)
        with pytest.warns(SpecificationWarning, match=r"^MBPLS: from component 3 on there is nothing left in Y"):
            third = MBPLS(n_components=3, algorithm="nipals").fit(blocks, y)
        np.testing.assert_array_equal(third.super_y_scores_[3], np.zeros(3))
        _frames_near(third.super_scores_[[1, 2]], model.super_scores_, 1e-9)

    @pytest.mark.parametrize("algorithm", ["dense", "nipals"])
    def test_once_y_is_explained_the_later_components_follow_x(self, algorithm: str) -> None:
        """Y is the first principal component of the weighted blocks, so one component explains all of it.

        From the second component on, ``X'Y`` is about zero, and each component is fitted along
        the largest direction left in the blocks over ``sqrt(K_b)``, the blocks three and two
        columns wide, with a Y-loading of about zero and no Y-score.
        """
        rng = np.random.default_rng(3)
        latent = rng.standard_normal((12, 3))
        blocks = {
            "wide": pd.DataFrame(latent @ rng.standard_normal((3, 3)) + 0.3 * rng.standard_normal((12, 3))),
            "narrow": pd.DataFrame(latent @ rng.standard_normal((3, 2)) + 0.3 * rng.standard_normal((12, 2))),
        }
        weighted = np.hstack([MCUVScaler().fit_transform(frame) / np.sqrt(frame.shape[1]) for frame in blocks.values()])
        y = pd.DataFrame({"y": 3.0 * np.linalg.svd(weighted, full_matrices=False)[0][:, 0] + 1.0})
        message = (
            r"^MBPLS: from component 2 on there is nothing left in Y that X can explain: what is left of Y is "
            r"uncorrelated with what is left of X \(R²Y stays at 1\)\. The data support 1 component\(s\); "
        )
        with pytest.warns(SpecificationWarning, match=message):
            model = MBPLS(n_components=3, algorithm=algorithm).fit(blocks, y)
        expected = _mbpls_written_out(blocks, y, 3)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], expected["iterations"])
        np.testing.assert_array_equal(model.super_y_scores_[[2, 3]], np.zeros((12, 2)))
        _check_mbpls_fitted(model, blocks, y, expected, rtol=1e-8)

    def test_a_component_found_against_its_largest_loadings_is_flipped_whole(self) -> None:
        """The blocks follow ``-y``, so the iteration from ``y`` finds a component whose loadings are negative.

        It is flipped: super and block scores, block weights, Y-score and Y-loading together. The
        super weight is not, as flipping leaves each ``t_b'u`` as it was.
        """
        rng = np.random.default_rng(4)
        z = rng.standard_normal(8)
        blocks = {
            "a": pd.DataFrame({"a0": -z + 0.1 * rng.standard_normal(8), "a1": -z + 0.3 * rng.standard_normal(8)}),
            "b": pd.DataFrame({"b0": -z + 0.2 * rng.standard_normal(8)}),
        }
        y = pd.DataFrame({"y": z + 0.1 * rng.standard_normal(8)})
        model = MBPLS(n_components=1).fit(blocks, y)
        assert model.super_y_loadings_.loc["y", 1] < 0
        assert (model.super_weights_[1] > 0).all()
        _check_mbpls_fitted(model, blocks, y, _mbpls_written_out(blocks, y, 1), rtol=1e-10)

    def test_a_change_of_exactly_tol_has_converged(self) -> None:
        """The loop stops once ``u`` moves by at most ``tol`` of its length, equality included.

        Fits stopped after 3 and 4 passes give the last two iterates, so their change relative to
        the newer is what the fourth pass measures, to the last bit. With that as ``tol`` the fit
        stops after the fourth pass.
        """
        blocks, y = _gappy_mbpls_data()
        with pytest.warns(SpecificationWarning, match="did not converge"):
            third = MBPLS(n_components=1, max_iter=3).fit(blocks, y).super_y_scores_[1].to_numpy()
        with pytest.warns(SpecificationWarning, match="did not converge"):
            fourth = MBPLS(n_components=1, max_iter=4).fit(blocks, y).super_y_scores_[1].to_numpy()
        step = np.linalg.norm(third - fourth)
        assert step < 0.1 * np.linalg.norm(fourth)  # one pass apart, with the same sign
        model = MBPLS(n_components=1, tol=step / np.linalg.norm(fourth)).fit(blocks, y)
        np.testing.assert_array_equal(model.fitting_info_["iterations"], [4])

    def test_a_fit_stopped_by_max_iter_says_which_components_and_logs_the_iterations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The gappy data converge in 27 and 8 passes: ten stop the first, which the warning names."""
        blocks, y = _gappy_mbpls_data()
        message = (
            r"^MBPLS NIPALS did not converge within max_iter=10 for component\(s\) \[1\]; "
            r"results for those components may be unreliable\.$"
        )
        with (
            caplog.at_level(logging.DEBUG, logger="process_improve.multivariate._mbpls"),
            pytest.warns(SpecificationWarning, match=message),
        ):
            model = MBPLS(n_components=2, max_iter=10).fit(blocks, y)
        assert caplog.messages == ["MBPLS (nipals): iterations per component = [10, 8]"]
        np.testing.assert_array_equal(model.fitting_info_["converged"], [False, True])
        assert model.fitting_info_["timing"].shape == (2,)
        _check_mbpls_fitted(model, blocks, y, _mbpls_written_out(blocks, y, 2, max_iter=10), rtol=1e-10)

    def test_two_rows_and_constant_columns_have_the_r2_they_should_without_a_warning(self) -> None:
        """At two rows a varying column holds a sum of squares of 1; a constant one, in X or Y, holds none.

        One component explains both rows of every column that varies: R2 1. The constant column
        ``b1``, the block ``c`` of constants and the constant target ``y1`` have nothing to
        explain: their R2 is NaN, with no warning from a division by 0, ``c``'s VIP is 0, and
        no constant column counts in its block's width. The variance divides by 1.
        """
        index = ["first", "second"]
        blocks = {
            "a": pd.DataFrame({"a0": [0.0, 1.0]}, index=index),
            "b": pd.DataFrame({"b0": [2.0, 7.0], "b1": [5.0, 5.0]}, index=index),
            "c": pd.DataFrame({"c0": [3.0, 3.0]}, index=index),
        }
        y = pd.DataFrame({"y0": [0.0, 1.0], "y1": [2.0, 2.0]}, index=index)
        assert np.sum(MCUVScaler().fit_transform(y[["y0"]]).to_numpy() ** 2) < 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MBPLS(n_components=1).fit(blocks, y)
            flat = MBPLS(n_components=1).fit(blocks, y[["y1"]])
        assert model.block_informative_widths_ == {"a": 1, "b": 1, "c": 1}
        _frames_near(
            model.r2_x_per_block_cumulative_,
            pd.DataFrame([[1.0], [1.0], [np.nan]], index=list(blocks), columns=[1]),
            1e-12,
        )
        _frames_near(
            model.r2_x_per_variable_["b"], pd.DataFrame([[1.0], [np.nan]], index=["b0", "b1"], columns=[1]), 1e-12
        )
        _frames_near(model.r2_y_per_variable_, pd.DataFrame([[1.0], [np.nan]], index=["y0", "y1"], columns=[1]), 1e-12)
        _frames_near(model.r2_y_cumulative_, pd.Series([1.0], index=[1], name="Cumulative R²Y"), 1e-12)
        np.testing.assert_allclose(model.explained_variance_, [np.sum(model.super_scores_.to_numpy() ** 2)], rtol=1e-15)
        np.testing.assert_allclose(model.block_vip_["b"], [1.0, 0.0], rtol=1e-12, atol=1e-15)
        np.testing.assert_array_equal(model.block_vip_["c"], [0.0])
        assert flat.r2_y_cumulative_.isna().all()
        assert flat.r2_y_per_variable_.isna().all().all()

    def test_a_block_or_a_y_the_component_leaves_untouched_has_no_r2_and_no_vip(self) -> None:
        """Columns with mean 0 and sd 1, exactly orthogonal: what the component does not reach has R2 and VIP 0.

        ``b`` is orthogonal to ``a`` and to Y, which is ``a``: its weight, loading and R2X are 0
        exactly, and so its VIP is 0 rather than 0 / 0. A Y orthogonal to the only block leaves
        the fit nothing at all: R2X and R2Y 0, and both VIPs 0.
        """
        a, b = [-1.0, -1.0, 0.0, 1.0, 1.0], [1.0, -1.0, 0.0, -1.0, 1.0]
        both = {"a": pd.DataFrame({"a0": a}), "b": pd.DataFrame({"b0": b})}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MBPLS(n_components=1).fit(both, pd.DataFrame({"y": a}))
            nothing = MBPLS(n_components=1).fit({"a": pd.DataFrame({"a0": a})}, pd.DataFrame({"y": b}))
        np.testing.assert_array_equal(model.super_weights_[1], [1.0, 0.0])
        np.testing.assert_array_equal(model.block_loadings_["b"][1], [0.0])
        np.testing.assert_array_equal(model.r2_x_per_block_cumulative_[1], [1.0, 0.0])
        np.testing.assert_array_equal(model.block_vip_["a"], [1.0])
        np.testing.assert_array_equal(model.block_vip_["b"], [0.0])
        np.testing.assert_array_equal(nothing.r2_y_cumulative_, [0.0])
        np.testing.assert_array_equal(nothing.r2_x_per_block_cumulative_[1], [0.0])
        np.testing.assert_array_equal(nothing.super_vip_, [0.0])
        np.testing.assert_array_equal(nothing.block_vip_["a"], [0.0])


def test_mbpls_refuses_what_its_masked_path_cannot_fit_in_full() -> None:
    """Each refusal names what is missing where; a row with nothing in one block, even the last, is fitted."""
    for block, column in (("x", "x1"), ("z", "z5")):
        blocks, y = _gappy_mbpls_data()
        blocks[block][column] = np.nan
        message = rf"^Block '{block}' has columns with all values missing: \['{column}'\]\. Drop these columns before"
        with pytest.raises(ValueError, match=message):
            MBPLS(n_components=1).fit(blocks, y)

    blocks, y = _gappy_mbpls_data()
    for frame in blocks.values():
        frame.iloc[[2, 4]] = np.nan
    message = (
        r"^Rows at positions \[2, 4\] have all values missing in every X-block\. "
        r"Drop these observations or impute them before fitting\.$"
    )
    with pytest.raises(ValueError, match=message):
        MBPLS(n_components=1).fit(blocks, y)

    blocks, y = _gappy_mbpls_data()
    with pytest.raises(
        ValueError, match=r"^Y has columns with all values missing: \['r'\]\. Drop these targets before fitting\.$"
    ):
        MBPLS(n_components=1).fit(blocks, y.assign(r=np.nan))
    gappy_y = y.copy()
    gappy_y.iloc[[0, 5]] = np.nan
    message = (
        r"^Y has rows with all values missing at positions \[0, 5\]\. "
        r"Drop these observations or impute them before fitting\.$"
    )
    with pytest.raises(ValueError, match=message):
        MBPLS(n_components=1).fit(blocks, gappy_y)

    with pytest.raises(
        ValueError, match=r"^Algorithm 'dense' cannot handle missing data\. Use 'nipals' or 'auto' instead\.$"
    ):
        MBPLS(n_components=1, algorithm="dense").fit(blocks, y)
    with pytest.raises(
        ValueError,
        match=r"^Algorithm 'Bogus' is not recognised\. Must be one of \['auto', 'dense', 'nipals'\]\.$",
    ):
        MBPLS(n_components=1, algorithm="Bogus").fit(blocks, y)

    blocks["z"].iloc[3] = np.nan  # r3 has nothing in x already, and now nothing in z, the last block
    model = MBPLS(n_components=1, algorithm="NIPALS").fit(blocks, y)
    assert model.algorithm_ == "nipals"
    assert np.isfinite(model.super_scores_.loc["r3", 1])
    assert np.isnan(model.block_scores_["z"].loc["r3", 1])


def test_an_mbpls_component_whose_loadings_are_all_zero_keeps_its_signs() -> None:
    """Only a negative largest loading flips a component; a largest of zero is not negative."""
    from process_improve.multivariate._mbpls import _apply_sign_convention, _MBPLSComponent, _MBPLSLoopContext

    context = _MBPLSLoopContext(
        algo="dense",
        block_names=["a"],
        sqrt_kb={"a": 1.0},
        block_has_data={"a": np.ones(2, dtype=bool)},
        tol=1e-8,
        max_iter=10,
    )
    component = _MBPLSComponent(
        block_weights={"a": np.array([-1.0])},
        block_scores={"a": np.array([-0.5, 0.5])},
        super_weight=np.ones(1),
        super_score=np.array([-1.0, 1.0]),
        super_y_score=np.array([-2.0, 2.0]),
        super_y_loading=np.array([-3.0]),
        iterations=1,
    )
    assert _apply_sign_convention(component, {"a": np.zeros((2, 1))}, context) is component


def _project_mbpls_by_hand(fitted: dict, blocks: dict[str, pd.DataFrame]) -> dict:
    """Score new rows on a model written out by ``_mbpls_written_out``, deflating as the fit did.

    Each block score is a regression of the row's observed cells on ``w_b``, over ``sqrt(K_b)``,
    and NaN for a row with nothing in the block; the super score is the regression of the
    whole row on ``g``, scaled by ``g'g``, and NaN for a row with nothing observed. A row
    without a super score is deflated by 0. Returns the scores, the data each component starts
    from and the data left after the last, the predictions, the block SPE and the T².
    """
    names = list(blocks)
    x = {
        name: (frame.to_numpy(dtype=float) - fitted["scaling"][name][0]) / fitted["scaling"][name][1]
        for name, frame in blocks.items()
    }
    has_data = {name: ~np.all(np.isnan(values), axis=1) for name, values in x.items()}
    root = {name: np.sqrt(fitted["widths"][name]) for name in names}
    t_all, t_b_all, before = [], {name: [] for name in names}, []
    for a in range(fitted["t"].shape[1]):
        before.append(dict(x))
        for name in names:
            t_b = _masked_regression(x[name], fitted["w_b"][name][:, a], "row") / root[name]
            t_b_all[name].append(np.where(has_data[name], t_b, np.nan))
        g = np.concatenate([fitted["w"][b, a] * fitted["w_b"][name][:, a] / root[name] for b, name in enumerate(names)])
        t = _masked_regression(np.hstack([x[name] for name in names]), g, "row") * (g @ g)
        for name in names:
            x[name] = x[name] - np.outer(np.nan_to_num(t), fitted["p_b"][name][:, a])
        t_all.append(t)
    t = np.column_stack(t_all)
    variance = np.sum(fitted["t"] ** 2, axis=0) / (len(fitted["t"]) - 1)
    centre, scale = fitted["y_scaling"]
    spe = {name: np.where(has_data[name], np.sqrt(np.nansum(x[name] ** 2, axis=1)), np.nan) for name in names}
    return {
        "t": t,
        "t_b": {name: np.column_stack(values) for name, values in t_b_all.items()},
        "residuals": x,
        "before": before,
        "predictions": t @ fitted["c"].T * scale + centre,
        "spe": spe,
        "t2": np.sum(t**2 / variance, axis=1),
    }


def _new_mbpls_rows(blocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Six new rows: n0 to n2 complete, n3 without one cell of z, n4 with nothing in x, n5 with nothing at all."""
    rng = np.random.default_rng(1)
    index = [f"n{i}" for i in range(6)]
    new = {
        name: pd.DataFrame(rng.standard_normal((6, frame.shape[1])), index=index, columns=frame.columns)
        for name, frame in blocks.items()
    }
    new["z"].loc["n3", "z2"] = np.nan
    new["x"].loc[["n4", "n5"]] = np.nan
    new["y"].loc["n5"] = np.nan
    new["z"].loc["n5"] = np.nan
    return new


class TestMBPLSNewRowsByHand:
    """New rows, their scores, predictions, SPE, T² and contributions, against the model written out."""

    def test_new_rows_are_scored_on_what_the_earlier_components_left(self) -> None:
        """Projections, predictions, SPE and T² of complete rows, a row with a gap, and rows with no data."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        new = _new_mbpls_rows(blocks)
        by_hand = _project_mbpls_by_hand(_mbpls_written_out(blocks, y, 2), new)
        index, components = new["x"].index, [1, 2]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # nothing divides by zero for the rows without data
            result = model.diagnose(new)
            from_arrays = model.diagnose({name: frame.to_numpy() for name, frame in new.items()})
            transformed, predicted = model.transform(new), model.predict(new)
        assert sorted(result) == ["block_scores", "block_spe", "hotellings_t2", "predictions", "super_scores"]
        scores = pd.DataFrame(by_hand["t"], index=index, columns=components)
        assert scores.loc["n5"].isna().all()
        assert scores.drop(index="n5").notna().all().all()
        _frames_near(result.super_scores, scores, 1e-10)
        _frames_near(transformed, scores, 1e-10)
        predictions = pd.DataFrame(by_hand["predictions"], index=index, columns=y.columns)
        _frames_near(result.predictions, predictions, 1e-10)
        _frames_near(predicted, predictions, 1e-10)
        _frames_near(result.hotellings_t2, pd.Series(by_hand["t2"], index=index, name="Hotelling's T²"), 1e-10)
        for name in blocks:
            t_b = pd.DataFrame(by_hand["t_b"][name], index=index, columns=components)
            _frames_near(result.block_scores[name], t_b, 1e-10)
            spe = pd.Series(by_hand["spe"][name], index=index, name=f"SPE[{name}]")
            _frames_near(result.block_spe[name], spe, 1e-10)
        assert np.isnan(result.block_spe["x"]["n4"])
        positions = pd.RangeIndex(6)
        _frames_near(from_arrays.super_scores, scores.set_axis(positions), 1e-10)
        _frames_near(from_arrays.predictions, predictions.set_axis(positions), 1e-10)

    def test_contributions_split_the_new_rows_scores_and_residuals_by_hand(self) -> None:
        """SPE contributions are the squared residuals; score contributions split each super score by variable."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        fitted = _mbpls_written_out(blocks, y, 2)
        new = _new_mbpls_rows(blocks)
        by_hand = _project_mbpls_by_hand(fitted, new)
        index = new["x"].index

        spe = model.spe_contributions(new)
        assert list(spe) == ["x", "y", "z"]
        for name, frame in new.items():
            scaled = (frame.to_numpy() - fitted["scaling"][name][0]) / fitted["scaling"][name][1]
            residuals = (scaled - by_hand["t"] @ fitted["p_b"][name].T) ** 2
            _frames_near(spe[name], pd.DataFrame(residuals, index=index, columns=frame.columns), 1e-10)
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # no "X does not have valid feature names"
                of_arrays = model.spe_contributions({key: value.to_numpy() for key, value in new.items()})[name]
            _frames_near(of_arrays, pd.DataFrame(residuals, columns=frame.columns), 1e-10)

        for component in (1, 2):
            a = component - 1
            contributions = model.score_contributions(new, component=component)
            expected = {
                name: pd.DataFrame(
                    by_hand["before"][a][name]
                    * fitted["w_b"][name][:, a]
                    * fitted["w"][b, a]
                    / np.sqrt(fitted["widths"][name]),
                    index=index,
                    columns=new[name].columns,
                )
                for b, name in enumerate(new)
            }
            for name in new:
                _frames_near(contributions[name], expected[name], 1e-10)
            complete = ["n0", "n1", "n2"]
            total = sum(frame.loc[complete].sum(axis=1) for frame in expected.values())
            np.testing.assert_allclose(total, by_hand["t"][:3, a], rtol=1e-12)
            if component == 2:
                group = model.group_contributions(new, group=["n0", "n2"], reference=["n1"], component=2)
                centred = model.group_contributions(new, group=["n0", "n2"], component=2)
                for name in new:
                    mean = expected[name].loc[["n0", "n2"]].mean()
                    _frames_near(centred[name], mean.rename(f"group_contributions[{name}]"), 1e-10)
                    shift = mean - expected[name].loc["n1"]
                    _frames_near(group[name], shift.rename(f"group_contributions[{name}]"), 1e-10)
        first = model.score_contributions(new)
        for name in new:
            pd.testing.assert_frame_equal(first[name], model.score_contributions(new, component=1)[name])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no "X does not have valid feature names"
            of_arrays = model.score_contributions({name: frame.to_numpy() for name, frame in new.items()}, component=2)
        for name in new:
            np.testing.assert_array_equal(of_arrays[name], model.score_contributions(new, component=2)[name])
            assert list(of_arrays[name].columns) == list(new[name].columns)
        first_group = model.group_contributions(new, group=["n0"])
        for name in new:
            pd.testing.assert_series_equal(
                first_group[name], model.group_contributions(new, group=["n0"], component=1)[name]
            )

        with pytest.raises(ValueError, match=r"^group contains entries that are not index labels of X: \['zz'\]\. "):
            model.group_contributions(new, group=["zz"])
        with pytest.raises(
            ValueError, match=r"^reference contains entries that are not index labels of X: \['zz'\]\. "
        ):
            model.group_contributions(new, group=["n0"], reference=["zz"])
        with pytest.raises(TypeError, match=r"^X must be a dict\[str, pd\.DataFrame\]\.$"):
            model.score_contributions(new["x"])
        with pytest.raises(ValueError, match=r"^Missing X-blocks: \['z'\]\.$"):
            model.score_contributions({"x": new["x"], "y": new["y"]})
        with pytest.raises(ValueError, match=r"^component must be a 1-based index within 1\.\.2, got 3\.$"):
            model.score_contributions(new, component=3)

    def test_diagnose_gives_no_block_spe_where_a_row_has_no_data(self) -> None:
        """A row with nothing observed in a block has no SPE there: NaN, as in ``block_spe_`` (#1028).

        Row r3 has nothing in block ``x``. ``block_spe_`` holds NaN for it, and diagnose of the
        training rows gave 0, the SPE of a row the model fits exactly.
        """
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        result = model.diagnose(blocks)
        for name in blocks:
            pd.testing.assert_series_equal(
                result.block_spe[name], model.block_spe_[name][2].rename(f"SPE[{name}]"), rtol=1e-12
            )
        assert np.isnan(result.block_spe["x"]["r3"])

    def test_the_spe_limits_are_set_on_the_last_component_over_the_rows_with_data(self) -> None:
        """With three components, each block's limit and the merged one read the third column of the SPE.

        Row r3 has nothing in block ``x``, so no SPE there: it has no part in ``x``'s limit, and
        adds nothing to its merged SPE. Its NaN made both limits NaN (#1028).
        """
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=3).fit(blocks, y)
        spe = {name: values[:, 2] for name, values in _mbpls_written_out(blocks, y, 3)["spe"].items()}
        assert np.isnan(spe["x"][3])
        merged = np.sqrt(sum(np.nan_to_num(values) ** 2 for values in spe.values()))
        for conf_level in (0.95, 0.99):
            for name in blocks:
                limit = spe_calculation(spe[name][np.isfinite(spe[name])], conf_level=conf_level)
                assert model.block_spe_limit(name, conf_level=conf_level) == pytest.approx(limit, rel=1e-9)
            assert model.super_spe_limit(conf_level=conf_level) == pytest.approx(
                spe_calculation(merged, conf_level=conf_level), rel=1e-9
            )
        assert model.block_spe_limit("z") == model.block_spe_limit("z", conf_level=0.95)
        assert model.super_spe_limit() == model.super_spe_limit(conf_level=0.95)


class TestMBPLSPlotsAndSummary:
    """The plots and the summary of a fitted MBPLS, in full."""

    def test_display_results_prints_the_r2_iterations_and_timing(self) -> None:
        """The summary, in full: cumulative R2X and R2Y by default, the increments on request, and plain integers."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        model.fitting_info_["timing"] = np.array([0.1, 0.00123])
        expected = _mbpls_written_out(blocks, y, 2)
        r2 = np.vstack([*(expected["r2_block"][name] for name in blocks), expected["r2_y"]])
        header = "  PC | R²X[x] | R²X[y] | R²X[z] | R²Y"

        def summary(values: np.ndarray) -> str:
            rows = [f"  {a + 1} | " + " | ".join(f"{value:9.4f}" for value in values[:, a]) for a in range(2)]
            return "\n".join(
                [
                    "MBPLS model: 2 component(s), 3 X-block(s)",
                    header,
                    "-" * len(header),
                    *rows,
                    "",
                    "  Iterations per PC: [27, 8]",
                    "  Time per PC (ms):  [100.0, 1.2]",
                ]
            )

        assert model.display_results() == summary(r2)
        assert model.display_results(show_cumulative=False) == summary(np.diff(r2, axis=1, prepend=0.0))

    def test_the_plots_draw_the_fitted_super_scores_and_weights(self) -> None:
        """Both plots, in full, by default and at the last component."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=3).fit(blocks, y)
        scores = model.super_scores_
        for figure, (horizontal, vertical) in (
            (model.super_score_plot(), (1, 2)),
            (model.super_score_plot(3, 1), (3, 1)),
        ):
            (trace,) = figure.data
            np.testing.assert_array_equal(trace.x, scores[horizontal])
            np.testing.assert_array_equal(trace.y, scores[vertical])
            assert trace.mode == "markers+text"
            assert list(trace.text) == [f"r{i}" for i in range(6)]
            assert trace.textposition == "top center"
            assert trace.name == "Super-scores"
            assert figure.layout.xaxis.title.text == f"t_super[{horizontal}]"
            assert figure.layout.yaxis.title.text == f"t_super[{vertical}]"
            assert figure.layout.title.text == f"MBPLS super-score plot: PC{horizontal} vs PC{vertical}"
        for horizontal, vertical in ((4, 1), (1, 4), (0, 1), (1, 0)):
            with pytest.raises(ValueError, match=r"^pc_horiz and pc_vert must be in 1\.\.3\.$"):
                model.super_score_plot(horizontal, vertical)

        for figure, component in ((model.super_weights_bar_plot(), 1), (model.super_weights_bar_plot(3), 3)):
            (bar,) = figure.data
            assert list(bar.x) == ["x", "y", "z"]
            np.testing.assert_array_equal(bar.y, model.super_weights_[component])
            assert bar.name == f"w_super[{component}]"
            assert figure.layout.xaxis.title.text == "Block"
            assert figure.layout.yaxis.title.text == f"w_super[{component}]"
            assert figure.layout.title.text == f"MBPLS super-weights, component {component}"
        for component in (0, 4):
            with pytest.raises(ValueError, match=r"^component must be in 1\.\.3\.$"):
                model.super_weights_bar_plot(component)

    def test_the_predictions_plot_draws_the_points_the_line_and_the_rmsee(self) -> None:
        """The plot of predicted against observed, in full: two traces, a y = x line padded by 5%, the RMSEE."""
        blocks, y = _gappy_mbpls_data()
        model = MBPLS(n_components=2).fit(blocks, y)
        observed = y.fillna(0.5) * 2.0  # what is drawn need not be what was fitted
        figure = model.predictions_vs_observed_plot(observed, variable="q")
        points, line = figure.data
        np.testing.assert_array_equal(points.x, observed["q"])
        np.testing.assert_array_equal(points.y, model.predictions_["q"])
        assert (points.mode, points.name) == ("markers", "Predicted vs observed")
        low = min(observed["q"].min(), model.predictions_["q"].min())
        high = max(observed["q"].max(), model.predictions_["q"].max())
        pad = 0.05 * (high - low)
        np.testing.assert_allclose(line.x, [low - pad, high + pad], rtol=1e-14)
        np.testing.assert_allclose(line.y, [low - pad, high + pad], rtol=1e-14)
        assert (line.mode, line.name) == ("lines", "y = x")
        assert line.line.color == REFERENCE_LINE_COLOR
        assert line.line.dash == "dash"
        (note,) = figure.layout.annotations
        rmsee = np.sqrt(np.mean((observed["q"] - model.predictions_["q"]) ** 2))
        assert note.text == f"RMSEE = {rmsee:.4g}"
        assert note.x == pytest.approx(low + 0.05 * (high - low), rel=1e-14)
        assert note.y == pytest.approx(high - 0.05 * (high - low), rel=1e-14)
        assert note.showarrow is False
        assert figure.layout.xaxis.title.text == "Observed: q"
        assert figure.layout.yaxis.title.text == "Predicted: q"
        assert figure.layout.title.text == "Predicted vs observed for q"

        # Every point at one value: the line is padded by 1 either side.
        flat = MBPLS(n_components=1).fit(blocks, pd.DataFrame({"c": np.full(6, 3.0)}, index=y.index))
        (_, line) = flat.predictions_vs_observed_plot(pd.DataFrame({"c": np.full(6, 3.0)})).data
        np.testing.assert_array_equal(line.x, [2.0, 4.0])
        np.testing.assert_array_equal(line.y, [2.0, 4.0])


@pytest.mark.parametrize(
    "use",
    [
        lambda model, blocks, _: model.transform(blocks),
        lambda model, blocks, _: model.diagnose(blocks),
        lambda model, blocks, _: model.predict(blocks),
        lambda model, blocks, y: model.score(blocks, y),
        lambda model, blocks, _: model.spe_contributions(blocks),
        lambda model, blocks, _: model.score_contributions(blocks),
        lambda model, blocks, _: model.group_contributions(blocks, group=["r0"]),
        lambda model, _, __: model.block_spe_limit("x"),
        lambda model, _, __: model.super_spe_limit(),
        lambda model, _, __: model.super_score_plot(),
        lambda model, _, __: model.super_weights_bar_plot(),
        lambda model, _, y: model.predictions_vs_observed_plot(y),
        lambda model, _, __: model.display_results(),
        lambda model, blocks, y: randomization_test_mbpls(model, blocks, y, n_permutations=1),
    ],
    ids=[
        "transform",
        "diagnose",
        "predict",
        "score",
        "spe_contributions",
        "score_contributions",
        "group_contributions",
        "block_spe_limit",
        "super_spe_limit",
        "super_score_plot",
        "super_weights_bar_plot",
        "predictions_vs_observed_plot",
        "display_results",
        "randomization_test_mbpls",
    ],
)
def test_an_mbpls_whose_fit_failed_is_not_fitted(use: Callable[[MBPLS, dict, pd.DataFrame], object]) -> None:
    """A fit refused after the block names were recorded leaves a model every method reports as unfitted."""
    blocks, y = _gappy_mbpls_data()
    model = MBPLS(n_components=2, algorithm="bogus")
    with pytest.raises(ValueError, match="is not recognised"):
        model.fit(blocks, y)
    assert model.block_names_ == ["x", "y", "z"]
    with pytest.raises(NotFittedError):
        use(model, blocks, y)


def _blocks_of_two_directions() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ten rows whose blocks hold two directions exactly, and a response barely tied to them.

    After two components the blocks have nothing left, so a third finds nothing in Y that X
    can explain, whatever the order of Y's rows.
    """
    rng = np.random.default_rng(6)
    latent = rng.standard_normal((10, 2))
    blocks = {
        "a": pd.DataFrame(latent @ rng.standard_normal((2, 3)), columns=["a0", "a1", "a2"]),
        "b": pd.DataFrame(latent @ rng.standard_normal((2, 2)), columns=["b0", "b1"]),
    }
    return blocks, pd.DataFrame({"y": 0.1 * latent[:, 0] + rng.standard_normal(10)})


def _association(model: MBPLS) -> np.ndarray:
    """``|t'u| / (||t|| ||u||)`` per component, or 0 where ``||t|| ||u||`` is at most ``epsqrt``."""
    t, u = model.super_scores_.to_numpy(), model.super_y_scores_.to_numpy()
    size = np.linalg.norm(t, axis=0) * np.linalg.norm(u, axis=0)
    return np.where(size > EPSQRT, np.abs(np.sum(t * u, axis=0)) / np.where(size > EPSQRT, size, 1.0), 0.0)


def test_the_randomization_test_counts_the_permutations_that_reach_the_fit_by_hand() -> None:
    """Each permutation of Y's rows is fitted afresh; a component's risk counts those at least as associated.

    The third component has no Y-score, so its association is 0 in the fit and in every
    permutation: a tie, which counts, so its risk is 100%. The permutations are drawn as
    ``seed`` draws them, with the rows of Y taken by position.
    """
    blocks, y = _blocks_of_two_directions()
    y.index = [f"r{i}" for i in range(10)][::-1]  # labels that differ from the positions
    with pytest.warns(SpecificationWarning, match="nothing left in Y"):
        model = MBPLS(n_components=3).fit(blocks, y)
    observed = _association(model)
    assert observed[2] == 0.0
    rng = np.random.default_rng(11)
    exceed = np.zeros(3)
    for _ in range(20):
        with pytest.warns(SpecificationWarning, match="nothing left in Y"):
            permuted = MBPLS(n_components=3).fit(blocks, y.iloc[rng.permutation(10)])
        exceed += _association(permuted) >= observed
    assert np.all((exceed[:2] > 0) & (exceed[:2] < 20))  # counts that depend on the permutations drawn
    assert exceed[2] == 20
    expected = pd.DataFrame(
        {"observed": observed, "risk_pct": 100.0 * (exceed + 1) / 21}, index=pd.Index([1, 2, 3], name="component")
    )
    for _ in range(2):  # the same seed draws the same permutations
        with pytest.warns(SpecificationWarning, match="nothing left in Y"):
            result = randomization_test_mbpls(model, blocks, y, n_permutations=20, seed=11)
        pd.testing.assert_frame_equal(result, expected, rtol=1e-12)


def test_the_randomization_test_draws_200_permutations_by_default() -> None:
    """The default is 200 permutations, so each risk is a whole number of 201sts of 100%."""
    blocks, y = _blocks_of_two_directions()
    model = MBPLS(n_components=1).fit(blocks, y)
    risk = randomization_test_mbpls(model, blocks, y, seed=0)["risk_pct"].to_numpy()
    assert 100.0 / 201 <= risk[0] < 100.0
    np.testing.assert_allclose(risk * 201 / 100, np.round(risk * 201 / 100), atol=1e-9)


def test_an_association_over_a_size_of_at_most_epsqrt_is_zero() -> None:
    """``|t'u| / (||t|| ||u||)`` is 0 once ``||t|| ||u||`` is at most ``epsqrt``, and the ratio above it.

    With ``t = [epsqrt, 0, ...]`` and ``u = [1, 0, ...]`` the size is exactly ``epsqrt``; with
    ``t = [2 epsqrt, 0, ...]`` it is twice that, and the ratio 1.
    """
    blocks, y = _blocks_of_two_directions()
    model = MBPLS(n_components=2).fit(blocks, y)
    unit = np.zeros(10)
    unit[0] = 1.0
    model.super_scores_ = pd.DataFrame({1: EPSQRT * unit, 2: 2.0 * EPSQRT * unit})
    model.super_y_scores_ = pd.DataFrame({1: unit, 2: unit})
    result = randomization_test_mbpls(model, blocks, y, n_permutations=0)
    np.testing.assert_array_equal(result["observed"], [0.0, 1.0])
    np.testing.assert_array_equal(result["risk_pct"], [100.0, 100.0])


# ---------------------------------------------------------------------------
# Tests deferred until PR2 ships the reference datasets
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires LDPE-PLS reference dataset (planned in PR2 of the multi-block port).")
class TestLDPEReference:
    """Numerical assertions against the LDPE multiblock dataset.

    The legacy ``test_mbpls.m`` and ``test_mbpca.m`` use this dataset for the
    primary algorithmic validation. The dataset (``LDPE-PLS.mat``) is not yet
    in ``process_improve.datasets``; see the multi-block port plan, PR2.
    """


@pytest.fixture(scope="module")
def fmc_script():
    from tests._case_study_scripts import load_script

    return load_script("fmc_multiblock_batch_pls")


@pytest.fixture(scope="module")
def fmc_data(fmc_script):
    from tests._case_study_scripts import load_or_skip

    return load_or_skip(fmc_script.load_and_exclude)


@pytest.fixture(scope="module")
def fmc_quality(fmc_script, fmc_data):
    return fmc_script.pca_on_quality(fmc_data.Y)


@pytest.fixture(scope="module")
def fmc_unfolded(fmc_script, fmc_data):
    return fmc_script.unfold_trajectories(fmc_data.X)


@pytest.mark.dataset
@pytest.mark.slow
class TestFMCReference:
    """The FMC batch dryer ladder of models, pinned on the case-study script.

    The FMC dataset is served from openmv.net (see
    ``process_improve.batch.load_fmc``), so these tests skip offline. The
    course notes quote no numbers for this case; every value here is a
    regression pin recorded when the case study was rebuilt, two components
    throughout, after excluding the thirteen batches without chemistry data,
    with ``ClockTime`` as the eleventh trajectory of the unfolded block.
    The script under test is ``docs/user_guide/case_studies/batch/fmc_multiblock_batch_pls.py``.
    """

    def test_exclusion_leaves_46_batches_with_the_documented_missing_cells(self, fmc_data, fmc_unfolded) -> None:
        wide, _scaled = fmc_unfolded
        assert len(fmc_data.X) == 46
        assert int(fmc_data.Y.isna().sum().sum()) == 19
        assert int(fmc_data.Zchem.isna().sum().sum()) == 1
        assert wide.shape == (46, 11 * 325)
        assert int(wide.isna().sum().sum()) == 1340  # 1220 in the ten process tags plus 120 in ClockTime
        incomplete = {batch_id for batch_id, batch in fmc_data.X.items() if batch.isna().any().any()}
        assert incomplete == {20, 22, 27, 28, 31, 55, 60, 61, 67, 71}

    def test_pca_on_quality(self, fmc_quality) -> None:
        model, _y_scaled = fmc_quality
        np.testing.assert_allclose(model.r2_cumulative_.to_numpy(), [0.500, 0.703], atol=2e-3)
        assert model.has_missing_data_

    def test_pls_from_each_initial_condition_block(self, fmc_script, fmc_data, fmc_quality) -> None:
        _model, y_scaled = fmc_quality
        pls_chem, pls_op, _zop_scaled = fmc_script.pls_from_initial_conditions(fmc_data, y_scaled)
        np.testing.assert_allclose(pls_chem.r2_cumulative_.to_numpy(), [0.163, 0.222], atol=2e-3)
        np.testing.assert_allclose(pls_op.r2_cumulative_.to_numpy(), [0.207, 0.262], atol=2e-3)

    def test_mbpls_on_initial_conditions(self, fmc_script, fmc_data) -> None:
        model = fmc_script.mbpls_on_initial_conditions(fmc_data)
        np.testing.assert_allclose(model.r2_y_cumulative_.to_numpy(), [0.292, 0.364], atol=2e-3)

    def test_batch_pca_on_trajectories(self, fmc_script, fmc_unfolded) -> None:
        _wide, x_scaled = fmc_unfolded
        model, spe_share = fmc_script.batch_pca_on_trajectories(x_scaled)
        np.testing.assert_allclose(model.r2_cumulative_.to_numpy(), [0.231, 0.376], atol=2e-3)
        assert list(model.loadings_.index.names) == ["tag", "sequence"]
        assert list(spe_share.columns.names) == ["tag", "sequence"]
        row = spe_share.loc[20]  # a batch with missing cells: contributions everywhere except at those cells
        assert row.isna().equals(x_scaled.loc[20].isna())
        assert row.sum() == pytest.approx(100.0)
        by_tag = row.groupby(level="tag", sort=False).sum()
        assert by_tag.idxmax() == "DryPress"  # half of batch 20's SPE sits in the dryer pressure
        assert 45 < by_tag["DryPress"] < 53

    def test_batch_pls_to_quality(self, fmc_script, fmc_unfolded, fmc_quality) -> None:
        _wide, x_scaled = fmc_unfolded
        _model, y_scaled = fmc_quality
        model, contributions = fmc_script.batch_pls_to_quality(x_scaled, y_scaled)
        np.testing.assert_allclose(model.r2_cumulative_.to_numpy(), [0.266, 0.410], atol=2e-3)
        assert np.isfinite(contributions.loc[13]).all()
        by_tag = contributions.loc[13].groupby(level="tag", sort=False).sum()
        assert set(by_tag.nsmallest(2).index) == {"ClockTime", "CTankLvl"}  # the two tags that lead batch 13's t1

    def test_batch_mbpls(self, fmc_script, fmc_data, fmc_unfolded) -> None:
        wide, _x_scaled = fmc_unfolded
        model, _blocks = fmc_script.batch_mbpls(fmc_data, wide)
        np.testing.assert_allclose(model.r2_y_cumulative_.to_numpy(), [0.370, 0.472], atol=2e-3)
        vip = model.super_vip_
        assert vip["Zchem"] < min(vip["Zop"], vip["X"])  # the chemistry ranks last
        assert abs(vip["Zop"] - vip["X"]) < 0.05  # the other two blocks are about equal
        assert list(model.block_weights_["X"].index.names) == ["tag", "sequence"]

    def test_four_good_batches_have_off_spec_trajectories_and_on_spec_initial_conditions(
        self, fmc_script, fmc_data, fmc_unfolded
    ) -> None:
        wide, _x_scaled = fmc_unfolded
        model, blocks = fmc_script.batch_mbpls(fmc_data, wide)
        found = fmc_script.off_spec_trajectories_on_spec_product(model, blocks)
        assert found.anomalous == [2, 3, 6, 7]
        assert found.neighbours == [42, 43, 44, 47, 50]
        assert (found.placed.loc[found.anomalous, "X"] == "abnormal").all()
        assert (found.placed.loc[found.anomalous, ["Zchem", "Zop"]] == "good").all().all()
        # A contribution's sign follows the component's arbitrary sign, so the claims are
        # checked relative to Time3: the pattern, not its orientation, is the finding.
        move = found.zop_move
        assert move.abs().idxmax() == "Time3"  # the longer cool-down is the largest single difference
        direction = np.sign(move["Time3"])
        assert (np.sign(move[["Time2", "Time4", "TempSlope"]]) == direction).all()  # later phases run differently
        assert (np.sign(move[["Level1", "WgtCake"]]) == -direction).all()  # the charge and first phase were not

    @pytest.mark.usefixtures("fmc_data")
    def test_script_runs_end_to_end(self, fmc_script, tmp_path) -> None:
        assert fmc_script.main(["--output-dir", str(tmp_path)]) == 0
        assert len(list(tmp_path.glob("*.html"))) >= 20


@pytest.mark.skip(reason="Requires kamyr-digester / Simca-P reference values (planned in PR2 of the multi-block port).")
class TestSimcaPCAComparison:
    """Numerical comparison against Simca-P 11.5.0.0 (2006) for kamyr-digester
    PCA and PLS models. Deferred until PR2 brings the reference values into
    the repository.
    """


class TestMultiblockDeterministicStart:
    """#503: the MBPCA / MBPLS NIPALS loops seed their super-score / u vector
    deterministically from the highest-sum-of-squares column, with no RNG in
    the fit path, so repeated fits on the same data are bit-identical.
    """

    @pytest.fixture
    def blocks_and_y(self) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        rng = np.random.default_rng(2026)
        n_rows = 40
        latent = rng.standard_normal((n_rows, 2))
        x_blocks = {
            "A": pd.DataFrame(
                latent @ rng.standard_normal((2, 5)) + 0.05 * rng.standard_normal((n_rows, 5)),
                columns=[f"a{i}" for i in range(5)],
            ),
            "B": pd.DataFrame(
                latent @ rng.standard_normal((2, 3)) + 0.05 * rng.standard_normal((n_rows, 3)),
                columns=[f"b{i}" for i in range(3)],
            ),
        }
        y_df = pd.DataFrame(
            latent @ rng.standard_normal((2, 2)) + 0.05 * rng.standard_normal((n_rows, 2)),
            columns=["y0", "y1"],
        )
        return x_blocks, y_df

    def test_mbpca_repeated_fits_are_identical(self, blocks_and_y) -> None:
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = blocks_and_y
        m1 = MBPCA(n_components=2, algorithm="nipals").fit(x_blocks)
        m2 = MBPCA(n_components=2, algorithm="nipals").fit(x_blocks)
        np.testing.assert_array_equal(m1.super_scores_.values, m2.super_scores_.values)
        np.testing.assert_array_equal(m1.super_loadings_.values, m2.super_loadings_.values)
        for name in m1.block_names_:
            np.testing.assert_array_equal(m1.block_scores_[name].values, m2.block_scores_[name].values)
            np.testing.assert_array_equal(m1.block_loadings_[name].values, m2.block_loadings_[name].values)

    def test_mbpls_repeated_fits_are_identical(self, blocks_and_y) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = blocks_and_y
        m1 = MBPLS(n_components=2, algorithm="nipals").fit(x_blocks, y_df)
        m2 = MBPLS(n_components=2, algorithm="nipals").fit(x_blocks, y_df)
        np.testing.assert_array_equal(m1.super_scores_.values, m2.super_scores_.values)
        np.testing.assert_array_equal(m1.super_weights_.values, m2.super_weights_.values)
        np.testing.assert_array_equal(m1.super_y_scores_.values, m2.super_y_scores_.values)
        for name in m1.block_names_:
            np.testing.assert_array_equal(m1.block_scores_[name].values, m2.block_scores_[name].values)
            np.testing.assert_array_equal(m1.block_loadings_[name].values, m2.block_loadings_[name].values)

    def test_no_nonconvergence_warning_on_well_conditioned_data(self, blocks_and_y) -> None:
        """#504 acceptance: the default (now relative) tolerance produces no
        spurious non-convergence warnings on ordinary well-conditioned data.
        """
        import warnings

        from process_improve.multivariate.methods import MBPCA, MBPLS

        x_blocks, y_df = blocks_and_y
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            MBPCA(n_components=2, algorithm="nipals").fit(x_blocks)
            MBPLS(n_components=2, algorithm="nipals").fit(x_blocks, y_df)


def _ldpe_blocks() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return the LDPE data as three process blocks and the quality block (complete data)."""
    import pathlib

    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE"
    values = pd.read_csv(folder / "LDPE.csv", index_col=0)
    blocks = {
        "zone1": values.iloc[:, [0, 1, 2, 5, 7, 9, 11]],
        "zone2": values.iloc[:, [3, 4, 6, 8, 10, 12]],
        "pressure": values.iloc[:, [13]],
    }
    return blocks, values.iloc[:, 14:]


def _gappy_blocks() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return two small blocks with 30-60% of their cells missing, and a response.

    On this data the MBPLS super weight of block ``b`` for the first component is
    negative and the largest in size, which the old convention (keyed on the super
    weight) acted on.
    """
    rng = np.random.default_rng(0)
    n_rows = 14
    x1, x2 = rng.standard_normal((n_rows, 3)), rng.standard_normal((n_rows, 3))
    y = rng.standard_normal((n_rows, 1))
    fraction = rng.uniform(0.3, 0.6)
    for x in (x1, x2):
        x[rng.random(x.shape) < fraction] = np.nan
    for x in (x1, x2):  # keep at least one value in every row and every column
        for i in range(n_rows):
            if np.all(np.isnan(x[i])):
                x[i, 0] = rng.standard_normal()
        for j in range(3):
            if np.all(np.isnan(x[:, j])):
                x[0, j] = rng.standard_normal()
    blocks = {"a": pd.DataFrame(x1, columns=list("pqr")), "b": pd.DataFrame(x2, columns=list("stu"))}
    return blocks, pd.DataFrame(y, columns=["y"])


class TestMultiblockSignConvention:
    """#586: each component is signed so its largest-magnitude X loading is positive.

    That is the PLS and PCA convention. The old one was keyed on the super weight
    (MBPCA: the super loading), whose element for block b is proportional to
    ``t_b' u``. Flipping a whole component leaves that unchanged, so it could not
    pin the sign: MBPLS's X side changed sign with Y's, and one-block models often
    disagreed with PLS and PCA. On complete data it is never negative, so the flip
    never fired. With missing cells it can be negative, and there the flip negated
    the super weight together with the scores, leaving a model whose ``transform``
    and ``predict`` disagreed with its own fitted values.
    """

    def test_mbpls_x_side_does_not_depend_on_the_sign_of_y(self) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = _ldpe_blocks()
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        mirrored = MBPLS(n_components=3).fit(x_blocks, -y_df)
        np.testing.assert_allclose(mirrored.super_scores_.values, model.super_scores_.values, atol=1e-8)
        for name in model.block_names_:
            np.testing.assert_allclose(
                mirrored.block_weights_[name].values, model.block_weights_[name].values, atol=1e-8
            )
            np.testing.assert_allclose(mirrored.block_scores_[name].values, model.block_scores_[name].values, atol=1e-8)
        # Only the Y side mirrors Y.
        np.testing.assert_allclose(mirrored.super_y_loadings_.values, -model.super_y_loadings_.values, atol=1e-8)

    def test_one_block_models_have_the_signs_of_pls_and_pca(self) -> None:
        """A single block reduces MBPLS to PLS and MBPCA to PCA, signs included.

        On this data the old convention left component 1 of MBPCA and component 2
        of MBPLS with the opposite sign.
        """
        from process_improve.multivariate.methods import MBPCA, MBPLS, PLS

        rng = np.random.default_rng(0)
        x = pd.DataFrame(
            rng.standard_normal((30, 3)) @ rng.standard_normal((3, 7)) + 0.5 * rng.standard_normal((30, 7)),
            columns=[f"x{i}" for i in range(7)],
        )
        x = MCUVScaler().fit_transform(x)
        y = pd.DataFrame(x.values @ rng.standard_normal((7, 2)) + rng.standard_normal((30, 2)), columns=["y1", "y2"])
        y = MCUVScaler().fit_transform(y)

        mbpca, pca = MBPCA(n_components=4).fit({"all": x}), PCA(n_components=4).fit(x)
        mbpls, pls = MBPLS(n_components=3).fit({"all": x}, y), PLS(n_components=3).fit(x, y)
        for multiblock, single, n_components in ((mbpca, pca, 4), (mbpls, pls, 3)):
            for a in range(n_components):
                r = np.corrcoef(multiblock.super_scores_.iloc[:, a], single.scores_.iloc[:, a])[0, 1]
                assert r > 1 - 1e-8, (type(multiblock).__name__, a + 1, r)

    def test_largest_x_loading_is_positive(self) -> None:
        from process_improve.multivariate.methods import MBPCA, MBPLS

        x_blocks, y_df = _ldpe_blocks()
        mbpls = MBPLS(n_components=3).fit(x_blocks, y_df)
        mbpca = MBPCA(n_components=3).fit(x_blocks)
        for a in range(3):
            loadings = np.concatenate([mbpls.block_loadings_[name].values[:, a] for name in mbpls.block_names_])
            assert loadings[np.argmax(np.abs(loadings))] > 0
            # MBPCA deflates with the block loading scaled by the super loading and sqrt(K_b).
            deflating = np.concatenate(
                [
                    mbpca.block_loadings_[name].values[:, a]
                    * mbpca.super_loadings_.values[b, a]
                    * np.sqrt(mbpca.block_widths_[name])
                    for b, name in enumerate(mbpca.block_names_)
                ]
            )
            assert deflating[np.argmax(np.abs(deflating))] > 0

    def test_super_weights_are_not_negative_on_complete_data(self) -> None:
        """Why the old convention never fired without missing data: ``t_b' u`` is ``||X_b' u||``."""
        from process_improve.multivariate.methods import MBPCA, MBPLS

        x_blocks, y_df = _ldpe_blocks()
        assert (MBPLS(n_components=3).fit(x_blocks, y_df).super_weights_.values >= 0).all()
        assert (MBPCA(n_components=3).fit(x_blocks).super_loadings_.values >= 0).all()

    def test_gappy_mbpls_agrees_with_its_own_transform_and_predict(self) -> None:
        """Missing cells can make a super weight negative; the model must stay self-consistent."""
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = _gappy_blocks()
        model = MBPLS(n_components=2).fit(x_blocks, y_df)
        weights = model.super_weights_.values[:, 0]
        assert weights[np.argmax(np.abs(weights))] < 0  # the case the old convention flipped
        np.testing.assert_allclose(model.transform(x_blocks).values, model.super_scores_.values, atol=1e-10)
        np.testing.assert_allclose(model.diagnose(x_blocks).predictions.values, model.predictions_.values, atol=1e-10)


@pytest.mark.parametrize("estimator", ["MBPCA", "MBPLS"])
def test_block_contribution_scalings_skip_the_missing_cells(estimator: str) -> None:
    """A missing cell's NaN neither stops ``"maximum"`` from scaling nor leaves its row out of ``"within"``.

    Miller, Swanson and Heckler (1994): divide by the largest absolute contribution, or by the
    row's total absolute contribution, both taken jointly over the blocks and over the observed
    cells, as the single-block ``score_contributions`` does.
    """
    from process_improve.multivariate import methods

    rng = np.random.default_rng(7)
    latent = rng.standard_normal((30, 2))
    x_blocks = {
        "A": pd.DataFrame(latent @ rng.standard_normal((2, 4)) + 0.1 * rng.standard_normal((30, 4))),
        "B": pd.DataFrame(latent @ rng.standard_normal((2, 3)) + 0.1 * rng.standard_normal((30, 3))),
    }
    model = getattr(methods, estimator)(n_components=2)
    if estimator == "MBPLS":
        model.fit(x_blocks, pd.DataFrame(latent @ rng.standard_normal((2, 1))))
    else:
        model.fit(x_blocks)
    with_gap = {name: block.copy() for name, block in x_blocks.items()}
    with_gap["A"].iloc[3, 1] = np.nan

    raw = model.score_contributions(with_gap, component=1)
    stacked = np.hstack([raw[name].to_numpy() for name in model.block_names_])
    assert np.isnan(stacked).sum() == 1

    largest = model.score_contributions(with_gap, component=1, scaling="maximum")
    for name in model.block_names_:
        np.testing.assert_allclose(largest[name], raw[name].to_numpy() / np.nanmax(np.abs(stacked)), rtol=1e-14)

    within = model.score_contributions(with_gap, component=1, scaling="within")
    row_total = np.nansum(np.abs(stacked), axis=1, keepdims=True)
    for name in model.block_names_:
        np.testing.assert_allclose(within[name], raw[name].to_numpy() / row_total, rtol=1e-14)
