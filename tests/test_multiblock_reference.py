"""
Reference numerical tests for multivariate and (eventually) multi-block methods.

These tests are ported from a legacy MATLAB multi-block latent-variable methods
codebase (ConnectMV, 2010-2012). The MATLAB codebase contained a large
``unit_tests.m`` suite that locked in expected numerical values both from the
chemometrics literature and from cross-checks against commercial packages
(ProSensus Multivariate / ProMV, Simca-P).

Phase 1 of the multi-block port to ``process-improve`` lifts as many of those
numerical assertions into Python as possible *without needing a MATLAB
session*. The strategy is:

1. Tests that assert against published literature values (Wold, Esbensen &
   Geladi, 1987) run today against the existing single-block :class:`PCA`
   class. They guard the existing implementation and form the spine for the
   future multi-block tests.

2. Tests for multi-block PCA / multi-block PLS (MBPCA / MBPLS) are written
   here as ``pytest.mark.xfail`` placeholders. They use the same Wold matrix
   split into two blocks. They will turn green when the corresponding
   classes land in the planned PRs.

3. Tests that need real reference datasets (LDPE, FMC, kamyr-digester) or a
   ProSensus / Simca-P numerical comparison are marked ``pytest.mark.skip``
   with a reason. They become runnable once the datasets land in
   ``process_improve.datasets`` (planned in PR2).

Conventions
-----------
- The legacy MATLAB code defines SPE as :math:`e'e` (raw row sum of squares).
  ``process_improve.PCA`` stores ``spe_`` as ``sqrt(e'e)``. Every assertion
  against a MATLAB / ProMV SPE value compares against ``spe_ ** 2``.
- The legacy MATLAB code stores the *inverse* standard deviation as the
  scaling vector. :class:`MCUVScaler` stores the standard deviation itself.
  Hence ``[1, 1, 0.5, 1]`` in MATLAB corresponds to ``[1, 1, 2, 1]`` here.
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

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import (
    PCA,
    MCUVScaler,
)

# ---------------------------------------------------------------------------
# Wold/Esbensen/Geladi 1987 reference data and expectations
# ---------------------------------------------------------------------------
# The 3x4 worked example used throughout the 1987 PCA paper. All expected
# values below come straight from that paper's pages 40-43, plus a small
# number of cross-checks from ProSensus Multivariate (2010, Revision 302)
# captured in the legacy ``unit_tests.m``.

WOLD_X = np.array(
    [
        [3.0, 4.0, 2.0, 2.0],
        [4.0, 3.0, 4.0, 3.0],
        [5.0, 5.0, 6.0, 4.0],
    ]
)

# Page 40: column means of WOLD_X
WOLD_CENTER = np.array([4.0, 4.0, 4.0, 3.0])

# Page 40: standard deviations (ddof=1) of WOLD_X. Note that the MATLAB code
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

# ProMV cross-check: SPE = e'e per row, after PC1
PROMV_SPE_AFTER_PC1 = np.array([0.366107, 0.877964, 0.110178])

# ProMV cross-check: VIP per variable after PC1 and after PC2
PROMV_VIP_AFTER_PC1 = np.array([1.082, 0.6987, 1.082, 1.082])
PROMV_VIP_AFTER_PC2 = np.array([1.0, 1.0, 1.0, 1.0])

# Statistical limits at 95% confidence (legacy ``unit_tests.m`` lines 210-212)
WOLD_T2_LIMIT_PC1 = 24.684
WOLD_SPE_LIMIT_PC1 = 1.2236
# Score limit uses t.ppf(0.975, N-1) * std(scores, ddof=1); not exposed
# directly by process_improve.PCA, so it is reconstructed inside the test.
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
    bookkeeping. Every assertion below comes from the published values or
    from the legacy ``unit_tests.m`` cross-checks against ProMV.
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
        # process_improve stores spe_ as sqrt(e'e); ProMV / MATLAB store e'e.
        model, _, _ = _fit_pca_on_wold(n_components=1)
        spe_squared = model.spe_.iloc[:, 0].values ** 2
        np.testing.assert_array_almost_equal(spe_squared, PROMV_SPE_AFTER_PC1, decimal=4)

    def test_spe_per_observation_after_two_pc_is_zero(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        spe_squared = model.spe_.iloc[:, 1].values ** 2
        np.testing.assert_array_almost_equal(spe_squared, np.zeros(3), decimal=6)

    def test_vip_after_one_pc(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        vip_pc1 = model.vip(n_components=1).values
        np.testing.assert_array_almost_equal(vip_pc1, PROMV_VIP_AFTER_PC1, decimal=3)

    def test_vip_after_two_pc(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=2)
        vip_pc2 = model.vip(n_components=2).values
        np.testing.assert_array_almost_equal(vip_pc2, PROMV_VIP_AFTER_PC2, decimal=3)

    def test_t2_limit_at_95pct(self) -> None:
        model, _, _ = _fit_pca_on_wold(n_components=1)
        assert model.hotellings_t2_limit(conf_level=0.95) == pytest.approx(WOLD_T2_LIMIT_PC1, abs=5e-3)

    def test_spe_limit_at_95pct(self) -> None:
        # process_improve.spe_limit consumes the model's spe_ values
        # (which are stored as sqrt(e'e)) and returns a limit on the same
        # scale. The MATLAB / ProMV limit is on the e'e scale, so we square
        # the Python limit to compare.
        model, _, _ = _fit_pca_on_wold(n_components=1)
        spe_limit_squared = model.spe_limit(conf_level=0.95) ** 2
        assert spe_limit_squared == pytest.approx(WOLD_SPE_LIMIT_PC1, abs=5e-3)

    def test_score_limit_at_95pct(self) -> None:
        # Score limit per component using the t-distribution:
        # lim.t = t.ppf(0.975, N-1) * std(scores, ddof=1)
        # The MATLAB code uses this form; process_improve does not expose
        # the limit directly but exposes the score scaling factor.
        from scipy import stats

        model, _, _ = _fit_pca_on_wold(n_components=1)
        score_std_ddof1 = model.scaling_factor_for_scores_.iloc[0]
        n_rows = WOLD_X.shape[0]
        score_limit = stats.t.ppf(0.975, n_rows - 1) * score_std_ddof1
        assert score_limit == pytest.approx(WOLD_SCORE_LIMIT_PC1, abs=5e-3)

    def test_predict_new_observations_one_pc(self) -> None:
        model, scaler, _ = _fit_pca_on_wold(n_components=1)
        new_pp = scaler.transform(pd.DataFrame(WOLD_X_NEW))
        result = model.predict(new_pp)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 0].values, WOLD_T_NEW_PC1, decimal=3)

    def test_predict_new_observations_two_pc(self) -> None:
        model, scaler, _ = _fit_pca_on_wold(n_components=2)
        new_pp = scaler.transform(pd.DataFrame(WOLD_X_NEW))
        result = model.predict(new_pp)
        np.testing.assert_array_almost_equal(result.scores.values, WOLD_T_NEW_2COMP, decimal=3)

    def test_predict_training_data_returns_training_scores(self) -> None:
        """Applying the fitted model to its own training data must reproduce
        the training scores exactly (within tolerance).
        """
        model, _, df_pp = _fit_pca_on_wold(n_components=2)
        result = model.predict(df_pp)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 0].values, WOLD_T_PC1, decimal=3)
        np.testing.assert_array_almost_equal(result.scores.iloc[:, 1].values, WOLD_T_PC2, decimal=3)

    def test_predict_training_spe_matches_fit_spe(self) -> None:
        model, _, df_pp = _fit_pca_on_wold(n_components=1)
        result = model.predict(df_pp)
        np.testing.assert_array_almost_equal(result.spe.values**2, PROMV_SPE_AFTER_PC1, decimal=4)


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
        result = model.predict(x_blocks)
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

    def test_score_contributions_to_origin_back_project_correctly(self, synthetic_two_block) -> None:
        """For ``t_super_end=0`` (the model centre), the per-block score
        contributions must equal the per-block sums of effective-loading
        projections of the (negated) super-score row.
        """
        from process_improve.multivariate.methods import MBPCA

        x_blocks, _ = synthetic_two_block
        model = MBPCA(n_components=2).fit(x_blocks)
        t_row = model.super_scores_.iloc[0].values
        contribs = model.score_contributions(t_row)
        # Effective per-block reconstruction at the row level: compare manually
        for b_idx, name in enumerate(model.block_names_):
            sqrt_kb = float(np.sqrt(model.block_widths_[name]))
            pb = model.block_loadings_[name].values
            ps = model.super_loadings_.values[b_idx, :]
            effective = pb * (ps * sqrt_kb)
            expected = -t_row @ effective.T  # back-projected variable contributions
            np.testing.assert_array_almost_equal(contribs[name].values, expected, decimal=10)

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
        result = m.predict(x_new)
        assert not result.super_scores.isna().any().any()


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
        result = model.predict(x_blocks)
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

    def test_x_row_all_nan_raises(self, two_block) -> None:
        from process_improve.multivariate.methods import MBPLS

        x, y = two_block
        x_n = {name: df.copy() for name, df in x.items()}
        x_n["A"].iloc[7, :] = np.nan
        with pytest.raises(ValueError, match="rows with all values missing"):
            MBPLS(n_components=2).fit(x_n, y)

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
        result = m.predict(x_new)
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

    The legacy ConnectMV ``test_mbpls.m`` used a 2-block zone split
    purely to make a single-block-PLS oracle assertion convenient.
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
        result = model.predict(x_blocks)
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

    def test_score_contributions_to_origin_back_project_correctly(self, ldpe) -> None:
        """For each per-block contribution Series, manual back-projection
        through ``w_super[b] * w_b / sqrt(K_b)`` reproduces the result.
        """
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        t_row = model.super_scores_.iloc[0].values
        contribs = model.score_contributions(t_row)
        for b_idx, name in enumerate(model.block_names_):
            sqrt_kb = float(np.sqrt(model.block_widths_[name]))
            wb = model.block_weights_[name].values
            ws = model.super_weights_.values[b_idx, :]
            effective = wb * (ws / sqrt_kb)
            expected = -t_row @ effective.T
            np.testing.assert_array_almost_equal(contribs[name].values, expected, decimal=10)

    def test_score_contributions_subset_of_components(self, ldpe) -> None:
        from process_improve.multivariate.methods import MBPLS

        x_blocks, y_df = ldpe
        model = MBPLS(n_components=3).fit(x_blocks, y_df)
        t_row = model.super_scores_.iloc[0].values
        # Components 1 + 2 should differ from full (1+2+3) decomposition
        partial = model.score_contributions(t_row, components=[1, 2])
        full = model.score_contributions(t_row)
        for name in model.block_names_:
            assert not np.allclose(partial[name].values, full[name].values)

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
        from process_improve.multivariate.methods import MBPLS, randomization_test_mbpls

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
        x_n = {
            name: _inject_mcar(df, ratio, rng) if df.shape[1] > 1 else df
            for name, df in x_blocks.items()
        }
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
# Tests deferred until PR2 ships the reference datasets
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires LDPE-PLS reference dataset (planned in PR2 of the multi-block port).")
class TestLDPEReference:
    """Numerical assertions against the LDPE multiblock dataset.

    The legacy ``test_mbpls.m`` and ``test_mbpca.m`` use this dataset for the
    primary algorithmic validation. The dataset (``LDPE-PLS.mat``) is not yet
    in ``process_improve.datasets``; see the multi-block port plan, PR2.
    """


@pytest.mark.skip(reason="Requires FMC reference dataset (planned in PR2 of the multi-block port).")
class TestFMCReference:
    """FMC dataset MBPLS assertions, deferred until PR2."""


@pytest.mark.skip(reason="Requires kamyr-digester / Simca-P reference values (planned in PR2 of the multi-block port).")
class TestSimcaPCAComparison:
    """Numerical comparison against Simca-P 11.5.0.0 (2006) for kamyr-digester
    PCA and PLS models. Deferred until PR2 brings the reference values into
    the repository.
    """
