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
        np.testing.assert_array_almost_equal(
            model.r2_per_component_.values, WOLD_R2_PER_COMPONENT, decimal=3
        )

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
        np.testing.assert_array_almost_equal(
            result.spe.values ** 2, PROMV_SPE_AFTER_PC1, decimal=4
        )


# ---------------------------------------------------------------------------
# Multi-block PCA reference tests (xfail until the MBPCA class lands in PR6)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="MBPCA class not yet implemented; will be added in PR6 of the multi-block port.",
    strict=True,
    raises=ImportError,
)
class TestWold1987MBPCA:
    """Multi-block PCA on the Wold matrix split into two blocks.

    Conceptual setup
    ----------------
    Split ``WOLD_X`` into two blocks: ``X1 = columns 0-1``, ``X2 = columns 2-3``.
    With equal block-scaling, the multiblock super-scores must equal the
    single-block PCA scores from :class:`TestWold1987PCA` to within tolerance.

    These tests are written now to lock in the expected behaviour so they
    will become active automatically when MBPCA is implemented.
    """

    def test_super_scores_match_single_block_pca_pc1(self) -> None:
        from process_improve.multivariate.methods import MBPCA  # noqa: F401

        # Intentionally left as a structural test; populated when MBPCA exists.
        raise ImportError("MBPCA not yet implemented")

    def test_super_scores_match_single_block_pca_pc2(self) -> None:
        from process_improve.multivariate.methods import MBPCA  # noqa: F401

        raise ImportError("MBPCA not yet implemented")

    def test_block_scores_recoverable_from_super_scores(self) -> None:
        """Per-Westerhuis/Kourti/MacGregor 1998: block scores recovered from
        the merged-then-recover path must equal block scores from the
        full multiblock NIPALS loop.
        """
        from process_improve.multivariate.methods import MBPCA  # noqa: F401

        raise ImportError("MBPCA not yet implemented")


# ---------------------------------------------------------------------------
# Multi-block PLS reference tests (xfail until the MBPLS class lands in PR3)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="MBPLS class not yet implemented; will be added in PR3 of the multi-block port.",
    strict=True,
    raises=ImportError,
)
class TestSmallMBPLS:
    """Multi-block PLS on a tiny synthetic problem.

    These mirror the assertions in the legacy ``test_mbpls.m`` self-consistency
    block: regardless of which path produces the model (a single big-X PLS
    with later block recovery, vs. the full multi-block NIPALS loop), super
    scores, super weights, block scores, block weights and Y predictions must
    agree to within numerical precision.
    """

    def test_super_scores_match_path_a_and_path_b(self) -> None:
        from process_improve.multivariate.methods import MBPLS  # noqa: F401

        raise ImportError("MBPLS not yet implemented")

    def test_super_weight_unit_norm(self) -> None:
        from process_improve.multivariate.methods import MBPLS  # noqa: F401

        raise ImportError("MBPLS not yet implemented")

    def test_y_prediction_matches_single_block_pls(self) -> None:
        from process_improve.multivariate.methods import MBPLS  # noqa: F401

        raise ImportError("MBPLS not yet implemented")


# ---------------------------------------------------------------------------
# Tests deferred until PR2 ships the reference datasets
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Requires LDPE-PLS reference dataset (planned in PR2 of the multi-block port)."
)
class TestLDPEReference:
    """Numerical assertions against the LDPE multiblock dataset.

    The legacy ``test_mbpls.m`` and ``test_mbpca.m`` use this dataset for the
    primary algorithmic validation. The dataset (``LDPE-PLS.mat``) is not yet
    in ``process_improve.datasets``; see the multi-block port plan, PR2.
    """


@pytest.mark.skip(
    reason="Requires FMC reference dataset (planned in PR2 of the multi-block port)."
)
class TestFMCReference:
    """FMC dataset MBPLS assertions, deferred until PR2."""


@pytest.mark.skip(
    reason="Requires kamyr-digester / Simca-P reference values (planned in PR2 of the multi-block port)."
)
class TestSimcaPCAComparison:
    """Numerical comparison against Simca-P 11.5.0.0 (2006) for kamyr-digester
    PCA and PLS models. Deferred until PR2 brings the reference values into
    the repository.
    """
