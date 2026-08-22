"""Regression tests for the 2026-08 repo-wide correctness audit: multivariate core.

Each test pins a specific defect that was found and fixed in the audit; the
test fails on the pre-fix code and passes afterwards. Grouped by module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate._common import SpecificationWarning
from process_improve.multivariate._limits import hotellings_t2_limit, spe_calculation
from process_improve.multivariate.methods import (
    PCA,
    PLS,
    TPLS,
    DataFrameDict,
    MCUVScaler,
    center,
    ellipse_coordinates,
    scale,
)


def _low_rank_matrix(n: int, k: int, rank: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, rank))
    loadings = np.linalg.qr(rng.standard_normal((k, rank)))[0]
    return scores @ loadings.T + noise * rng.standard_normal((n, k))


class TestTSR:
    """PCA(algorithm="tsr"): the N <= K SVD branch and the centring mismatch."""

    def test_wide_matrix_with_missing_data_fits(self) -> None:
        """N < K previously crashed with IndexError inside the imputation loop."""
        x = _low_rank_matrix(n=12, k=25, rank=2, noise=0.01, seed=1)
        x -= x.mean(axis=0)
        rng = np.random.default_rng(2)
        mask = rng.random(x.shape) < 0.05
        x_missing = x.copy()
        x_missing[mask] = np.nan

        model = PCA(n_components=2, algorithm="tsr").fit(pd.DataFrame(x_missing))
        gram = model.loadings_.values.T @ model.loadings_.values
        assert np.allclose(gram, np.eye(2), atol=1e-2)
        assert np.isfinite(model.scores_.values).all()

    def test_square_matrix_imputation_recovers_low_rank_cells(self) -> None:
        """N == K previously ran the imputation against a transposed matrix.

        No exception was raised; the imputed values were garbage. On low-rank
        data the reconstruction at the masked cells must be close to the true
        (hidden) values.
        """
        n = 30
        x = _low_rank_matrix(n=n, k=n, rank=2, noise=0.01, seed=3)
        x -= x.mean(axis=0)
        rng = np.random.default_rng(4)
        flat = rng.choice(n * n, size=25, replace=False)
        mask = np.zeros((n, n), dtype=bool)
        mask.ravel()[flat] = True
        x_missing = x.copy()
        x_missing[mask] = np.nan

        model = PCA(n_components=2, algorithm="tsr").fit(pd.DataFrame(x_missing))
        reconstruction = model.scores_.values @ model.loadings_.values.T
        residual_at_masked = reconstruction[mask] - x[mask]
        assert np.sqrt(np.mean(residual_at_masked**2)) < 0.2 * float(x.std())

    def test_scores_match_projection_of_complete_data(self) -> None:
        """TSR scores must agree with transform() on the same (complete) data.

        The pre-fix code centred the scores inside fit() while transform()
        projects without centring, so fitted and projected scores disagreed by
        the column-mean offset for any not-exactly-centred input.
        """
        x = _low_rank_matrix(n=40, k=6, rank=2, noise=0.05, seed=5) + 3.0  # deliberate offset
        frame = pd.DataFrame(x)
        model = PCA(n_components=2, algorithm="tsr").fit(frame)
        projected = model.transform(frame)
        np.testing.assert_allclose(projected.values, model.scores_.values, atol=1e-8)

    def test_sign_convention_matches_svd(self) -> None:
        """Largest-magnitude loading element is positive, matching _fit_svd."""
        x = _low_rank_matrix(n=25, k=8, rank=3, noise=0.05, seed=6)
        x -= x.mean(axis=0)
        model = PCA(n_components=3, algorithm="tsr").fit(pd.DataFrame(x))
        for a in range(3):
            column = model.loadings_.values[:, a]
            assert column[np.argmax(np.abs(column))] > 0


class TestEllipse:
    """Score-plot T2 ellipse: bivariate limit, not the full-model limit."""

    @pytest.fixture
    def model(self) -> PCA:
        x = _low_rank_matrix(n=50, k=8, rank=6, noise=0.2, seed=7)
        frame = pd.DataFrame(MCUVScaler().fit_transform(pd.DataFrame(x)))
        return PCA(n_components=5).fit(frame)

    def test_uses_two_component_limit(self, model: PCA) -> None:
        x_coords, y_coords = model.ellipse_coordinates(1, 2)
        expected_radius = np.sqrt(hotellings_t2_limit(0.95, n_components=2, n_rows=50))
        # rtol accounts for the discrete parametrisation: no sample point
        # lands exactly on the semi-axis for the vertical direction.
        assert np.isclose(np.max(x_coords), expected_radius * model.scaling_factor_for_scores_.iloc[0], rtol=1e-3)
        assert np.isclose(np.max(y_coords), expected_radius * model.scaling_factor_for_scores_.iloc[1], rtol=1e-3)
        # And it must be strictly tighter than the (wrong) full-model limit.
        full_radius = np.sqrt(hotellings_t2_limit(0.95, n_components=5, n_rows=50))
        assert np.max(x_coords) < full_radius * model.scaling_factor_for_scores_.iloc[0]

    def test_rejects_degenerate_n_points(self, model: PCA) -> None:
        with pytest.raises(ValueError, match="n_points"):
            model.ellipse_coordinates(1, 2, n_points=1)

    def test_standalone_function_agrees(self, model: PCA) -> None:
        x_m, y_m = model.ellipse_coordinates(1, 2)
        x_f, y_f = ellipse_coordinates(
            score_horiz=1,
            score_vert=2,
            n_components=model.n_components,
            scaling_factor_for_scores=model.scaling_factor_for_scores_,
            n_rows=model.n_samples_,
        )
        np.testing.assert_allclose(x_m, x_f)
        np.testing.assert_allclose(y_m, y_f)


class TestPlotLimits:
    """spe_plot / t2_plot must draw the limit for the plotted component count."""

    @pytest.fixture
    def model(self) -> PCA:
        x = _low_rank_matrix(n=40, k=6, rank=4, noise=0.3, seed=8)
        frame = pd.DataFrame(MCUVScaler().fit_transform(pd.DataFrame(x)))
        return PCA(n_components=3).fit(frame)

    def test_t2_plot_limit_matches_sub_model(self, model: PCA) -> None:
        fig = model.t2_plot(with_a=1)
        expected = hotellings_t2_limit(0.95, n_components=1, n_rows=model.n_samples_)
        limit_line = fig.layout.shapes[0]
        assert np.isclose(limit_line.y0, expected)
        # y-axis label is the data series name, not the limit legend entry.
        assert "T2 values" in fig.layout.yaxis.title.text

    def test_spe_plot_limit_matches_sub_model(self, model: PCA) -> None:
        fig = model.spe_plot(with_a=1)
        expected = spe_calculation(model.spe_[1], conf_level=0.95)
        limit_line = fig.layout.shapes[0]
        assert np.isclose(limit_line.y0, expected)
        assert "SPE values" in fig.layout.yaxis.title.text

    def test_full_model_plots_unchanged(self, model: PCA) -> None:
        fig = model.t2_plot()
        expected = hotellings_t2_limit(0.95, n_components=3, n_rows=model.n_samples_)
        assert np.isclose(fig.layout.shapes[0].y0, expected)


class TestSelectNComponentsScales:
    """PCA CV: the 1-SE band and the Q2 null model."""

    def test_se_press_is_on_the_press_scale(self) -> None:
        x = _low_rank_matrix(n=30, k=8, rank=3, noise=0.3, seed=9)
        n_folds = 5
        result = PCA.select_n_components(pd.DataFrame(x), max_components=4, cv=n_folds, random_state=0)
        per_fold = result.per_fold_press.to_numpy()
        counts = np.maximum(1, np.sum(~np.isnan(per_fold), axis=1))
        expected_se = np.nanstd(per_fold, axis=1, ddof=1) / np.sqrt(counts) * n_folds
        np.testing.assert_allclose(result.se_press.to_numpy(), expected_se)

    def test_q2_null_model_is_centred(self) -> None:
        rng = np.random.default_rng(10)
        x = _low_rank_matrix(n=30, k=6, rank=2, noise=0.4, seed=11) + 100.0  # large offset
        x += rng.standard_normal(x.shape) * 0.01
        result = PCA.select_n_components(pd.DataFrame(x), max_components=3, cv=4, random_state=0)
        x_arr = np.asarray(x, dtype=float)
        null_ss = float(np.nansum((x_arr - np.nanmean(x_arr, axis=0)) ** 2))
        expected_q2 = 1.0 - result.press.to_numpy() / null_ss
        np.testing.assert_allclose(result.q2.to_numpy(), expected_q2)
        # With the uncentred sum(x^2) null model the offset made every Q2
        # indistinguishable from 1; the centred reference must discriminate.
        assert result.q2.iloc[0] < 0.9999


class TestPLSInference:
    """K-fold beta CIs and the NIPALS non-convergence warning."""

    @staticmethod
    def _xy(seed: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(seed)
        x = pd.DataFrame(rng.standard_normal((40, 5)), columns=[f"x{i}" for i in range(5)])
        beta = np.array([1.0, -0.5, 0.0, 0.25, 0.0])
        y = pd.DataFrame({"y": x.values @ beta + 0.1 * rng.standard_normal(40)})
        return x, y

    def test_kfold_beta_std_uses_delete_a_block_jackknife(self) -> None:
        x, y = self._xy()
        model = PLS(n_components=2).fit(x, y)
        result = model.cross_validate(x, y, cv=5, show_progress=False, random_state=0)
        k = result.beta_samples.shape[0]
        deviations_sq = np.sum((result.beta_samples - result.beta_samples.mean(axis=0)) ** 2, axis=0)
        expected_std = np.sqrt((k - 1) / k * deviations_sq)
        np.testing.assert_allclose(result.beta_std.to_numpy(), expected_std)

    def test_pls_warns_when_nipals_hits_max_iter(self) -> None:
        x, y = self._xy(seed=13)
        with pytest.warns(SpecificationWarning, match="maximum number of iterations"):
            PLS(n_components=1, max_iter=1).fit(x, y)

    def test_pca_warns_when_nipals_hits_max_iter(self) -> None:
        x, _ = self._xy(seed=14)
        with pytest.warns(SpecificationWarning, match="maximum number of iterations"):
            PCA(n_components=1, algorithm="nipals", missing_data_settings={"md_max_iter": 1}).fit(x)


class TestRankDeficiencyGuards:
    """T2 must stay finite when a trailing component has ~zero variance."""

    def test_pca_full_rank_request_on_centred_data(self) -> None:
        rng = np.random.default_rng(15)
        x = rng.standard_normal((6, 10))
        x -= x.mean(axis=0)  # rank is now at most N - 1 = 5
        model = PCA(n_components=6).fit(pd.DataFrame(x))
        assert np.isfinite(model.hotellings_t2_.values).all()

    def test_pca_rejects_zero_components(self) -> None:
        x = pd.DataFrame(np.random.default_rng(16).standard_normal((10, 4)))
        with pytest.raises(ValueError, match="n_components"):
            PCA(n_components=0).fit(x)

    def test_pls_rejects_zero_components(self) -> None:
        rng = np.random.default_rng(17)
        x = pd.DataFrame(rng.standard_normal((10, 4)))
        y = pd.DataFrame(rng.standard_normal((10, 1)))
        with pytest.raises(ValueError, match="n_components"):
            PLS(n_components=0).fit(x, y)


class TestTargetProjection:
    """The TP direction must live in the model's internal (scaled) space."""

    def test_raw_and_prescaled_fits_agree(self) -> None:
        rng = np.random.default_rng(18)
        n = 60
        # Columns on wildly different raw scales; y driven by the first two.
        x_raw = pd.DataFrame(
            {
                "a": 1000.0 * rng.standard_normal(n) + 5000.0,
                "b": 0.001 * rng.standard_normal(n),
                "c": rng.standard_normal(n),
                "d": 10.0 * rng.standard_normal(n),
            }
        )
        y_raw = pd.DataFrame({"y": 0.002 * x_raw["a"] + 800.0 * x_raw["b"] + 0.05 * rng.standard_normal(n)})

        model_raw = PLS(n_components=2).fit(x_raw, y_raw)  # scale=True, raw input
        tp_raw = model_raw.target_projection(x_raw)

        x_scaled = MCUVScaler().fit_transform(x_raw)
        y_scaled = MCUVScaler().fit_transform(y_raw)
        model_pre = PLS(n_components=2).fit(x_scaled, y_scaled)
        tp_pre = model_pre.target_projection(x_scaled)

        # Same underlying model in two parameterisations: the TP scores must
        # agree (up to a global sign). Pre-fix, the raw-units beta was used as
        # a direction in scaled space and the two disagreed wildly.
        corr = np.corrcoef(tp_raw.scores.to_numpy(), tp_pre.scores.to_numpy())[0, 1]
        assert abs(corr) > 0.999

    def test_tp_scores_track_the_response(self) -> None:
        rng = np.random.default_rng(19)
        n = 80
        x_raw = pd.DataFrame(
            {
                "big": 1e4 * rng.standard_normal(n),
                "small": 1e-3 * rng.standard_normal(n),
                "noise": rng.standard_normal(n),
            }
        )
        y_raw = pd.DataFrame({"y": 5000.0 * x_raw["small"] + 0.01 * rng.standard_normal(n)})
        model = PLS(n_components=2).fit(x_raw, y_raw)
        tp = model.target_projection(x_raw)
        corr = np.corrcoef(tp.scores.to_numpy(), y_raw["y"].to_numpy())[0, 1]
        assert abs(corr) > 0.99


class TestPreprocessingGuards:
    """MCUVScaler degenerate columns and center()/scale() axis=1."""

    def test_mcuv_single_observation_column(self) -> None:
        x = pd.DataFrame({"good": [1.0, 2.0, 3.0], "lonely": [5.0, np.nan, np.nan]})
        scaler = MCUVScaler().fit(x)
        assert scaler.scale_["lonely"] == 1.0
        out = scaler.transform(x)
        assert np.isfinite(out["lonely"].iloc[0])

    def test_mcuv_all_nan_column(self) -> None:
        x = pd.DataFrame({"good": [1.0, 2.0, 3.0], "empty": [np.nan, np.nan, np.nan]})
        scaler = MCUVScaler().fit(x)
        assert scaler.scale_["empty"] == 1.0
        assert scaler.center_["empty"] == 0.0
        out = scaler.transform(x)
        # The NaN cells pass through; the finite column is untouched.
        assert out["empty"].isna().all()
        assert np.isfinite(out["good"]).all()

    def test_center_axis_1(self) -> None:
        x = np.arange(12, dtype=float).reshape(3, 4)
        centred = center(x, axis=1)
        np.testing.assert_allclose(np.asarray(centred).mean(axis=1), 0.0, atol=1e-12)

    def test_scale_axis_1(self) -> None:
        rng = np.random.default_rng(20)
        x = rng.standard_normal((3, 5)) * np.array([[1.0], [10.0], [100.0]])
        scaled = scale(x, axis=1)
        np.testing.assert_allclose(np.asarray(scaled).std(axis=1), 1.0, atol=1e-12)


class TestTPLSDiagnoseMissing:
    """TPLS.diagnose must neutralise NaN cells like fit() does."""

    @pytest.fixture
    def fitted(self) -> tuple[TPLS, dict]:
        rng = np.random.default_rng(21)
        n, n_materials = 24, 5
        d = pd.DataFrame(
            rng.standard_normal((n_materials, 3)),
            index=[f"m{i}" for i in range(n_materials)],
            columns=["p1", "p2", "p3"],
        )
        f = pd.DataFrame(
            np.abs(rng.standard_normal((n, n_materials))),
            columns=d.index,
            index=[f"obs{i}" for i in range(n)],
        )
        f = f.div(f.sum(axis=1), axis=0)  # ratios per observation
        z = pd.DataFrame(rng.standard_normal((n, 2)), columns=["z1", "z2"], index=f.index)
        y = pd.DataFrame(
            {"q1": f.values @ rng.standard_normal(n_materials) + 0.1 * z["z1"].values},
            index=f.index,
        )
        data = {"F": {"G": f}, "Z": {"Cond": z}, "Y": {"Quality": y}}
        model = TPLS(n_components=2, d_matrix={"G": d}).fit(DataFrameDict(data))
        return model, data

    def test_single_missing_cell_stays_finite(self, fitted: tuple[TPLS, dict]) -> None:
        model, data = fitted
        f_new = data["F"]["G"].iloc[:4].copy()
        z_new = data["Z"]["Cond"].iloc[:4].copy()
        f_new.iloc[1, 2] = np.nan  # a single missing formula entry
        result = model.diagnose(DataFrameDict({"F": {"G": f_new}, "Z": {"Cond": z_new}}))
        assert np.isfinite(result.t_scores_super.values).all()
        assert np.isfinite(result.hotellings_t2.values).all()
        for frame in result.hat.values():
            assert np.isfinite(frame.values).all()

    def test_rows_without_missing_cells_are_unaffected(self, fitted: tuple[TPLS, dict]) -> None:
        model, data = fitted
        f_new = data["F"]["G"].iloc[:4].copy()
        z_new = data["Z"]["Cond"].iloc[:4].copy()
        clean = model.diagnose(DataFrameDict({"F": {"G": f_new.copy()}, "Z": {"Cond": z_new.copy()}}))
        f_new.iloc[1, 2] = np.nan
        poked = model.diagnose(DataFrameDict({"F": {"G": f_new}, "Z": {"Cond": z_new}}))
        other_rows = [0, 2, 3]
        np.testing.assert_allclose(
            poked.t_scores_super.iloc[other_rows].values,
            clean.t_scores_super.iloc[other_rows].values,
        )
