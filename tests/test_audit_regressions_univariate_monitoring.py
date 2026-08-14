"""Regression tests for the 2026-08 repo-wide correctness audit: univariate + monitoring.

Each test pins a specific defect found and fixed in the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from process_improve.monitoring.control_charts import ControlChart, rho
from process_improve.monitoring.metrics import calculate_cpk
from process_improve.univariate import metrics as univariate


class TestGeneralizedESD:
    """detect_outliers_esd: last crossing, and the classical default."""

    def test_number_of_outliers_is_largest_crossing(self) -> None:
        """NIST: the outlier count is the LARGEST i with R_i > lambda_i.

        Build a masking scenario: a tight cluster plus three far points close
        to one another. R_1 dips (masking), and only at i=3 do the statistics
        clear the critical values again; the pre-fix code stopped at the
        first crossing and under-reported.
        """
        rng = np.random.default_rng(0)
        base = rng.normal(0.0, 1.0, size=40)
        contaminated = np.concatenate([base, [12.0, 12.5, 13.0]])
        outliers, details = univariate.detect_outliers_esd(contaminated, max_outliers_detected=5)
        crossings = np.where(np.array(details["R_i"]) >= np.array(details["lambda"]))[0]
        assert details["cutoff"] == crossings[-1]
        assert len(outliers) == 3
        assert set(outliers) == {40, 41, 42}

    def test_default_is_classical_statistic(self) -> None:
        """The MAD-scaled variant is anti-conservative against classical
        critical values, so it must be opt-in, not the default."""
        rng = np.random.default_rng(1)
        clean_sample = rng.normal(10.0, 2.0, size=60)
        outliers, _ = univariate.detect_outliers_esd(clean_sample, max_outliers_detected=6)
        # Clean normal data: a calibrated 5% test should find (almost) nothing.
        assert len(outliers) <= 1

    def test_nist_rosner_example_still_reproduces(self) -> None:
        """The classical path must still match the published NIST example."""
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
        # (Rosner, 1983): 54 observations, 3 outliers at the 5% level.
        rosner = np.array(
            [
                -0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43,
                1.49, 1.49, 1.55, 1.56, 1.58, 1.65, 1.69, 1.70, 1.76, 1.77,
                1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15,
                2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64,
                2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30,
                4.64, 5.34, 5.42, 6.01,
            ]
        )  # fmt: skip
        outliers, details = univariate.detect_outliers_esd(
            rosner, max_outliers_detected=10, robust_variant=False
        )
        assert len(outliers) == 3


class TestMedianConfidenceInterval:
    """The robust CI must use the median's standard error, sqrt(pi/2) * sigma / sqrt(n)."""

    def test_robust_interval_carries_median_se_factor(self) -> None:
        rng = np.random.default_rng(2)
        data = pd.DataFrame({"v": rng.normal(50.0, 5.0, size=200)})
        robust = univariate.confidence_interval(data, "v", conflevel=0.95, style="robust")
        mad = univariate.median_absolute_deviation(data["v"].to_numpy(), nan_policy="omit")
        n = 200
        c_t = univariate.t_value(0.975, n - 1)
        expected_half_width = c_t * mad * np.sqrt(np.pi / 2.0) / np.sqrt(n)
        observed_half_width = (robust[1] - robust[0]) / 2.0
        assert observed_half_width == pytest.approx(expected_half_width, rel=1e-12)

    def test_robust_coverage_on_normal_data(self) -> None:
        """~95% of intervals must contain the true median (was ~87% pre-fix)."""
        rng = np.random.default_rng(3)
        hits = 0
        n_sim = 400
        for _ in range(n_sim):
            sample = pd.DataFrame({"v": rng.normal(0.0, 1.0, size=60)})
            low, high = univariate.confidence_interval(sample, "v", conflevel=0.95, style="robust")
            hits += low <= 0.0 <= high
        coverage = hits / n_sim
        assert coverage > 0.92


class TestVarianceDecomposition:
    def test_between_stddev_is_the_variance_component(self) -> None:
        df = pd.DataFrame(data={"Result": [101, 102, 94, 95], "Repeat": [1, 1, 2, 2]})
        out = univariate.variance_decomposition(df, measured="Result", repeat="Repeat")
        assert out["between_ms"] == pytest.approx(49.0)
        # sigma_between^2 = (MS_between - MS_within) / n = (49 - 0.5) / 2.
        assert out["between_stddev"] == pytest.approx(np.sqrt(24.25))
        assert out["within_stddev"] == pytest.approx(np.sqrt(0.5))

    def test_noise_only_data_reports_zero_between_component(self) -> None:
        """When groups do not differ, the between component must clip at 0."""
        rng = np.random.default_rng(4)
        df = pd.DataFrame({"v": rng.normal(0, 1, size=40), "g": np.repeat(np.arange(8), 5)})
        out = univariate.variance_decomposition(df, measured="v", repeat="g")
        assert out["between_stddev"] < out["within_stddev"]


class TestBiweightMidvariance:
    def test_consistent_with_variance_on_normal_data(self) -> None:
        """With c = 9 the estimator tracks sigma^2 on clean Gaussian data.

        The previous c = 6 (the biweight LOCATION constant) rejected too much
        of the sample and was biased low.
        """
        rng = np.random.default_rng(5)
        sample = rng.normal(0.0, 2.0, size=5000)
        bw = univariate.biweight_midvariance(sample)
        assert bw == pytest.approx(4.0, rel=0.10)


class TestHoltWintersChart:
    def test_rho_is_normalised_for_consistency(self) -> None:
        """E[rho(Z)] = 1 for standard-normal Z; pre-fix it was ~0.77."""
        expectation, _ = integrate.quad(lambda z: rho(z) * stats.norm.pdf(z), -12, 12, limit=200)
        assert expectation == pytest.approx(1.0, abs=1e-3)

    def test_limits_track_sigma_on_clean_data(self) -> None:
        """+/-3S limits on well-behaved N(mu, sigma) data must be ~3 sigma wide.

        Pre-fix the biweight constant made S ~ 0.88 sigma, i.e. limits at
        +/-2.6 sigma and a ~3x inflated false-alarm rate.
        """
        rng = np.random.default_rng(6)
        y = pd.Series(rng.normal(100.0, 2.0, size=400))
        cc = ControlChart(variant="HW")
        cc.calculate_limits(y)
        assert cc.s == pytest.approx(2.0, rel=0.15)

    def test_warm_up_residuals_remove_the_trend(self) -> None:
        """A strong warm-up trend must not inflate sigma_0."""
        rng = np.random.default_rng(7)
        n = 400
        trend = 0.5 * np.arange(n)
        y = pd.Series(trend + rng.normal(0.0, 1.0, size=n))
        cc = ControlChart(variant="HW")
        cc._apply_tuning_kwargs({})
        cc.df["y"] = y.values
        cc.N = n
        cc.warm_up["M"] = cc.warm_up_M = 20
        cc.train_samples = [int(i) for i in np.arange(20, n)]
        cc.ld_1 = cc.ld_2 = None
        cc._holt_winters_parameter_fit()
        # sigma_0 estimated from de-trended residuals: ~1, nowhere near the
        # ~ MAD of the raw trending window (~ 2.5+).
        assert cc.warm_up["sigma_0"] < 2.0

    def test_small_sample_grid_search_is_not_degenerate(self) -> None:
        """10 <= N < 20: the training window includes row 0, whose error is
        unset. Pre-fix, every grid cell was NaN and (0.1, 0.1) always 'won'."""
        rng = np.random.default_rng(8)
        y = pd.Series(rng.normal(50.0, 3.0, size=15))
        cc = ControlChart(variant="HW")
        cc.calculate_limits(y)
        assert np.isfinite(cc._residuals_HW).all()
        assert cc.s is not None and np.isfinite(cc.s)

    def test_explicit_zero_lambda_is_respected(self) -> None:
        rng = np.random.default_rng(9)
        y = pd.Series(rng.normal(10.0, 1.0, size=60))
        cc = ControlChart(variant="HW")
        cc.calculate_limits(y, ld_1=0.0, ld_2=0.5)
        # 0.0 must survive as the user's choice, not trigger the grid search.
        assert cc.ld_1 == 0.0
        assert cc.ld_2 == 0.5

    def test_unknown_variant_rejected_up_front(self) -> None:
        with pytest.raises(ValueError, match="not implemented"):
            ControlChart(variant="cusum")


class TestCpkTool:
    def test_rsd_is_of_the_data_not_the_spec_distance(self) -> None:
        rng = np.random.default_rng(10)
        values = rng.normal(100.0, 2.0, size=200)
        df = pd.DataFrame({"v": values})
        result_a = calculate_cpk(df, "v", specifications=(90.0, 110.0))
        result_b = calculate_cpk(df, "v", specifications=(50.0, 150.0))
        # Moving the spec limits must not change the data's RSD.
        assert result_a.rsd == pytest.approx(result_b.rsd, rel=1e-9)
        assert result_a.rsd == pytest.approx(2.0, rel=0.25)  # ~ 2/100 * 100%

    def test_nan_cpk_reports_undefined_not_poor(self) -> None:
        from process_improve.monitoring.tools import ProcessCapabilityInput, process_capability

        rng = np.random.default_rng(11)
        spec = ProcessCapabilityInput(values=list(rng.normal(10, 1, size=20)))  # no spec limits
        result = process_capability(spec)
        assert "error" not in result
        assert "could not be computed" in result["interpretation"]
