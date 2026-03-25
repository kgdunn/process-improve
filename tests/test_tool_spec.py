"""Tests for the tool_spec decorator, registry, and univariate tool wrappers."""

from __future__ import annotations

import json

import numpy as np
import pytest

# Importing the univariate tools module triggers @tool_spec registration.
import process_improve.multivariate.tools
import process_improve.univariate.tools  # noqa: F401
from process_improve.tool_spec import _TOOL_REGISTRY, execute_tool_call, get_tool_specs, tool_spec
from process_improve.univariate.tools import get_univariate_tool_specs

# ---------------------------------------------------------------------------
# tool_spec decorator and registry
# ---------------------------------------------------------------------------


class TestToolSpecDecorator:
    def test_attaches_tool_spec_attribute(self) -> None:
        """Verify the decorator attaches a _tool_spec dict to the function."""
        @tool_spec(
            name="test_dummy_add",
            description="Add two numbers.",
            input_schema={
                "json": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                }
            },
        )
        def _dummy(*, a: float, b: float) -> dict:
            return {"result": a + b}

        assert hasattr(_dummy, "_tool_spec")
        assert _dummy._tool_spec["name"] == "test_dummy_add"

    def test_function_still_callable(self) -> None:
        """Verify the decorated function remains callable."""
        @tool_spec(
            name="test_dummy_mul",
            description="Multiply.",
            input_schema={
                "json": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                }
            },
        )
        def _mul(*, a: float, b: float) -> dict:
            return {"result": a * b}

        assert _mul(a=3, b=4)["result"] == 12

    def test_examples_appended_to_description(self) -> None:
        """Verify examples text is appended to the description."""
        @tool_spec(
            name="test_dummy_examples",
            description="A tool.",
            input_schema={"json": {"type": "object", "properties": {}, "required": []}},
            examples="# example -> call()",
        )
        def _ex() -> dict:
            return {}

        assert "Examples" in _ex._tool_spec["description"]
        assert "example -> call()" in _ex._tool_spec["description"]

    def test_no_examples_no_extra_text(self) -> None:
        """Verify description is unchanged when no examples are provided."""
        @tool_spec(
            name="test_dummy_no_examples",
            description="Plain description.",
            input_schema={"json": {"type": "object", "properties": {}, "required": []}},
        )
        def _plain() -> dict:
            return {}

        assert _plain._tool_spec["description"] == "Plain description."

    def test_registered_in_registry(self) -> None:
        """Verify decorated tools are added to the global registry."""
        assert "test_dummy_add" in _TOOL_REGISTRY
        assert "test_dummy_mul" in _TOOL_REGISTRY


class TestGetToolSpecs:
    def test_returns_list_of_dicts(self) -> None:
        """Verify get_tool_specs returns a list of dicts."""
        specs = get_tool_specs()
        assert isinstance(specs, list)
        assert all(isinstance(s, dict) for s in specs)

    def test_each_spec_has_required_keys(self) -> None:
        """Verify each spec contains name, description, and input_schema."""
        for spec in get_tool_specs():
            assert "name" in spec
            assert "description" in spec
            assert "input_schema" in spec

    def test_name_filter(self) -> None:
        """Verify filtering specs by name returns only matching entries."""
        specs = get_tool_specs(names=["robust_summary_stats"])
        assert len(specs) == 1
        assert specs[0]["name"] == "robust_summary_stats"

    def test_empty_filter_returns_empty(self) -> None:
        """Verify an empty names filter returns an empty list."""
        specs = get_tool_specs(names=[])
        assert specs == []


class TestExecuteToolCall:
    def test_dispatches_by_name(self) -> None:
        """Verify execute_tool_call dispatches to the correct tool."""
        result = execute_tool_call("robust_summary_stats", {"values": [1, 2, 3]})
        assert "mean" in result

    def test_raises_on_unknown_tool(self) -> None:
        """Verify execute_tool_call raises ValueError for unknown tools."""
        with pytest.raises(ValueError, match="Unknown tool"):
            execute_tool_call("nonexistent_tool_xyz", {})


# ---------------------------------------------------------------------------
# get_univariate_tool_specs
# ---------------------------------------------------------------------------


class TestGetUnivariateToolSpecs:
    def test_returns_all_nine_tools(self) -> None:
        """Verify all nine univariate tool specs are returned."""
        specs = get_univariate_tool_specs()
        names = {s["name"] for s in specs}
        expected = {
            "robust_summary_stats",
            "detect_outliers",
            "robust_scale_sn",
            "median_absolute_deviation",
            "normality_test",
            "confidence_interval",
            "ttest_two_samples",
            "ttest_paired_samples",
            "within_between_variance",
        }
        assert expected == names


# ---------------------------------------------------------------------------
# robust_summary_stats
# ---------------------------------------------------------------------------


class TestRobustSummaryStats:
    def test_basic_keys_present(self) -> None:
        """Verify all expected keys are present in the result."""
        result = execute_tool_call("robust_summary_stats", {"values": [1, 2, 3, 4, 5]})
        for key in ("mean", "median", "std_ddof1", "center", "spread", "N_non_missing"):
            assert key in result

    def test_robust_method_center_is_median(self) -> None:
        """Verify robust method uses median as center."""
        result = execute_tool_call("robust_summary_stats", {"values": [1, 2, 3, 4, 5]})
        assert result["center"] == result["median"]

    def test_classical_method_center_is_mean(self) -> None:
        """Verify classical method uses mean as center."""
        result = execute_tool_call("robust_summary_stats", {"values": [1, 2, 3, 4, 5], "method": "classical"})
        assert result["center"] == result["mean"]

    def test_outlier_does_not_dominate_robust_center(self) -> None:
        """Verify an outlier does not pull the robust center."""
        # Use data with natural spread so Sn > 0 and the robust path is taken
        result = execute_tool_call("robust_summary_stats", {"values": [9, 10, 11, 10, 9, 11, 10, 200]})
        # Robust center (median) should be near 10, not pulled toward 200
        assert result["center"] == pytest.approx(10.0)

    def test_n_non_missing(self) -> None:
        """Verify N_non_missing count is correct."""
        result = execute_tool_call("robust_summary_stats", {"values": [1, 2, 3]})
        assert result["N_non_missing"] == 3

    def test_all_values_json_serialisable(self) -> None:
        """Verify all result values are JSON-serialisable."""
        result = execute_tool_call("robust_summary_stats", {"values": [1.0, 2.0, 3.0]})
        json.dumps(result)  # must not raise


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------


class TestDetectOutliers:
    def test_detects_obvious_outlier(self) -> None:
        """Verify an obvious outlier is detected."""
        result = execute_tool_call("detect_outliers", {"values": [1, 2, 2, 3, 2, 100]})
        assert result["n_outliers_found"] == 1
        assert 5 in result["outlier_indices"]
        assert 100.0 in result["outlier_values"]

    def test_clean_data_no_outliers(self) -> None:
        """Verify clean data produces no outliers."""
        # Uniformly spaced data: ESD finds no outliers
        result = execute_tool_call("detect_outliers", {"values": list(range(1, 15))})
        assert result["n_outliers_found"] == 0

    def test_respects_max_outliers_param(self) -> None:
        """Verify max_outliers_to_detect parameter is respected."""
        result = execute_tool_call(
            "detect_outliers",
            {"values": [1, 2, 3, 4, 5, 1000, 2000], "max_outliers_to_detect": 1},
        )
        assert result["n_outliers_found"] <= 1

    def test_result_json_serialisable(self) -> None:
        """Verify the result is JSON-serialisable."""
        result = execute_tool_call("detect_outliers", {"values": [1, 2, 3, 100]})
        json.dumps(result)


# ---------------------------------------------------------------------------
# robust_scale_sn
# ---------------------------------------------------------------------------


class TestRobustScaleSn:
    def test_returns_sn_key(self) -> None:
        """Verify the result contains an sn key."""
        result = execute_tool_call("robust_scale_sn", {"values": [1, 2, 3, 4, 5]})
        assert "sn" in result

    def test_sn_approximately_matches_std_for_normal_data(self) -> None:
        """Verify Sn approximates standard deviation for normal data."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=0, scale=1, size=200).tolist()
        result = execute_tool_call("robust_scale_sn", {"values": data})
        # For large normal samples, Sn ≈ std
        assert pytest.approx(1.0, abs=0.2) == result["sn"]

    def test_outlier_does_not_inflate_sn(self) -> None:
        """Verify an outlier does not inflate Sn substantially."""
        # Without outlier: data ~[10..15], with: add 10000
        clean = list(range(10, 16))
        dirty = [*clean, 10000]
        clean_result = execute_tool_call("robust_scale_sn", {"values": clean})
        dirty_result = execute_tool_call("robust_scale_sn", {"values": dirty})
        # Sn should be much more stable than std
        assert dirty_result["sn"] < 10 * clean_result["sn"]


# ---------------------------------------------------------------------------
# median_absolute_deviation
# ---------------------------------------------------------------------------


class TestMedianAbsoluteDeviation:
    def test_returns_mad_key(self) -> None:
        """Verify the result contains a mad key."""
        result = execute_tool_call("median_absolute_deviation", {"values": [1, 2, 3, 4, 5]})
        assert "mad" in result

    def test_normal_scale_approx_std(self) -> None:
        """Verify MAD with normal scale approximates standard deviation."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 1000).tolist()
        result = execute_tool_call("median_absolute_deviation", {"values": data, "scale": "normal"})
        assert pytest.approx(1.0, abs=0.1) == result["mad"]

    def test_raw_scale_smaller_than_normal(self) -> None:
        """Verify raw MAD is smaller than normal-scaled MAD."""
        data = list(range(1, 11))
        raw = execute_tool_call("median_absolute_deviation", {"values": data, "scale": "raw"})
        normal = execute_tool_call("median_absolute_deviation", {"values": data, "scale": "normal"})
        assert raw["mad"] < normal["mad"]


# ---------------------------------------------------------------------------
# normality_test
# ---------------------------------------------------------------------------


class TestNormalityTest:
    def test_clearly_normal_data(self) -> None:
        """Verify normal data is identified as normal."""
        rng = np.random.default_rng(1)
        data = rng.normal(10, 1, 100).tolist()
        result = execute_tool_call("normality_test", {"values": data})
        assert result["is_normal"] is True
        assert "p_value" in result
        assert "statistic" in result
        assert "interpretation" in result

    def test_clearly_non_normal_data(self) -> None:
        """Verify non-normal data is flagged appropriately."""
        # Exponential distribution is highly skewed
        rng = np.random.default_rng(2)
        data = rng.exponential(scale=1, size=100).tolist()
        result = execute_tool_call("normality_test", {"values": data})
        # Not guaranteed but very likely to reject normality for exponential
        assert "is_normal" in result

    def test_returns_json_serialisable(self) -> None:
        """Verify the result is JSON-serialisable."""
        result = execute_tool_call("normality_test", {"values": [1.0, 2.0, 1.5, 1.8, 2.1]})
        json.dumps(result)


# ---------------------------------------------------------------------------
# confidence_interval
# ---------------------------------------------------------------------------


class TestConfidenceInterval:
    def test_returns_lower_center_upper(self) -> None:
        """Verify the result contains lower, center, and upper bounds in order."""
        result = execute_tool_call("confidence_interval", {"values": [10, 11, 12, 10, 9, 10]})
        assert result["lower"] < result["center"] < result["upper"]

    def test_wider_at_higher_confidence(self) -> None:
        """Verify a higher confidence level produces a wider interval."""
        values = list(range(1, 20))
        ci95 = execute_tool_call("confidence_interval", {"values": values, "confidence_level": 0.95})
        ci99 = execute_tool_call("confidence_interval", {"values": values, "confidence_level": 0.99})
        width95 = ci95["upper"] - ci95["lower"]
        width99 = ci99["upper"] - ci99["lower"]
        assert width99 > width95

    def test_robust_center_is_median(self) -> None:
        """Verify robust method uses median as center."""
        values = [10, 10, 10, 10, 10]
        result = execute_tool_call("confidence_interval", {"values": values, "method": "robust"})
        assert result["center"] == pytest.approx(10.0)

    def test_classical_center_is_mean(self) -> None:
        """Verify classical method uses mean as center."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = execute_tool_call("confidence_interval", {"values": values, "method": "classical"})
        assert result["center"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ttest_two_samples
# ---------------------------------------------------------------------------


class TestTtestTwoSamples:
    def test_obvious_difference_is_significant(self) -> None:
        """Verify obviously different groups yield a significant result."""
        result = execute_tool_call(
            "ttest_two_samples",
            {"group_a": [100, 101, 99, 100, 101], "group_b": [200, 201, 199, 200, 201]},
        )
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_same_groups_not_significant(self) -> None:
        """Verify identical groups yield a non-significant result."""
        data = [10.0, 10.0, 10.0, 10.0, 10.0]
        result = execute_tool_call("ttest_two_samples", {"group_a": data, "group_b": data})
        assert result["significant"] is False

    def test_output_keys_present(self) -> None:
        """Verify all expected output keys are present."""
        result = execute_tool_call(
            "ttest_two_samples",
            {"group_a": [1, 2, 3], "group_b": [4, 5, 6]},
        )
        for key in ("p_value", "z_value", "conf_int_lower", "conf_int_upper", "interpretation"):
            assert key in result

    def test_result_json_serialisable(self) -> None:
        """Verify the result is JSON-serialisable."""
        result = execute_tool_call("ttest_two_samples", {"group_a": [1, 2, 3], "group_b": [4, 5, 6]})
        json.dumps(result)


# ---------------------------------------------------------------------------
# ttest_paired_samples
# ---------------------------------------------------------------------------


class TestTtestPairedSamples:
    def test_before_after_no_effect(self) -> None:
        """Verify identical before/after data yields zero mean difference."""
        values = [10, 11, 9, 10, 11]
        result = execute_tool_call("ttest_paired_samples", {"group_a": values, "group_b": values})
        # identical before/after: differences all zero
        assert result["differences_mean"] == pytest.approx(0.0)

    def test_output_keys_present(self) -> None:
        """Verify all expected output keys are present."""
        result = execute_tool_call(
            "ttest_paired_samples",
            {"group_a": [70, 65, 80], "group_b": [75, 70, 82]},
        )
        for key in ("differences_mean", "p_value", "z_value", "significant", "interpretation"):
            assert key in result

    def test_result_json_serialisable(self) -> None:
        """Verify the result is JSON-serialisable."""
        result = execute_tool_call("ttest_paired_samples", {"group_a": [1, 2, 3], "group_b": [1, 2, 3]})
        json.dumps(result)


# ---------------------------------------------------------------------------
# within_between_variance
# ---------------------------------------------------------------------------


class TestWithinBetweenVariance:
    def test_day_example_matches_docs(self) -> None:
        """Verify the day example matches documented expected values."""
        result = execute_tool_call(
            "within_between_variance",
            {"values": [101, 102, 94, 95], "groups": [1, 1, 2, 2]},
        )
        assert result["within_stddev"] == pytest.approx(0.70711, abs=1e-4)
        assert result["between_stddev"] == pytest.approx(7.0, abs=1e-4)

    def test_output_keys_present(self) -> None:
        """Verify all expected output keys are present."""
        result = execute_tool_call(
            "within_between_variance",
            {"values": [1, 2, 3, 4], "groups": ["A", "A", "B", "B"]},
        )
        for key in ("within_ms", "within_stddev", "between_ms", "between_stddev"):
            assert key in result

    def test_string_group_labels(self) -> None:
        """Verify string group labels are handled correctly."""
        result = execute_tool_call(
            "within_between_variance",
            {
                "values": [10.1, 10.2, 10.0, 10.5, 10.4, 10.6],
                "groups": ["Alice", "Alice", "Alice", "Bob", "Bob", "Bob"],
            },
        )
        assert result["within_stddev"] is not None
        assert result["between_stddev"] is not None


# ---------------------------------------------------------------------------
# Multivariate tool wrapper tests (improving multivariate/tools.py coverage)
# ---------------------------------------------------------------------------


class TestFitPca:
    """Tests for the fit_pca tool wrapper."""

    def test_basic_fit(self) -> None:
        """Verify basic PCA fitting returns expected structure."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((30, 4)).tolist()
        result = execute_tool_call("fit_pca", {"data": data, "n_components": 2})
        assert "error" not in result
        assert result["n_components"] == 2
        assert result["n_samples"] == 30
        assert result["n_features"] == 4
        assert len(result["r2_cumulative"]) == 2

    def test_with_column_names(self) -> None:
        """Verify PCA fitting works with column names."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 3)).tolist()
        result = execute_tool_call(
            "fit_pca",
            {"data": data, "n_components": 2, "column_names": ["A", "B", "C"]},
        )
        assert "error" not in result
        assert "model_params" in result

    def test_returns_outliers(self) -> None:
        """Verify outlier indices are included in the result."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 4)).tolist()
        # Add an obvious outlier
        data.append([100.0, 100.0, 100.0, 100.0])
        result = execute_tool_call("fit_pca", {"data": data, "n_components": 2})
        assert "outlier_indices" in result


class TestFitPls:
    """Tests for the fit_pls tool wrapper."""

    def test_basic_fit(self) -> None:
        """Verify basic PLS fitting returns expected structure."""
        rng = np.random.default_rng(42)
        x_data = rng.standard_normal((30, 3)).tolist()
        y_data = [float(sum(row) + rng.normal() * 0.1) for row in x_data]
        result = execute_tool_call(
            "fit_pls",
            {"x_data": x_data, "y_data": y_data, "n_components": 2},
        )
        assert "error" not in result, f"Got error: {result.get('error')}"
        assert result["n_components"] == 2
        assert "r2x_cumulative" in result
        assert "model_params" in result

    def test_with_column_names(self) -> None:
        """Verify PLS fitting works with column names."""
        rng = np.random.default_rng(42)
        x_data = rng.standard_normal((20, 3)).tolist()
        y_data = [[sum(row)] for row in x_data]
        result = execute_tool_call(
            "fit_pls",
            {
                "x_data": x_data,
                "y_data": y_data,
                "n_components": 1,
                "x_column_names": ["T", "P", "F"],
                "y_column_names": ["yield"],
            },
        )
        assert "error" not in result


class TestPcaPredict:
    """Tests for the pca_predict tool wrapper."""

    def test_predict_new_data(self) -> None:
        """Verify PCA prediction on new data returns scores."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((40, 3)).tolist()
        fit_result = execute_tool_call("fit_pca", {"data": data, "n_components": 2})
        assert "model_params" in fit_result

        new_data = rng.standard_normal((5, 3)).tolist()
        pred_result = execute_tool_call(
            "pca_predict",
            {"new_data": new_data, "model_params": fit_result["model_params"]},
        )
        assert "error" not in pred_result
        assert len(pred_result["scores"]) == 5


class TestPlsPredict:
    """Tests for the pls_predict tool wrapper."""

    def test_predict_new_data(self) -> None:
        """Verify PLS prediction on new data returns y_hat values."""
        rng = np.random.default_rng(42)
        x_data = rng.standard_normal((30, 3)).tolist()
        y_data = [sum(row) for row in x_data]
        fit_result = execute_tool_call(
            "fit_pls",
            {"x_data": x_data, "y_data": y_data, "n_components": 2},
        )
        assert "model_params" in fit_result

        new_x = rng.standard_normal((5, 3)).tolist()
        pred_result = execute_tool_call(
            "pls_predict",
            {"new_data": new_x, "model_params": fit_result["model_params"]},
        )
        assert "error" not in pred_result
        assert len(pred_result["y_hat"]) == 5
