"""Tests for the SEC-19 (#268) DoS caps.

Each cap fires with a clear ``ValueError`` *before* the algorithm
allocates the oversized structure. The tests use small overrides on
``process_improve.config.settings`` so they finish quickly without
needing to fabricate a 32k-row matrix in memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.config import settings


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    """Drop the cache before and after each test so overrides leak nowhere."""
    settings.reload()
    yield
    settings.reload()


# ---------------------------------------------------------------------------
# Sub-item 1: combinatorial design generators
# ---------------------------------------------------------------------------


class TestCombinatorialGenerators:
    def test_full_factorial_above_cap_rejected(self) -> None:
        from process_improve.experiments.designs_factorial import full_factorial

        settings.max_factors_combinatorial = 5
        with pytest.raises(ValueError, match="exceeds the SEC-19 combinatorial cap"):
            full_factorial(6)

    def test_full_factorial_at_cap_accepted(self) -> None:
        from process_improve.experiments.designs_factorial import full_factorial

        settings.max_factors_combinatorial = 5
        cols = full_factorial(5)
        assert len(cols) == 5

    def test_simplex_centroid_above_cap_rejected(self) -> None:
        from process_improve.experiments.designs_mixture import _simplex_centroid

        settings.max_factors_combinatorial = 4
        with pytest.raises(ValueError, match="exceeds the SEC-19 cap"):
            _simplex_centroid(5)

    def test_simplex_lattice_above_cap_rejected(self) -> None:
        from process_improve.experiments.designs_mixture import _simplex_lattice

        settings.max_factors_combinatorial = 4
        with pytest.raises(ValueError, match="exceeds the SEC-19 cap"):
            _simplex_lattice(5, degree=2)

    def test_simplex_lattice_iteration_cap(self) -> None:
        """``_simplex_lattice`` rejects ``(degree+1)**k > 1M`` even when
        ``k`` is under the factor cap.
        """
        from process_improve.experiments.designs_mixture import _simplex_lattice

        # 11**6 ~ 1.77M > 1M but k = 6 < default cap of 15.
        with pytest.raises(ValueError, match="1M iteration cap"):
            _simplex_lattice(6, degree=10)


# ---------------------------------------------------------------------------
# Sub-item 2: O(N^2) regression kernels
# ---------------------------------------------------------------------------


class TestRegressionPointsCap:
    def test_repeated_median_above_cap_rejected(self) -> None:
        from process_improve.regression.methods import repeated_median_slope

        settings.max_regression_points = 50
        x = np.arange(60, dtype=float)
        y = x * 2.0 + 1.0
        with pytest.raises(ValueError, match="exceeds the SEC-19"):
            repeated_median_slope(x, y)

    def test_repeated_median_at_cap_accepted(self) -> None:
        from process_improve.regression.methods import repeated_median_slope

        settings.max_regression_points = 100
        x = np.arange(50, dtype=float)
        y = x * 2.0 + 1.0
        slope = repeated_median_slope(x, y)
        assert slope == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Sub-item 3: multivariate matrix dimensions
# ---------------------------------------------------------------------------


class TestMatrixDimensionCap:
    def test_fit_pca_too_many_rows_rejected(self) -> None:
        from process_improve.multivariate.tools import fit_pca

        settings.max_matrix_rows = 5
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 3)).tolist()
        result = fit_pca(data=data, n_components=2)
        assert "error" in result
        assert "max_matrix_rows" in result["error"]

    def test_fit_pca_too_many_cols_rejected(self) -> None:
        from process_improve.multivariate.tools import fit_pca

        settings.max_matrix_cols = 4
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 6)).tolist()
        result = fit_pca(data=data, n_components=2)
        assert "error" in result
        assert "max_matrix_cols" in result["error"]

    def test_fit_pca_within_caps_accepted(self) -> None:
        from process_improve.multivariate.tools import fit_pca

        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 3)).tolist()
        result = fit_pca(data=data, n_components=2)
        assert "error" not in result
        assert result["n_components"] == 2


# ---------------------------------------------------------------------------
# Sub-item 4: new _SCALAR_CAPS keys
# ---------------------------------------------------------------------------


class TestScalarCapsExtension:
    """Each new key in ``_SCALAR_CAPS`` rejects an oversize int via
    ``validate_input`` (the function ``safe_execute_tool_call`` uses).
    """

    @pytest.mark.parametrize(
        ("key", "limit"),
        [
            ("n_steps", 100),
            ("n_additional_runs", 500),
            ("center_points", 50),
            ("replicates", 50),
            ("n_factors", 15),
        ],
    )
    def test_new_scalar_cap_enforced(self, key: str, limit: int) -> None:
        from process_improve.tool_safety import ToolInputTooLargeError, validate_input

        with pytest.raises(ToolInputTooLargeError, match=key):
            validate_input({key: limit + 1})


# ---------------------------------------------------------------------------
# Sub-item 5: fit_linear_model -- formula chars + expanded terms + data rows
# ---------------------------------------------------------------------------


class TestFitLinearModelCaps:
    def _make_data(self, n_rows: int = 8) -> list[dict[str, float]]:
        rng = np.random.default_rng(0)
        cols = ["A", "B", "C", "y"]
        return [
            {c: float(rng.standard_normal()) for c in cols} for _ in range(n_rows)
        ]

    def test_too_long_formula_rejected(self) -> None:
        from process_improve.experiments.tools import fit_linear_model

        settings.max_formula_chars = 50
        long_formula = "y ~ " + " + ".join([f"A_{i}" for i in range(40)])
        assert len(long_formula) > 50
        result = fit_linear_model(formula=long_formula, data=self._make_data())
        assert "error" in result
        assert "max_formula_chars" in result["error"]

    def test_too_many_data_rows_rejected(self) -> None:
        from process_improve.experiments.tools import fit_linear_model

        settings.max_matrix_rows = 5
        data = self._make_data(n_rows=10)
        result = fit_linear_model(formula="y ~ A + B + C", data=data)
        assert "error" in result
        assert "max_matrix_rows" in result["error"]

    def test_too_many_expanded_terms_rejected(self) -> None:
        from process_improve.experiments.tools import fit_linear_model

        # 5 factors, degree=5 -> 2**5 = 32 terms; cap below that triggers.
        settings.max_formula_terms = 5
        rng = np.random.default_rng(0)
        data = [
            {**{c: float(rng.standard_normal()) for c in "ABCDE"}, "y": float(rng.standard_normal())}
            for _ in range(20)
        ]
        result = fit_linear_model(formula="y ~ (A + B + C + D + E) ** 3", data=data)
        # The cap fires; the underlying ValueError is captured and surfaced
        # in result["error"] by the tool wrapper.
        assert "error" in result
        assert "max_formula_terms" in result["error"]

    def test_normal_use_still_works(self) -> None:
        from process_improve.experiments.tools import fit_linear_model

        result = fit_linear_model(formula="y ~ A + B + C", data=self._make_data())
        assert "error" not in result
        assert result["r2"] >= 0.0
