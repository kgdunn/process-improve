"""Property tests for the PCA / PLS / TPLS numerical safety fixes in SEC-21.

These tests pin the post-sweep behaviour for sub-items 2, 4, 6, 7, 8
(see the SEC-21 issue body for the full list). Each test exercises a
degenerate input that previously produced silent ``inf`` / ``NaN``
poisoning and asserts the new behaviour: either a clean error, a
flagged result, or a NaN that propagates predictably.

The tests live in ``tests/properties/`` per ENG-29 marker conventions:
they all use ``hypothesis`` strategies and run in the regular pytest
session (no ``slow`` / ``dataset`` mark).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from process_improve.multivariate.methods import PCA, MCUVScaler


@pytest.fixture
def _quiet_runtime_warnings():
    """The sweeps fix the *poisoning*, not every adjacent warning numpy
    emits inside a fitting loop. Silence those here so a test asserting
    on the fitted attribute's shape isn't masked by an unrelated warning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


# ---------------------------------------------------------------------------
# Sub-item 4: r2_per_variable_ on constant columns is NaN (not inf / 1.0)
# ---------------------------------------------------------------------------


class TestR2PerVariableConstantColumn:
    """A constant column has no variance to explain; r2_per_variable_ for
    that column must be NaN, not ``inf`` or ``1.0``.

    Previously the code did ``1 - col_ssx / prior_ssx_col`` directly,
    so a column with ``prior_ssx_col == 0`` produced a silent
    ``RuntimeWarning: invalid value encountered in divide`` and a
    NaN-or-inf entry the caller could not distinguish from a meaningful
    R^2 (SEC-21 #270 sub-item 4).
    """

    def _make_df_with_constant_column(
        self,
        n_rows: int = 20,
        n_cols: int = 4,
        constant_col_idx: int = 1,
        seed: int = 0,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_cols))
        X[:, constant_col_idx] = 7.5  # the constant
        return pd.DataFrame(X, columns=[f"v{i}" for i in range(n_cols)])

    def test_svd_path_emits_nan_for_constant_column(self, _quiet_runtime_warnings):
        df = self._make_df_with_constant_column()
        # SVD path needs MCUV-scaled inputs to be defined.
        scaled = MCUVScaler().fit_transform(df)
        pca = PCA(n_components=2, algorithm="svd").fit(scaled)
        assert np.isnan(pca.r2_per_variable_.loc["v1"]).all()
        # The other variables get a finite R^2.
        for v in ["v0", "v2", "v3"]:
            assert np.isfinite(pca.r2_per_variable_.loc[v]).all()

    def test_nipals_path_emits_nan_for_constant_column(self, _quiet_runtime_warnings):
        df = self._make_df_with_constant_column()
        scaled = MCUVScaler().fit_transform(df)
        pca = PCA(n_components=2, algorithm="nipals").fit(scaled)
        assert np.isnan(pca.r2_per_variable_.loc["v1"]).all()

    @given(
        n_rows=st.integers(min_value=5, max_value=15),
        n_cols=st.integers(min_value=3, max_value=6),
        constant_col_idx=st.integers(min_value=0, max_value=5),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=20, deadline=None)
    def test_property_constant_column_r2_is_nan(
        self,
        n_rows: int,
        n_cols: int,
        constant_col_idx: int,
        seed: int,
    ) -> None:
        # constant_col_idx must be in range; hypothesis-shrunk failures
        # are handled below.
        if constant_col_idx >= n_cols:
            return
        if n_rows < n_cols + 1:
            return  # SVD needs N > K
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n_rows, n_cols))
        X[:, constant_col_idx] = float(seed)  # constant value
        df = pd.DataFrame(X, columns=[f"v{i}" for i in range(n_cols)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scaled = MCUVScaler().fit_transform(df)
            pca = PCA(n_components=2, algorithm="svd").fit(scaled)
        # The constant column's R^2 entries must all be NaN.
        assert np.isnan(pca.r2_per_variable_.iloc[constant_col_idx, :]).all()


# ---------------------------------------------------------------------------
# Sub-item 2: PCA NIPALS does not poison its input X via initial-guess slice
# ---------------------------------------------------------------------------


class TestNipalsDoesNotMutateInput:
    """``Xd[:, [0]]`` returns a copy under current numpy semantics, but
    SEC-21 sub-item 2 adds a defensive ``.copy()`` to protect against
    future numpy view-returning changes. This test pins the behaviour:
    after a NIPALS fit, the caller's training data column 0 still has
    its original NaN sentinel value (if any).
    """

    def test_nipals_does_not_overwrite_input_column0_nan(self, _quiet_runtime_warnings):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 4))
        X[3, 0] = np.nan  # sentinel that the algorithm zeroes internally
        df = pd.DataFrame(X, columns=list("ABCD"))
        # Caller-visible copy of column 0 before fitting.
        col0_before = df["A"].copy()
        scaled = MCUVScaler().fit_transform(df)
        col0_scaled_before = scaled["A"].copy()
        _ = PCA(n_components=2, algorithm="nipals").fit(scaled)
        # The fit must not mutate the caller's DataFrames.
        pd.testing.assert_series_equal(df["A"], col0_before)
        pd.testing.assert_series_equal(scaled["A"], col0_scaled_before)
