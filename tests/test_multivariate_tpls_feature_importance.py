"""Coverage for TPLS ``feature_importance`` / ``vip()`` (issue #192).

The TPLS VIP-based feature importance for the D-block (material properties) and
F-block (formulation variables) previously had no tests pinning its behaviour.
These tests lock in the current structure and invariants so that any future
re-derivation (e.g. the deflated-matrix variant noted in #192) is a deliberate,
reviewed change rather than a silent one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import TPLS, DataFrameDict


@pytest.fixture
def tpls_pyphi_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Load the pyphi TPLS example dataset (Z, D, F, Y blocks)."""
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate" / "tpls-pyphi"
    properties = {
        f"Group {i}": pd.read_csv(folder / f"properties_Group{i}.csv", sep=",", index_col=0, header=0).astype("float64")
        for i in range(1, 6)
    }
    formulas = {
        f"Group {i}": pd.read_csv(folder / f"formulas_Group{i}.csv", sep=",", index_col=0, header=0).astype("float64")
        for i in range(1, 6)
    }
    process_conditions = {
        "Conditions": pd.read_csv(folder / "process_conditions.csv", sep=",", index_col=0, header=0).astype("float64"),
    }
    quality_indicators = {
        "Quality": pd.read_csv(folder / "quality_indicators.csv", sep=",", index_col=0, header=0).astype("float64"),
    }
    return {"Z": process_conditions, "D": properties, "F": formulas, "Y": quality_indicators}


@pytest.fixture
def fitted_tpls(tpls_pyphi_data: dict) -> TPLS:
    d_matrix = tpls_pyphi_data.pop("D")
    model = TPLS(n_components=2, d_matrix=d_matrix)
    model.fit(DataFrameDict(tpls_pyphi_data))
    return model


class TestTPLSFeatureImportance:
    def test_feature_importance_structure(self, fitted_tpls: TPLS) -> None:
        fi = fitted_tpls.feature_importance
        assert set(fi.keys()) == {"D", "F", "Z", "Y"}
        # D and F are populated, one Series per group.
        assert set(fi["D"].keys()) == {f"Group {i}" for i in range(1, 6)}
        assert set(fi["F"].keys()) == {f"Group {i}" for i in range(1, 6)}

    def test_vip_values_are_finite_nonnegative(self, fitted_tpls: TPLS) -> None:
        for block in ("D", "F"):
            for group, series in fitted_tpls.feature_importance[block].items():
                assert isinstance(series, pd.Series), f"{block}/{group}"
                values = series.to_numpy()
                assert np.isfinite(values).all(), f"{block}/{group} has non-finite VIP"
                assert (values >= 0).all(), f"{block}/{group} has negative VIP"
        # There is real signal: not every VIP is zero.
        all_d = np.concatenate([s.to_numpy() for s in fitted_tpls.feature_importance["D"].values()])
        assert all_d.max() > 0

    def test_vip_series_indexed_by_feature_names(self, fitted_tpls: TPLS) -> None:
        for group, series in fitted_tpls.feature_importance["D"].items():
            assert list(series.index) == list(fitted_tpls.property_names[group])
        for group, series in fitted_tpls.feature_importance["F"].items():
            assert list(series.index) == list(fitted_tpls.material_names[group])

    def test_vip_method_matches_attribute(self, fitted_tpls: TPLS) -> None:
        assert fitted_tpls.vip() is fitted_tpls.feature_importance
        assert fitted_tpls.vip("D") is fitted_tpls.feature_importance["D"]
        assert fitted_tpls.vip("F") is fitted_tpls.feature_importance["F"]

    def test_vip_rejects_unknown_block(self, fitted_tpls: TPLS) -> None:
        with pytest.raises(ValueError, match="block must be"):
            fitted_tpls.vip("Z")

    def test_vip_before_fit_raises(self, tpls_pyphi_data: dict) -> None:
        model = TPLS(n_components=2, d_matrix=tpls_pyphi_data["D"])
        with pytest.raises((RuntimeError, ValueError)):
            model.vip()

    def test_vip_deflated_method_structure(self, fitted_tpls: TPLS) -> None:
        """The opt-in deflated direct-weights variant has the same shape and stays finite/non-negative."""
        deflated = fitted_tpls.vip(method="deflated")
        default = fitted_tpls.feature_importance
        for block in ("D", "F"):
            assert set(deflated[block].keys()) == set(default[block].keys())
            for group, series in deflated[block].items():
                assert isinstance(series, pd.Series)
                assert list(series.index) == list(default[block][group].index)
                values = series.to_numpy()
                assert np.isfinite(values).all()
                assert (values >= 0).all()

    def test_vip_deflated_d_block_matches_default(self, fitted_tpls: TPLS) -> None:
        """For the D-block the rotation S(V^T S)^-1 is ~identity (V^T S ~= I by construction), so the
        deflated importance coincides with the standard VIP. This is a useful sanity result: the TODO's
        deflated-matrix concern is effectively a no-op for the D-block.
        """
        default_d = np.concatenate([s.to_numpy() for s in fitted_tpls.vip("D").values()])
        deflated_d = np.concatenate([s.to_numpy() for s in fitted_tpls.vip("D", method="deflated").values()])
        assert np.allclose(default_d, deflated_d, atol=1e-2)

    def test_vip_deflated_f_block_differs(self, fitted_tpls: TPLS) -> None:
        """For the F-block P^T P is not identity, so P(P^T P)^-1 genuinely rescales the weights and the
        deflated importance differs from the standard VIP.
        """
        default_f = np.concatenate([s.to_numpy() for s in fitted_tpls.vip("F").values()])
        deflated_f = np.concatenate([s.to_numpy() for s in fitted_tpls.vip("F", method="deflated").values()])
        assert not np.allclose(default_f, deflated_f)

    def test_vip_deflated_does_not_mutate_stored_importance(self, fitted_tpls: TPLS) -> None:
        """Requesting the deflated variant must not overwrite the stored (default) feature_importance."""
        before = fitted_tpls.feature_importance["D"]["Group 1"].copy()
        _ = fitted_tpls.vip(method="deflated")
        pd.testing.assert_series_equal(fitted_tpls.feature_importance["D"]["Group 1"], before)
        # And the default accessor still returns the stored VIP.
        assert fitted_tpls.vip("D") is fitted_tpls.feature_importance["D"]

    def test_vip_rejects_unknown_method(self, fitted_tpls: TPLS) -> None:
        with pytest.raises(ValueError, match="method must be"):
            fitted_tpls.vip(method="bogus")


class TestTPLSDBlockScaling:
    """The D-block is block-scaled by 1/sqrt(P_i * M_i) per Garcia-Munoz (2014), section 2.1."""

    def test_block_factor_is_sqrt_lots_times_properties(self, tpls_pyphi_data: dict) -> None:
        d_matrix = tpls_pyphi_data.pop("D")
        model = TPLS(n_components=2, d_matrix=d_matrix)
        model.fit(DataFrameDict(tpls_pyphi_data))
        for group, df_d in d_matrix.items():
            n_lots, n_properties = df_d.shape
            expected = np.sqrt(n_lots * n_properties)
            assert model.preproc_["D"][group]["block"][0] == pytest.approx(expected)

    def test_scaling_equalises_block_trace(self, tpls_pyphi_data: dict) -> None:
        """After centring, auto-scaling and block-scaling, trace(X_i^T X_i) ~= 1 for every D-block.

        This is the paper's stated goal (removing bias toward blocks with more lots or properties) and is
        an independent anchor for the scaling: the previous 1/sqrt(M_i) factor left trace = P_i - 1, i.e.
        wildly unequal across blocks (e.g. 161 vs 8 on this dataset).
        """
        d_matrix = tpls_pyphi_data["D"]
        for df_d in d_matrix.values():
            n_lots, n_properties = df_d.shape
            autoscaled = (df_d - df_d.mean()) / df_d.std(ddof=1)
            block_scaled = autoscaled / np.sqrt(n_lots * n_properties)
            trace = np.nansum(block_scaled.to_numpy() ** 2)
            # Exactly (P-1)/P for complete data (the ddof=1 vs ddof=0 gap); slightly less with missing data.
            assert trace == pytest.approx(1.0, abs=0.12)
