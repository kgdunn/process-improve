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
