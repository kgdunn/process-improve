"""A floor that decides "no spread" or "zero" is relative to the data, not to their units (#1070, #1114).

Each estimator is fitted to real data in units of 1, and to the same data in other units: 1e9 times
smaller or larger, and, where the arithmetic would otherwise overflow or underflow on squaring a
value, 1e150 times larger or 1e160 times smaller. The fits must agree: scores, predictions and SPE in
proportion to the units where the model works in the units it is given, and equal where it scales;
Hotelling's T², contribution bars and component counts equal. A floor in data units, such as
1.5e-8, read data in small units as having no spread and gave other numbers, with no warning.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import MBPCA, MBPLS, PCA, PLS, PRM, TPLS, DataFrameDict, MCUVScaler

DATA = Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
LDPE = pd.read_csv(DATA / "LDPE" / "LDPE.csv", index_col=0)
#: The fourteen process variables, centred, so a model that does not scale sees them in their units.
X = LDPE.iloc[:, :14] - LDPE.iloc[:, :14].mean()
#: The five quality variables; Mw reaches 1.7e5, which is where 1e150 overflows a square.
Y = LDPE.iloc[:, 14:]
BLOCKS = {
    "zone1": X.iloc[:, [0, 1, 2, 5, 7, 9, 11]],
    "zone2": X.iloc[:, [3, 4, 6, 8, 10, 12]],
    "pressure": X[["Press"]],
}

#: What each case returns: a named array, already brought back to units of 1.
Outputs = dict[str, np.ndarray]


def _pca(units: float) -> Outputs:
    """Fit PCA, which scales nothing: scores and SPE move with the units, T² and scaled bars do not."""
    model = PCA(n_components=4).fit(X * units)
    return {
        "scores": model.scores_.to_numpy() / units,
        "spe": model.spe_.to_numpy() / units,
        "T2": model.hotellings_t2_.to_numpy(),
        "maximum": model.score_contributions(X * units, component=2, scaling="maximum").to_numpy(),
        "within": model.score_contributions(X * units, component=2, scaling="within").to_numpy(),
    }


def _pca_scaled(units: float) -> Outputs:
    """Autoscale, then fit, as usual: nothing depends on the units."""
    model = PCA(n_components=4).fit(MCUVScaler().fit_transform(X * units))
    return {"scores": model.scores_.to_numpy(), "T2": model.hotellings_t2_.to_numpy()}


def _pca_count(units: float) -> Outputs:
    """Count the components, each fold left in the units it was given."""
    found = PCA.select_n_components(X * units, max_components=8, scale_inside_folds=False, random_state=0)
    return {"count": np.array([found.n_components]), "Q2": found.q2.to_numpy()}


def _pls(units: float, *, scale: bool) -> Outputs:
    """Fit PLS: unscaled, its scores and predictions move with the units; T² never does."""
    model = PLS(n_components=3, scale=scale).fit(X * units, Y * units)
    return {
        "scores": model.scores_.to_numpy() / (1.0 if scale else units),
        "predictions": model.predict(X * units).to_numpy() / units,
        "T2": model.hotellings_t2_.to_numpy(),
    }


def _prm(units: float, *, scale: bool) -> Outputs:
    """Fit PRM: the robust weights, and so the predictions, are the same in any units (#1114)."""
    model = PRM(n_components=2, scale=scale).fit(X * units, Y[["Conv"]] * units)
    return {"predictions": model.predict(X * units).to_numpy() / units, "weights": model.robust_weights_}


def _mbpca(units: float) -> Outputs:
    model = MBPCA(n_components=3).fit({name: block * units for name, block in BLOCKS.items()})
    bars = model.score_contributions({name: block * units for name, block in BLOCKS.items()}, scaling="within")
    return {
        "scores": model.super_scores_.to_numpy(),
        "T2": model.super_hotellings_t2_.to_numpy(),
        "within": np.hstack([bars[name].to_numpy() for name in BLOCKS]),
    }


def _mbpls(units: float) -> Outputs:
    blocks = {name: block * units for name, block in BLOCKS.items()}
    model = MBPLS(n_components=3).fit(blocks, Y * units)
    bars = model.score_contributions(blocks, scaling="maximum")
    return {
        "scores": model.super_scores_.to_numpy(),
        "predictions": model.predict(blocks).to_numpy() / units,
        "maximum": np.hstack([bars[name].to_numpy() for name in BLOCKS]),
    }


def _tpls(units: float) -> Outputs:
    """Fit TPLS to its example data; F holds the mixture fractions, which have no units."""

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(DATA / "tpls-pyphi" / name, index_col=0).astype("float64")

    groups = {f"Group {i}": i for i in range(1, 6)}
    z = {"Conditions": read("process_conditions.csv") * units}
    f = {group: read(f"formulas_Group{i}.csv") for group, i in groups.items()}
    d = {group: read(f"properties_Group{i}.csv") * units for group, i in groups.items()}
    y = {"Quality": read("quality_indicators.csv") * units}
    model = TPLS(n_components=3, d_matrix=d).fit(DataFrameDict({"Z": z, "F": f, "Y": y}))
    return {
        "scores": model.t_scores_super.to_numpy(),
        "predictions": model.hat["Quality"].to_numpy() / units,
        "T2": model.hotellings_t2.to_numpy(),
    }


#: Each case, and the units it is fitted in besides 1.
CASES: dict[str, tuple[Callable[[float], Outputs], tuple[float, ...]]] = {
    "PCA": (_pca, (1e-9, 1e9)),
    "PCA on autoscaled data": (_pca_scaled, (1e-9, 1e9, 1e150, 1e-160)),
    "PCA.select_n_components": (_pca_count, (1e-9, 1e9)),
    "PLS": (lambda units: _pls(units, scale=True), (1e-9, 1e9, 1e150, 1e-160)),
    "PLS(scale=False)": (lambda units: _pls(units, scale=False), (1e-9, 1e9)),
    "PRM": (lambda units: _prm(units, scale=True), (1e-9, 1e9, 1e150, 1e-160)),
    "PRM(scale=False)": (lambda units: _prm(units, scale=False), (1e-9, 1e9)),
    "MBPCA": (_mbpca, (1e-9, 1e9, 1e150, 1e-160)),
    "MBPLS": (_mbpls, (1e-9, 1e9, 1e150, 1e-160)),
    "TPLS": (_tpls, (1e-9, 1e9, 1e150, 1e-160)),
}


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("case", "units"),
    [
        pytest.param(case, units, id=f"{case}-{units:g}")
        for case, (_, all_units) in CASES.items()
        for units in all_units
    ],
)
def test_the_same_data_in_other_units_give_the_same_model(case: str, units: float) -> None:
    """Every output agrees with the fit in units of 1, to the accuracy of the iterations that produce it.

    The iterations stop at a relative change of about 1.5e-8, so an entry near 0 can differ by that
    much of the largest; a floor in data units changes the outputs by far more, or raises.
    """
    fit = CASES[case][0]
    expected, found = fit(1.0), fit(units)
    for name, values in expected.items():
        largest = float(np.max(np.abs(values)))
        np.testing.assert_allclose(found[name], values, rtol=1e-6, atol=1e-6 * largest, err_msg=f"{case}: {name}")
