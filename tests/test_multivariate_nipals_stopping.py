"""When the NIPALS loops of PCA, PLS, MBPLS and MBPCA stop, and where their warnings point.

- No variance left is judged against the block's own size, so the same data in other units
  fit the same components (#1070).
- A component that converges on its last allowed pass has converged: only one whose last
  pass missed the tolerance is reported as not converged (#1080).
- A warning about non-convergence names the caller's line, not a line in the library or in
  scikit-learn's ``_fit_context`` wrapper (#1079).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import (
    MBPCA,
    MBPLS,
    PCA,
    PLS,
    MCUVScaler,
    NotEnoughVarianceError,
    SpecificationWarning,
)

_RNG = np.random.default_rng(0)
#: Four autoscaled columns, and two responses, so that every loop takes several passes per component.
X = MCUVScaler().fit_transform(pd.DataFrame(_RNG.normal(size=(30, 4)), columns=list("abcd")))
_SIGNAL = X.to_numpy() @ np.array([[1.0, 0.0], [-0.5, 0.4], [0.3, 0.2], [0.0, 1.0]])
Y = MCUVScaler().fit_transform(pd.DataFrame(_SIGNAL + 0.5 * _RNG.normal(size=(30, 2)), columns=["y1", "y2"]))
BLOCKS = {"b1": X[["a", "b"]], "b2": X[["c", "d"]]}

#: One component of each estimator, with the iteration cap given. Each fit is called from this file.
FITS: dict[str, Callable[[int], object]] = {
    "PCA": lambda max_iter: PCA(n_components=1, algorithm="nipals", max_iter=max_iter).fit(X),
    "PLS": lambda max_iter: PLS(n_components=1, max_iter=max_iter).fit(X, Y),
    "MBPLS": lambda max_iter: MBPLS(n_components=1, algorithm="nipals", max_iter=max_iter).fit(BLOCKS, Y),
    "MBPCA": lambda max_iter: MBPCA(n_components=1, algorithm="nipals", max_iter=max_iter).fit(BLOCKS),
}


def _non_convergence(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if re.search("without converging|did not converge", str(w.message))]


@pytest.mark.parametrize("estimator", list(FITS))
def test_a_component_that_converges_on_its_last_allowed_pass_has_converged(estimator: str) -> None:
    """``max_iter`` set to the passes the component needs gives no warning; one pass fewer still warns (#1080)."""
    fit = FITS[estimator]
    needed = int(fit(500).fitting_info_["iterations"][0])  # type: ignore[attr-defined]
    assert needed >= 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exact = fit(needed)
    assert _non_convergence(caught) == []
    assert np.all(exact.fitting_info_.get("converged", True))  # type: ignore[attr-defined]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        short = fit(needed - 1)
    assert len(_non_convergence(caught)) == 1
    assert not np.any(short.fitting_info_.get("converged", False))  # type: ignore[attr-defined]


@pytest.mark.parametrize("estimator", list(FITS))
def test_the_non_convergence_warning_names_the_callers_line(estimator: str) -> None:
    """The warning points at the line in this file that called ``fit`` (#1079)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FITS[estimator](1)
    found = _non_convergence(caught)
    assert len(found) == 1
    assert found[0].filename == __file__
    assert found[0].category is SpecificationWarning


def test_the_imputation_warning_of_pls_names_the_callers_line() -> None:
    """The imputed fit's warning that its cells were still moving points here too."""
    gappy = X.copy()
    gappy.iloc[::4, 0] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PLS(n_components=1, max_iter=1, missing_data_settings={"md_method": "tsr"}).fit(gappy, Y)
    moving = [w for w in caught if "still moving" in str(w.message)]
    assert len(moving) == 1
    assert moving[0].filename == __file__


@pytest.mark.parametrize("units", [1e-9, 1e-5, 1e5])
def test_pca_fits_the_same_components_in_any_units(units: float) -> None:
    """Scores scale with the data, loadings do not: the block is judged against its own size (#1070)."""
    unit = PCA(n_components=3, algorithm="nipals").fit(X)
    scaled = PCA(n_components=3, algorithm="nipals").fit(X * units)
    np.testing.assert_allclose(scaled.scores_, unit.scores_ * units, rtol=1e-8)
    np.testing.assert_allclose(scaled.loadings_, unit.loadings_, rtol=1e-8)


@pytest.mark.parametrize(("x_units", "y_units"), [(1e-5, 1e-5), (1e-9, 1.0), (1.0, 1e-9), (1e5, 1e5)])
def test_pls_without_scaling_fits_the_same_components_in_any_units(x_units: float, y_units: float) -> None:
    """``scale=False`` on X and Y in other units: scores and predictions in proportion (#1070)."""
    unit = PLS(n_components=2, scale=False).fit(X, Y)
    scaled = PLS(n_components=2, scale=False).fit(X * x_units, Y * y_units)
    np.testing.assert_allclose(scaled.scores_, unit.scores_ * x_units, rtol=1e-8)
    np.testing.assert_allclose(scaled.predictions_, unit.predictions_ * y_units, rtol=1e-8)


@pytest.mark.parametrize("units", [1e-9, 1.0, 1e9])
def test_a_block_with_nothing_left_is_refused_in_any_units(units: float) -> None:
    """Two proportional columns hold one component, and a block of zeros holds none, whatever the units."""
    one = X[["a"]].assign(twice=2 * X["a"]) * units
    for model, data, block in (
        (PCA(n_components=2, algorithm="nipals"), (one,), ""),
        (PLS(n_components=2, scale=False), (one, Y), " for X"),
    ):
        message = f"There is no variance left in the data array{block}: cannot compute any more components beyond "
        with pytest.raises(NotEnoughVarianceError, match="^" + re.escape(message + "component 1.") + "$"):
            model.fit(*data)  # type: ignore[attr-defined]
    zeros = X * 0.0
    with pytest.raises(NotEnoughVarianceError, match=r"beyond component 0\.$"):
        PCA(n_components=1, algorithm="nipals").fit(zeros)
    for x_block, y_block, label in ((zeros, Y * units, "X"), (X * units, Y * 0.0, "Y")):
        with pytest.raises(NotEnoughVarianceError, match=rf"for {label}: .* beyond component 0\.$"):
            PLS(n_components=1, scale=False).fit(x_block, y_block)
