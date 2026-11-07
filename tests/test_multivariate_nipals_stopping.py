"""When the NIPALS loops of PCA, PLS, MBPLS, MBPCA and TPLS stop, and where their warnings point.

- No variance left is judged against the block's own size, so the same data in other units
  fit the same components (#1070).
- A component that converges on its last allowed pass has converged: only one whose last
  pass missed the tolerance is reported as not converged (#1080).
- A warning about non-convergence names the caller's line, not a line in the library or in
  scikit-learn's ``_fit_context`` wrapper (#1079).
- A TPLS component that never converges is reported as the others are, and not left to the
  state its last pass happened to stop in (#1163).
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
    TPLS,
    DataFrameDict,
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
#: The same rows as a T-shaped data set: blends of six materials of one group, with their properties, and Z.
PROPERTIES = pd.DataFrame(_RNG.normal(size=(6, 3)), index=[f"m{i}" for i in range(6)], columns=["p1", "p2", "p3"])
T_SHAPED = DataFrameDict(
    {
        "F": {"g": pd.DataFrame(_RNG.dirichlet(np.ones(6), size=30), columns=PROPERTIES.index)},
        "Z": {"Z": X[["c", "d"]]},
        "Y": {"Y": Y},
    }
)

#: One component of each estimator, with the iteration cap given. Each fit is called from this file.
FITS: dict[str, Callable[[int], object]] = {
    "PCA": lambda max_iter: PCA(n_components=1, algorithm="nipals", max_iter=max_iter).fit(X),
    "PLS": lambda max_iter: PLS(n_components=1, max_iter=max_iter).fit(X, Y),
    "MBPLS": lambda max_iter: MBPLS(n_components=1, algorithm="nipals", max_iter=max_iter).fit(BLOCKS, Y),
    "MBPCA": lambda max_iter: MBPCA(n_components=1, algorithm="nipals", max_iter=max_iter).fit(BLOCKS),
    "TPLS": lambda max_iter: TPLS(n_components=1, d_matrix={"g": PROPERTIES}, max_iter=max_iter).fit(T_SHAPED),
}


def _fit_info(model: object) -> dict:
    """Return the per-component record of a fit: ``fitting_info_``, which TPLS calls ``fitting_statistics``."""
    return model.fitting_statistics if isinstance(model, TPLS) else model.fitting_info_  # type: ignore[attr-defined]


def _non_convergence(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if re.search("without converging|did not converge", str(w.message))]


@pytest.mark.parametrize("estimator", list(FITS))
def test_a_component_that_converges_on_its_last_allowed_pass_has_converged(estimator: str) -> None:
    """``max_iter`` set to the passes the component needs gives no warning; one pass fewer still warns (#1080)."""
    fit = FITS[estimator]
    needed = int(_fit_info(fit(500))["iterations"][0])
    assert needed >= 2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exact = fit(needed)
    assert _non_convergence(caught) == []
    assert np.all(_fit_info(exact).get("converged", True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        short = fit(needed - 1)
    assert len(_non_convergence(caught)) == 1
    assert not np.any(_fit_info(short).get("converged", False))


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


def _two_groups(seed: int) -> tuple[dict[str, pd.DataFrame], DataFrameDict]:
    """Thirty blends of two groups of materials, A (two latent traits) and B (one), with Z and Y."""
    rng = np.random.default_rng(seed)
    traits = {"A": rng.standard_normal((8, 2)), "B": rng.standard_normal((4, 1))}
    d_matrix = {
        group: pd.DataFrame(
            traits[group] @ rng.standard_normal((traits[group].shape[1], n_properties))
            + 0.3 * rng.standard_normal((len(traits[group]), n_properties)),
            index=[f"{group}{i}" for i in range(len(traits[group]))],
        )
        for group, n_properties in (("A", 5), ("B", 3))
    }
    f_a, f_b = rng.dirichlet(np.ones(8), size=30), np.eye(4)[rng.integers(0, 4, 30)]
    z = rng.standard_normal((30, 2))
    y = (
        f_a @ traits["A"] @ rng.standard_normal((2, 3))
        + f_b @ traits["B"] @ rng.standard_normal((1, 3))
        + z @ rng.standard_normal((2, 3))
        + 0.3 * rng.standard_normal((30, 3))
    )
    f = {"A": pd.DataFrame(f_a, columns=d_matrix["A"].index), "B": pd.DataFrame(f_b, columns=d_matrix["B"].index)}
    return d_matrix, DataFrameDict({"F": f, "Z": {"Z": pd.DataFrame(z)}, "Y": {"Y": pd.DataFrame(y)}})


def test_a_tpls_component_that_cycles_for_ever_is_reported() -> None:
    """The TPLS loop can settle into two states that it alternates between for ever (#1163).

    It is not a power iteration, as the NIPALS loop of PLS is, and nothing makes it converge. Here
    the first component cycles, before any block is deflated. The model is the state of the pass
    the loop stopped on, so one pass more gives other scores, and two passes more the same ones:
    the fit warns, and records the component as not converged.
    """
    d_matrix, blocks = _two_groups(seed=110)
    message = r"^TPLS NIPALS did not converge within max_iter=200 for component\(s\) \[1\]; results for those"
    with pytest.warns(SpecificationWarning, match=message):
        stopped = TPLS(n_components=1, d_matrix=d_matrix, max_iter=200).fit(blocks)
    assert stopped.fitting_statistics["converged"] == [False]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SpecificationWarning)
        one_more, two_more = (TPLS(1, d_matrix=d_matrix, max_iter=n).fit(blocks) for n in (201, 202))
    assert np.max(np.abs(one_more.t_scores_super - stopped.t_scores_super).to_numpy()) > 0.01
    np.testing.assert_allclose(two_more.t_scores_super, stopped.t_scores_super, rtol=0, atol=1e-10)


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
    # An X with no variance at all is refused before the first component, in the family's words (#1144)
    with pytest.raises(NotEnoughVarianceError, match=r"^There is no variance in X: every column is constant"):
        PCA(n_components=1, algorithm="nipals").fit(zeros)
    with pytest.raises(NotEnoughVarianceError, match=r"^There is no variance in X: every column is constant"):
        PLS(n_components=1, scale=False).fit(zeros, Y * units)
    # So is a Y of zeros, unscaled, in the words of the family's constant Y (#1149)
    with pytest.raises(NotEnoughVarianceError, match=r"^There is no variance in Y: every response is constant"):
        PLS(n_components=1, scale=False).fit(X * units, Y * 0.0)
