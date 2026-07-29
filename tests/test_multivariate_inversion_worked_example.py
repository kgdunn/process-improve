# (c) Kevin Dunn, 2010-2026. MIT License.
"""Pin the numbers quoted in the model-inversion worked example.

The cheddar-cheese example is written up in *Process Improvement using Data*,
in "Using a PLS model backwards: model inversion and the null space". Most of
the numbers it quotes are printed by code blocks the reader can run, so a
change in this library would be visible there. A handful are not: the
Hotelling's :math:`T^2` and SPE limits, the score norms either side of the
direct-inversion solution, the angle between each weight and its own loading,
the agreement between the PLS and O-PLS regression coefficients, and the way
O-PLS splits the sum of squares in ``X``.

Those quantities are quoted in the prose without appearing in any output, so
nothing else would catch a silent change to them. This module is that catch.
The tolerances are set to the precision the text actually quotes, so a failure
here means a sentence in the book needs rewriting, not merely that a digit
moved somewhere far downstream.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate.methods import OPLS, PLS

#: The book fits on cheeses 5 to 30 and holds out the first four as design targets.
FIRST_CALIBRATION_ROW = 4
N_COMPONENTS = 2
TARGET_TASTE = 20.9


@pytest.fixture
def calibration() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the 26 calibration cheeses, exactly as the worked example splits them."""
    folder = pathlib.Path(__file__).parents[1] / "src" / "process_improve" / "datasets" / "multivariate"
    path = folder / "cheddar-cheese.csv"
    if not path.exists():
        pytest.skip("cheddar-cheese.csv fixture not present")
    data = pd.read_csv(path).iloc[FIRST_CALIBRATION_ROW:]
    return data[["Acetic", "H2S", "Lactic"]], data[["Taste"]]


@pytest.fixture
def fitted(calibration: tuple[pd.DataFrame, pd.DataFrame]) -> PLS:
    """Fit the two-component PLS model the worked example uses throughout."""
    x_block, y_block = calibration
    return PLS(n_components=N_COMPONENTS).fit(x_block, y_block)


def test_diagnostic_limits(fitted: PLS) -> None:
    """The limits quoted when judging whether a proposed design is supported."""
    assert float(fitted.hotellings_t2_limit(0.95)) == pytest.approx(7.36, abs=0.005)
    assert float(fitted.hotellings_t2_limit(0.99)) == pytest.approx(12.14, abs=0.005)
    assert float(fitted.spe_limit(0.99)) == pytest.approx(1.60, abs=0.005)


def test_direct_inversion_is_the_minimum_norm_point(fitted: PLS) -> None:
    """Pythagoras: stepping along the null space adds ``s**2`` to the squared norm."""
    result = fitted.invert(y_desired=TARGET_TASTE)
    tau = result.scores.to_numpy().ravel()
    g = result.null_space_basis.to_numpy().ravel()

    assert np.linalg.norm(tau) == pytest.approx(0.281, abs=0.0005)
    for step in (-1.0, 1.0):
        assert np.linalg.norm(tau + step * g) == pytest.approx(1.039, abs=0.0005)
        # The identity the book states, rather than only the value it takes here.
        assert np.linalg.norm(tau + step * g) ** 2 == pytest.approx(np.linalg.norm(tau) ** 2 + step**2)


def test_smallest_norm_is_not_smallest_t2(fitted: PLS) -> None:
    """The two distance measures are different parabolas, least at different steps."""
    result = fitted.invert(y_desired=TARGET_TASTE)
    tau = result.scores.to_numpy().ravel()
    g = result.null_space_basis.to_numpy().ravel()
    sf = fitted.scaling_factor_for_scores_.to_numpy()

    step_of_least_t2 = -float((tau / sf**2) @ g) / float((g / sf**2) @ g)
    assert step_of_least_t2 == pytest.approx(-0.103, abs=0.0005)
    assert (((tau + step_of_least_t2 * g) / sf) ** 2).sum() == pytest.approx(0.043, abs=0.0005)
    assert ((tau / sf) ** 2).sum() == pytest.approx(0.064, abs=0.0005)


def test_weight_and_loading_angles(calibration: tuple[pd.DataFrame, pd.DataFrame], fitted: PLS) -> None:
    """Removing the orthogonal variation first brings weight and loading together."""

    def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
        cosine = first @ second / (np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    weight = fitted.x_weights_.to_numpy()[:, 0]
    loading = fitted.x_loadings_.to_numpy()[:, 0]
    assert angle_degrees(weight, loading) == pytest.approx(5.49, abs=0.005)
    # w'p = 1 for every component, which is what makes p - w a plain subtraction.
    assert float(weight @ loading) == pytest.approx(1.0)

    x_block, y_block = calibration
    opls = OPLS(n_orthogonal_components=N_COMPONENTS - 1).fit(x_block, y_block)
    predictive_weight = opls.predictive_weights_.to_numpy().ravel()
    predictive_loading = opls.predictive_loadings_.to_numpy().ravel()
    assert angle_degrees(predictive_weight, predictive_loading) == pytest.approx(0.31, abs=0.005)


def test_pls_and_opls_agree_on_the_regression_coefficients(
    calibration: tuple[pd.DataFrame, pd.DataFrame], fitted: PLS
) -> None:
    """The book quotes agreement to 1.4e-13, so anything near 1e-12 is a change."""
    x_block, y_block = calibration
    opls = OPLS(n_orthogonal_components=N_COMPONENTS - 1).fit(x_block, y_block)
    difference = np.abs(fitted.beta_coefficients_.to_numpy().ravel() - opls.beta_coefficients_.to_numpy().ravel())
    assert difference.max() < 1e-12


def test_opls_splits_the_sum_of_squares_in_x(calibration: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """The 68.7 / 18.3 / 13.0 split of |X|, which the book quotes but never prints."""
    x_block, y_block = calibration
    opls = OPLS(n_orthogonal_components=N_COMPONENTS - 1).fit(x_block, y_block)

    scaled = ((x_block - x_block.mean()) / x_block.std()).to_numpy()
    predictive = opls.predictive_scores_.to_numpy().reshape(-1, 1) @ opls.predictive_loadings_.to_numpy().reshape(1, -1)
    orthogonal = opls.orthogonal_scores_.to_numpy().reshape(-1, 1) @ opls.orthogonal_loadings_.to_numpy().reshape(1, -1)
    total = (scaled**2).sum()

    assert 100 * (predictive**2).sum() / total == pytest.approx(68.7, abs=0.05)
    assert 100 * (orthogonal**2).sum() / total == pytest.approx(18.3, abs=0.05)
    assert 100 * ((scaled - predictive - orthogonal) ** 2).sum() / total == pytest.approx(13.0, abs=0.05)


def test_one_opls_component_explains_what_two_pls_components_do(
    calibration: tuple[pd.DataFrame, pd.DataFrame], fitted: PLS
) -> None:
    """Same explained variance in the response, carried by one component instead of two."""
    x_block, y_block = calibration
    opls = OPLS(n_orthogonal_components=N_COMPONENTS - 1).fit(x_block, y_block)

    centred = (y_block - y_block.mean()).to_numpy().ravel()
    predictive_score = opls.predictive_scores_.to_numpy().ravel()
    correlation = float(np.corrcoef(predictive_score, centred)[0, 1])

    assert fitted.r2_cumulative_.iloc[0] == pytest.approx(0.642, abs=0.0005)
    assert fitted.r2_cumulative_.iloc[-1] == pytest.approx(0.672, abs=0.0005)
    # The single predictive score reaches the two-component R-squared on its own.
    assert correlation**2 == pytest.approx(float(fitted.r2_cumulative_.iloc[-1]), abs=0.0005)


def test_specification_region_box_and_its_corners(fitted: PLS) -> None:
    """Every corner of the box of three ranges fails, and for two distinct reasons."""
    taste_low, taste_high = 20.0, 30.0
    t2_limit = float(fitted.hotellings_t2_limit(0.95))

    region = [
        candidate.x_new
        for target in np.linspace(taste_low, taste_high, 11)
        for step in np.linspace(-2.5, 2.5, 50)
        if (candidate := fitted.invert(target, null_space_coordinates=[step])).hotellings_t2 <= t2_limit
    ]
    region = pd.DataFrame(region)
    assert len(region) == 415

    corners = pd.DataFrame(
        np.array(np.meshgrid(*zip(region.min(), region.max(), strict=True))).reshape(3, -1).T,
        columns=region.columns,
    )
    taste = fitted.predict(corners).to_numpy().ravel()
    t2 = fitted.diagnose(corners).hotellings_t2.to_numpy()

    in_window = (taste >= taste_low) & (taste <= taste_high)
    assert in_window.sum() == 2, "two corners predict an acceptable taste"
    assert (t2[in_window] > t2_limit).all(), "and both of those are extrapolations"
    assert not ((taste >= taste_low) & (taste <= taste_high) & (t2 <= t2_limit)).any(), "so no corner is acceptable"
