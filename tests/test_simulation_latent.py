"""Tests for the known-latent-structure generator (``simulation/latent.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.multivariate import PLS, compare_cv_criteria
from process_improve.simulation import LatentStructure


def test_loadings_are_orthonormal_with_equal_row_norms() -> None:
    process = LatentStructure(x_sd=[3.0, 1.0, 0.5], y_coefficients=[1.0, 0.0, 0.0], n_features=8)
    P = process.loadings.to_numpy()
    np.testing.assert_allclose(P.T @ P, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.abs(P), 1 / np.sqrt(8))
    # Every X column has the same population variance, so autoscaling is a uniform rescale.
    column_variance = np.diag(P @ np.diag(np.square(process.x_sd)) @ P.T)
    np.testing.assert_allclose(column_variance, column_variance[0])
    assert (process.n_latent, process.n_targets, process.n_relevant) == (3, 1, 1)


def test_sample_shapes_names_and_reproducibility() -> None:
    process = LatentStructure(x_sd=[2.0, 1.0], y_coefficients=[[1.0, 0.0], [0.0, 1.0]])
    first = process.sample(25, random_state=3)
    assert first.X.shape == (25, 16)
    assert first.Y.shape == (25, 2)
    assert first.scores.shape == (25, 2)
    assert list(first.Y.columns) == ["y0", "y1"]
    assert list(first.scores.columns) == ["t1", "t2"]
    second = process.sample(25, random_state=np.random.default_rng(3))
    np.testing.assert_array_equal(first.X.to_numpy(), second.X.to_numpy())


def test_population_truth_matches_a_large_sample() -> None:
    """The closed-form best linear predictor is what least squares converges to."""
    process = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0])
    big = process.sample(100_000, random_state=0)
    beta = np.linalg.lstsq(big.X.to_numpy(), big.Y.to_numpy(), rcond=None)[0]
    np.testing.assert_allclose(beta, process.population_coefficients().to_numpy(), atol=0.02)
    fresh = process.sample(100_000, random_state=1)
    residual = fresh.Y.to_numpy() - fresh.X.to_numpy() @ beta
    r2 = 1 - residual.var() / fresh.Y.to_numpy().var()
    assert r2 == pytest.approx(process.population_r2().iloc[0], abs=0.005)


def test_missing_cells_are_added_after_the_values() -> None:
    process = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0], n_features=4)
    complete = process.sample(200, random_state=0)
    gappy = process.sample(200, missing_fraction=0.6, random_state=0)
    observed = gappy.X.notna().to_numpy()
    np.testing.assert_array_equal(gappy.X.to_numpy()[observed], complete.X.to_numpy()[observed])
    assert observed.any(axis=1).all()  # no row loses every X cell
    assert 0.5 < gappy.Y.isna().to_numpy().mean() < 0.7


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"x_sd": [], "y_coefficients": []}, "x_sd"),
        ({"x_sd": [1.0, -1.0], "y_coefficients": [1.0, 1.0]}, "x_sd"),
        ({"x_sd": [1.0, 1.0], "y_coefficients": [1.0]}, "one row per latent variable"),
        ({"x_sd": [1.0], "y_coefficients": [1.0], "n_features": 12}, "power of two"),
        ({"x_sd": [1.0, 1.0], "y_coefficients": [1.0, 1.0], "n_features": 2}, "power of two"),
        ({"x_sd": [1.0], "y_coefficients": [1.0], "noise_sd": -0.1}, "noise_sd"),
    ],
)
def test_invalid_structures_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LatentStructure(**kwargs)


def test_invalid_sample_arguments_raise() -> None:
    process = LatentStructure(x_sd=[1.0], y_coefficients=[1.0])
    with pytest.raises(ValueError, match="n_samples"):
        process.sample(0)
    with pytest.raises(ValueError, match="missing_fraction"):
        process.sample(10, missing_fraction=1.0)


def test_y_orthogonal_variation_tilts_the_first_pls_weight() -> None:
    """A stronger latent variable that Y does not see pulls the sample w1 away from the relevant loading."""
    process = LatentStructure(x_sd=[2.0, 3.0, 1.5], y_coefficients=[1.0, 0.0, 0.0])
    train = process.sample(60, random_state=0)
    w1 = PLS(n_components=1).fit(train.X, train.Y).x_weights_.to_numpy()[:, 0]
    tilt = np.degrees(np.arccos(abs(w1 @ process.loadings["t1"].to_numpy())))
    assert tilt > 3
    assert process.n_relevant == 1


def test_compare_cv_criteria_recovers_the_simulated_count() -> None:
    process = LatentStructure(x_sd=[3.0, 1.0], y_coefficients=[1.0, 1.0])
    train = process.sample(60, random_state=0)
    result = compare_cv_criteria(
        train.X, train.Y, max_components=4, random_state=0, n_permutations=199, n_cv_permutations=99
    )
    assert set(result.recommendations["n_components"]) == {process.n_relevant}
    # Two components reach what the best linear predictor achieves on new rows; one does not.
    ceiling = process.population_r2().iloc[0]
    assert result.table.loc[2, "q2y"] == pytest.approx(ceiling, abs=0.01)
    assert result.table.loc[1, "q2y"] < ceiling - 0.05
