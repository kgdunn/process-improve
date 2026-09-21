"""Partial Robust M-regression and its primitives (#191).

The tests are written around the property that justifies the method: a few bad
rows should move the fit a little, where they move an ordinary least-squares
PLS a lot. A test that only checks "PRM runs and returns coefficients" would
pass just as well against plain PLS, so each case here contrasts the two.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize

from process_improve.multivariate._robust import fair_weights, l1_median


def _cost(centre: np.ndarray, X: np.ndarray) -> float:
    """Total Euclidean distance from ``centre`` to every row: what the L1 median minimises."""
    return float(np.linalg.norm(X - centre, axis=1).sum())


class TestFairWeights:
    def test_weight_is_one_at_zero_and_a_quarter_at_the_cutoff(self) -> None:
        """The cutoff is a scale, not a threshold: it names where the weight reaches 1/4."""
        weights = fair_weights(np.array([0.0, 4.0]), cutoff=4.0)
        assert weights[0] == pytest.approx(1.0)
        assert weights[1] == pytest.approx(0.25)

    def test_never_reaches_zero(self) -> None:
        """Why Fair and not a hard cutoff: nothing is ever switched fully off.

        A weight that can hit exactly zero lets an observation leave and re-enter
        the fit between iterations, and the reweighting loop cycles instead of
        settling.
        """
        assert fair_weights(np.array([1e6]), cutoff=4.0)[0] > 0

    def test_decreasing_and_sign_blind(self) -> None:
        distances = np.array([0.0, 0.5, 1.0, 5.0, 50.0])
        weights = fair_weights(distances, cutoff=4.0)
        assert np.all(np.diff(weights) < 0)
        np.testing.assert_allclose(fair_weights(-distances, cutoff=4.0), weights)

    def test_shape_is_preserved(self) -> None:
        assert fair_weights(np.zeros((3, 2)), cutoff=4.0).shape == (3, 2)

    @pytest.mark.parametrize("cutoff", [0.0, -1.0, np.nan, np.inf])
    def test_bad_cutoff_rejected(self, cutoff: float) -> None:
        with pytest.raises(ValueError, match="cutoff must be a positive finite number"):
            fair_weights(np.array([1.0]), cutoff=cutoff)

    def test_non_finite_distances_rejected(self) -> None:
        with pytest.raises(ValueError, match="distances must be finite"):
            fair_weights(np.array([1.0, np.nan]), cutoff=4.0)


class TestL1Median:
    def test_matches_the_optimiser_it_is_a_shortcut_for(self) -> None:
        """The definition is an argmin, so check against a general-purpose argmin.

        Contaminated and clean data both, because the iteration's awkward case is
        a badly spread cloud rather than a nice one.
        """
        rng = np.random.default_rng(0)
        worst = 0.0
        for trial in range(40):
            n_samples, n_features = int(rng.integers(3, 30)), int(rng.integers(1, 5))
            X = rng.normal(size=(n_samples, n_features))
            if trial % 2:
                bad = max(1, n_samples // 4)
                X[:bad] += rng.normal(scale=50, size=(bad, n_features))
            found = _cost(l1_median(X), X)
            reference = _cost(
                minimize(
                    _cost,
                    np.median(X, axis=0),
                    args=(X,),
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
                ).x,
                X,
            )
            worst = max(worst, abs(found - reference) / max(reference, 1e-12))
        assert worst < 1e-6

    def test_centre_of_a_symmetric_cloud(self) -> None:
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(l1_median(corners), [0.5, 0.5], atol=1e-8)

    def test_one_dimensional_case_is_the_ordinary_median(self) -> None:
        values = np.array([[1.0], [2.0], [10.0]])
        np.testing.assert_allclose(l1_median(values), [np.median(values)])

    def test_iterate_landing_on_a_data_point(self) -> None:
        """The case plain Weiszfeld divides by zero on; Vardi-Zhang is why it is here.

        The origin is both the answer and one of the rows, so the very first
        iteration has a row at distance zero.
        """
        star = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        np.testing.assert_allclose(l1_median(star), [0.0, 0.0], atol=1e-8)

    def test_degenerate_inputs(self) -> None:
        np.testing.assert_allclose(l1_median(np.array([[3.0, 7.0]])), [3.0, 7.0])
        np.testing.assert_allclose(l1_median(np.tile([2.0, 5.0], (6, 1))), [2.0, 5.0])

    def test_one_wild_row_barely_moves_it(self) -> None:
        """The reason PRM measures leverage against this and not the mean."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(21, 3))
        clean = l1_median(X)
        X[0] = 1e6
        assert np.linalg.norm(l1_median(X) - clean) < 1.0
        assert np.linalg.norm(X.mean(axis=0) - clean) > 1e4

    def test_rotation_equivariant(self) -> None:
        """Unlike the coordinate-wise median, which is why the scores use this one."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 2))
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        np.testing.assert_allclose(l1_median(X @ rotation.T), rotation @ l1_median(X), atol=1e-6)

    def test_bad_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 2-dimensional"):
            l1_median(np.zeros(5))
        with pytest.raises(ValueError, match="at least one row"):
            l1_median(np.zeros((0, 3)))
        with pytest.raises(ValueError, match="must be finite"):
            l1_median(np.array([[1.0, np.nan]]))
