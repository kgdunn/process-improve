"""Uniform sampling of design regions: affine detection, chain starts and mixing (review fixes)."""

from __future__ import annotations

import numpy as np

from process_improve.experiments import Constraint, DesignRegion, Factor
from process_improve.experiments._uniform_sampling import _affine_form


def _box(*names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def test_piecewise_linear_constraint_is_not_taken_as_affine() -> None:
    """``abs(a - 0.95) >= 0.01`` is linear wherever the old probes looked, but not near the face a = 1."""
    region = DesignRegion(
        _box("a", "b"),
        [Constraint(expression="abs(a - 0.95) >= 0.01", type="nonlinear"), Constraint(expression="b >= 0.99")],
    )
    assert _affine_form(region.inequalities[0], np.array([-1.0, -1.0]), np.array([1.0, 1.0])) is None
    x = region.sample(20_000, np.random.default_rng(0))
    assert region.feasible(x).all()
    # The part a >= 0.96 holds 0.04 / 1.98 of the region; it used to get no samples at all.
    assert 0.012 < (x[:, 0] >= 0.96).mean() < 0.03


def test_affine_constraints_are_still_recognised() -> None:
    region = DesignRegion(_box("a", "b"), [Constraint(expression="a + 2*b <= 1.1")])
    form = _affine_form(region.inequalities[0], np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert form is not None
    np.testing.assert_allclose(form[0], [1.0, 2.0], atol=1e-9)


def test_disconnected_region_is_shared_by_size_between_its_pieces() -> None:
    """Two separate strips: the left one holds 0.1 / 0.15 of the region, whatever the seed."""
    region = DesignRegion(
        _box("a", "b"),
        [
            Constraint(expression="(a - 0.95)*(a + 0.9) >= 0", type="nonlinear"),
            Constraint(expression="b >= 0.95"),
        ],
    )
    shares = [(region.sample(20_000, np.random.default_rng(s))[:, 0] < 0).mean() for s in range(6)]
    # The chains used to stay in the strip they started in: shares from 0.56 to 0.78.
    assert np.max(np.abs(np.array(shares) - 0.1 / 0.15)) < 0.04


def test_thin_mixture_sliver_is_sampled_along_its_length() -> None:
    """In ``x1 + 3*x2 <= 0.05`` the blend runs from pure x3 to pure x4; the chains must reach both ends."""
    factors = [Factor(name=f"x{i + 1}", type="mixture") for i in range(4)]
    region = DesignRegion(factors, [Constraint(expression="x1 + 3*x2 <= 0.05")])
    x = region.sample(20_000, np.random.default_rng(1))
    assert region.feasible(x).all()
    low, high = np.quantile(x[:, 2], [0.05, 0.95])
    # Exact uniform sampling gives 0.05 and 0.93; the isotropic chains gave 0.15 and 0.76.
    assert abs(low - 0.05) < 0.03
    assert abs(high - 0.93) < 0.03


def test_region_of_categorical_factors_only_samples_no_columns() -> None:
    region = DesignRegion([Factor(name="C", type="categorical", levels=["a", "b"])])
    assert region.sample(5, np.random.default_rng(0)).shape == (5, 0)
