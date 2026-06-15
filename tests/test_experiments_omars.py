"""Tests for OMARS (Orthogonal Minimally Aliased Response Surface) designs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from process_improve.experiments.designs import generate_design
from process_improve.experiments.designs_omars import dispatch_omars, is_omars, omars_properties
from process_improve.experiments.factor import Factor


def _continuous_factors(n: int, names: str | None = None) -> list[Factor]:
    """Create n continuous factors with default ranges."""
    if names and len(names) >= n:
        return [Factor(name=names[i], low=0, high=10) for i in range(n)]
    return [Factor(name=f"X{i + 1}", low=0, high=10) for i in range(n)]


def _manual_omars_check(matrix: np.ndarray, tol: float = 1e-9) -> None:
    """Independently assert the OMARS properties (does not call omars_properties)."""
    matrix = np.asarray(matrix, dtype=float)
    n_factors = matrix.shape[1]

    # Three-level and balanced.
    assert set(np.unique(matrix).tolist()) <= {-1.0, 0.0, 1.0}
    assert np.all(np.abs(matrix.sum(axis=0)) <= tol)

    # Main effects mutually orthogonal.
    gram = matrix.T @ matrix
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.abs(off_diagonal).max() <= tol

    # Main effects orthogonal to every second-order term.
    second_order = [matrix[:, i] * matrix[:, i] for i in range(n_factors)]
    second_order += [matrix[:, i] * matrix[:, j] for i, j in itertools.combinations(range(n_factors), 2)]
    for me in range(n_factors):
        for term in second_order:
            assert abs(float(matrix[:, me] @ term)) <= tol


class TestOMARSProperties:
    """Unit tests for the dependency-free OMARS verifier."""

    def test_minimal_3_factor_design_is_omars(self) -> None:
        """The minimal conference-foldover 3-factor design is a valid OMARS."""
        coded, meta = dispatch_omars(_continuous_factors(3, "ABC"))
        props = omars_properties(coded)
        assert props["is_omars"] is True
        assert props["is_three_level"] is True
        assert props["is_balanced"] is True
        assert props["main_effects_orthogonal"] is True
        assert props["main_effects_clear_of_second_order"] is True
        assert meta["omars_verified"] is True

    def test_full_factorial_is_not_omars(self) -> None:
        """A 2-level full factorial has inestimable quadratics, so it is not OMARS."""
        # 2^3 factorial: never takes the middle level -> quadratics are constant.
        ff = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=float)
        props = omars_properties(ff)
        assert props["quadratics_estimable"] is False
        assert props["is_omars"] is False
        assert is_omars(ff) is False

    def test_main_effect_aliased_with_interaction_fails(self) -> None:
        """A design whose main effect correlates with an interaction is not OMARS."""
        # x3 deliberately equals x1*x2 on the non-zero rows -> ME3 aliased with x1:x2.
        bad = np.array(
            [
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=float,
        )
        props = omars_properties(bad)
        assert props["main_effects_clear_of_second_order"] is False
        assert props["is_omars"] is False

    def test_tolerance_is_respected(self) -> None:
        """A tiny perturbation below tol still verifies; above tol does not."""
        coded, _ = dispatch_omars(_continuous_factors(4))
        perturbed = coded.copy()
        perturbed[0, 0] += 1e-6
        assert is_omars(perturbed, tol=1e-3) is True
        assert is_omars(perturbed, tol=1e-12) is False


class TestOMARSVerifierEdgeCases:
    """Edge cases that exercise the verifier's defensive branches."""

    def test_single_factor_is_trivially_omars(self) -> None:
        """A single three-level factor has no aliasing and is trivially OMARS."""
        props = omars_properties(np.array([[-1.0], [0.0], [1.0]]))
        assert props["n_factors"] == 1
        # Only one quadratic term, so there is no second-order correlation.
        assert props["max_second_order_correlation"] == 0.0
        assert props["is_omars"] is True

    def test_all_constant_second_order_terms(self) -> None:
        """An all-zero design has only constant second-order terms (no correlation)."""
        props = omars_properties(np.zeros((4, 2)))
        assert props["max_second_order_correlation"] == 0.0

    def test_zero_factor_matrix(self) -> None:
        """A matrix with no factors has no second-order terms to alias."""
        props = omars_properties(np.zeros((3, 0)))
        assert props["n_factors"] == 0
        assert props["max_main_vs_second_order_inner_product"] == 0.0

    def test_verify_false_skips_verification(self) -> None:
        """dispatch_omars(verify=False) does not record the omars_verified flag."""
        _, meta = dispatch_omars(_continuous_factors(4), verify=False)
        assert "omars_verified" not in meta
        assert meta["family"] == "conference_foldover"


class TestOMARSDispatch:
    """Tests for dispatch_omars and the generate_design integration."""

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_generated_designs_are_omars(self, k: int) -> None:
        """Designs generated for k=3..6 factors satisfy the OMARS properties."""
        coded, meta = dispatch_omars(_continuous_factors(k))
        _manual_omars_check(coded)
        assert meta["omars_verified"] is True
        assert meta["family"] == "conference_foldover"

    def test_run_counts_match_conference_foldover(self) -> None:
        """Even k -> 2k+1 runs; odd k -> 2k+3 runs (the DSD family)."""
        even, _ = dispatch_omars(_continuous_factors(4))
        odd, _ = dispatch_omars(_continuous_factors(5))
        assert even.shape == (2 * 4 + 1, 4)
        assert odd.shape == (2 * 5 + 3, 5)

    def test_requires_3_factors(self) -> None:
        """OMARS designs require at least three factors."""
        with pytest.raises(ValueError, match="at least 3"):
            dispatch_omars(_continuous_factors(2, "AB"))

    def test_generate_design_integration(self) -> None:
        """generate_design(design_type='omars') returns a verified OMARS design."""
        result = generate_design(_continuous_factors(5), design_type="omars", center_points=0)
        assert result.design_type == "omars"
        assert result.metadata["omars_verified"] is True
        x = result.design[result.factor_names].values.astype(float)
        _manual_omars_check(x)

    def test_values_in_range(self) -> None:
        """Coded OMARS values lie within [-1, +1]."""
        result = generate_design(_continuous_factors(5), design_type="omars", center_points=0)
        for col in result.factor_names:
            assert np.all(np.abs(result.design[col].values) <= 1.0 + 1e-10)
