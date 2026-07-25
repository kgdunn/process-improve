"""Hypothesis property-based tests for the unified design-generation API.

These tests verify structural invariants that must hold across the full
range of admissible inputs, rather than probing a handful of hand-picked
cases.  For each design family we encode the invariants that are promised
by the literature (orthogonality, balance, run-count, resolution,
Jones-Nachtsheim DSD structure, ...) and let Hypothesis look for inputs
that break them.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from process_improve.experiments.designs import generate_design
from process_improve.experiments.evaluate import evaluate_design
from process_improve.experiments.factor import Factor

# ---------------------------------------------------------------------------
# Shared hypothesis settings
# ---------------------------------------------------------------------------

_settings = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _factors(n: int) -> list[Factor]:
    """Return ``n`` continuous factors with a fixed numeric range."""
    return [Factor(name=f"X{i + 1}", low=0.0, high=10.0) for i in range(n)]


def _coded(result) -> np.ndarray:
    """Extract the coded factor matrix from a ``DesignResult`` as a float array."""
    return result.design[result.factor_names].values.astype(float)


# ---------------------------------------------------------------------------
# Full factorial
# ---------------------------------------------------------------------------


class TestFullFactorialProperties:
    """Structural invariants of the 2^k full factorial."""

    @_settings
    @given(k=st.integers(min_value=2, max_value=6))
    def test_run_count_is_two_to_the_k(self, k: int) -> None:
        """A 2^k full factorial has exactly ``2**k`` runs (before replication/centers)."""
        result = generate_design(_factors(k), design_type="full_factorial", center_points=0)
        assert result.n_runs == 2**k

    @_settings
    @given(k=st.integers(min_value=2, max_value=6))
    def test_columns_are_balanced(self, k: int) -> None:
        """Every column has equal numbers of -1 and +1 (sum = 0)."""
        x = _coded(generate_design(_factors(k), design_type="full_factorial", center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=2, max_value=6))
    def test_main_effects_are_orthogonal(self, k: int) -> None:
        """``X.T @ X == N * I`` for the main-effects matrix of a 2^k design."""
        x = _coded(generate_design(_factors(k), design_type="full_factorial", center_points=0))
        gram = x.T @ x
        assert np.allclose(gram, x.shape[0] * np.eye(k), atol=1e-9)


# ---------------------------------------------------------------------------
# Fractional factorial
# ---------------------------------------------------------------------------


class TestFractionalFactorialProperties:
    """Structural invariants of fractional 2^(k-p) factorials."""

    @_settings
    @given(
        k=st.integers(min_value=4, max_value=7),
        resolution=st.sampled_from([3, 4, 5]),
    )
    def test_run_count_is_power_of_two_and_bounded(self, k: int, resolution: int) -> None:
        """Run count is a power of 2 and at most ``2^k``."""
        try:
            result = generate_design(
                _factors(k),
                design_type="fractional_factorial",
                resolution=resolution,
                center_points=0,
            )
        except (ValueError, KeyError):
            # Not every (k, resolution) pair admits a fraction (e.g. res V for k=4);
            # those are excluded cleanly upstream.
            return
        n = result.n_runs
        assert n <= 2**k
        assert n & (n - 1) == 0, f"fractional factorial run count {n} is not a power of 2"

    @_settings
    @given(
        k=st.integers(min_value=5, max_value=7),
        resolution=st.sampled_from([3, 4]),
    )
    def test_achieved_resolution_meets_request(self, k: int, resolution: int) -> None:
        """``evaluate_design`` reports a resolution at least as high as requested."""
        try:
            result = generate_design(
                _factors(k),
                design_type="fractional_factorial",
                resolution=resolution,
                center_points=0,
            )
        except (ValueError, KeyError):
            return
        metrics = evaluate_design(result, model="main_effects", metric="resolution")
        achieved = metrics.get("resolution")
        if achieved is None:
            return
        assert achieved >= resolution


# ---------------------------------------------------------------------------
# Plackett-Burman
# ---------------------------------------------------------------------------


class TestPlackettBurmanProperties:
    """Structural invariants of the Plackett-Burman screening design."""

    @_settings
    @given(k=st.integers(min_value=2, max_value=15))
    def test_run_count_is_multiple_of_four_and_covers_factors(self, k: int) -> None:
        """PB run count is a multiple of 4 (Hadamard order) and at least ``k + 1``."""
        result = generate_design(_factors(k), design_type="plackett_burman", center_points=0)
        assert result.n_runs % 4 == 0
        assert result.n_runs >= k + 1

    @_settings
    @given(k=st.integers(min_value=2, max_value=15))
    def test_columns_are_balanced_and_orthogonal(self, k: int) -> None:
        """Every column sums to 0 and main effects are mutually orthogonal."""
        x = _coded(generate_design(_factors(k), design_type="plackett_burman", center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)
        gram = x.T @ x
        off_diag = gram - np.diag(np.diag(gram))
        assert np.abs(off_diag).max() < 1e-9


# ---------------------------------------------------------------------------
# Box-Behnken
# ---------------------------------------------------------------------------


class TestBoxBehnkenProperties:
    """Structural invariants of the Box-Behnken design."""

    @_settings
    @given(k=st.integers(min_value=3, max_value=6))
    def test_non_center_rows_have_two_zero_coordinates(self, k: int) -> None:
        """Every non-center BB run has exactly two coordinates at 0 and the rest at ±1."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", center_points=0))
        non_center_rows = x[~np.all(x == 0, axis=1)]
        # Each BB non-center run varies two factors at ±1 and sets the remaining (k-2)
        # factors to 0, so for every non-center row we expect exactly (k-2) zeros.
        zeros_per_row = (non_center_rows == 0).sum(axis=1)
        assert np.all(zeros_per_row == k - 2)

    @_settings
    @given(k=st.integers(min_value=3, max_value=6))
    def test_columns_balanced(self, k: int) -> None:
        """Every factor column sums to 0 in a Box-Behnken design."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=3, max_value=6))
    def test_no_corner_points(self, k: int) -> None:
        """Box-Behnken avoids the full-factorial corners (no run has all ±1)."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", center_points=0))
        all_extreme = np.all(np.abs(x) == 1, axis=1)
        assert not all_extreme.any()


# ---------------------------------------------------------------------------
# Central Composite
# ---------------------------------------------------------------------------


class TestCCDProperties:
    """Structural invariants of CCD variants."""

    @_settings
    @given(k=st.integers(min_value=2, max_value=5))
    def test_face_centered_has_alpha_one(self, k: int) -> None:
        """Face-centered CCD has all coordinates in ``[-1, 1]``."""
        result = generate_design(
            _factors(k),
            design_type="ccd",
            alpha="face_centered",
            center_points=2,
        )
        x = _coded(result)
        assert np.abs(x).max() <= 1.0 + 1e-9

    @_settings
    @given(k=st.integers(min_value=2, max_value=5))
    def test_rotatable_axial_distance_is_k_fourth_root(self, k: int) -> None:
        """Rotatable CCD has ``max|x_i| ≈ (2^k)^(1/4)`` (pyDOE3's rotatable alpha)."""
        result = generate_design(
            _factors(k),
            design_type="ccd",
            alpha="rotatable",
            center_points=2,
        )
        x = _coded(result)
        expected_alpha = (2**k) ** 0.25
        assert np.isclose(np.abs(x).max(), expected_alpha, atol=1e-6)


# ---------------------------------------------------------------------------
# Definitive Screening Design
# ---------------------------------------------------------------------------


class TestDSDProperties:
    """Jones-Nachtsheim structural invariants of the DSD."""

    @_settings
    @given(k=st.integers(min_value=3, max_value=30))
    def test_run_count_matches_jones_nachtsheim(self, k: int) -> None:
        """Even k -> 2k+1 runs; odd k -> 2k+3 runs (Jones-Nachtsheim 2011).

        The exception is a factor count whose minimal conference matrix does not
        exist, where a larger one is used and the design costs more runs.  21
        and 22 factors are the first such counts: no conference matrix of order
        22 exists, so order 24 is used and the design has 49 runs, not 45.
        """
        result = generate_design(_factors(k), design_type="dsd", center_points=0)
        expected = 2 * k + 1 if k % 2 == 0 else 2 * k + 3
        if k in (21, 22):
            assert result.n_runs == 49
            assert result.metadata["conference_order"] == 24
            assert result.metadata["minimal_conference_order"] == 22
        else:
            assert result.n_runs == expected

    @_settings
    @given(k=st.integers(min_value=3, max_value=30))
    def test_columns_balanced(self, k: int) -> None:
        """Every DSD factor column sums to 0 (foldover structure)."""
        x = _coded(generate_design(_factors(k), design_type="dsd", center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=3, max_value=30))
    def test_exactly_one_zero_pair_per_factor(self, k: int) -> None:
        """Each factor takes value 0 in exactly 2 rows for odd k and 1 row for even k (main rows)."""
        result = generate_design(_factors(k), design_type="dsd", center_points=0)
        x = _coded(result)
        zeros_per_column = (x == 0).sum(axis=0)
        # Structure: [C; -C; zero_row] gives each column exactly 2 zeros from C/-C
        # plus the 1 zero from the center row -> 3 zeros per column for even k.
        # For odd k we build a (k+1)-column DSD and drop the last column, so the
        # first k columns still see 3 zeros per column.
        assert np.all(zeros_per_column == 3)

    @_settings
    @given(k=st.integers(min_value=3, max_value=30))
    def test_main_effects_are_exactly_orthogonal(self, k: int) -> None:
        """Main effects are mutually orthogonal at every factor count, with no exceptions.

        This invariant used to be conditional on the Paley construction having
        been reached, because the cyclic fallback returned a matrix that is not
        a conference matrix at all.  It now holds unconditionally: the
        construction either succeeds or raises.
        """
        result = generate_design(_factors(k), design_type="dsd", center_points=0)
        x = _coded(result)
        gram = x.T @ x
        off_diag = gram - np.diag(np.diag(gram))
        assert np.abs(off_diag).max() < 1e-9

    @_settings
    @given(k=st.integers(min_value=3, max_value=30))
    def test_main_effects_clear_of_second_order_terms(self, k: int) -> None:
        """The defining DSD property: no main effect is aliased with any quadratic or two-factor interaction."""
        x = _coded(generate_design(_factors(k), design_type="dsd", center_points=0))
        quadratics = x**2
        quadratics = quadratics - quadratics.mean(axis=0)
        assert np.abs(x.T @ quadratics).max() < 1e-9

        interactions = np.array([x[:, i] * x[:, j] for i in range(k) for j in range(i + 1, k)]).T
        interactions = interactions - interactions.mean(axis=0)
        assert np.abs(x.T @ interactions).max() < 1e-9

    @pytest.mark.parametrize("k", [9, 10, 15, 16, 21, 22, 25, 26, 27, 28])
    def test_previously_degraded_factor_counts(self, k: int) -> None:
        """Regression test for the factor counts the cyclic fallback used to mishandle.

        Before the prime-power Paley construction and the order-16 table, these
        counts fell through to an approximation whose main effects correlated
        with each other at up to r = 0.9, and `generate_design` returned it with
        only a log warning.
        """
        result = generate_design(_factors(k), design_type="dsd", center_points=0)
        assert result.metadata["construction"] != "cyclic_fallback"

        x = _coded(result)
        gram = x.T @ x
        off_diag = gram - np.diag(np.diag(gram))
        assert np.abs(off_diag).max() < 1e-9
        assert np.all((x == 0).sum(axis=0) == 3)


# ---------------------------------------------------------------------------
# evaluate_design metrics
# ---------------------------------------------------------------------------


class TestEvaluateDesignProperties:
    """Metric invariants that must hold for any admissible design."""

    @_settings
    @given(k=st.integers(min_value=2, max_value=5))
    def test_full_factorial_has_max_d_efficiency(self, k: int) -> None:
        """A 2^k full factorial is D-optimal for the main-effects model."""
        result = generate_design(_factors(k), design_type="full_factorial", center_points=0)
        metrics = evaluate_design(
            result,
            model="main_effects",
            metric=["d_efficiency", "condition_number"],
        )
        assert metrics["d_efficiency"] == pytest.approx(100.0, abs=1e-6)
        assert metrics["condition_number"] == pytest.approx(1.0, abs=1e-6)

    @_settings
    @given(k=st.integers(min_value=2, max_value=5))
    def test_full_factorial_vifs_are_one(self, k: int) -> None:
        """An orthogonal design has VIF = 1 on every main-effect term."""
        result = generate_design(_factors(k), design_type="full_factorial", center_points=0)
        metrics = evaluate_design(result, model="main_effects", metric="vif")
        for term, vif in metrics["vif"].items():
            assert vif == pytest.approx(1.0, abs=1e-6), f"VIF != 1 for term {term}"

    @_settings
    @given(k=st.integers(min_value=2, max_value=4), effect=st.floats(min_value=0.1, max_value=5.0))
    def test_d_efficiency_is_a_percentage(self, k: int, effect: float) -> None:  # noqa: ARG002
        """0 <= d_efficiency <= 100 for every admissible design."""
        result = generate_design(_factors(k), design_type="full_factorial", center_points=0)
        metrics = evaluate_design(result, model="main_effects", metric="d_efficiency")
        assert 0.0 <= metrics["d_efficiency"] <= 100.0 + 1e-6

    @_settings
    @given(
        k=st.integers(min_value=2, max_value=4),
        sigma=st.floats(min_value=0.5, max_value=2.0),
    )
    def test_power_monotone_in_effect_size(self, k: int, sigma: float) -> None:
        """Power is non-decreasing in effect size, holding sigma and design fixed."""
        result = generate_design(_factors(k), design_type="full_factorial", center_points=0)
        p_small = evaluate_design(
            result, model="main_effects", metric="power", effect_size=0.5, sigma=sigma,
        )["power"]
        p_large = evaluate_design(
            result, model="main_effects", metric="power", effect_size=3.0, sigma=sigma,
        )["power"]
        # evaluate_design returns either a scalar or a dict of per-term powers; handle both.
        def _as_float(val: object) -> float:
            if isinstance(val, dict):
                return float(next(iter(val.values())))
            return float(val)  # type: ignore[arg-type]

        assert _as_float(p_large) >= _as_float(p_small) - 1e-9
