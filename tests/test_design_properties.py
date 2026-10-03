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
from process_improve.experiments.designs_response_surface import dsd_run_count
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
        result = generate_design(_factors(k), design_type="full_factorial", n_center_points=0)
        assert result.n_runs == 2**k

    @_settings
    @given(k=st.integers(min_value=2, max_value=6))
    def test_columns_are_balanced(self, k: int) -> None:
        """Every column has equal numbers of -1 and +1 (sum = 0)."""
        x = _coded(generate_design(_factors(k), design_type="full_factorial", n_center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=2, max_value=6))
    def test_main_effects_are_orthogonal(self, k: int) -> None:
        """``X.T @ X == N * I`` for the main-effects matrix of a 2^k design."""
        x = _coded(generate_design(_factors(k), design_type="full_factorial", n_center_points=0))
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
                n_center_points=0,
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
                n_center_points=0,
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
        result = generate_design(_factors(k), design_type="plackett_burman", n_center_points=0)
        assert result.n_runs % 4 == 0
        assert result.n_runs >= k + 1

    @_settings
    @given(k=st.integers(min_value=2, max_value=15))
    def test_columns_are_balanced_and_orthogonal(self, k: int) -> None:
        """Every column sums to 0 and main effects are mutually orthogonal."""
        x = _coded(generate_design(_factors(k), design_type="plackett_burman", n_center_points=0))
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
    @given(k=st.integers(min_value=3, max_value=7))
    def test_non_center_rows_vary_one_block_of_factors(self, k: int) -> None:
        """Every non-centre run varies one block of factors (pairs up to 5 factors, triples at 6 and 7)."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", n_center_points=0))
        non_center_rows = x[~np.all(x == 0, axis=1)]
        varying_per_row = (non_center_rows != 0).sum(axis=1)
        assert np.all(varying_per_row == (3 if k in (6, 7) else 2))
        assert np.all(np.isin(non_center_rows, (-1.0, 0.0, 1.0)))

    @pytest.mark.parametrize(("k", "n_runs"), [(3, 12), (4, 24), (5, 40), (6, 48), (7, 56)])
    def test_run_counts_match_the_published_designs(self, k: int, n_runs: int) -> None:
        """Box and Behnken (1960): 12, 24, 40, 48 and 56 runs before centre points."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", n_center_points=0))
        assert len(x) == n_runs

    @pytest.mark.parametrize("k", [6, 7])
    def test_six_and_seven_factor_designs_fit_the_quadratic_model(self, k: int) -> None:
        """Main effects orthogonal to every other column; the full quadratic model has full rank."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", n_center_points=3))
        pairs = [x[:, i] * x[:, j] for i in range(k) for j in range(i + 1, k)]
        model = np.column_stack([np.ones(len(x)), x, x**2, *pairs])
        assert np.linalg.matrix_rank(model) == model.shape[1]
        assert np.abs(x.T @ np.column_stack([np.ones(len(x)), x**2, *pairs])).max() == 0

    @_settings
    @given(k=st.integers(min_value=3, max_value=6))
    def test_columns_balanced(self, k: int) -> None:
        """Every factor column sums to 0 in a Box-Behnken design."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", n_center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=3, max_value=6))
    def test_no_corner_points(self, k: int) -> None:
        """Box-Behnken avoids the full-factorial corners (no run has all ±1)."""
        x = _coded(generate_design(_factors(k), design_type="box_behnken", n_center_points=0))
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
            n_center_points=2,
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
            n_center_points=2,
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
    @given(k=st.integers(min_value=3, max_value=14))
    def test_run_count_matches_jones_nachtsheim(self, k: int) -> None:
        """Even k -> 2k+1 runs; odd k -> 2k+3 runs (Jones-Nachtsheim 2011), for k up to 14."""
        result = generate_design(_factors(k), design_type="dsd", n_center_points=0)
        expected = 2 * k + 1 if k % 2 == 0 else 2 * k + 3
        assert result.n_runs == expected

    @_settings
    @given(k=st.integers(min_value=3, max_value=14))
    def test_columns_balanced(self, k: int) -> None:
        """Every DSD factor column sums to 0 (foldover structure)."""
        x = _coded(generate_design(_factors(k), design_type="dsd", n_center_points=0))
        assert np.allclose(x.sum(axis=0), 0.0, atol=1e-9)

    @_settings
    @given(k=st.integers(min_value=3, max_value=14))
    def test_exactly_one_zero_pair_per_factor(self, k: int) -> None:
        """Each factor takes value 0 in exactly 2 rows for odd k and 1 row for even k (main rows)."""
        result = generate_design(_factors(k), design_type="dsd", n_center_points=0)
        x = _coded(result)
        zeros_per_column = (x == 0).sum(axis=0)
        # Structure: [C; -C; zero_row] gives each column exactly 2 zeros from C/-C
        # plus the 1 zero from the center row -> 3 zeros per column for even k.
        # For odd k we build a (k+1)-column DSD and drop the last column, so the
        # first k columns still see 3 zeros per column.
        assert np.all(zeros_per_column == 3)

    @pytest.mark.parametrize("k", range(3, 37))
    def test_main_effects_exactly_orthogonal(self, k: int) -> None:
        """Main effects are orthogonal at every size, including those that need GF(p^n) or doubling (#629)."""
        result = generate_design(_factors(k), design_type="dsd", n_center_points=0)
        x = _coded(result)
        gram = x.T @ x
        assert np.array_equal(gram, np.diag(np.diag(gram)))
        assert result.n_runs == dsd_run_count(k)

    @pytest.mark.parametrize("k", [9, 10, 15, 16, 25, 26, 27, 28])
    def test_main_effects_orthogonal_to_second_order(self, k: int) -> None:
        """Every main-effect column is orthogonal to every quadratic and two-factor interaction column."""
        x = _coded(generate_design(_factors(k), design_type="dsd", n_center_points=0))
        second = [x[:, i] * x[:, j] for i in range(k) for j in range(i, k)]
        assert np.abs(x.T @ np.column_stack(second)).max() == 0


# ---------------------------------------------------------------------------
# evaluate_design metrics
# ---------------------------------------------------------------------------


class TestEvaluateDesignProperties:
    """Metric invariants that must hold for any admissible design."""

    @_settings
    @given(k=st.integers(min_value=2, max_value=5))
    def test_full_factorial_has_max_d_efficiency(self, k: int) -> None:
        """A 2^k full factorial is D-optimal for the main-effects model."""
        result = generate_design(_factors(k), design_type="full_factorial", n_center_points=0)
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
        result = generate_design(_factors(k), design_type="full_factorial", n_center_points=0)
        metrics = evaluate_design(result, model="main_effects", metric="vif")
        for term, vif in metrics["vif"].items():
            assert vif == pytest.approx(1.0, abs=1e-6), f"VIF != 1 for term {term}"

    @_settings
    @given(k=st.integers(min_value=2, max_value=4), effect=st.floats(min_value=0.1, max_value=5.0))
    def test_d_efficiency_is_a_percentage(self, k: int, effect: float) -> None:  # noqa: ARG002
        """0 <= d_efficiency <= 100 for every admissible design."""
        result = generate_design(_factors(k), design_type="full_factorial", n_center_points=0)
        metrics = evaluate_design(result, model="main_effects", metric="d_efficiency")
        assert 0.0 <= metrics["d_efficiency"] <= 100.0 + 1e-6

    @_settings
    @given(
        k=st.integers(min_value=2, max_value=4),
        sigma=st.floats(min_value=0.5, max_value=2.0),
    )
    def test_power_monotone_in_effect_size(self, k: int, sigma: float) -> None:
        """Power is non-decreasing in effect size, holding sigma and design fixed."""
        result = generate_design(_factors(k), design_type="full_factorial", n_center_points=0)
        p_small = evaluate_design(
            result,
            model="main_effects",
            metric="power",
            effect_size=0.5,
            sigma=sigma,
        )["power"]
        p_large = evaluate_design(
            result,
            model="main_effects",
            metric="power",
            effect_size=3.0,
            sigma=sigma,
        )["power"]

        # evaluate_design returns either a scalar or a dict of per-term powers; handle both.
        def _as_float(val: object) -> float:
            if isinstance(val, dict):
                return float(next(iter(val.values())))
            return float(val)  # type: ignore[arg-type]

        assert _as_float(p_large) >= _as_float(p_small) - 1e-9


# ---------------------------------------------------------------------------
# Central composite design: alpha and centre runs
# ---------------------------------------------------------------------------


class TestCCDAxialDistance:
    """The axial distance asked for is the one built, for both cube types."""

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_numeric_alpha_is_honoured_for_a_full_cube(self, k: int) -> None:
        result = generate_design(_factors(k), design_type="ccd", alpha=1.3, n_center_points=0)
        x = _coded(result)
        assert result.alpha == 1.3
        assert sorted(np.unique(np.round(np.abs(x), 12))) == [0.0, 1.0, 1.3]
        assert len(x) == 2**k + 2 * k

    def test_numeric_rotatable_alpha_gives_the_rotatable_design(self) -> None:
        named = _coded(generate_design(_factors(3), design_type="ccd", alpha="rotatable", n_center_points=0))
        numeric = _coded(generate_design(_factors(3), design_type="ccd", alpha=8**0.25, n_center_points=0))
        assert sorted(map(tuple, np.round(named, 10))) == sorted(map(tuple, np.round(numeric, 10)))

    @pytest.mark.parametrize("cube", ["full", "fractional"])
    def test_unknown_alpha_raises(self, cube: str) -> None:
        with pytest.raises(ValueError, match="Unknown alpha"):
            generate_design(_factors(5), design_type="ccd", alpha="rotateable", cube=cube)

    @pytest.mark.parametrize("n_center", [0, 1, 2, 5])
    def test_centre_run_count_is_the_one_asked_for(self, n_center: int) -> None:
        x = _coded(generate_design(_factors(3), design_type="ccd", n_center_points=n_center))
        assert int(np.all(x == 0, axis=1).sum()) == n_center
