"""Tests for the teaching simulators in ``process_improve.experiments.simulations``.

The three simulators (``popcorn``, ``grocery``, ``manufacture``) are ports of
the R ``pid`` package. The reference values below are the deterministic part of
each model, computed by hand from the published formula; the tests bracket the
returned value by the known noise range rather than pinning an exact draw, so
they hold for any RNG stream.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from process_improve.experiments.simulations import grocery, manufacture, popcorn


class TestPopcorn:
    def test_returns_int_with_defaults(self):
        assert isinstance(popcorn(), int)

    def test_matches_the_published_model(self):
        """At t = 135 s the coded time is zero, so y = 93 + U(0, 1) * 6 - 3."""
        values = [popcorn(t=135, random_state=seed) for seed in range(50)]
        assert all(90 <= v <= 96 for v in values)

    def test_response_peaks_near_the_optimum(self):
        """The quadratic peaks at coded = 15 / (2 * 2.4), i.e. t ~ 181 s."""
        rng = np.random.default_rng(0)
        at_peak = np.mean([popcorn(t=182, random_state=rng) for _ in range(200)])
        for elsewhere in (100, 140, 250):
            assert at_peak > np.mean([popcorn(t=elsewhere, random_state=rng) for _ in range(200)])

    def test_never_returns_a_negative_count(self):
        """Far past the optimum the quadratic goes negative; the count cannot."""
        assert popcorn(t=1000, random_state=0) == 0

    def test_uppercase_alias_matches_lowercase(self):
        """``popcorn(T=...)`` is the R spelling and must agree with ``t``."""
        assert popcorn(T=135, random_state=7) == popcorn(t=135, random_state=7)

    def test_uppercase_alias_wins_when_both_given(self):
        assert popcorn(t=90, T=135, random_state=7) == popcorn(t=135, random_state=7)

    def test_same_seed_reproduces(self):
        assert popcorn(t=140, random_state=99) == popcorn(t=140, random_state=99)

    def test_generator_advances_between_calls(self):
        """Passing a Generator lets the caller own the state, so draws differ."""
        rng = np.random.default_rng(3)
        draws = {popcorn(t=140, random_state=rng) for _ in range(30)}
        assert len(draws) > 1

    @pytest.mark.parametrize("bad", [76, 0, -10])
    def test_too_short_a_cooking_time_rejected(self, bad):
        with pytest.raises(ValueError, match="cook for a longer time"):
            popcorn(t=bad)

    def test_lower_boundary_is_inclusive(self):
        """77 seconds is the first supported time."""
        assert isinstance(popcorn(t=77, random_state=0), int)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_time_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            popcorn(t=bad)

    @pytest.mark.parametrize("bad", [[100, 120], np.array([100.0, 120.0])])
    def test_vector_input_rejected(self, bad):
        with pytest.raises(ValueError, match="parallel"):
            popcorn(t=bad)


class TestGrocery:
    def test_returns_int_with_defaults(self):
        assert isinstance(grocery(), int)

    def test_same_seed_reproduces(self):
        assert grocery(P=3.5, H=150, random_state=5) == grocery(P=3.5, H=150, random_state=5)

    def test_matches_the_published_model(self):
        """At P = 3.2, H = 50 both coded factors are zero, so y = 600 + noise."""
        values = [grocery(P=3.2, H=50, random_state=seed) for seed in range(50)]
        assert all(abs(v - 600) < 12 for v in values)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_price_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            grocery(P=bad, H=150.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_height_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            grocery(P=3.5, H=bad)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="positive sales price"):
            grocery(P=-1.0, H=150.0)

    def test_negative_height_rejected(self):
        with pytest.raises(ValueError, match="height of the shelving"):
            grocery(P=3.5, H=-1.0)

    def test_list_input_rejected(self):
        with pytest.raises(ValueError, match="parallel"):
            grocery(P=[3.5, 4.0], H=150.0)

    def test_array_input_rejected(self):
        with pytest.raises(ValueError, match="parallel"):
            grocery(P=3.5, H=np.array([100.0, 150.0]))


class TestManufacture:
    def test_returns_int_with_defaults(self):
        assert isinstance(manufacture(), int)

    def test_matches_the_published_model(self):
        """At P = 1.5, T = 320 both coded factors are zero.

        That leaves y = 50 * 12 + 2 sin(320) + 2 cos(1.5) + noise.
        """
        deterministic = 600 + 2 * math.sin(320) + 2 * math.cos(1.5)
        values = [manufacture(p=1.5, t=320, random_state=seed) for seed in range(50)]
        assert all(abs(v - deterministic) < 12 for v in values)

    def test_response_peaks_near_the_optimum(self):
        """The quadratic in the coded factors peaks at P = 1.5, T = 320."""
        rng = np.random.default_rng(0)
        at_peak = np.mean([manufacture(p=1.5, t=320, random_state=rng) for _ in range(200)])
        for p, t in [(0.75, 325), (2.5, 300), (1.5, 380), (0.1, 250)]:
            assert at_peak > np.mean([manufacture(p=p, t=t, random_state=rng) for _ in range(200)])

    def test_uppercase_aliases_match_lowercase(self):
        assert manufacture(P=1.5, T=320, random_state=7) == manufacture(p=1.5, t=320, random_state=7)

    def test_same_seed_reproduces(self):
        assert manufacture(random_state=11) == manufacture(random_state=11)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_price_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            manufacture(p=bad)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_throughput_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            manufacture(t=bad)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="positive sales price"):
            manufacture(p=-1.0)

    def test_negative_throughput_rejected(self):
        with pytest.raises(ValueError, match="throughput"):
            manufacture(t=-1.0)

    @pytest.mark.parametrize("bad", [[0.5, 1.0], np.array([0.5, 1.0])])
    def test_vector_price_rejected(self, bad):
        with pytest.raises(ValueError, match="parallel"):
            manufacture(p=bad)

    def test_vector_throughput_rejected(self):
        with pytest.raises(ValueError, match="parallel"):
            manufacture(t=np.array([300.0, 325.0]))
