"""Tests for `process_improve.experiments.simulations.grocery`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from process_improve.experiments.simulations import grocery


class TestGroceryDefaults:
    def test_returns_int_with_defaults(self):
        result = grocery()
        assert isinstance(result, int)


class TestGroceryInputChecks:
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
