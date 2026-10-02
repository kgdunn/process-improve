"""Optimal and constrained designs: review fixes for the exchange engines and their metadata."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from process_improve.experiments.optimal import point_exchange


def test_point_exchange_reaches_the_half_fraction_from_every_seed() -> None:
    """Four runs from the 3^3 grid, first-order model: the 2^(3-1) half fraction, det(X'X) = 4^4."""
    grid = pd.DataFrame(list(itertools.product([-1, 0, 1], repeat=3)), columns=list("abc"))
    values = [point_exchange(grid, 4, random_state=seed)[1] for seed in range(20)]
    # A single pass stopped at det(X'X) = 64 or below for about half of all seeds.
    np.testing.assert_allclose(values, -np.log(4.0**4))
