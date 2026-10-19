"""augment_design inside a region: new runs keep to the limits on the settings (#809)."""

from __future__ import annotations

import doctest

import numpy as np
import pytest

from process_improve.experiments import Factor
from process_improve.experiments.region import DesignRegion

#: A reactor whose temperature and dosing rate cannot both be high: 3*T + 5*D <= 600 cuts the (+, +) corner.
FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
LIMIT = "3*T + 5*D <= 600"


class TestBrokenLimits:
    def test_each_point_names_the_ranges_it_leaves_and_the_limits_it_fails(self) -> None:
        region = DesignRegion(FACTORS, [LIMIT, "100 <= T + D <= 190"])
        points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.5, -1.0], [2.0, 1.0]])
        assert region.broken_limits(points) == [
            [LIMIT, "100 <= T + D <= 190"],  # a chained limit is named once
            [],
            ["100 <= T <= 150"],
            ["100 <= T <= 150", LIMIT, "100 <= T + D <= 190"],
        ]

    def test_a_point_breaks_no_limit_exactly_when_it_is_feasible(self) -> None:
        region = DesignRegion(FACTORS, [LIMIT, "T*D <= 7000", "sqrt(T - 110) >= 0"])
        points = np.random.default_rng(3).uniform(-1.3, 1.3, size=(400, 2))
        with np.errstate(invalid="ignore"):
            feasible = region.feasible(points)
        assert [not broken for broken in region.broken_limits(points)] == feasible.tolist()

    def test_a_curved_limit_that_cannot_be_evaluated_is_broken(self) -> None:
        """``sqrt`` of a negative number: :meth:`feasible` cannot run the setting either."""
        region = DesignRegion(FACTORS, ["sqrt(T - 110) >= 1"])
        assert region.broken_limits(np.array([[-1.0, 0.0]])) == [["sqrt(T - 110) >= 1"]]

    def test_a_mixture_region_is_refused(self) -> None:
        blend = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"]
        with pytest.raises(ValueError, match=r"DesignRegion.broken_limits\(\) describes a box"):
            DesignRegion(blend).broken_limits(np.array([[0.2, 0.3, 0.5]]))

    def test_the_example_runs(self) -> None:
        finder = doctest.DocTestFinder()
        [test] = finder.find(
            DesignRegion.broken_limits,
            "broken_limits",
            globs={"DesignRegion": DesignRegion, "Factor": Factor, "np": np},
        )
        result = doctest.DocTestRunner().run(test)
        assert (result.failed, result.attempted) == (0, 2)
