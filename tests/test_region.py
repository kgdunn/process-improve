"""Tests for the region diagnostics of :class:`DesignRegion` and for renaming factors in constraints."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import Constraint, DesignRegion, Factor
from process_improve.experiments.designs import _as_constraints
from process_improve.experiments.factor import as_constraints

# The heat limit of tests/test_designs_constrained.py: it cuts the corner (150, 60) off the box.
HEAT_FACTORS = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT = "3*T + 5*D <= 600"


# ---------------------------------------------------------------------------
# Constraints as text
# ---------------------------------------------------------------------------


class TestConstraintsAsText:
    def test_a_bare_string_a_dict_and_a_constraint_give_the_same_region(self):
        points = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.2, 0.9]])
        expected = DesignRegion(HEAT_FACTORS, [Constraint(expression=HEAT)]).feasible(points)
        for given in (HEAT, {"expression": HEAT}, {"expression": HEAT, "type": "linear"}):
            region = DesignRegion(HEAT_FACTORS, [given])
            assert region.constraints == [Constraint(expression=HEAT)]
            np.testing.assert_array_equal(region.feasible(points), expected)

    def test_forms_mix_freely_and_keep_their_order(self):
        region = DesignRegion(HEAT_FACTORS, ["T <= 140", Constraint(expression="D >= 25"), {"expression": HEAT}])
        assert [c.expression for c in region.constraints] == ["T <= 140", "D >= 25", HEAT]

    def test_a_tuple_of_constraints_is_accepted(self):
        assert len(DesignRegion(HEAT_FACTORS, (HEAT,)).constraints) == 1

    def test_anything_else_is_refused(self):
        with pytest.raises(TypeError, match="Constraint, a dict or an expression string"):
            DesignRegion(HEAT_FACTORS, [42])  # type: ignore[list-item]

    def test_the_designs_module_keeps_the_old_name(self):
        assert _as_constraints is as_constraints
        assert as_constraints(None) is None

    def test_a_text_region_round_trips_through_its_dict(self):
        region = DesignRegion(HEAT_FACTORS, [HEAT])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.constraints == region.constraints
