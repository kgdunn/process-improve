"""Tests for the region diagnostics of :class:`DesignRegion` and for renaming factors in constraints."""

from __future__ import annotations

import numpy as np
import pytest

from process_improve.experiments import Constraint, DesignRegion, Factor, rename_factors
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


# ---------------------------------------------------------------------------
# Renaming factors inside a constraint
# ---------------------------------------------------------------------------


class TestRenameFactors:
    def test_a_rename_gives_the_canonical_text(self):
        assert rename_factors(HEAT, {"T": "Temp"}) == "3 * Temp + 5 * D <= 600"

    def test_a_swap_is_done_all_at_once(self):
        assert rename_factors(HEAT, {"T": "D", "D": "T"}) == "3 * D + 5 * T <= 600"

    def test_a_cycle_of_three_is_done_all_at_once(self):
        assert rename_factors("A + 2*B + 3*C <= 6", {"A": "B", "B": "C", "C": "A"}) == "B + 2 * C + 3 * A <= 6"

    def test_function_names_are_untouched(self):
        assert rename_factors("sqrt(log) + log(log) <= 2", {"log": "L", "sqrt": "S"}) == "sqrt(L) + log(L) <= 2"

    def test_a_factor_may_be_renamed_to_a_function_name(self):
        text = rename_factors("sqrt(A) <= 2", {"A": "sqrt"})
        assert text == "sqrt(sqrt) <= 2"
        region = DesignRegion([Factor(name="sqrt", low=1, high=9)], [text])
        np.testing.assert_array_equal(region.feasible(np.array([[-1.0], [1.0]])), [True, False])

    def test_the_inverse_rename_gives_back_the_canonical_text(self):
        for expression in (HEAT, "0.5 <= A/B <= 2", "abs(A - 1e-5) + exp(-B) < 3", "-A**2 >= -4"):
            forward = {"A": "Alpha", "B": "Beta", "T": "Temp", "D": "Dose"}
            inverse = {new: old for old, new in forward.items()}
            assert rename_factors(rename_factors(expression, forward), inverse) == rename_factors(expression, {})

    def test_an_empty_rename_gives_the_canonical_text(self):
        assert rename_factors("3*T+5*D<=600", {}) == "3 * T + 5 * D <= 600"
        assert rename_factors("0.5<=A/B<=2", {}) == "0.5 <= A / B <= 2"

    def test_names_the_expression_does_not_use_are_ignored(self):
        assert rename_factors("T <= 140", {"Z": "Zeta"}) == "T <= 140"

    def test_the_constraint_method_keeps_the_type(self):
        renamed = Constraint(expression="A*B <= 3", type="nonlinear").renamed({"A": "B", "B": "A"})
        assert renamed == Constraint(expression="B * A <= 3", type="nonlinear")

    @pytest.mark.parametrize(
        ("expression", "names", "match"),
        [
            ("T <= 3", {"T": "1x"}, "not a valid identifier"),
            ("T <= 3", {"T": "for"}, "not a valid identifier"),
            ("T <= 3", {"T": "Temp (C)"}, "not a valid identifier"),
            ("T + D <= 3", {"T": "D"}, "merge the factors"),
            ("T == 3", {}, "only <, <=, >, >="),
            ("T = 3", {}, "not a valid expression"),
            ("T + 3", {}, "must be an inequality"),
            ("T(3) <= 1", {}, "Unsupported syntax"),
            ("__import__('os') <= 1", {}, "Unsupported syntax"),
            ("T" * 501 + " <= 1", {}, "longer than 500"),
        ],
    )
    def test_bad_input_is_refused(self, expression, names, match):
        with pytest.raises(ValueError, match=match):
            rename_factors(expression, names)

    def test_a_rename_that_makes_the_text_too_long_is_refused(self):
        expression = " + ".join(["A"] * 120) + " <= 1"
        with pytest.raises(ValueError, match="longer than 500"):
            rename_factors(expression, {"A": "Alpha"})
