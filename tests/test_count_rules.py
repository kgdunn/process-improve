"""Limits on how many factors a run uses: ``count_nonzero(A, B, C) <= 2``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import DesignRegion, Factor, generate_design, optimize_responses
from process_improve.experiments import _uniform_sampling as sampling
from process_improve.experiments import designs_constrained as engine
from process_improve.experiments._count_rules import (
    CountRule,
    UsedSets,
    count_rule_syntax,
    count_rules,
    is_count_rule,
    used_sets,
    without_count_rules,
)
from process_improve.experiments.augment import augment_design
from process_improve.experiments.designs_constrained import (
    build_candidates,
    piece_vertices,
    region_with_rules,
    rename_factors,
)
from process_improve.experiments.designs_mixture_constrained import candidate_blends
from process_improve.experiments.factor import DesignResult, as_constraints

NAMES = ["A", "B", "C", "D"]
#: Four amounts, each left out or between 2 and 10.
AMOUNTS = [Factor(name=name, low=0, high=10, min_when_used=2) for name in NAMES]
#: The same four without a minimum: used at any amount above 0.
PLAIN = [Factor(name=name, low=0, high=10) for name in NAMES]
AT_MOST_TWO = "count_nonzero(A, B, C, D) <= 2"


def _used(result: DesignResult, names: list[str]) -> np.ndarray:
    """Return how many of ``names`` each run of a design uses."""
    return (result.design_actual[names].to_numpy(dtype=float) > 1e-9).sum(axis=1)


class TestSyntax:
    @pytest.mark.parametrize(
        ("expression", "limits"),
        [
            ("count_nonzero(A, B, C) <= 2", (None, 2)),
            ("count_nonzero(A, B, C) < 2", (None, 1)),
            ("count_nonzero(A, B, C) >= 1", (1, None)),
            ("count_nonzero(A, B, C) > 1", (2, None)),
            ("count_nonzero(A, B, C) == 1", (1, 1)),
            ("2 >= count_nonzero(A, B, C)", (None, 2)),
            ("1 < count_nonzero(A, B, C)", (2, None)),
            ("1 <= count_nonzero(A, B, C) <= 2", (1, 2)),
            ("3 > count_nonzero(A, B, C) > 0", (1, 2)),
            ("count_nonzero(A, B, C) <= 2.0", (None, 2)),
        ],
    )
    def test_a_rule_is_read_as_a_lower_and_an_upper_limit(self, expression: str, limits: tuple) -> None:
        assert is_count_rule(expression)
        assert count_rule_syntax(expression) == (["A", "B", "C"], *limits)

    @pytest.mark.parametrize(
        ("expression", "message"),
        [
            ("count_nonzero(A, B) + 1 <= 2", "stands alone on one side of a comparison"),
            ("count_nonzero(A, B) <= 1 or A <= 2", "stands alone on one side of a comparison"),
            ("count_nonzero(A, B) <= count_nonzero(C, D)", "stands alone on one side of a comparison"),
            ("abs(count_nonzero(A, B)) <= 1", "stands alone on one side of a comparison"),
            ("count_nonzero() <= 1", "takes factor names"),
            ("count_nonzero(A, 2*B) <= 1", "takes factor names"),
            ("count_nonzero(A, B, at_most=1) <= 1", "takes factor names"),
            ("count_nonzero(A, B) <= 1.5", "is compared with whole numbers"),
            ("count_nonzero(A, B) <= A", "is compared with whole numbers"),
            ("count_nonzero(A, B) != 1", "only <, <=, >, >= and == compare"),
            ("0 <= 1 <= count_nonzero(A, B)", "each comparison must have count_nonzero"),
            ("count_nonzero(A, B <= 1", "is not a valid expression"),
            ("count_nonzero(A, B) <= 1" + " " * 3000, "is longer than"),
        ],
    )
    def test_a_rule_written_another_way_is_refused(self, expression: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            count_rule_syntax(expression)

    def test_other_constraints_are_not_count_rules(self) -> None:
        constraints = as_constraints(["A + B <= 12", AT_MOST_TWO, "A + B <= ("]) or []
        assert [is_count_rule(c.expression) for c in constraints] == [False, True, False]
        assert [c.expression for c in without_count_rules(constraints)] == ["A + B <= 12", "A + B <= ("]
        assert without_count_rules(None) == []

    def test_a_rule_names_the_factors_it_counts(self) -> None:
        rules = count_rules(as_constraints(["A + B <= 12", "1 <= count_nonzero(A, B, C) <= 2"]), AMOUNTS)
        assert rules == [CountRule(("A", "B", "C"), 1, 2, "1 <= count_nonzero(A, B, C) <= 2")]
        assert count_rules(None, AMOUNTS) == []

    def test_a_factor_that_cannot_be_left_out_always_counts(self) -> None:
        factors = [Factor(name="A", low=1, high=10), Factor(name="B", low=0, high=10), Factor(name="C", low=0, high=10)]
        (rule,) = count_rules(as_constraints(["count_nonzero(A, B, C) <= 2"]), factors)
        assert (rule.at_least, rule.at_most) == (1, 2)
        # At least one of them is in every run, so "at least 1" asks nothing, and the rule is dropped.
        assert count_rules(as_constraints(["count_nonzero(A, B, C) >= 1"]), factors) == []

    def test_a_rule_that_cannot_bind_is_dropped(self) -> None:
        assert count_rules(as_constraints(["count_nonzero(A, B, C) <= 3", "count_nonzero(A, B) >= 0"]), AMOUNTS) == []

    @pytest.mark.parametrize(
        ("expression", "message"),
        [
            ("count_nonzero(A, Z) <= 1", r"counts unknown factor\(s\) \['Z'\]; the factors are \[A, B, C, D, T\]"),
            ("count_nonzero(A, A) <= 1", "names a factor more than once"),
            ("count_nonzero(A, kind) <= 1", "counts 'kind', which is not an amount"),
            ("count_nonzero(A, T) <= 1", "counts 'T', which is not an amount"),
            ("count_nonzero(A, B) >= 3", "No setting meets .* between 0 and 2 of them are used"),
            ("count_nonzero(A, B) < 0", "No setting meets"),
        ],
    )
    def test_a_rule_that_cannot_be_counted_is_refused(self, expression: str, message: str) -> None:
        factors = [
            *AMOUNTS,
            Factor(name="T", low=-5, high=5),
            Factor(name="kind", type="categorical", levels=["x", "y"]),
        ]
        with pytest.raises(ValueError, match=message):
            count_rules(as_constraints([expression]), factors)

    def test_factors_are_renamed_inside_a_rule(self) -> None:
        assert rename_factors("1 <= count_nonzero(A, B, C) <= 2", {"A": "X"}) == "1 <= count_nonzero(X, B, C) <= 2"
        with pytest.raises(ValueError, match="stands alone"):
            rename_factors("count_nonzero(A, B) * 2 <= 2", {"A": "X"})


def _sets(factors: list[Factor], rules: list[str]) -> UsedSets:
    names = [f.name for f in factors]
    found = used_sets(names, factors, count_rules(as_constraints(rules), factors), dict.fromkeys(names, True))
    assert found is not None
    return found


class TestUsedSets:
    def test_there_are_no_sets_without_a_rule(self) -> None:
        assert used_sets(NAMES, AMOUNTS, [], dict.fromkeys(NAMES, True)) is None

    def test_a_factor_is_used_above_zero(self) -> None:
        sets = _sets(AMOUNTS, [AT_MOST_TWO])
        points = np.array([[-1, -1, -1, -1], [0.5, 0.5, 0.5, -1], [1, -1, -0.6, -1]], dtype=float)
        np.testing.assert_array_equal(sets.used(points).sum(axis=1), [0, 3, 2])
        np.testing.assert_array_equal(sets.allows(points), [True, False, True])
        ((expression, breaks),) = sets.broken(points)
        assert expression == AT_MOST_TWO
        np.testing.assert_array_equal(breaks, [False, True, False])

    def test_where_a_factor_sits_depends_on_its_units(self) -> None:
        coded = _sets(AMOUNTS, [AT_MOST_TWO])
        assert coded.optional == [(j, -1.0, pytest.approx(-0.6)) for j in range(4)]  # 2 of 10 is coded -0.6
        assert _sets(PLAIN, [AT_MOST_TWO]).optional == [(j, -1.0, -1.0) for j in range(4)]
        shares = [Factor(name=name, type="mixture", min_when_used=0.1) for name in NAMES]
        blends = used_sets(
            NAMES, shares, count_rules(as_constraints([AT_MOST_TWO]), shares), dict.fromkeys(NAMES, False)
        )
        assert blends is not None
        assert blends.optional == [(j, 0.0, 0.1) for j in range(4)]

    def test_only_the_factors_that_can_be_left_out_are_optional(self) -> None:
        factors = [Factor(name="A", low=1, high=10), *AMOUNTS[1:], Factor(name="E", low=0, high=10)]
        sets = _sets(factors, ["count_nonzero(A, B, C) <= 2"])
        # A cannot be left out, D can (its minimum), and E has neither a minimum nor a rule.
        assert [column for column, _zero, _minimum in sets.optional] == [1, 2, 3]
        allowed = sets.sets(100)
        assert len(allowed) == 6  # B and C not both in use, with D either way
        assert not np.any(allowed[:, 0] & allowed[:, 1])

    def test_every_allowed_set_of_a_few_factors_is_listed(self) -> None:
        sets = _sets(AMOUNTS, [AT_MOST_TWO])
        allowed = sets.sets(100)
        assert len(allowed) == 11  # none, 4 single factors, 6 pairs
        assert allowed.sum(axis=1).max() == 2
        fewer = sets.sets(5)
        assert len(fewer) == 5
        assert {tuple(row) for row in fewer} <= {tuple(row) for row in allowed}

    def test_sets_of_many_factors_are_drawn_and_meet_every_rule(self) -> None:
        names = [f"G{g}_{i}" for g in range(3) for i in range(5)]
        factors = [Factor(name=name, low=0, high=10, min_when_used=1) for name in names]
        rules = [f"1 <= count_nonzero({', '.join(names[5 * g : 5 * g + 5])}) <= 2" for g in range(3)]
        rules.append(f"count_nonzero({', '.join(names)}) <= 5")
        sets = _sets(factors, rules)
        drawn = sets.sets(60)
        assert len(drawn) == 60
        assert len({tuple(row) for row in drawn}) == 60
        per_group = drawn.reshape(60, 3, 5).sum(axis=2)
        assert per_group.min() >= 1
        assert per_group.max() <= 2
        assert drawn.sum(axis=1).max() <= 5
        assert {int(n) for n in drawn.sum(axis=1)} == {3, 4, 5}  # small and large sets both come up
        np.testing.assert_array_equal(drawn, sets.sets(60))  # the sets do not depend on a random state

    def test_rules_that_cannot_all_hold_give_no_set(self) -> None:
        names = [f"X{i}" for i in range(12)]
        factors = [Factor(name=name, low=0, high=10) for name in names]
        sets = _sets(
            factors, [f"count_nonzero({', '.join(names)}) <= 2", f"count_nonzero({', '.join(names[:6])}) >= 3"]
        )
        assert sets.sets(10).shape == (0, 12)

    def test_a_piece_leaves_factors_out_and_keeps_the_others_at_their_minimum_or_above(self) -> None:
        sets = _sets(AMOUNTS, [AT_MOST_TWO])
        piece = sets.bounds(np.array([True, False, True, False]), [(-1.0, 1.0)] * 4)
        assert piece == [(pytest.approx(-0.6), 1.0), (-1.0, -1.0), (pytest.approx(-0.6), 1.0), (-1.0, -1.0)]


class TestPieceVertices:
    def test_every_point_lies_on_a_piece(self) -> None:
        sets = _sets(AMOUNTS, [AT_MOST_TWO])
        total = np.ones((1, 4)) / 2.0  # A + B + C + D <= 14 in amounts, a unit-length row in coded units
        points = piece_vertices(sets, total, np.array([-0.6]), [(-1.0, 1.0)] * 4)
        assert len(points) > 11
        assert sets.allows(points).all()
        assert np.all(points @ total.T <= -0.6 + 1e-9)
        amounts = (points + 1.0) * 5.0
        assert not np.any((amounts > 1e-9) & (amounts < 2 - 1e-9))
        assert len(np.unique(points.round(10), axis=0)) == len(points)

    def test_points_between_the_vertices_of_a_piece_are_included(self) -> None:
        sets = _sets(AMOUNTS, [AT_MOST_TWO])
        points = piece_vertices(sets, np.empty((0, 4)), np.empty(0), [(-1.0, 1.0)] * 4)
        amounts = (points + 1.0) * 5.0
        assert np.any((amounts > 2 + 1e-6) & (amounts < 10 - 1e-6))  # between the minimum and the upper limit

    def test_a_piece_that_no_point_meets_gives_nothing(self) -> None:
        sets = _sets(AMOUNTS, ["count_nonzero(A, B, C, D) == 1"])
        total = -np.ones((1, 4)) / 2.0  # A + B + C + D >= 15: more than one factor at its upper limit of 10
        assert piece_vertices(sets, total, np.array([-1.0]), [(-1.0, 1.0)] * 4).shape == (0, 4)

    def test_no_allowed_set_gives_nothing(self) -> None:
        names = [f"X{i}" for i in range(12)]
        factors = [Factor(name=name, low=0, high=10) for name in names]
        sets = _sets(
            factors, [f"count_nonzero({', '.join(names)}) <= 2", f"count_nonzero({', '.join(names[:6])}) >= 3"]
        )
        assert piece_vertices(sets, np.empty((0, 12)), np.empty(0), [(-1.0, 1.0)] * 12).shape == (0, 12)


class TestDesigns:
    @pytest.mark.parametrize(("model_type", "budget"), [("main_effects", 8), ("interactions", 14), ("quadratic", 20)])
    @pytest.mark.parametrize("factors", [AMOUNTS, PLAIN], ids=["with-a-minimum", "without"])
    def test_no_run_uses_more_factors_than_the_rule_allows(
        self, factors: list[Factor], model_type: str, budget: int
    ) -> None:
        result = generate_design(
            factors, budget=budget, model_type=model_type, constraints=[AT_MOST_TWO], random_state=1
        )
        assert result.design_type == "d_optimal"
        assert _used(result, NAMES).max() <= 2
        amounts = result.design_actual[NAMES].to_numpy(dtype=float)
        if factors is AMOUNTS:
            assert not np.any((amounts > 1e-9) & (amounts < 2 - 1e-9))
        assert result.metadata["constraints"] == [AT_MOST_TWO]
        assert DesignRegion.from_dict(result.metadata["region"]).feasible((amounts - 5.0) / 5.0).all()

    def test_exactly_one_of_a_group_is_used(self) -> None:
        result = generate_design(
            AMOUNTS, budget=8, model_type="main_effects", constraints=["count_nonzero(A, B, C) == 1"], random_state=1
        )
        np.testing.assert_array_equal(_used(result, ["A", "B", "C"]), 1)
        assert set(_used(result, ["D"])) == {0, 1}  # the rule does not count D

    def test_interactions_of_factors_never_used_together_are_left_out(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = generate_design(
                AMOUNTS,
                budget=12,
                model_type="interactions",
                constraints=["count_nonzero(A, B, C) == 1"],
                random_state=1,
            )
        assert result.metadata["inestimable_columns"] == {"A:B": 1, "A:C": 1, "B:C": 1}
        assert "The constraints leave the A:B term with 0 of its 1 column(s) estimable" in caplog.text
        np.testing.assert_array_equal(_used(result, ["A", "B", "C"]), 1)

    def test_a_rule_that_cannot_bind_changes_nothing(self) -> None:
        with_rule = generate_design(
            PLAIN, budget=12, model_type="interactions", constraints=["count_nonzero(A, B, C, D) <= 4"], random_state=3
        )
        without = generate_design(PLAIN, design_type="d_optimal", budget=12, model_type="interactions", random_state=3)
        np.testing.assert_array_equal(
            with_rule.design_actual[NAMES].to_numpy(dtype=float), without.design_actual[NAMES].to_numpy(dtype=float)
        )

    def test_a_linear_limit_is_kept_beside_the_rule(self) -> None:
        result = generate_design(
            AMOUNTS,
            budget=10,
            model_type="main_effects",
            constraints=[AT_MOST_TWO, "A + B + C + D <= 14"],
            random_state=1,
        )
        amounts = result.design_actual[NAMES].to_numpy(dtype=float)
        assert amounts.sum(axis=1).max() <= 14 + 1e-9
        assert _used(result, NAMES).max() <= 2

    def test_candidates_that_break_the_rule_are_dropped(self) -> None:
        candidates = pd.DataFrame(
            [[0, 0, 0, 0], [10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10], [10, 10, 0, 0], [10, 10, 10, 0]],
            columns=NAMES,
        )
        result = generate_design(
            AMOUNTS,
            design_type="d_optimal",
            budget=6,
            model_type="main_effects",
            constraints=[AT_MOST_TWO],
            candidates=candidates,
            random_state=1,
        )
        assert result.metadata["n_candidates"] == 6
        assert _used(result, NAMES).max() <= 2

    def test_a_rule_holds_beside_a_rule_on_levels(self) -> None:
        factors = [*AMOUNTS, Factor(name="kind", type="categorical", levels=["x", "y"])]
        limits = [AT_MOST_TWO, "not (kind == 'y' and A > 5)"]
        result = generate_design(factors, budget=14, model_type="main_effects", constraints=limits, random_state=1)
        runs = result.design_actual
        assert _used(result, NAMES).max() <= 2
        assert runs.loc[runs["kind"] == "y", "A"].max() <= 5 + 1e-9
        coded = (runs[NAMES].to_numpy(dtype=float) - 5.0) / 5.0
        assert DesignRegion(factors, limits).feasible(coded, labels=runs).all()

    @pytest.mark.parametrize("design_type", ["ccd", "full_factorial", "box_behnken"])
    def test_a_classical_design_is_refused(self, design_type: str) -> None:
        with pytest.raises(
            ValueError, match=r"cannot follow the constraints .* which limit how many factors a run uses"
        ):
            generate_design(AMOUNTS, design_type=design_type, constraints=[AT_MOST_TWO])

    def test_a_design_drawn_from_the_region_keeps_the_rule(self) -> None:
        result = generate_design(AMOUNTS, design_type="maximin", budget=12, constraints=[AT_MOST_TWO], random_state=1)
        assert _used(result, NAMES).max() <= 2

    def test_runs_are_added_inside_the_rule(self) -> None:
        design = generate_design(
            AMOUNTS, budget=8, model_type="main_effects", constraints=[AT_MOST_TWO], random_state=1
        )
        added = augment_design(
            design.design[NAMES],
            "add_runs_optimal",
            n_additional_runs=4,
            target_model="main_effects",
            region=design.metadata["region"],
        )
        new_runs = pd.DataFrame(added["new_runs"])[NAMES].to_numpy(dtype=float)
        assert len(new_runs) == 4
        assert DesignRegion.from_dict(design.metadata["region"]).feasible(new_runs).all()

    @pytest.mark.slow
    def test_a_formulation_of_many_amounts_in_groups_keeps_the_rule(self) -> None:
        names = [f"G{g}_{i}" for g in range(4) for i in range(6)]
        groups = [names[6 * g : 6 * g + 6] for g in range(4)]
        factors = [Factor(name=name, low=0, high=40, units="%", min_when_used=4) for name in names]
        limits = [f"12 <= {' + '.join(group)} <= 40" for group in groups]
        limits += [f"{' + '.join(names)} <= 100", f"6 <= count_nonzero({', '.join(names)}) <= 10"]
        result = generate_design(factors, budget=40, model_type="main_effects", constraints=limits, random_state=1)
        used = _used(result, names)
        assert used.min() >= 6
        assert used.max() <= 10
        amounts = result.design_actual[names].to_numpy(dtype=float)
        assert not np.any((amounts > 1e-6) & (amounts < 4 - 1e-6))
        for group in groups:
            total = result.design_actual[group].sum(axis=1)
            assert total.min() >= 12 - 1e-6
            assert total.max() <= 40 + 1e-6
        assert np.linalg.matrix_rank(np.column_stack([np.ones(40), amounts])) == 25


class TestCandidates:
    def test_the_pieces_supply_candidates_a_grid_misses(self) -> None:
        names = [f"X{i}" for i in range(14)]
        factors = [Factor(name=name, low=0, high=10, min_when_used=1) for name in names]
        rule = f"6 <= count_nonzero({', '.join(names)}) <= 8"
        region = region_with_rules(factors, as_constraints([rule]) or [])
        coded, cats, counts = build_candidates(region, model_type="main_effects")
        assert counts["n_vertex_points"] > 0
        used = (coded > -1 + 1e-9).sum(axis=1)
        assert used.min() >= 6
        assert used.max() <= 8
        assert np.all(region.slack(coded, cats) <= 1e-9)

    def test_no_allowed_piece_leaves_no_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine, "_SAMPLED_VERTICES", 64)
        names = [f"X{i}" for i in range(12)]
        factors = [Factor(name=name, low=0, high=10) for name in names]
        limits = [f"count_nonzero({', '.join(names)}) <= 2", f"count_nonzero({', '.join(names[:6])}) >= 3"]
        with pytest.raises(ValueError, match="No candidate point satisfies all the constraints"):
            generate_design(factors, budget=20, model_type="main_effects", constraints=limits, random_state=1)


class TestRegion:
    def test_a_run_that_uses_too_many_factors_is_outside_the_region(self) -> None:
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO, "A + B <= 15"])
        coded = np.array([[-1, -1, -1, -1], [0.5, 0.5, 0.5, -1], [0.5, -0.9, -1, -1], [1, 1, -1, -1]], dtype=float)
        np.testing.assert_array_equal(region.feasible(coded), [True, False, False, False])
        assert region.broken_limits(coded) == [[], [AT_MOST_TWO], ["B = 0 or B >= 2"], ["A + B <= 15"]]
        assert [rule.expression for rule in region.counts] == [AT_MOST_TWO]
        assert region.gaps == [(j, pytest.approx(-0.6)) for j in range(4)]

    def test_a_factor_without_a_minimum_can_be_left_out_too(self) -> None:
        region = DesignRegion(PLAIN, [AT_MOST_TWO])
        assert region.gaps == [(j, -1.0) for j in range(4)]
        np.testing.assert_array_equal(
            region.feasible(np.array([[-0.99, 0.0, -1, -1], [-0.99, 0.0, 0.5, -1]])), [True, False]
        )

    def test_the_region_is_written_and_read_with_its_rule(self) -> None:
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.counts == region.counts
        np.testing.assert_array_equal(
            again.feasible(np.array([[1.0, 1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0]])), [False, True]
        )

    @pytest.mark.parametrize("method", ["vertices", "volume_fraction", "centroid", "implied_bounds"])
    def test_a_diagnostic_of_one_region_is_refused(self, method: str) -> None:
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO])
        with pytest.raises(ValueError, match=r"the limits on how many factors are used .* give one for each set"):
            getattr(region, method)()

    def test_a_region_is_empty_when_no_piece_holds_a_setting(self) -> None:
        assert DesignRegion(AMOUNTS, [AT_MOST_TWO]).is_empty() is False
        # Two factors at 10 at most reach 20, so no run with at most two in use totals 25.
        assert DesignRegion(AMOUNTS, [AT_MOST_TWO, "A + B + C + D >= 25"]).is_empty() is True

    def test_the_support_points_keep_the_rule(self) -> None:
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO])
        points = region.support_points()
        assert len(points) > 11
        assert region.feasible(points).all()

    def test_the_sets_are_drawn_as_often_as_one_another(self) -> None:
        # Each factor is left out half the time before the rule acts, so the 11 sets of at most two of the four
        # are as likely as one another: A is left out in 7 of them, and 1, 4 and 6 of them use 0, 1 and 2 factors.
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO])
        points = region.sample(20_000, np.random.default_rng(1))
        assert region.feasible(points).all()
        used = points > -1 + 1e-9
        assert (~used[:, 0]).mean() == pytest.approx(7 / 11, abs=0.02)
        shares = np.bincount(used.sum(axis=1), minlength=3) / len(points)
        assert shares == pytest.approx([1 / 11, 4 / 11, 6 / 11], abs=0.02)

    def test_the_chains_keep_the_rule_and_those_shares(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sampling, "_REJECTION_DRAWS", 10)
        region = DesignRegion(AMOUNTS, [AT_MOST_TWO])
        points = region.sample(20_000, np.random.default_rng(1))
        assert region.feasible(points).all()
        used = points > -1 + 1e-9
        shares = np.bincount(used.sum(axis=1), minlength=3) / len(points)
        assert shares == pytest.approx([1 / 11, 4 / 11, 6 / 11], abs=0.04)

    def test_the_rule_holds_at_each_combination_of_levels(self) -> None:
        factors = [*AMOUNTS, Factor(name="kind", type="categorical", levels=["x", "y"])]
        region = DesignRegion(factors, [AT_MOST_TWO, "not (kind == 'y' and A > 5)"])
        one = region.given(kind="y")
        assert [rule.expression for rule in one.counts] == [AT_MOST_TWO]
        np.testing.assert_array_equal(
            one.feasible(np.array([[0.0, 1.0, -1.0, -1.0], [0.0, 1.0, 1.0, -1.0]])), [True, False]
        )
        frame = region.sample_frame(500, np.random.default_rng(2))
        assert region.feasible(frame[NAMES].to_numpy(dtype=float), labels=frame).all()


def _model(coefficients: dict[str, float], names: list[str]) -> dict:
    """Return a fitted model as ``optimize_responses`` takes it."""
    return {
        "response_name": "y",
        "factor_names": names,
        "coefficients": [{"term": term, "coefficient": value} for term, value in coefficients.items()],
    }


class TestSearch:
    """The optimum uses a number of factors that the rule allows."""

    @pytest.mark.parametrize(
        ("rule", "amounts", "response"),
        [
            ("count_nonzero(A, B, C, D) <= 1", [10, 0, 0, 0], 9.5),
            ("count_nonzero(A, B, C, D) <= 2", [10, 10, 0, 0], 13.5),
            ("count_nonzero(A, B) == 1", [10, 0, 10, 10], 12.5),
            ("count_nonzero(B, C, D) < 1", [10, 0, 0, 0], 9.5),
        ],
    )
    def test_the_best_factors_the_rule_allows_are_used(self, rule: str, amounts: list[float], response: float) -> None:
        # Coded: y = 10 + 3 A + 2 B + C + 0.5 D, so each factor in use adds to it and A adds the most.
        model = _model({"Intercept": 10.0, "A": 3.0, "B": 2.0, "C": 1.0, "D": 0.5}, NAMES)
        goals = [{"response": "y", "goal": "maximize", "low": 0.0, "high": 30.0}]
        found = optimize_responses(
            [model],
            goals=goals,
            region=DesignRegion(AMOUNTS, [rule]),
            factor_ranges={name: {"low": 0, "high": 10} for name in NAMES},
        )["desirability"]
        assert [found["optimal_actual"][name] for name in NAMES] == pytest.approx(amounts, abs=1e-5)
        assert found["predicted_responses"]["y"] == pytest.approx(response, abs=1e-5)
        assert found["within_region"] is True


SHARES = [Factor(name=name, type="mixture", min_when_used=0.1) for name in ["a", "b", "c", "d", "e", "f"]]
BLEND = [f.name for f in SHARES]
TWO_OR_THREE = "2 <= count_nonzero(a, b, c, d, e, f) <= 3"


class TestMixtures:
    """A blend uses a number of components that the rule allows."""

    @pytest.mark.parametrize(("model_type", "budget"), [("scheffe_linear", 10), ("scheffe_quadratic", 26)])
    def test_a_blend_uses_two_or_three_components(self, model_type: str, budget: int) -> None:
        result = generate_design(
            SHARES, budget=budget, model_type=model_type, constraints=[TWO_OR_THREE], random_state=1
        )
        blends = result.design_actual[BLEND].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
        used = (blends > 1e-9).sum(axis=1)
        assert used.min() >= 2
        assert used.max() <= 3
        assert not np.any((blends > 1e-9) & (blends < 0.1 - 1e-9))
        assert DesignRegion.from_dict(result.metadata["region"]).feasible(blends).all()

    def test_without_a_budget_the_design_is_d_optimal(self) -> None:
        result = generate_design(SHARES, model_type="scheffe_linear", constraints=[TWO_OR_THREE], random_state=1)
        assert result.metadata["method"] == "d_optimal_extreme_vertices"
        assert len(result.design_actual) == 11  # 6 terms and 5 more runs
        used = (result.design_actual[BLEND].to_numpy(dtype=float) > 1e-9).sum(axis=1)
        assert set(used) <= {2, 3}

    def test_the_candidate_blends_keep_the_rule(self) -> None:
        blends = candidate_blends(SHARES, as_constraints([TWO_OR_THREE, "a + b <= 0.8"]))
        used = (blends > 1e-9).sum(axis=1)
        assert len(blends) > 30
        assert set(used) == {2, 3}
        assert (blends[:, 0] + blends[:, 1]).max() <= 0.8 + 1e-9

    def test_supplied_blends_that_break_the_rule_are_dropped(self) -> None:
        blends = pd.DataFrame(
            [
                [1.0, 0, 0, 0, 0, 0],  # one component
                [0.7, 0.3, 0, 0, 0, 0],
                [0, 0.5, 0.5, 0, 0, 0],
                [0, 0, 0.5, 0.5, 0, 0],
                [0, 0, 0, 0.5, 0.5, 0],
                [0, 0, 0, 0, 0.5, 0.5],
                [0.5, 0, 0, 0, 0, 0.5],
                [0.25, 0.25, 0.25, 0.25, 0, 0],  # four components
            ],
            columns=BLEND,
        )
        result = generate_design(
            SHARES,
            design_type="d_optimal",
            budget=8,
            model_type="scheffe_linear",
            constraints=[TWO_OR_THREE],
            candidates=blends,
            random_state=1,
        )
        assert result.metadata["n_candidates"] == 6
        assert set((result.design_actual[BLEND].to_numpy(dtype=float) > 1e-9).sum(axis=1)) == {2}

    def test_no_blend_left_is_refused(self) -> None:
        tight = [Factor(name="a", type="mixture"), Factor(name="b", type="mixture"), Factor(name="c", type="mixture")]
        with pytest.raises(ValueError, match="uses a number of components that a count_nonzero limit does not allow"):
            generate_design(
                tight,
                budget=4,
                model_type="scheffe_linear",
                constraints=["count_nonzero(a, b, c) <= 1", "a <= 0.6", "b <= 0.6", "c <= 0.6"],
            )

    def test_the_region_knows_the_rule_and_is_not_sampled(self) -> None:
        region = DesignRegion(SHARES, [TWO_OR_THREE])
        blends = np.array([[1.0, 0, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25, 0, 0]])
        np.testing.assert_array_equal(region.feasible(blends), [False, True, False])
        assert region.feasible(region.support_points()).all()
        with pytest.raises(ValueError, match="cannot be sampled"):
            region.sample(5, np.random.default_rng(1))

    def test_the_optimum_uses_two_components(self) -> None:
        # y = 9 a + 7 b + 5 c + 3 d + e + f: all of a is best, and the rule asks for a second component, at 0.1.
        model = _model({"a": 9.0, "b": 7.0, "c": 5.0, "d": 3.0, "e": 1.0, "f": 1.0}, BLEND)
        goals = [{"response": "y", "goal": "maximize", "low": 0.0, "high": 12.0}]
        found = optimize_responses([model], goals=goals, region=DesignRegion(SHARES, [TWO_OR_THREE]))["desirability"]
        assert found["optimal_coded"] == pytest.approx({"a": 0.9, "b": 0.1, "c": 0, "d": 0, "e": 0, "f": 0}, abs=1e-4)
        assert found["within_region"] is True

    def test_blends_beside_a_process_factor_keep_both_rules(self) -> None:
        factors = [*SHARES, *AMOUNTS[:2], Factor(name="T", low=20, high=80)]
        limits = [TWO_OR_THREE, "count_nonzero(A, B) <= 1"]
        result = generate_design(factors, budget=20, model_type="main_effects", constraints=limits, random_state=1)
        blends = result.design_actual[BLEND].to_numpy(dtype=float)
        assert set((blends > 1e-9).sum(axis=1)) <= {2, 3}
        assert _used(result, ["A", "B"]).max() <= 1

    def test_blends_beside_a_process_factor_can_all_break_the_rule(self) -> None:
        tight = [
            Factor(name="a", type="mixture"),
            Factor(name="b", type="mixture"),
            Factor(name="c", type="mixture"),
            Factor(name="T", low=20, high=80),
        ]
        limits = ["count_nonzero(a, b, c) <= 1", "a <= 0.6", "b <= 0.6", "c <= 0.6"]
        with pytest.raises(ValueError, match="uses a number of components that a count_nonzero limit does not allow"):
            generate_design(tight, budget=8, model_type="main_effects", constraints=limits)

    @pytest.mark.slow
    def test_a_blend_of_many_components_in_groups_keeps_the_rule(self) -> None:
        names = [f"G{g}_{i}" for g in range(4) for i in range(6)]
        groups = [names[6 * g : 6 * g + 6] for g in range(4)]
        factors = [Factor(name=name, type="mixture", high=0.40, min_when_used=0.04) for name in names]
        limits = [f"0.12 <= {' + '.join(group)} <= 0.40" for group in groups]
        limits.append(f"6 <= count_nonzero({', '.join(names)}) <= 10")
        result = generate_design(factors, budget=40, model_type="main_effects", constraints=limits, random_state=1)
        blends = result.design_actual[names].to_numpy(dtype=float)
        np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
        used = (blends > 1e-9).sum(axis=1)
        assert used.min() >= 6
        assert used.max() <= 10
        assert not np.any((blends > 1e-9) & (blends < 0.04 - 1e-9))
        for group in groups:
            share = result.design_actual[group].sum(axis=1)
            assert share.min() >= 0.12 - 1e-9
            assert share.max() <= 0.40 + 1e-9
        assert np.linalg.matrix_rank(blends) == 24
