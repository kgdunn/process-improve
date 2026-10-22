"""Tests for constraints that name categorical factors: disallowed combinations and limits on some levels only."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Constraint, Factor
from process_improve.experiments._rules import MAX_CLAUSES, compile_rules
from process_improve.experiments.designs_constrained import (
    _FEASIBILITY_TOL,
    ConstrainedOptions,
    Criterion,
    PolishLattice,
    build_candidates,
    constrained_optimal_design,
    fedorov_exchange,
    model_matrix,
    polish_runs,
    region_with_rules,
)

LEVELS = {"catalyst": ["A", "B", "C"], "solvent": ["X", "Y"]}
TEMP = Factor(name="T", low=150, high=200)
PRESSURE = Factor(name="P", low=1, high=5)
CATALYST = Factor(name="catalyst", type="categorical", levels=["A", "B", "C"])
SOLVENT = Factor(name="solvent", type="categorical", levels=["X", "Y"])
FACTORS = [TEMP, PRESSURE, CATALYST, SOLVENT]
NO_B_WITH_Y = "not (catalyst == 'B' and solvent == 'Y')"
B_UP_TO_180 = "not (catalyst == 'B' and T > 180)"


def _compile(*expressions: str, levels: dict | None = None) -> tuple:
    """Compile ``expressions`` over T, P and the categorical ``levels``."""
    return compile_rules([Constraint(expression=e) for e in expressions], ["T", "P"], levels or LEVELS)


def _allowed(*expressions: str) -> list[tuple[str, ...]]:
    """Return the combinations the rules allow, as tuples of levels."""
    _plain, rules = _compile(*expressions)
    table, _ = rules.allowed_table()
    return [tuple(row) for row in table.to_numpy()]


def _limits(*expressions: str) -> list[tuple[str, list[tuple[str, ...]]]]:
    """Return each conditional limit and the combinations it applies on."""
    _plain, rules = _compile(*expressions)
    table, applies = rules.allowed_table()
    rows = [tuple(row) for row in table.to_numpy()]
    return [
        (clause.limit, [row for row, on in zip(rows, applies[:, j], strict=True) if on])
        for j, clause in enumerate(rules.limits)
    ]


def _design(*expressions: str, **options: object) -> tuple[pd.DataFrame, dict]:
    """Build a design over FACTORS, returned in actual units for T and P."""
    budget = int(options.pop("budget", 20))  # type: ignore[call-overload]
    seed = options.pop("random_state", 1)
    matrix, meta = constrained_optimal_design(
        FACTORS,
        budget,
        [Constraint(expression=e) for e in expressions],
        ConstrainedOptions(**options),  # type: ignore[arg-type]
        random_state=seed,  # type: ignore[arg-type]
    )
    design = pd.DataFrame(matrix, columns=[f.name for f in FACTORS])
    design["T"] = 175 + 25 * design["T"].astype(float)
    design["P"] = 3 + 2 * design["P"].astype(float)
    return design, meta


# ---------------------------------------------------------------------------
# Reading the rules
# ---------------------------------------------------------------------------


class TestCompile:
    def test_a_disallowed_pair_removes_one_combination(self) -> None:
        assert _allowed(NO_B_WITH_Y) == [("A", "X"), ("A", "Y"), ("B", "X"), ("C", "X"), ("C", "Y")]

    @pytest.mark.parametrize(
        "expression",
        [
            "not (catalyst == 'B' and solvent == 'Y')",
            "catalyst != 'B' or solvent != 'Y'",
            "not ('B' == catalyst and 'Y' == solvent)",
            "catalyst in ('A', 'C') or solvent not in ['Y']",
            "not (catalyst not in {'A', 'C'} and solvent in ('Y',))",
        ],
    )
    def test_every_way_of_writing_the_pair_reads_the_same(self, expression: str) -> None:
        assert _allowed(expression) == _allowed(NO_B_WITH_Y)

    def test_a_limit_under_not_applies_on_its_levels_only(self) -> None:
        assert _limits(B_UP_TO_180) == [("T <= 180", [("B",)])]

    def test_a_limit_joined_by_and_gives_one_limit_each(self) -> None:
        assert _limits("catalyst != 'C' or (T <= 170 and P <= 4)") == [
            ("T <= 170", [("C",)]),
            ("P <= 4", [("C",)]),
        ]

    def test_limits_without_levels_apply_everywhere(self) -> None:
        plain, rules = _compile("T <= 170 and P <= 4", "3*T + 5*P <= 1000")
        assert plain == [(0, "T <= 170"), (0, "P <= 4"), (1, "3*T + 5*P <= 1000")]
        assert rules is None

    def test_a_clause_that_always_holds_is_dropped(self) -> None:
        plain, rules = _compile("catalyst in ('A', 'B', 'C') or T <= 160")
        assert (plain, rules) == ([], None)

    def test_numeric_levels_compare_by_their_text(self) -> None:
        plain, rules = _compile("not (grade == -1 and T > 180)", levels={"grade": [-1, 0, 1]})
        assert plain == []
        table, applies = rules.allowed_table()
        assert table["grade"].tolist() == ["-1", "0", "1"]
        assert applies[:, 0].tolist() == [True, False, False]

    def test_a_negated_range_is_refused_as_not_convex(self) -> None:
        with pytest.raises(ValueError, match="not form a convex region"):
            _compile("not (150 <= T <= 170) or catalyst == 'A'")

    def test_two_limits_joined_by_or_are_refused(self) -> None:
        with pytest.raises(ValueError, match="'T <= 190' or 'P <= 3'"):
            _compile("not (catalyst in ('A', 'C') and T > 190 and P > 3)")

    def test_a_level_typo_is_refused_with_the_levels(self) -> None:
        with pytest.raises(ValueError, match=r"has no level 'b'; its levels are \['A', 'B', 'C'\]"):
            _compile("not (catalyst == 'b' and T > 180)")

    @pytest.mark.parametrize("expression", ["catalyst < 'C'", "catalyst + T <= 3", "catalyst == solvent"])
    def test_a_categorical_factor_is_compared_with_its_levels_only(self, expression: str) -> None:
        with pytest.raises(ValueError, match="compare the categorical factor"):
            _compile(expression)

    def test_a_bare_name_is_not_a_comparison(self) -> None:
        with pytest.raises(ValueError, match="is not a comparison"):
            _compile("catalyst == 'A' or T")

    def test_a_level_with_no_allowed_run_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no allowed run with catalyst = 'C'"):
            _compile("catalyst != 'C'")

    def test_rules_that_allow_nothing_are_refused(self) -> None:
        with pytest.raises(ValueError, match="disallow every combination"):
            _compile("catalyst == 'A'", "catalyst == 'B'")

    def test_a_long_expansion_is_refused(self) -> None:
        pairs = [f"(T <= {150 + i} and P <= {1 + i / 10})" for i in range(7)]  # 2**7 clauses
        assert MAX_CLAUSES < 2**7
        with pytest.raises(ValueError, match="clauses"):
            _compile(" or ".join(pairs))

    def test_a_plain_inequality_is_passed_on_as_written(self) -> None:
        plain, rules = _compile("3*T + 5*P <= 1000")
        assert (plain, rules) == ([(0, "3*T + 5*P <= 1000")], None)


# ---------------------------------------------------------------------------
# Designs that follow the rules
# ---------------------------------------------------------------------------


class TestDesign:
    @pytest.mark.parametrize("criterion", ["d_optimal", "i_optimal"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_no_run_breaks_a_rule(self, criterion: str, seed: int) -> None:
        design, _meta = _design(NO_B_WITH_Y, B_UP_TO_180, criterion=criterion, random_state=seed)
        pairs = set(zip(design["catalyst"], design["solvent"], strict=True))
        assert ("B", "Y") not in pairs
        assert design.loc[design["catalyst"] == "B", "T"].max() <= 180 + 1e-9
        assert design.loc[design["catalyst"] != "B", "T"].max() == pytest.approx(200)

    def test_a_missing_cell_drops_the_interaction_column_it_costs(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _frame, meta = _design(NO_B_WITH_Y)
        assert meta["inestimable_columns"] == {"catalyst:solvent": 1}
        assert "no allowed run has catalyst = 'B' with solvent = 'Y'" in caplog.text

    def test_a_main_effects_model_loses_nothing(self) -> None:
        _design_frame, meta = _design(NO_B_WITH_Y, model_type="main_effects", budget=10)
        assert "inestimable_columns" not in meta

    def test_a_level_the_limits_leave_empty_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No setting the constraints allow has catalyst = 'C'"):
            _design("not (catalyst == 'C' and T > 140)")

    def test_a_fixed_run_in_a_disallowed_cell_is_counted_outside(self) -> None:
        fixed = pd.DataFrame({"T": [0.0], "P": [0.0], "catalyst": ["B"], "solvent": ["Y"]})
        _frame, meta = _design(NO_B_WITH_Y, fixed_runs=fixed)
        assert meta["n_fixed_runs_outside_region"] == 1

    def test_candidates_that_break_a_rule_are_dropped(self) -> None:
        grid = list(itertools.product([150.0, 200.0], [1.0, 5.0], LEVELS["catalyst"], LEVELS["solvent"]))
        candidates = pd.DataFrame(grid, columns=["T", "P", "catalyst", "solvent"])
        _frame, meta = _design(NO_B_WITH_Y, B_UP_TO_180, candidates=candidates, model_type="main_effects")
        # (B, Y) at 4 settings, and B with X at T = 200 at 2 settings
        assert meta["n_candidates_infeasible"] == 6

    def test_the_d_optimum_matches_a_search_of_every_design(self) -> None:
        """With categorical factors only, the 5 allowed cells are the candidates: every 6-run design is listed."""
        factors = [CATALYST, SOLVENT]
        cells = [("A", "X"), ("A", "Y"), ("B", "X"), ("C", "X"), ("C", "Y")]
        region = region_with_rules(factors, [Constraint(expression=NO_B_WITH_Y)])
        index = {level: i for i, level in enumerate(LEVELS["catalyst"])}, {"X": 0, "Y": 1}
        cats = np.array([[index[0][a], index[1][b]] for a, b in cells])
        f = model_matrix(region, np.empty((5, 0)), cats, "main_effects")
        best = max(
            np.linalg.slogdet(f[list(rows)].T @ f[list(rows)])[1]
            for rows in itertools.combinations_with_replacement(range(5), 6)
        )
        matrix, meta = constrained_optimal_design(
            factors,
            6,
            [Constraint(expression=NO_B_WITH_Y)],
            ConstrainedOptions(model_type="main_effects"),
            random_state=0,
        )
        chosen = np.array([[index[0][a], index[1][b]] for a, b in matrix])
        x = model_matrix(region, np.empty((6, 0)), chosen, "main_effects")
        assert np.linalg.slogdet(x.T @ x)[1] == pytest.approx(best)
        assert meta["optimality_criterion"] == "d_optimal"

    def test_polish_keeps_a_limit_on_its_levels(self) -> None:
        region = region_with_rules(FACTORS, [Constraint(expression=B_UP_TO_180)])
        coded, cats, _ = build_candidates(region, n_levels=3)
        f = model_matrix(region, coded, cats, "quadratic")
        criterion = Criterion.d()
        no_fixed = np.empty((0, f.shape[1]))
        rows, _value = fedorov_exchange(f, 20, no_fixed, np.random.default_rng(0), criterion)
        lattice = PolishLattice(
            5,
            lambda c, k: model_matrix(region, c, k, "quadratic"),
            lambda c, k: region.slack(c, k) <= _FEASIBILITY_TOL,
        )
        polished, kept_cats, _n_moves = polish_runs(coded[rows], cats[rows], no_fixed, criterion, lattice)
        assert (region.slack(polished, kept_cats) <= _FEASIBILITY_TOL).all()

    def test_the_slack_needs_the_levels(self) -> None:
        region = region_with_rules(FACTORS, [Constraint(expression=B_UP_TO_180)])
        with pytest.raises(ValueError, match="needs each run's levels"):
            region.slack(np.zeros((1, 2)))
