"""Tests for constraints that name categorical factors: disallowed combinations and limits on some levels only."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import (
    Constraint,
    DesignRegion,
    Factor,
    augment_design,
    compare_designs,
    evaluate_design,
    generate_design,
)
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
from process_improve.experiments.factor import DesignResult

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
        with pytest.raises(ValueError, match="No candidate the constraints allow has catalyst = 'C'"):
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


# ---------------------------------------------------------------------------
# The region, and evaluating a design over it
# ---------------------------------------------------------------------------

RULES = [NO_B_WITH_Y, B_UP_TO_180]


@pytest.fixture(scope="module")
def region() -> DesignRegion:
    return DesignRegion(FACTORS, [*RULES, "3*T + 5*P <= 1000"])


class TestRegion:
    def test_allowed_combinations_lists_each_with_its_limits(self, region: DesignRegion) -> None:
        table = region.allowed_combinations()
        assert [tuple(row) for row in table[["catalyst", "solvent"]].to_numpy()] == [
            ("A", "X"),
            ("A", "Y"),
            ("B", "X"),
            ("C", "X"),
            ("C", "Y"),
        ]
        assert table["limits"].tolist() == [[], [], ["T <= 180"], [], []]

    def test_given_returns_the_region_of_one_combination(self, region: DesignRegion) -> None:
        assert region.given(catalyst="B", solvent="X").implied_bounds() == {"T": (150.0, 180.0), "P": (1.0, 5.0)}
        assert region.given(catalyst="A", solvent="Y").implied_bounds() == {"T": (150.0, 200.0), "P": (1.0, 5.0)}

    def test_given_refuses_a_disallowed_combination(self, region: DesignRegion) -> None:
        with pytest.raises(ValueError, match="do not allow the combination"):
            region.given(catalyst="B", solvent="Y")

    def test_feasible_and_broken_limits_read_the_levels(self, region: DesignRegion) -> None:
        x = np.array([[1.0, -1.0], [0.0, 0.0], [1.0, -1.0]])  # T = 200, 175, 200
        labels = {"catalyst": ["B", "B", "A"], "solvent": ["X", "Y", "Y"]}
        assert region.feasible(x, labels=labels).tolist() == [False, False, True]
        assert region.broken_limits(x, labels=labels) == [[B_UP_TO_180], [NO_B_WITH_Y], []]

    def test_feasible_needs_the_levels(self, region: DesignRegion) -> None:
        with pytest.raises(ValueError, match="pass each point's levels"):
            region.feasible(np.zeros((1, 2)))

    def test_sample_frame_weighs_each_allowed_combination_equally(self, region: DesignRegion) -> None:
        frame = region.sample_frame(10_000, np.random.default_rng(0))
        shares = frame.groupby(["catalyst", "solvent"]).size() / len(frame)
        assert len(shares) == 5
        assert np.allclose(shares, 0.2, atol=0.02)
        assert region.feasible(frame[region.names].to_numpy(), labels=frame).all()
        temperature = 175 + 25 * frame["T"]
        assert temperature[frame["catalyst"] == "B"].max() <= 180 + 1e-9
        assert temperature[frame["catalyst"] != "B"].max() > 195

    def test_support_frame_holds_each_combination(self, region: DesignRegion) -> None:
        frame = region.support_frame()
        assert set(zip(frame["catalyst"], frame["solvent"], strict=True)) == {
            ("A", "X"),
            ("A", "Y"),
            ("B", "X"),
            ("C", "X"),
            ("C", "Y"),
        }
        assert region.feasible(frame[region.names].to_numpy(), labels=frame).all()

    @pytest.mark.parametrize("method", ["vertices", "is_empty", "implied_bounds", "volume_fraction"])
    def test_a_diagnostic_of_one_region_points_to_given(self, region: DesignRegion, method: str) -> None:
        with pytest.raises(ValueError, match=r"region\.given\(catalyst='A', solvent='X'\)"):
            getattr(region, method)()

    def test_sample_and_support_points_point_to_the_frames(self, region: DesignRegion) -> None:
        with pytest.raises(ValueError, match="sample_frame"):
            region.sample(3, np.random.default_rng(0))
        with pytest.raises(ValueError, match="support_frame"):
            region.support_points()

    def test_linear_forms_gives_none_for_a_rule(self, region: DesignRegion) -> None:
        assert region.linear_forms() == [None, None, ({"T": 3.0, "P": 5.0}, None, 1000.0)]

    def test_the_region_survives_a_round_trip(self, region: DesignRegion) -> None:
        again = DesignRegion.from_dict(region.to_dict())
        assert again.allowed_combinations().equals(region.allowed_combinations())

    def test_rules_beside_mixture_components_are_refused(self) -> None:
        parts = [Factor(name=n, type="mixture") for n in ("a", "b", "c")]
        with pytest.raises(ValueError, match="not supported beside mixture components"):
            DesignRegion([*parts, TEMP, CATALYST], ["not (catalyst == 'B' and T > 180)"])


class TestGenerateAndEvaluate:
    @pytest.fixture(scope="class")
    def design(self) -> DesignResult:
        return generate_design(FACTORS, design_type="d_optimal", budget=20, constraints=RULES, random_state=1)

    def test_the_design_records_its_rules_and_lost_columns(self, design: DesignResult) -> None:
        assert design.metadata["inestimable_columns"] == {"catalyst:solvent": 1}
        assert DesignRegion.from_dict(design.metadata["region"]).rules is not None

    def test_evaluation_drops_the_same_column_and_matches_the_engine(self, design: DesignResult) -> None:
        result = evaluate_design(design, metric="d_efficiency")
        assert "inestimable_columns" in result["notes"]
        p, n = 14, 20  # 15 interaction-model coefficients, one lost to the missing cell
        log_det = p * np.log(result["d_efficiency"] * n / 100)
        assert log_det == pytest.approx(design.metadata["log_det_information"])

    def test_a_factor_outside_the_region_is_held_at_the_designs_values(self, design: DesignResult) -> None:
        frame = design.design.copy()
        frame["extra"] = np.resize([-1.0, 1.0], len(frame))
        result = evaluate_design(frame, region=DesignRegion(FACTORS, RULES), metric="d_efficiency")
        assert result["d_efficiency"] is not None
        assert "inestimable_columns" in result["notes"]

    def test_the_average_variance_is_over_the_allowed_region(self) -> None:
        design = generate_design(FACTORS, design_type="i_optimal", budget=20, constraints=RULES, random_state=1)
        average = evaluate_design(design, metric="average_prediction_variance")["average_prediction_variance"]
        assert average == pytest.approx(design.metadata["trace_criterion"], rel=0.03)


class TestOtherEntryPoints:
    def test_a_design_that_cannot_follow_a_rule_refuses_it(self) -> None:
        with pytest.raises(ValueError, match="'full_factorial' cannot follow the constraints"):
            generate_design([TEMP, CATALYST, SOLVENT], design_type="full_factorial", constraints=[NO_B_WITH_Y])

    def test_a_design_beside_mixture_components_refuses_a_rule(self) -> None:
        parts = [Factor(name=n, type="mixture") for n in ("a", "b", "c")]
        with pytest.raises(ValueError, match="not supported beside mixture components"):
            generate_design([*parts, TEMP, CATALYST], design_type="d_optimal", constraints=[B_UP_TO_180])

    def test_the_automatic_choice_follows_the_rules(self) -> None:
        design = generate_design(FACTORS, budget=20, constraints=RULES, random_state=1)
        assert design.design_type == "d_optimal"
        pairs = set(zip(design.design["catalyst"], design.design["solvent"], strict=True))
        assert ("B", "Y") not in pairs

    def test_optimize_responses_points_to_one_combination(self, region: DesignRegion) -> None:
        from process_improve.experiments.optimization import _search_space

        with pytest.raises(ValueError, match=r"region\.given\(catalyst='A', solvent='X'\)"):
            _search_space(["T", "P"], None, region, np.random.default_rng(0), 3)


# ---------------------------------------------------------------------------
# compare_designs and augment_design work over one region of numeric settings (#783)
# ---------------------------------------------------------------------------

_SQUARE = pd.DataFrame({"T": [-1.0, 1, -1, 1, 0], "P": [-1.0, -1, 1, 1, 0]})
_THIN = pd.DataFrame({"T": [-1.0, 0.2, -1, 0.2], "P": [-1.0, -1, 1, 1]})


def test_compare_designs_refuses_a_region_with_rules_and_names_the_region_of_one_level():
    """The designs are numeric, so they hold no level to choose a region by: the message says how to compare."""
    region = DesignRegion([TEMP, PRESSURE, CATALYST], [B_UP_TO_180])
    with pytest.raises(ValueError, match=r"limits on \['catalyst'\].*region\.given\(catalyst='A'\)"):
        compare_designs({"square": _SQUARE, "thin": _THIN}, "main_effects", region=region)


def test_the_region_of_one_level_is_compared_as_its_numeric_limits_are():
    """At catalyst B the rule leaves T <= 180, and the comparison over that region is the one over the limit itself."""
    region = DesignRegion([TEMP, PRESSURE, CATALYST], [B_UP_TO_180])
    designs = {"square": _SQUARE, "thin": _THIN}
    at_b = compare_designs(designs, "main_effects", region=region.given(catalyst="B")).to_dict()
    plain = compare_designs(designs, "main_effects", region=DesignRegion([TEMP, PRESSURE], ["T <= 180"])).to_dict()
    assert at_b == plain


def test_augment_design_refuses_a_region_with_rules_and_names_the_region_of_one_level():
    """augment_design works in coded numeric factors; its message points to the region at one level."""
    region = DesignRegion([TEMP, PRESSURE, CATALYST], [B_UP_TO_180])
    with pytest.raises(ValueError, match=r"are categorical\. Augment the runs at one level, over region\.given\("):
        augment_design(_SQUARE, "add_center_points", region=region)
