"""Tests for recommend_next_step: the next round of a designed experiment from its results."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import process_improve.experiments.tools  # noqa: F401 - registers the experiments tools
from process_improve.experiments import Factor, TrueProcess, generate_design, recommend_next_step
from process_improve.experiments.next_step import (
    _CURVATURE,
    ACTIONS,
    _code,
    _design,
    _lenth_null,
    _select,
    _stands_out,
    _Study,
    _with_blocks,
)
from process_improve.tool_spec import execute_tool_call

ADHESION = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)
CURVED = TrueProcess({"Intercept": 60, "A": 2, "B": -1.5, "I(A ** 2)": -6, "I(B ** 2)": -4, "A:B": 1}, sigma=0.7)


def _factors(names: str) -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


def _runs(factors: list[Factor], truth: TrueProcess, seed: int, **design: object) -> pd.DataFrame:
    """Return a generated design's runs in actual units, with the truth's response as ``y``."""
    names = [f.name for f in factors]
    runs = generate_design(factors, **design).design_actual[names].astype(float)  # type: ignore[arg-type]
    return runs.assign(y=truth.simulate(runs, random_state=seed).to_numpy())


def _signal(factors: list[Factor], runs: pd.DataFrame) -> bool:
    """Say whether anything stands out in the runs, as recommend_next_step judges them, without deciding the step."""
    study = _Study(factors, _with_blocks(_code(factors, runs, "y")), "y")
    design = _design(study)
    study.data[_CURVATURE] = design.center.astype(float)
    return _select(study, design, None, 0.05).signal


def _made(step: dict, truth: TrueProcess, seed: int) -> pd.DataFrame:
    """Return the runs a step asks for, made on the truth."""
    runs = pd.DataFrame(step["runs"]).astype(float)
    return runs.assign(y=truth.simulate(runs, random_state=seed).to_numpy())


@pytest.fixture(scope="module")
def dsd_study() -> tuple[list[Factor], pd.DataFrame]:
    """Return an 8-factor definitive screening design with five centre runs, on the adhesion process."""
    factors = _factors("ABCDEFGH")
    return factors, _runs(factors, ADHESION, 1, design_type="dsd", n_center_points=5)


@pytest.fixture(scope="module")
def drifted() -> tuple[list[Factor], pd.DataFrame]:
    """Return a 2^3 with centre runs, then the axial runs it calls for, made after the process rose by 8."""
    factors = _factors("ABC")
    first = _runs(factors, CURVED, 2, design_type="full_factorial", n_center_points=3).assign(Round=1)
    step = recommend_next_step(factors, first, "y", block="Round")
    assert step["action"] == "augment_to_rsm"
    second = _made(step, CURVED, 3).assign(Round=2)
    second["y"] += 8.0
    return factors, pd.concat([first, second], ignore_index=True)


class TestScreeningResults:
    """What follows a screening design, by what its results show."""

    def test_dsd_finds_the_process_and_confirms_inside_fixed_ranges(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "confirm"
        assert step["evidence"]["active_terms"] == ["A", "C", "G", "A:C"]
        assert step["model"]["judged_by"].startswith("the staged analysis of a definitive screening design")
        assert step["n_runs"] == 3
        assert {k: step["runs"][0][k] for k in "ACG"} == {"A": 1.0, "C": 1.0, "G": -1.0}
        low, high = step["evidence"]["optimum"]["interval_for_mean"]
        assert low < step["evidence"]["optimum"]["predicted"] < high

    def test_dsd_moves_past_the_ranges_when_they_may_move(self, dsd_study: tuple) -> None:
        """The best settings sit on three edges, and the slope there is many times the noise."""
        factors, runs = dsd_study
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "move_region"
        assert step["n_runs"] == 4
        path = pd.DataFrame(step["runs"])
        assert (path["A"].diff().dropna() > 0).all()  # each run further out than the one before
        assert (path["G"].diff().dropna() < 0).all()
        assert len(step["path_predictions"]) == 4

    def test_plackett_burman_with_centre_runs_tests_the_interaction_it_cannot_alias_away(self) -> None:
        """Main effects alone would leave A:C in the error, where it hides the effects."""
        factors = _factors("ABCDEFGH")
        runs = _runs(factors, ADHESION, 7, design_type="plackett_burman", n_center_points=3)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "confirm"
        assert step["evidence"]["active_terms"] == ["A", "C", "G", "A:C"]
        assert "stepwise search (AICc)" in step["model"]["judged_by"]

    def test_resolution_iii_fraction_folds_over_an_active_effect_aliased_with_an_interaction(self) -> None:
        """In 2^(7-4) with D = AB, the D the analysis sees may be A:B: A and B are active, so A:B is plausible."""
        factors = _factors("ABCDEFG")
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": 5, "A:B": 4}, sigma=0.8)
        runs = _runs(factors, truth, 4, design_type="fractional_factorial", resolution=3, n_center_points=0)
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "de_alias"
        assert step["model"]["judged_by"].startswith("Lenth's method on the 7 contrasts")
        assert {"term": "D", "aliased_with": "A:B"} in [
            {k: a[k] for k in ("term", "aliased_with")} for a in step["evidence"]["aliasing"]
        ]
        switched = pd.DataFrame(step["runs"])
        assert len(switched) == 8
        assert set(map(tuple, switched.to_numpy())) == set(map(tuple, -runs[[f.name for f in factors]].to_numpy()))

    def test_curvature_at_the_centre_adds_axial_runs_in_every_factor(self) -> None:
        """Near an optimum a factor's linear effect vanishes but its square does not: the curvature could be C's."""
        factors = _factors("ABC")
        runs = _runs(factors, CURVED, 2, design_type="full_factorial", n_center_points=3)
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "augment_to_rsm"
        assert step["evidence"]["curvature"]["source"] == "center_runs"
        assert step["evidence"]["curvature"]["significant"]
        added = pd.DataFrame(step["runs"])
        axial = added[(added != 0).sum(axis=1) == 1]
        assert sorted(axial.abs().idxmax(axis=1).unique()) == ["A", "B", "C"]
        assert ((added.abs() <= 1).all()).all()  # face-centred: no run leaves the ranges

    def test_response_surface_confirms_an_interior_optimum(self) -> None:
        factors = _factors("ABC")
        runs = _runs(factors, CURVED, 3, design_type="ccd", alpha="face_centered", n_center_points=4)
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "confirm"
        optimum = step["evidence"]["optimum"]["coded"]
        assert abs(optimum["A"]) < 0.5  # inside the ranges, near the true 0.15, -0.17
        assert abs(optimum["B"]) < 0.5
        assert step["evidence"]["optimum"]["edges"] == []


class TestSquaresAliasedWithEachOther:
    """Centre runs give every factor's square one column: curvature in many factors is not yet anyone's."""

    CURVED_8 = TrueProcess({"Intercept": 100, "A": 2, "I(A ** 2)": -6, "I(C ** 2)": -4}, sigma=1)

    def test_runs_that_tell_every_square_apart_come_before_the_optimum(self) -> None:
        """With axial runs in A alone, B's square stood for C's, and studies chased B."""
        factors = _factors("ABCDEFGH")
        runs = _runs(factors, self.CURVED_8, 0, design_type="fractional_factorial", resolution=4, n_center_points=3)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "augment_to_rsm"
        assert "tell the squares of B, C, D, E, F, G, H apart" in step["headline"]
        added = pd.DataFrame(step["runs"])
        assert (added[list("BCDEFGH")] != 0).any().all()  # every suspect varies
        after = recommend_next_step(
            factors, pd.concat([runs, _made(step, self.CURVED_8, 1)]), "y", ranges_can_move=False
        )
        assert {"A", "I(A ** 2)", "I(C ** 2)"} <= set(after["evidence"]["active_terms"])
        assert after["action"] == "confirm"


class TestNothingStandsOut:
    """The gate on whether anything stands out from the noise, and what follows when nothing does."""

    def test_noise_widens_the_ranges_once_then_stops(self) -> None:
        factors = _factors("ABCD")
        runs = _runs(
            factors,
            TrueProcess({"Intercept": 20, "A": 0.1}, sigma=2),
            5,
            design_type="full_factorial",
            n_center_points=0,
        )
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "widen_ranges"
        assert step["evidence"]["active_terms"] == []
        assert step["new_ranges"] == {name: [-1.5, 1.5] for name in "ABCD"}
        again = recommend_next_step(factors, runs, "y", history=["widen_ranges"])
        assert again["action"] == "stop"
        assert again["headline"] == "Stop: no factor moves the response by more than the noise."

    def test_fixed_ranges_replicate_instead_of_widening(self) -> None:
        factors = _factors("ABCD")
        runs = _runs(
            factors, TrueProcess({"Intercept": 20}, sigma=2), 5, design_type="full_factorial", n_center_points=0
        )
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "replicate"
        assert step["n_runs"] == 16
        assert recommend_next_step(factors, runs, "y", ranges_can_move=False, history=["replicate"])["action"] == "stop"

    def test_an_unreplicated_two_by_two_gets_centre_runs(self) -> None:
        factors = _factors("AB")
        runs = _runs(
            factors,
            TrueProcess({"Intercept": 5, "A": 2, "B": 1}, sigma=0.5),
            6,
            design_type="full_factorial",
            n_center_points=0,
        )
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "add_center_points"
        assert step["runs"] == [{"A": 0.0, "B": 0.0}] * 3

    def test_too_few_error_degrees_of_freedom_add_runs_before_doubting_the_ranges(self) -> None:
        """A Plackett-Burman design of eight factors leaves its interactions too few runs to show a signal."""
        factors = _factors("ABCDEFGH")
        runs = _runs(
            factors,
            TrueProcess({"Intercept": 100, "A": 3, "C": 2}, sigma=3),
            3,
            design_type="plackett_burman",
            n_center_points=0,
        )
        step = recommend_next_step(factors, runs, "y")
        assert step["evidence"]["active_terms"] == []  # a seed in which A and C, at one noise sd, do not show
        assert step["evidence"]["error_df"] == 3
        assert step["action"] == "add_center_points"
        assert "degree(s) of freedom" in step["reasons"][0]

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("design", "n_draws"),
        [
            ({"design_type": "full_factorial", "n_center_points": 0}, 400),
            ({"design_type": "plackett_burman", "n_center_points": 3}, 400),
        ],
    )
    def test_the_false_alarm_rate_on_pure_noise_is_alpha(self, design: dict, n_draws: int) -> None:
        """The combined Bonferroni and F-test gate has its critical value simulated for the design."""
        names = "ABCD" if design["design_type"] == "full_factorial" else "ABCDEFGH"
        runs = generate_design(_factors(names), **design).design[list(names)].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(runs)), runs])
        rng = np.random.default_rng(0)
        alarms = sum(
            _stands_out(x, rng.standard_normal(len(runs)), list(range(1, x.shape[1])), 0.05) for _ in range(n_draws)
        )
        assert 0.02 < alarms / n_draws < 0.08


class TestLenth:
    """Lenth's method, with critical values simulated rather than read off a t distribution."""

    @pytest.mark.parametrize(("m", "published"), [(7, 2.30), (15, 2.16)])
    def test_individual_critical_values_match_ye_and_hamada(self, m: int, published: float) -> None:
        """Ye and Hamada (2000), Table 1: the individual error rate's critical values at 0.05."""
        one, *_ = _lenth_null(m)
        assert np.quantile(one, 0.95) == pytest.approx(published, abs=0.04)

    def test_an_8_run_factorial_finds_two_clear_effects(self) -> None:
        """Lenth's own simultaneous margin with seven contrasts (9.0 PSE) misses both; the pair of ratios does not."""
        factors = _factors("ABC")
        truth = TrueProcess({"Intercept": 50, "A": 2, "B": -1.5}, sigma=0.4)
        runs = _runs(factors, truth, 3, design_type="full_factorial", n_center_points=0)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["model"]["judged_by"].startswith("Lenth's method on the 7 contrasts")
        assert step["model"]["signal"]
        assert {"A", "B"} <= set(step["evidence"]["active_terms"])

    def test_a_fraction_with_three_centre_runs_is_judged_on_its_two_level_runs(self) -> None:
        """t-tests in the interactions model had 2 error df: nothing stood out in a quarter of adhesion studies."""
        factors = _factors("ABCDEFGH")
        design = {"design_type": "fractional_factorial", "resolution": 4, "n_center_points": 3}
        runs = _runs(factors, ADHESION, 1, **design)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["model"]["judged_by"].startswith("Lenth's method on the 15 contrasts of the two-level runs")
        assert {"A", "C", "G"} <= set(step["evidence"]["active_terms"])
        assert step["evidence"]["curvature"]["source"] == "center_runs"
        five = _runs(factors, ADHESION, 1, **(design | {"n_center_points": 5}))  # 4 error df: t-tests see enough
        assert recommend_next_step(factors, five, "y")["model"]["judged_by"].startswith("t-tests")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("truth", "low", "high"),
        [
            (TrueProcess({"Intercept": 100}, sigma=3), 0.02, 0.09),  # the false alarm rate: alpha
            (TrueProcess({"Intercept": 100, "I(A ** 2)": -6, "I(C ** 2)": -4}, sigma=1), 0.95, 1.0),
        ],
    )
    def test_the_curvature_of_the_centre_runs_counts_toward_the_signal(
        self, truth: TrueProcess, low: float, high: float
    ) -> None:
        """No term of the two-level runs moves here, so only the centre runs can show the squares."""
        factors = _factors("ABCDEFGH")
        design = generate_design(factors, design_type="fractional_factorial", resolution=4, n_center_points=3)
        coded = design.design[[f.name for f in factors]]
        signals = [_signal(factors, coded.assign(y=truth.simulate(coded, random_state=seed))) for seed in range(300)]
        assert low <= np.mean(signals) <= high


class TestConfirmation:
    """Confirmation runs, judged by their mean against the interval for a mean of that many runs."""

    @pytest.fixture
    def confirm_step(self, dsd_study: tuple) -> dict:
        factors, runs = dsd_study
        return recommend_next_step(factors, runs, "y", ranges_can_move=False)

    def test_runs_that_agree_stop_the_study(self, dsd_study: tuple, confirm_step: dict) -> None:
        factors, runs = dsd_study
        made = _made(confirm_step, ADHESION, 11)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, confirmation_runs=made)
        assert step["action"] == "stop"
        verdict = step["evidence"]["confirmation"][0]
        assert verdict["within_PI"]
        assert verdict["at_optimum"]
        assert verdict["n_runs"] == 3

    def test_runs_that_agree_still_move_on_when_the_ranges_may_move(self, dsd_study: tuple, confirm_step: dict) -> None:
        factors, runs = dsd_study
        made = _made(confirm_step, ADHESION, 11)
        assert recommend_next_step(factors, runs, "y", confirmation_runs=made)["action"] == "move_region"

    def test_a_small_miss_is_confirmed_again_and_judged_on_its_own(self, dsd_study: tuple, confirm_step: dict) -> None:
        factors, runs = dsd_study
        _, high = confirm_step["evidence"]["optimum"]["interval_for_mean"]
        made = _made(confirm_step, ADHESION, 11)
        made["y"] += high - made["y"].mean() + 1.0  # just past the interval
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, confirmation_runs=made)
        assert step["action"] == "confirm"
        assert step["headline"].startswith("Repeat the confirmation")
        assert "with only these new runs as the confirmation runs" in step["after"]

    def test_a_second_miss_calls_for_a_look_at_the_process(self, dsd_study: tuple, confirm_step: dict) -> None:
        factors, runs = dsd_study
        _, high = confirm_step["evidence"]["optimum"]["interval_for_mean"]
        made = _made(confirm_step, ADHESION, 11)
        made["y"] += high - made["y"].mean() + 1.0
        step = recommend_next_step(
            factors, runs, "y", ranges_can_move=False, confirmation_runs=made, history=["confirm", "confirm"]
        )
        assert step["action"] == "investigate"

    def test_a_gross_miss_calls_for_a_look_at_the_process(self, dsd_study: tuple, confirm_step: dict) -> None:
        factors, runs = dsd_study
        made = _made(confirm_step, ADHESION, 11).assign(y=lambda d: d["y"] - 25)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, confirmation_runs=made)
        assert step["action"] == "investigate"
        assert step["n_runs"] == 0

    def test_a_miss_on_a_two_level_design_adds_what_the_design_could_not_show(self) -> None:
        """Axial runs first: the corners and centre cannot say which factor curves."""
        factors = _factors("ABC")
        truth = TrueProcess({"Intercept": 50, "A": 5, "B": 3}, sigma=0.5)
        runs = _runs(factors, truth, 8, design_type="full_factorial", n_center_points=3)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "confirm"
        made = _made(step, truth, 1).assign(y=lambda d: d["y"] - 3)
        after = recommend_next_step(factors, runs, "y", ranges_can_move=False, confirmation_runs=made)
        assert after["action"] == "augment_to_rsm"
        assert after["reasons"][0].startswith("The confirmation runs' mean")


class TestHardLimits:
    """A range that is a limit of the process is never moved or widened."""

    def test_the_path_moves_only_the_factors_without_hard_limits(self, dsd_study: tuple) -> None:
        """G would improve past its low end, as A and C past their high ends."""
        _, runs = dsd_study
        factors = [Factor(name=n, low=-1, high=1, hard_limits=n == "G") for n in "ABCDEFGH"]
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "move_region"
        assert set(step["direction"]) == {"A", "C"}
        assert (pd.DataFrame(step["runs"])["G"] == -1).all()
        assert {"factor": "G", "end": "low", "improves_beyond": True, "hard_limit": True} in step["evidence"][
            "optimum"
        ]["edges"]
        assert "past the end of G, but its range is a hard limit: it stays at that end." in step["reasons"][-1]

    def test_with_every_edge_at_a_hard_limit_the_optimum_is_confirmed(self, dsd_study: tuple) -> None:
        _, runs = dsd_study
        factors = [Factor(name=n, low=-1, high=1, hard_limits=n in "ACG") for n in "ABCDEFGH"]
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "confirm"
        assert step["reasons"][-1].startswith("The response would improve past the ends of A, C, G")

    def test_noise_widens_only_the_ranges_without_hard_limits(self) -> None:
        factors = [Factor(name=n, low=-1, high=1, hard_limits=n == "B") for n in "ABCD"]
        noise = TrueProcess({"Intercept": 20, "A": 0.1}, sigma=2)
        runs = _runs(factors, noise, 5, design_type="full_factorial", n_center_points=0)
        step = recommend_next_step(factors, runs, "y")
        assert step["action"] == "widen_ranges"
        assert step["new_ranges"] == {name: [-1.5, 1.5] for name in "ACD"}
        assert pd.DataFrame(step["runs"]).abs().max().to_dict() == {"A": 1.5, "B": 1.0, "C": 1.5, "D": 1.5}
        assert "The range of B is a hard limit, and stays as it is." in step["reasons"]
        held = [Factor(name=n, low=-1, high=1, hard_limits=True) for n in "ABCD"]
        assert recommend_next_step(held, runs, "y")["action"] == "replicate"

    def test_the_tool_takes_hard_limits_in_the_factors(self, dsd_study: tuple) -> None:
        _, runs = dsd_study
        factors = [{"name": n, "low": -1, "high": 1, "hard_limits": n == "G"} for n in "ABCDEFGH"]
        reply = execute_tool_call(
            "recommend_next_step",
            {"factors": factors, "runs": runs.to_dict(orient="records"), "response_column": "y"},
        )
        assert set(reply["direction"]) == {"A", "C"}


class TestOtherGoals:
    def test_a_target_is_met_nearest_the_middle_of_the_ranges(self) -> None:
        factors = _factors("ABC")
        truth = TrueProcess({"Intercept": 50, "A": 5, "B": 3}, sigma=0.5)
        runs = _runs(factors, truth, 8, design_type="full_factorial", n_center_points=3)
        step = recommend_next_step(factors, runs, "y", goal="target", target=52)
        assert step["action"] == "confirm"
        assert step["evidence"]["optimum"]["predicted"] == pytest.approx(52, abs=1e-3)
        coded = step["evidence"]["optimum"]["coded"]
        assert np.hypot(coded["A"], coded["B"]) < 0.5  # the line A, B meeting 52 passes 0.34 from the middle

    def test_minimize_takes_the_other_ends(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        step = recommend_next_step(factors, runs, "y", goal="minimize", ranges_can_move=False)
        coded = step["evidence"]["optimum"]["coded"]
        assert coded["G"] == 1.0
        assert coded["A"] * coded["C"] < 0  # A:C pulls A and C to opposite ends


class TestRunsInTheFactorsOwnUnits:
    def test_runs_are_in_real_units_and_inside_fixed_ranges(self) -> None:
        factors = [Factor(name="T", low=150, high=180, units="degC"), Factor(name="t", low=30, high=90, units="min")]
        coded = generate_design(_factors("AB"), design_type="full_factorial", n_center_points=3).design[["A", "B"]]
        runs = pd.DataFrame({"T": 165 + 15 * coded["A"], "t": 60 + 30 * coded["B"]})
        runs["y"] = (
            TrueProcess({"Intercept": 60, "A": 2, "I(A ** 2)": -5, "I(B ** 2)": -3}, sigma=0.4)
            .simulate(coded, random_state=1)
            .to_numpy()
        )
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "augment_to_rsm"
        added = pd.DataFrame(step["runs"])
        assert added["T"].between(150, 180).all()
        assert added["t"].between(30, 90).all()
        assert {150.0, 165.0, 180.0} <= set(added["T"])

    def test_a_categorical_factor_is_set_at_a_level(self) -> None:
        factors = [Factor(name="Supplier", type="categorical", levels=["a", "b"]), *_factors("BC")]
        design = generate_design(factors, design_type="full_factorial", n_center_points=0, n_replicates=2)
        runs = design.design_actual[["Supplier", "B", "C"]]
        coded = runs.assign(Supplier=runs["Supplier"].map({"a": -1, "b": 1}))
        y = TrueProcess({"Intercept": 10, "Supplier": 2, "B": 3}, sigma=0.5).simulate(coded, random_state=3)
        step = recommend_next_step(factors, runs.assign(y=y.to_numpy()), "y", ranges_can_move=False)
        assert step["action"] == "confirm"
        assert step["runs"][0]["Supplier"] == "b"
        assert step["evidence"]["optimum"]["settings"]["Supplier"] == "b"


class TestBudget:
    def test_a_step_that_needs_more_runs_falls_back_to_confirming(self) -> None:
        factors = _factors("ABCDEFG")
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": 5, "A:B": 4}, sigma=0.8)
        runs = _runs(factors, truth, 4, design_type="fractional_factorial", resolution=3, n_center_points=0)
        step = recommend_next_step(factors, runs, "y", budget=3)
        assert step["action"] == "confirm"
        assert step["reasons"][0].startswith("Fold the design over")

    def test_too_few_runs_left_to_confirm_stops(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, budget=2)
        assert step["action"] == "stop"
        assert step["headline"] == "Stop: 2 run(s) are left, fewer than the next step needs."


class TestBlocks:
    """A process that drifts between rounds: each round a block, its shift fitted and kept out of the effects."""

    def test_the_shift_between_rounds_is_fitted_as_a_block(self, drifted: tuple) -> None:
        factors, runs = drifted
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, block="Round")
        assert step["evidence"]["n_blocks"] == 2
        assert step["evidence"]["active_terms"] == ["A", "B", "I(A ** 2)", "I(B ** 2)"]
        coded = step["evidence"]["optimum"]["coded"]
        assert coded["A"] == pytest.approx(0.15, abs=0.15)  # the true optimum is at A = 0.15, B = -0.17
        assert coded["B"] == pytest.approx(-0.17, abs=0.15)
        assert all(not c["term"].startswith("blk") for c in step["model"]["coefficients"])

    def test_without_the_blocks_the_shift_hides_the_effects(self, drifted: tuple) -> None:
        factors, runs = drifted
        blocked = recommend_next_step(factors, runs, "y", ranges_can_move=False, block="Round")
        ignored = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert ignored["evidence"]["noise_sd"] > 3 * blocked["evidence"]["noise_sd"]

    def test_the_next_runs_are_predicted_at_the_latest_block(self, drifted: tuple) -> None:
        """The confirmation runs, made with the drift still there, agree: they are judged as the process is now."""
        factors, runs = drifted
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, block="Round")
        assert step["evidence"]["optimum"]["predicted"] == pytest.approx(60.27 + 8.0, abs=1.0)
        made = _made(step, CURVED, 4).assign(y=lambda d: d["y"] + 8.0)
        done = recommend_next_step(factors, runs, "y", ranges_can_move=False, block="Round", confirmation_runs=made)
        assert done["action"] == "stop"

    def test_a_missing_block_column_is_refused(self, drifted: tuple) -> None:
        factors, runs = drifted
        with pytest.raises(ValueError, match="no block column"):
            recommend_next_step(factors, runs, "y", block="Lot")


class TestModelGiven:
    def test_the_given_model_is_used(self) -> None:
        factors = _factors("ABC")
        runs = _runs(factors, CURVED, 3, design_type="ccd", alpha="face_centered", n_center_points=4)
        step = recommend_next_step(factors, runs, "y", model_terms=["A", "B", "I(A ** 2)", "I(B ** 2)", "A:B"])
        assert step["model"]["judged_by"] == "t-tests in the model given"
        assert set(step["model"]["judged"]) == {"A", "B", "I(A ** 2)", "I(B ** 2)", "A:B"}


class TestInputs:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"goal": "target"}, "needs its target"),
            ({"goal": "best"}, "goal must be"),
            ({"n_confirmation_runs": 0}, "at least 1"),
            ({"history": ["fold"]}, "not actions"),
        ],
    )
    def test_bad_choices_are_refused(self, dsd_study: tuple, change: dict, message: str) -> None:
        factors, runs = dsd_study
        with pytest.raises(ValueError, match=message):
            recommend_next_step(factors, runs, "y", **change)

    def test_bad_factors_and_runs_are_refused(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        with pytest.raises(ValueError, match="no column for the factor"):
            recommend_next_step(factors, runs.drop(columns="B"), "y")
        with pytest.raises(ValueError, match="no response column"):
            recommend_next_step(factors, runs, "z")
        with pytest.raises(ValueError, match="At least two runs"):
            recommend_next_step(factors, runs.head(1), "y")
        with pytest.raises(ValueError, match="mixture component"):
            recommend_next_step([Factor(name="M", type="mixture", low=0, high=1)], runs, "y")
        three = Factor(name="S", type="categorical", levels=["a", "b", "c"])
        with pytest.raises(ValueError, match="at two levels"):
            recommend_next_step([three], runs.assign(S="a"), "y")
        with pytest.raises(ValueError, match="not among"):
            recommend_next_step([Factor(name="S", type="categorical", levels=["a", "b"])], runs.assign(S="c"), "y")

    def test_every_action_returned_is_one_of_the_actions(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        assert recommend_next_step(factors, runs, "y")["action"] in ACTIONS


class TestThroughTheTool:
    def test_the_tool_returns_json_and_no_library_warnings(self, dsd_study: tuple) -> None:
        factors, runs = dsd_study
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = execute_tool_call(
                "recommend_next_step",
                {
                    "factors": [{"name": f.name, "low": f.low, "high": f.high} for f in factors],
                    "runs": runs.to_dict(orient="records"),
                    "response_column": "y",
                    "ranges_can_move": False,
                },
            )
        assert result["action"] == "confirm"
        assert "library_warnings" not in result
        json.dumps(result, allow_nan=False)

    def test_the_tool_reports_an_error(self, dsd_study: tuple) -> None:
        _, runs = dsd_study
        result = execute_tool_call(
            "recommend_next_step",
            {
                "factors": [{"name": "A", "low": -1, "high": 1}],
                "runs": runs.to_dict(orient="records"),
                "response_column": "nope",
            },
        )
        assert "no response column" in result["error"]


def _follow(truth: TrueProcess, factors: list[Factor], runs: pd.DataFrame, *, seed: int, moves: bool) -> tuple:
    """Follow the recommendations round by round, as a user would; return the actions and the final settings."""
    history: list[str] = []
    confirmation = None
    rng = np.random.default_rng(seed + 1000)  # not the first runs' seed: a fold-over would repeat their noise
    names = [f.name for f in factors]
    for _ in range(10):
        step = recommend_next_step(
            factors, runs, "y", confirmation_runs=confirmation, ranges_can_move=moves, history=history
        )
        history.append(step["action"])
        if step["action"] in ("stop", "investigate"):
            break
        new = pd.DataFrame(step["runs"])[names].astype(float)
        new["y"] = truth.simulate(new, random_state=rng).to_numpy()
        if confirmation is not None:
            runs, confirmation = pd.concat([runs, confirmation], ignore_index=True), None
        if step["action"] == "confirm":
            confirmation = new
        elif step["action"] == "move_region":
            ys = new["y"].to_numpy()
            best = new.iloc[int(np.argmax(ys[: next((i for i in range(1, 4) if ys[i] < ys[i - 1]), 4)]))]
            factors = [Factor(name=n, low=best[n] - 1, high=best[n] + 1) for n in names]
            runs = _runs(factors, truth, int(rng.integers(1000)), design_type="full_factorial", n_center_points=3)
        else:
            runs = pd.concat([runs, new], ignore_index=True)
    settings = pd.DataFrame([step["evidence"]["optimum"]["settings"]])[names].astype(float)
    return history, float(truth.mean(settings)[0])


@pytest.mark.slow
class TestRoundByRound:
    """Following the recommendations from the first design to the last round reaches the optimum."""

    @pytest.mark.parametrize("seed", range(4))
    def test_an_interior_optimum_is_reached_through_a_response_surface(self, seed: int) -> None:
        factors = _factors("ABC")
        runs = _runs(factors, CURVED, seed, design_type="full_factorial", n_center_points=3)
        history, value = _follow(CURVED, factors, runs, seed=seed, moves=True)
        assert history[0] == "augment_to_rsm"
        assert history[-1] == "stop"
        assert value > 60.27 - 0.1 * CURVED.sigma  # the true maximum is 60.27, at A = 0.15, B = -0.17

    @pytest.mark.parametrize("seed", range(3))
    def test_an_optimum_outside_the_ranges_is_reached_by_moving(self, seed: int) -> None:
        truth = TrueProcess({"Intercept": 80, "A": 12, "B": -9, "I(A ** 2)": -3, "I(B ** 2)": -3}, sigma=1.0)
        factors = _factors("ABC")
        runs = _runs(factors, truth, seed, design_type="full_factorial", n_center_points=3)
        history, value = _follow(truth, factors, runs, seed=seed, moves=True)
        assert "move_region" in history
        assert history[-1] == "stop"
        assert value > 98.75 - 0.5 * truth.sigma  # the true maximum is 98.75, at A = 2, B = -1.5

    @pytest.mark.parametrize("seed", range(3))
    def test_curvature_in_one_of_eight_factors_is_found_and_confirmed(self, seed: int) -> None:
        truth = TestSquaresAliasedWithEachOther.CURVED_8
        factors = _factors("ABCDEFGH")
        runs = _runs(factors, truth, seed, design_type="fractional_factorial", resolution=4, n_center_points=3)
        history, value = _follow(truth, factors, runs, seed=seed, moves=False)
        assert history[-1] == "stop"
        assert value > 100.17 - 0.1 * truth.sigma  # the true maximum is 100.17, at A = 1/6, C = 0

    @pytest.mark.parametrize("seed", range(3))
    def test_an_aliased_interaction_is_resolved_before_the_optimum_is_confirmed(self, seed: int) -> None:
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": 5, "A:B": 4}, sigma=0.8)
        factors = _factors("ABCDEFG")
        runs = _runs(factors, truth, seed, design_type="fractional_factorial", resolution=3, n_center_points=0)
        history, value = _follow(truth, factors, runs, seed=seed, moves=False)
        assert history == ["de_alias", "confirm", "stop"]
        assert value == pytest.approx(65.0)
