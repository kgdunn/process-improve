"""Edge cases of recommending the next experiment, evaluating a design, its projections, and the strategy planner."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import (
    Factor,
    TrueProcess,
    evaluate_design,
    evaluate_projections,
    generate_design,
    recommend_next_step,
)
from process_improve.experiments.projections import projection_formula
from process_improve.experiments.region import DesignRegion
from process_improve.experiments.strategy.engine import recommend_strategy
from tests.test_next_step import ADHESION, CURVED, _factors, _runs
from tests.test_recommend_strategy import _continuous, _maximise, _stage


def _rank(frame: pd.DataFrame, columns: list[str]) -> int:
    """Return the rank of a model matrix: the intercept, then each column, a product written ``"A*B"``."""
    parts = [np.ones(len(frame))]
    for column in columns:
        factors = column.split("*")
        parts.append(np.prod([frame[f].to_numpy(dtype=float) for f in factors], axis=0))
    return int(np.linalg.matrix_rank(np.column_stack(parts)))


# --------------------------------------------------------------------------- recommend_next_step
class TestRefusedInputs:
    def test_a_study_without_factors_is_refused(self) -> None:
        """An empty factor list is refused before the runs are read."""
        runs = pd.DataFrame({"y": [1.0, 2.0]})
        with pytest.raises(ValueError, match="At least one factor is required"):
            recommend_next_step([], runs, "y")

    def test_a_model_term_that_names_no_factor_is_refused(self) -> None:
        """A term of the model given must multiply the study's factors only."""
        factors = _factors("AB")
        runs = _runs(factors, CURVED, 1, design_type="full_factorial", n_center_points=3)
        with pytest.raises(ValueError, match=r"The model term 'A:Z' names \['Z'\], which are not factors"):
            recommend_next_step(factors, runs, "y", model_terms=["A", "A:Z"])


class TestHowTheTermsAreJudged:
    """The judging falls through to the next method when the runs cannot take the one before."""

    def test_an_intercept_only_model_has_no_term_that_can_stand_out(self) -> None:
        """The intercept is implied, so the model given tests nothing, and nothing stands out."""
        factors = _factors("ABC")
        corners = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=list("ABC"))
        truth = TrueProcess({"Intercept": 50, "A": 5, "B": 3}, sigma=0.5)
        runs = corners.assign(y=truth.simulate(corners, random_state=8).to_numpy())
        step = recommend_next_step(factors, runs, "y", model_terms=["Intercept"], ranges_can_move=False)
        assert step["model"]["terms"] == []
        assert step["model"]["judged"] == {}
        assert not step["model"]["signal"]
        fitted = [(c["term"], c["coefficient"]) for c in step["model"]["coefficients"]]
        assert fitted == [("Intercept", pytest.approx(runs["y"].mean()))]  # the intercept alone fits the mean
        assert step["action"] == "replicate"
        assert step["n_runs"] == 8
        assert "with 0 terms judged" in step["reasons"][0]

    def test_an_unbalanced_two_level_design_is_judged_by_t_tests_not_lenth(self) -> None:
        """A repeated corner leaves the contrasts unbalanced, so Lenth's method cannot judge the interactions model."""
        factors = _factors("AB")
        x = pd.DataFrame([[-1, -1], [1, -1], [-1, 1], [1, 1], [-1, -1]], columns=list("AB"), dtype=float)
        runs = x.assign(
            y=TrueProcess({"Intercept": 10, "A": 3, "B": 2}, sigma=0.3).simulate(x, random_state=2).to_numpy()
        )
        step = recommend_next_step(factors, runs, "y")
        assert step["model"]["judged_by"] == "t-tests in the main effects model"
        assert step["evidence"]["error_df"] == 5 - 3
        expected = np.linalg.lstsq(np.column_stack([np.ones(5), x]), runs["y"], rcond=None)[0]
        fitted = {c["term"]: c["coefficient"] for c in step["model"]["coefficients"]}
        assert [fitted["Intercept"], fitted["A"], fitted["B"]] == pytest.approx(expected)
        assert step["evidence"]["active_terms"] == ["A", "B"]

    def test_a_blocked_definitive_screening_design_is_judged_by_the_stepwise_search(self) -> None:
        """The staged analysis has no place for blocks, so the AICc search proposes the second-order terms."""
        factors = _factors("ABCDEFGH")
        runs = _runs(factors, ADHESION, 1, design_type="dsd", n_center_points=5)
        runs["Round"] = [1] * 11 + [2] * (len(runs) - 11)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False, block="Round")
        assert step["model"]["judged_by"] == (
            "t-tests of every main effect and the quadratic terms a stepwise search (AICc) proposed"
        )
        assert step["evidence"]["n_blocks"] == 2
        assert step["evidence"]["active_terms"] == ["A", "C", "G", "A:C"]
        assert step["evidence"]["error_df"] == len(runs) - 1 - 1 - 8 - 1  # intercept, block, mains, A:C

    def test_a_definitive_screening_design_without_error_df_is_judged_by_the_stepwise_search(self) -> None:
        """Six factors and one centre run: the full second-order model leaves the staged analysis no error."""
        factors = _factors("ABCDEF")
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": -4, "A:B": 3}, sigma=1)
        with pytest.warns(UserWarning, match="leaves no degrees of freedom for error"):
            runs = _runs(factors, truth, 1, design_type="dsd", n_center_points=1)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["model"]["judged_by"] == (
            "t-tests of every main effect and the quadratic terms a stepwise search (AICc) proposed"
        )
        assert step["evidence"]["active_terms"] == ["A", "B", "A:B"]
        assert step["evidence"]["error_df"] == 13 - 1 - 6 - 1
        assert step["action"] == "confirm"

    def test_a_proposal_the_runs_cannot_hold_falls_back_to_the_main_effects_model(self) -> None:
        """Ten factors in a 12-run Plackett-Burman design and two centre runs: A:C beside them leaves one error df."""
        factors = _factors("ABCDEFGHJK")
        runs = _runs(factors, ADHESION, 0, design_type="plackett_burman", n_center_points=2)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["model"]["judged_by"] == "t-tests in the main effects model"
        assert step["evidence"]["error_df"] == 14 - 1 - 10 - 1  # intercept, mains, curvature
        assert not step["model"]["signal"]
        assert step["action"] == "replicate"
        assert step["n_runs"] == 14

    def test_runs_too_few_for_any_model_are_replicated_and_the_centre_runs_still_test_curvature(self) -> None:
        """Eleven factors in 14 runs: nothing is judged, and curvature is tested against the two centre runs alone."""
        factors = _factors("ABCDEFGHJKL")
        runs = _runs(factors, ADHESION, 0, design_type="plackett_burman", n_center_points=2)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["model"]["judged_by"] == "none: too few runs to judge any term"
        assert step["evidence"]["error_df"] == 0
        assert step["action"] == "replicate"
        assert step["n_runs"] == 14
        centre = (runs[[f.name for f in factors]] == 0).all(axis=1)
        y_centre, y_corner = runs.loc[centre, "y"], runs.loc[~centre, "y"]
        n_c, n_f = len(y_centre), len(y_corner)
        f_ratio = n_f * n_c * (y_corner.mean() - y_centre.mean()) ** 2 / (n_f + n_c) / y_centre.var(ddof=1)
        curvature = step["evidence"]["curvature"]
        assert curvature["source"] == "center_runs"
        assert curvature["p_value"] == pytest.approx(stats.f.sf(f_ratio, 1, n_c - 1))


class TestInteractionsTheRunsCannotSee:
    """An active term's partner the runs cannot tell apart gets runs chosen (D-optimal) for both."""

    def test_one_factor_at_a_time_runs_get_the_corners_for_the_interaction(self) -> None:
        """Varying one factor at a time sets A:B to zero in every run: only runs at the corners can estimate it."""
        factors = _factors("AB")
        ofat = pd.DataFrame(
            [[0, 0], [0, 0], [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 0], [1, 0], [0, -1], [0, 1]],
            columns=list("AB"),
            dtype=float,
        )
        truth = TrueProcess({"Intercept": 50, "A": 5, "B": 3, "A:B": 4}, sigma=0.5)
        runs = ofat.assign(y=truth.simulate(ofat, random_state=0).to_numpy())
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "de_alias"
        assert step["evidence"]["aliasing"] == [{"term": "A", "aliased_with": "A:B", "weight": 0.0}]
        assert step["headline"] == (
            "Add 4 runs chosen to estimate each active term apart from the interaction it is aliased with."
        )
        added = pd.DataFrame(step["runs"])
        assert set(map(tuple, added[["A", "B"]].to_numpy())) == set(itertools.product([-1.0, 1.0], repeat=2))
        assert _rank(ofat, ["A", "B", "A*B"]) == 3
        assert _rank(pd.concat([ofat, added], ignore_index=True), ["A", "B", "A*B"]) == 4

    def test_an_active_interaction_aliased_with_another_gets_runs_chosen_for_both(self) -> None:
        """In the half fraction D = ABC, A:B is C:D: a fold-over would not part two interactions."""
        factors = _factors("ABCD")
        truth = TrueProcess({"Intercept": 50, "A": 6, "B": 5, "C": 4, "D": 3, "A:B": 4}, sigma=0.5)
        runs = _runs(factors, truth, 0, design_type="fractional_factorial", resolution=4, n_center_points=3)
        step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
        assert step["action"] == "de_alias"
        assert [(a["term"], a["aliased_with"]) for a in step["evidence"]["aliasing"]] == [("A:B", "C:D")]
        assert step["evidence"]["aliasing"][0]["weight"] == pytest.approx(1.0)
        assert step["n_runs"] == max(1, 7 - 11) + 3  # the partners, or the coefficients the runs lack; and 3 spare
        assert step["reasons"][-1] == (
            "Runs chosen for the model that holds both terms of each pair (D-optimal) let it tell them apart."
        )
        model = ["A", "B", "C", "D", "A*B", "C*D"]
        assert _rank(runs, model) == 6
        assert _rank(pd.concat([runs, pd.DataFrame(step["runs"])], ignore_index=True), model) == 7


class TestConfirmationRunsAwayFromTheOptimum:
    def test_they_join_the_runs_in_the_latest_block(self) -> None:
        """Runs made at the centre, not the optimum, are data; made after every block, they count in the latest one."""
        factors = _factors("AB")
        cube = pd.DataFrame(
            [[-1, -1], [1, -1], [-1, 1], [1, 1], [0, 0], [0, 0], [0, 0]], columns=list("AB"), dtype=float
        )
        axial = pd.DataFrame([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [0, 0]], columns=list("AB"), dtype=float)
        first = cube.assign(y=CURVED.simulate(cube, random_state=1).to_numpy(), Round=1)
        second = axial.assign(y=CURVED.simulate(axial, random_state=2).to_numpy() + 8.0, Round=2)
        runs = pd.concat([first, second], ignore_index=True)
        centre = pd.DataFrame(np.zeros((3, 2)), columns=list("AB"))
        made = centre.assign(y=CURVED.simulate(centre, random_state=3).to_numpy() + 8.0)  # the process as it is now
        step = recommend_next_step(factors, runs, "y", block="Round", ranges_can_move=False, confirmation_runs=made)
        assert not step["evidence"]["confirmation"][0]["at_optimum"]
        assert step["reasons"][-1] == (
            "The confirmation runs were not at the optimum the model now predicts, so they join the other runs."
        )
        assert step["evidence"]["n_runs"] == 16
        assert step["evidence"]["n_blocks"] == 2
        every = pd.concat([runs, made.assign(Round=2)], ignore_index=True)
        a, b = every["A"].to_numpy(), every["B"].to_numpy()
        x = np.column_stack([np.ones(16), a, b, a * b, a**2, b**2, np.where(every["Round"] == 1, 1.0, -1.0)])
        residual = every["y"].to_numpy() - x @ np.linalg.lstsq(x, every["y"].to_numpy(), rcond=None)[0]
        assert step["evidence"]["noise_sd"] == pytest.approx(np.sqrt(residual @ residual / (16 - 7)))


# --------------------------------------------------------------------------- evaluate_design
def _rules_region() -> DesignRegion:
    """Return the square in T and D with two categorical factors whose rule forbids catalyst B in water."""
    return DesignRegion(
        [
            Factor(name="T", low=-1, high=1),
            Factor(name="D", low=-1, high=1),
            Factor(name="catalyst", type="categorical", levels=["A", "B"]),
            Factor(name="solvent", type="categorical", levels=["water", "oil"]),
        ],
        ["not (catalyst == 'B' and solvent == 'water')"],
    )


_SQUARE = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=2)), columns=["T", "D"])


class TestRegionsWithRules:
    """A 2^2 in T and D, main effects: X'X = 4I, so d(x) = (1 + T^2 + D^2) / 4."""

    def test_rules_on_factors_the_design_lacks_drop_no_column(self) -> None:
        """The rules name only categorical factors the design never set: T and D stay uniform on the square."""
        found = evaluate_design(
            _SQUARE,
            model="T + D",
            region=_rules_region(),
            metric=["average_prediction_variance", "g_efficiency"],
            n_samples=20_000,
        )
        assert "notes" not in found
        assert found["average_prediction_variance"] == pytest.approx((1 + 2 / 3) / 4, abs=2e-3)
        assert found["max_prediction_variance"] == pytest.approx(3 / 4)  # at the corners, which the support adds
        assert found["g_efficiency"] == pytest.approx(100.0)

    def test_without_vertices_the_region_is_only_sampled(self) -> None:
        """The same sample gives the same average; without the corners the maximum falls short of 3/4."""
        options = {"model": "T + D", "region": _rules_region(), "n_samples": 20_000}
        metrics = ["average_prediction_variance", "g_efficiency"]
        sampled = evaluate_design(_SQUARE, metric=metrics, include_vertices=False, **options)
        with_vertices = evaluate_design(_SQUARE, metric=metrics, include_vertices=True, **options)
        assert sampled["average_prediction_variance"] == with_vertices["average_prediction_variance"]
        assert 0.7 < sampled["max_prediction_variance"] < 3 / 4

    def test_a_design_that_ran_none_of_the_region_levels_keeps_every_column(self) -> None:
        """The rules cannot be judged on levels the design never ran: D-efficiency is that of X'X = 4I, 100."""
        design = pd.DataFrame({"T": [-1.0, 1.0, -1.0, 1.0], "catalyst": ["x", "x", "y", "y"]})
        region = DesignRegion(
            [Factor(name="T", low=-1, high=1), Factor(name="catalyst", type="categorical", levels=["A", "B"])],
            ["not (catalyst == 'B' and T > 0.5)"],
        )
        found = evaluate_design(design, model="T + catalyst", region=region, metric="d_efficiency")
        assert found == {"d_efficiency": pytest.approx(100.0)}


class TestEvaluateDesignEdges:
    def test_an_exact_average_over_eight_factors_falls_back_to_the_sample(self) -> None:
        """5^8 quadrature points pass the limit. A 16-run 2^(8-4): X'X = 16I, so the average is (1 + 8/3)/16."""
        factors = _factors("ABCDEFGH")
        design = generate_design(factors, design_type="fractional_factorial", resolution=4, n_center_points=0)
        coded = design.design[list("ABCDEFGH")]
        exact = evaluate_design(
            coded, model="main_effects", metric="average_prediction_variance", integration="exact", n_samples=20_000
        )
        sampled = evaluate_design(coded, model="main_effects", metric="average_prediction_variance", n_samples=20_000)
        assert exact["notes"]["average_prediction_variance"].endswith("of at most 250,000 points: sampled.")
        assert exact["average_prediction_variance"] == sampled["average_prediction_variance"]
        assert exact["average_prediction_variance"] == pytest.approx(11 / 48, abs=2e-3)

    def test_a_scheffe_model_is_aliased_only_against_the_blending_interactions(self) -> None:
        """A Scheffe model has no squares; the alias matrix is (X1'X1)^-1 X1'X2 against the three products."""
        blends = pd.DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]],
            columns=list("ABC"),
        )
        found = evaluate_design(blends, model="scheffe_linear", metric="alias_matrix", alias_terms="quadratic")
        alias = found["alias_matrix"]
        assert alias["alias_terms"] == ["A:B", "A:C", "B:C"]
        x1 = blends.to_numpy()
        x2 = np.column_stack([x1[:, 0] * x1[:, 1], x1[:, 0] * x1[:, 2], x1[:, 1] * x1[:, 2]])
        assert np.array(alias["matrix"]) == pytest.approx(np.linalg.solve(x1.T @ x1, x1.T @ x2))

    def test_a_full_quadratic_model_has_nothing_outside_it_to_alias_against(self) -> None:
        """Every interaction and square is in the model, so the alias matrix is empty, with a note."""
        grid = pd.DataFrame(list(itertools.product([-1.0, 0.0, 1.0], repeat=2)), columns=list("AB"))
        found = evaluate_design(grid, model="quadratic", metric="alias_matrix", alias_terms="quadratic")
        assert found["alias_matrix"]["alias_terms"] == []
        assert found["alias_matrix"]["max_abs"] == 0.0
        assert found["alias_matrix"]["note"] == "No second-order terms outside the model to alias against."

    def test_one_factor_is_studied_for_its_linear_and_quadratic_effects_only(self) -> None:
        """With A at -1, -1, 0, 0, 1, 1: var(b1) = 1/4 on 4 df, and var(b11) = 3/4 on 3 df; no pair to interact."""
        design = pd.DataFrame({"A": [-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]})
        hypotheses = evaluate_design(design, model="A", metric="screening_power")["screening_power"]["hypotheses"]
        assert [h["hypothesis"] for h in hypotheses] == [1, 2]
        assert all(h["status"] == "estimable" and h["n_aliased"] == 0 for h in hypotheses)

        def power(ncp: float, df: int) -> float:  # two-sided t test at 0.05, by the noncentral t
            t = stats.t.ppf(0.975, df)
            return float(stats.nct.sf(t, df, ncp) + stats.nct.cdf(-t, df, ncp))

        assert hypotheses[0]["average_power"] == pytest.approx([power(e / 0.5, 4) for e in (1, 2, 3)])
        assert hypotheses[1]["average_power"] == pytest.approx([power(e / np.sqrt(0.75), 3) for e in (1, 2, 3)])


# --------------------------------------------------------------------------- projections
class TestProjectionEdges:
    def test_an_unknown_model_is_refused_by_the_formula(self) -> None:
        """Only the three nested models have a formula."""
        with pytest.raises(ValueError, match="model must be one of"):
            projection_formula(pd.DataFrame({"A": [-1.0, 1.0]}), ["A"], "cubic")

    def test_the_main_effects_formula_is_the_factors_alone(self) -> None:
        """The main effects model joins the factors, with no products and no squares."""
        frame = pd.DataFrame({"A": [-1.0, 0.0, 1.0], "B": [-1.0, 1.0, 0.0]})
        assert projection_formula(frame, ["A", "B"], "main_effects") == "A + B"

    def test_a_projection_with_a_label_factor_is_judged_by_its_information_matrix(self) -> None:
        """R follows S (a: -1, b: +1), so the projection on S and R cannot estimate both; the others can."""
        frame = pd.DataFrame(
            {
                "Q": [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
                "S": ["a", "a", "a", "b", "b", "b"],
                "R": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            }
        )
        found = evaluate_projections(frame, 2, model="main_effects")
        assert {tuple(p["factors"]): p["estimable"] for p in found["projections"]} == {
            ("Q", "S"): True,
            ("Q", "R"): True,
            ("S", "R"): False,
        }
        assert found["n_estimable"] == 2


# --------------------------------------------------------------------------- recommend_strategy
class TestStrategyEdges:
    def test_prior_knowledge_past_the_length_cap_is_refused(self) -> None:
        """Text of 4096 characters is read; one character more is refused, naming both lengths."""
        recommend_strategy(factors=_continuous(3), prior_knowledge="x" * 4096)  # at the cap: accepted
        with pytest.raises(ValueError, match="prior_knowledge text is 4097 characters; the maximum is 4096"):
            recommend_strategy(factors=_continuous(3), prior_knowledge="x" * 4097)

    def test_mixture_components_cannot_be_hard_to_change(self) -> None:
        """A split-plot design cannot hold mixture components."""
        factors = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"]
        with pytest.raises(ValueError, match="not supported for mixture components"):
            recommend_strategy(factors=factors, hard_to_change_factors=["A"])

    def test_eight_factors_known_well_get_a_ccd_on_a_half_fraction(self) -> None:
        """Screening is skipped; the cube is 2^(8-1) = 128 runs, with 16 axial and 3 centre runs."""
        strategy = recommend_strategy(
            factors=_continuous(8), responses=_maximise(), prior_knowledge="Validated in published work."
        )
        stage = strategy["stages"][0]
        assert (stage["stage_name"], stage["design_type"]) == ("Optimization", "ccd")
        assert stage["design_params"] == {"n_center_points": 3, "alpha": "face_centered", "cube": "fractional"}
        assert stage["estimated_runs"] == 2**7 + 2 * 8 + 3

    def test_a_dsd_of_an_odd_number_of_factors_without_extra_centre_runs_warns_of_its_one_error_df(self) -> None:
        """A two-level categorical factor takes the DSD's centre runs away; seven factors leave one df for error."""
        factors = [Factor(name="S", type="categorical", levels=["a", "b"]), *_continuous(6)]
        strategy = recommend_strategy(factors=factors, responses=_maximise(), domain="pharma_formulation")
        assert _stage(strategy, "Screening")["design_params"] == {}
        assert (
            "analyze_omars, whose 1 degree(s) of freedom for error can miss a smaller active effect"
            in _stage(strategy, "Optimization")["run_if"]
        )

    def test_a_tight_budget_takes_a_dsd_without_centre_runs_straight_to_plackett_burman(self) -> None:
        """There are no centre runs to drop first; seven factors fit an 8-run Plackett-Burman design."""
        factors = [Factor(name="S", type="categorical", levels=["a", "b"]), *_continuous(6)]
        strategy = recommend_strategy(factors=factors, responses=_maximise(), domain="pharma_formulation", budget=25)
        screening = _stage(strategy, "Screening")
        assert (screening["design_type"], screening["design_params"]) == ("plackett_burman", {"n_center_points": 0})
        assert screening["estimated_runs"] == 8
        assert any("Screening: dsd" in risk and "became plackett_burman (8 runs)" in risk for risk in strategy["risks"])

    def test_an_optimal_stage_with_a_hard_to_change_factor_keeps_its_parameters(self) -> None:
        """A three-level categorical factor: main effects take 1 + 2 + 3 = 6 coefficients, so 9 runs."""
        factors = [Factor(name="A", type="categorical", levels=["x", "y", "z"]), *_continuous(3)]
        strategy = recommend_strategy(factors=factors, responses=_maximise(), hard_to_change_factors=["X0"])
        screening = _stage(strategy, "Screening")
        assert screening["design_type"] == "d_optimal"
        assert screening["design_params"] == {"model_type": "main_effects", "budget": 9, "hard_to_change": ["X0"]}
        assert "Split-plot: X0 change only between whole plots" in screening["purpose"]

    def test_existing_data_counts_in_the_strategy_id_only_when_it_has_columns(self) -> None:
        """A frame is summarised into the id; a list has no columns, so it is as if no data were given."""
        plain = recommend_strategy(factors=_continuous(3))["strategy_id"]
        frame = recommend_strategy(factors=_continuous(3), existing_data=pd.DataFrame({"X0": [1.0], "y": [2.0]}))
        listed = recommend_strategy(factors=_continuous(3), existing_data=[1, 2, 3])
        assert frame["strategy_id"] != plain
        assert listed["strategy_id"] == plain
