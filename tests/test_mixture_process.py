"""Tests for mixture-process models and designs: mixture components and process factors together."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from process_improve.experiments import (
    Constraint,
    DesignRegion,
    Factor,
    analyze_experiment,
    evaluate_design,
    generate_design,
    optimize_responses,
)
from process_improve.experiments.designs_mixture_process import mixture_process_model

#: Three bounded components and two process factors, used for the region and the designs.
BLENDS = [
    Factor(name="x1", type="mixture", low=0.1, high=0.6),
    Factor(name="x2", type="mixture", low=0.1, high=0.6),
    Factor(name="x3", type="mixture", low=0.1, high=0.6),
]
SETTINGS = [Factor(name="T", low=20, high=80, units="degC"), Factor(name="S", low=100, high=300, units="rpm")]


def _points(q: int, m: int, n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Random blends of ``q`` components, which sum to 1, with ``m`` process factors in coded units."""
    rng = np.random.default_rng(seed)
    blends = rng.dirichlet(np.ones(q), size=n)
    data = {f"x{i + 1}": blends[:, i] for i in range(q)}
    data.update({f"z{k + 1}": rng.uniform(-1, 1, size=n) for k in range(m)})
    return pd.DataFrame(data)


def _names(q: int, m: int) -> tuple[list[str], list[str]]:
    return [f"x{i + 1}" for i in range(q)], [f"z{k + 1}" for k in range(m)]


def _rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=1e-9))


class TestModelSize:
    """The parameter counts that Goos and Jones (2011, section 6.3) give for each model."""

    @pytest.mark.parametrize(("q", "m"), [(2, 1), (3, 2), (4, 3), (5, 2)])
    def test_kowalski_cornell_vining(self, q: int, m: int) -> None:
        expected = q + q * (q - 1) // 2 + q * m + m * (m - 1) // 2 + m
        quadratic = mixture_process_model(*_names(q, m), "quadratic")
        assert quadratic.n_parameters == expected
        assert mixture_process_model(*_names(q, m), "kcv").terms == quadratic.terms

    @pytest.mark.parametrize(("q", "m"), [(2, 1), (3, 2), (4, 3)])
    def test_crossed(self, q: int, m: int) -> None:
        expected = (q + q * (q - 1) // 2) * (1 + m + m * (m - 1) // 2)
        assert mixture_process_model(*_names(q, m), "scheffe_quadratic * interactions").n_parameters == expected

    def test_crossed_three_components_two_process_factors(self) -> None:
        """The 24 parameters the book quotes for three ingredients and two process variables."""
        assert mixture_process_model(*_names(3, 2), "scheffe_quadratic * interactions").n_parameters == 24

    def test_additive_table_6_2(self) -> None:
        """Three mills' second-order Scheffé model plus two process main effects: 8 terms."""
        assert mixture_process_model(*_names(3, 2), "scheffe_quadratic + main_effects").n_parameters == 8

    @pytest.mark.parametrize(("q", "m"), [(3, 1), (4, 3)])
    def test_first_order(self, q: int, m: int) -> None:
        assert mixture_process_model(*_names(q, m), "main_effects").n_parameters == q + m


class TestOrderBasedModels:
    """The order-based models are the full polynomial in every factor, less what sum(x) = 1 makes redundant."""

    @staticmethod
    def _full_polynomial(data: pd.DataFrame, squares: bool) -> np.ndarray:
        names = list(data.columns)
        rhs = f"({' + '.join(names)}) ** 2"
        if squares:
            rhs += "".join(f" + I({n} ** 2)" for n in names)
        return np.asarray(dmatrix(rhs, data))

    @pytest.mark.parametrize(("order", "squares"), [("interactions", False), ("quadratic", True)])
    @pytest.mark.parametrize(("q", "m"), [(3, 2), (4, 3)])
    def test_same_span_as_the_full_polynomial(self, order: str, squares: bool, q: int, m: int) -> None:
        data = _points(q, m)
        model = mixture_process_model(*_names(q, m), order).matrix(data)
        full = self._full_polynomial(data, squares)
        # On blends that sum to 1 the full polynomial loses exactly the redundant terms, and the model
        # holds the rest: its columns are independent and span every column of the full polynomial.
        assert _rank(model) == model.shape[1] == _rank(full)
        assert _rank(np.hstack([model, full])) == model.shape[1]


class TestMatrixAndFormulaAgree:
    """The model matrix the design engine optimises is the one the recorded formula builds."""

    @pytest.mark.parametrize(
        "model_type",
        [
            "main_effects",
            "interactions",
            "quadratic",
            "scheffe_quadratic + main_effects",
            "scheffe_special_cubic * interactions",
            "scheffe_cubic * main_effects",
            "scheffe_linear + quadratic",
        ],
    )
    def test_columns_match(self, model_type: str) -> None:
        data = _points(3, 2)
        model = mixture_process_model(*_names(3, 2), model_type)
        ours = model.matrix(data)
        theirs = np.asarray(dmatrix(model.formula_rhs(), data))
        assert theirs.shape == ours.shape
        # Each column appears once on each side, though patsy orders the terms by degree.
        matches = np.isclose(ours[:, :, None], theirs[:, None, :]).all(axis=0)
        assert (matches.sum(axis=1) == 1).all()
        assert (matches.sum(axis=0) == 1).all()

    def test_cubic_difference_values(self) -> None:
        data = _points(3, 1)
        model = mixture_process_model(*_names(3, 1), "scheffe_cubic + main_effects")
        names = [":".join(a.code for a in term) for term in model.terms]
        column = model.matrix(data)[:, names.index("I(x1 * x2 * (x1 - x2))")]
        np.testing.assert_allclose(column, data.x1 * data.x2 * (data.x1 - data.x2))

    def test_formula_has_no_intercept(self) -> None:
        assert mixture_process_model(*_names(3, 2), "interactions").formula_rhs().startswith("-1 + x1 + x2 + x3")


class TestModelNames:
    @pytest.mark.parametrize("model_type", ["cubic", "scheffe_quadratic", "linear * interactions x", "full"])
    def test_unknown_names_are_refused(self, model_type: str) -> None:
        with pytest.raises(ValueError, match=r"not a mixture-process model|Unknown"):
            mixture_process_model(*_names(3, 2), model_type)

    def test_unknown_process_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="Unknown process model 'cubic'"):
            mixture_process_model(*_names(3, 2), "scheffe_quadratic * cubic")

    def test_combined_name_is_normalised(self) -> None:
        model = mixture_process_model(*_names(3, 2), "quadratic*main_effects")
        assert model.name == "scheffe_quadratic * main_effects"


class TestRegion:
    """A mixture-process region is the product of a mixture region and a box region."""

    def test_kind_columns_and_bounds(self) -> None:
        region = DesignRegion([BLENDS[0], SETTINGS[0], *BLENDS[1:], SETTINGS[1]])
        assert region.kind == "mixture_process"
        assert region.names == ["x1", "T", "x2", "x3", "S"]
        assert region.bounds == [(0.1, 0.6), (-1.0, 1.0), (0.1, 0.6), (0.1, 0.6), (-1.0, 1.0)]

    def test_feasible_needs_blends_summing_to_one_and_settings_in_the_box(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 + x2 <= 0.7", "T + S <= 300"])
        points = np.array(
            [
                [0.3, 0.3, 0.4, 0.0, 0.0],  # feasible
                [0.3, 0.3, 0.3, 0.0, 0.0],  # the blend sums to 0.9
                [0.4, 0.4, 0.2, 0.0, 0.0],  # x1 + x2 = 0.8
                [0.3, 0.3, 0.4, 1.2, 0.0],  # T beyond its range
                [0.3, 0.3, 0.4, 1.0, 1.0],  # T + S = 380 in actual units
            ]
        )
        np.testing.assert_array_equal(region.feasible(points), [True, False, False, False, False])

    def test_sample_is_feasible_and_spread_over_both_parts(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 <= 0.4"])
        points = region.sample(4000, np.random.default_rng(1))
        assert region.feasible(points).all()
        np.testing.assert_allclose(points[:, :3].sum(axis=1), 1.0)
        np.testing.assert_allclose(points[:, 3:].mean(axis=0), 0.0, atol=0.05)

    def test_support_points_cross_the_parts(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS])
        blends = DesignRegion(BLENDS).support_points()
        settings = DesignRegion(SETTINGS).support_points()
        points = region.support_points()
        assert len(points) == len(blends) * len(settings)
        assert region.feasible(points).all()

    def test_volume_fraction_is_the_product_of_the_parts(self) -> None:
        region = DesignRegion([Factor(name=n, type="mixture") for n in "ABC"] + SETTINGS[:1], ["A <= 0.5", "T <= 50"])
        assert region.volume_fraction() == pytest.approx(0.75 * 0.5)

    def test_round_trip_keeps_the_kind(self) -> None:
        region = DesignRegion([*BLENDS, *SETTINGS], ["x1 <= 0.4"])
        again = DesignRegion.from_dict(region.to_dict())
        assert again.kind == "mixture_process"
        assert again.is_constrained

    def test_a_constraint_joining_both_parts_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"names both mixture components \['x1'\] and process factors \['T'\]"):
            DesignRegion([*BLENDS, *SETTINGS], ["x1 + 0.01*T <= 1"])

    def test_box_diagnostics_are_refused(self) -> None:
        with pytest.raises(ValueError, match="describes a box of factor settings"):
            DesignRegion([*BLENDS, *SETTINGS]).centroid()


#: Goos and Jones (2011), Table 6.2: the 12-run D-optimal design for the reflectivity of aluminium sheet. The
#: three mills share the reduction in thickness (each 0.1 to 0.8 of it), and spray volume and oil-water are
#: two-level process variables, here -1 (Low) and +1 (High). The model is the second-order Scheffé model in the
#: mills plus the main effects of the two process variables.
TABLE_6_2 = pd.DataFrame(
    [
        [0.1, 0.8, 0.1, -1, -1],
        [0.8, 0.1, 0.1, -1, -1],
        [0.1, 0.45, 0.45, -1, -1],
        [0.1, 0.1, 0.8, 1, -1],
        [0.45, 0.45, 0.1, 1, -1],
        [0.45, 0.1, 0.45, 1, -1],
        [0.45, 0.1, 0.45, -1, 1],
        [0.1, 0.1, 0.8, -1, 1],
        [0.45, 0.45, 0.1, -1, 1],
        [0.8, 0.1, 0.1, 1, 1],
        [0.1, 0.8, 0.1, 1, 1],
        [0.1, 0.45, 0.45, 1, 1],
    ],
    columns=["mill1", "mill2", "mill3", "spray", "oil"],
)
MILLS = [Factor(name=f"mill{i}", type="mixture", low=0.1, high=0.8) for i in (1, 2, 3)]
COOLANT = [Factor(name="spray", low=-1, high=1), Factor(name="oil", low=-1, high=1)]
TABLE_6_2_MODEL = "scheffe_quadratic + main_effects"


def _model_rows(design: pd.DataFrame, factors: list[Factor], model_type: str) -> np.ndarray:
    mixture = [f.name for f in factors if f.type.value == "mixture"]
    process = [f.name for f in factors if f.type.value != "mixture"]
    model = mixture_process_model(mixture, process, model_type)
    return model.matrix({n: design[n].to_numpy(dtype=float) for n in mixture + process})


def _log_det(x: np.ndarray) -> float:
    return float(np.linalg.slogdet(x.T @ x)[1])


@pytest.fixture(scope="module")
def design() -> pd.DataFrame:
    """Return the library's 12-run D-optimal design for the model of Table 6.2 (every seed tried finds the optimum)."""
    result = generate_design(
        [*MILLS, *COOLANT], design_type="d_optimal", model_type=TABLE_6_2_MODEL, budget=12, random_state=3
    )
    return result.design.drop(columns="RunOrder").round(6)


class TestPublishedDesign:
    """The D-optimal design of Goos and Jones (2011), Table 6.2, is found again."""

    def test_as_informative_as_the_published_design(self, design: pd.DataFrame) -> None:
        factors = [*MILLS, *COOLANT]
        ours = _log_det(_model_rows(design, factors, TABLE_6_2_MODEL))
        assert ours == pytest.approx(_log_det(_model_rows(TABLE_6_2, factors, TABLE_6_2_MODEL)), abs=1e-9)

    def test_six_blends_each_twice_once_at_each_process_setting(self, design: pd.DataFrame) -> None:
        """Goos and Jones, p. 119: six mill combinations, each twice, once at each level of each coolant factor."""
        blends = design.groupby(["mill1", "mill2", "mill3"])
        assert blends.ngroups == 6
        assert (blends.size() == 2).all()
        assert (blends["spray"].sum() == 0).all()
        assert (blends["oil"].sum() == 0).all()
        assert sorted(map(tuple, design[["mill1", "mill2", "mill3"]].drop_duplicates().to_numpy())) == sorted(
            map(tuple, TABLE_6_2[["mill1", "mill2", "mill3"]].drop_duplicates().to_numpy())
        )

    def test_each_coolant_combination_three_times(self, design: pd.DataFrame) -> None:
        assert (design.groupby(["spray", "oil"]).size() == 3).all()

    def test_leverage_is_two_thirds_at_every_run(self, design: pd.DataFrame) -> None:
        """Goos and Jones, p. 120: the worst relative variance of prediction is two-thirds, 8 terms over 12 runs."""
        x = _model_rows(design, [*MILLS, *COOLANT], TABLE_6_2_MODEL)
        leverage = np.einsum("ij,jk,ik->i", x, np.linalg.inv(x.T @ x), x)
        np.testing.assert_allclose(leverage, 8 / 12)


class TestGenerateDesign:
    """generate_design builds mixture-process designs with the optimal families."""

    FACTORS = (*BLENDS, *SETTINGS)

    def test_automatic_choice_is_d_optimal_for_the_interactions_model(self) -> None:
        result = generate_design(list(self.FACTORS))
        assert result.design_type == "d_optimal"
        assert result.metadata["model_type"] == "interactions"
        assert result.n_runs == result.metadata["n_parameters"] + 3 == 16

    def test_blends_are_proportions_and_settings_actual_units(self) -> None:
        result = generate_design(list(self.FACTORS), budget=16)
        names = [f.name for f in BLENDS]
        for frame in (result.design, result.design_actual):
            np.testing.assert_allclose(frame[names].sum(axis=1), 1.0)
            assert frame[names].min().min() >= 0.1 - 1e-9
            assert frame[names].max().max() <= 0.6 + 1e-9
        assert set(result.design["T"]) <= {-1.0, 0.0, 1.0}
        np.testing.assert_allclose(result.design_actual["T"], 50 + 30 * result.design["T"])

    def test_recorded_region_holds_every_run(self) -> None:
        constraints = ["x1 + x2 <= 0.7", "T + 0.1*S <= 90"]
        result = generate_design(list(self.FACTORS), budget=16, constraints=constraints)
        region = DesignRegion.from_dict(result.metadata["region"])
        assert region.kind == "mixture_process"
        runs = result.design[region.names].to_numpy(dtype=float)
        assert region.feasible(runs).all()
        assert result.metadata["constraints_enforced"]

    def test_squared_process_factors_get_a_middle_level(self) -> None:
        result = generate_design(list(self.FACTORS), design_type="d_optimal", model_type="quadratic", budget=20)
        assert result.metadata["n_parameters"] == 15
        assert (result.design["T"] == 0).any()
        assert (result.design["S"] == 0).any()

    @pytest.mark.parametrize(
        "design_type", ["d_optimal", "i_optimal", "a_optimal", "e_optimal", "g_optimal", "k_optimal"]
    )
    def test_every_optimal_family(self, design_type: str) -> None:
        result = generate_design(list(self.FACTORS), design_type=design_type, model_type="main_effects", budget=8)
        x = _model_rows(result.design, list(self.FACTORS), "main_effects")
        assert result.n_runs == 8
        assert np.linalg.matrix_rank(x) == 5
        assert result.metadata["optimality_criterion"] == design_type

    def test_i_optimal_beats_d_optimal_on_average_variance(self) -> None:
        def average_variance(design_type: str) -> float:
            result = generate_design(list(self.FACTORS), design_type=design_type, budget=16, random_state=0)
            region = DesignRegion.from_dict(result.metadata["region"])
            sample = pd.DataFrame(region.sample(20_000, np.random.default_rng(5)), columns=region.names)
            x = _model_rows(result.design, list(self.FACTORS), "interactions")
            f = _model_rows(sample, list(self.FACTORS), "interactions")
            return float(np.mean(np.einsum("ij,jk,ik->i", f, np.linalg.inv(x.T @ x), f)))

        assert average_variance("i_optimal") < average_variance("d_optimal")

    def test_a_small_budget_gets_a_smaller_model_when_chosen_automatically(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = generate_design(list(self.FACTORS), model_type="quadratic", budget=10)
        assert result.metadata["model_type"] == "main_effects"
        assert "cannot estimate the 'quadratic' model" in caplog.text

    def test_a_budget_below_the_model_size_is_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            result = generate_design(list(self.FACTORS), design_type="d_optimal", budget=10)
        assert result.n_runs == 13
        assert result.metadata["budget_requested"] == 10


class TestRefusals:
    FACTORS = (*BLENDS, *SETTINGS)

    @pytest.mark.parametrize("design_type", ["mixture", "full_factorial", "ccd", "latin_hypercube", "sobol"])
    def test_other_families_are_refused(self, design_type: str) -> None:
        with pytest.raises(ValueError, match=r"process factors \['T', 'S'\] need an optimal design"):
            generate_design(list(self.FACTORS), design_type=design_type, budget=16)

    @pytest.mark.parametrize(
        ("argument", "value"),
        [
            ("hard_to_change", ["T"]),
            ("n_blocks", 2),
            ("fixed_runs", pd.DataFrame([[0.3, 0.3, 0.4, 0.0, 0.0]], columns=["x1", "x2", "x3", "T", "S"])),
            ("candidates", pd.DataFrame([[0.3, 0.3, 0.4, 50.0, 200.0]], columns=["x1", "x2", "x3", "T", "S"])),
        ],
    )
    def test_options_not_supported_yet(self, argument: str, value: object) -> None:
        with pytest.raises(ValueError, match=f"Mixture-process designs do not take {argument} yet"):
            generate_design(list(self.FACTORS), design_type="d_optimal", budget=16, **{argument: value})

    def test_categorical_process_factor(self) -> None:
        factors = [*BLENDS, Factor(name="catalyst", type="categorical", levels=["A", "B"])]
        with pytest.raises(ValueError, match=r"must be continuous; \['catalyst'\] are categorical"):
            generate_design(factors, design_type="d_optimal", budget=12)

    def test_a_constraint_joining_blends_and_settings(self) -> None:
        with pytest.raises(ValueError, match="names both mixture components"):
            generate_design(list(self.FACTORS), budget=16, constraints=[Constraint(expression="x1 + 0.001*T <= 0.6")])

    def test_conflicting_process_constraints(self) -> None:
        with pytest.raises(ValueError, match="No process setting satisfies"):
            generate_design(list(self.FACTORS), budget=16, constraints=["T >= 90"])


class TestEvaluateDesign:
    """evaluate_design reads the model and the region a mixture-process design was generated for."""

    FACTORS = (*BLENDS, *SETTINGS)

    def test_d_efficiency_is_for_the_recorded_model(self) -> None:
        result = generate_design(list(self.FACTORS), design_type="d_optimal", model_type="quadratic", budget=20)
        x = _model_rows(result.design, list(self.FACTORS), "quadratic")
        expected = 100 * np.exp(_log_det(x) / x.shape[1]) / len(x)
        assert evaluate_design(result)["d_efficiency"] == pytest.approx(expected)

    def test_average_variance_is_over_the_mixture_process_region(self) -> None:
        """The exchange's I-criterion and evaluate_design's average are over the same region."""
        result = generate_design(list(self.FACTORS), design_type="i_optimal", budget=16, constraints=["x1 <= 0.4"])
        average = evaluate_design(result, metric="average_prediction_variance")["average_prediction_variance"]
        assert average == pytest.approx(result.metadata["trace_criterion"], rel=0.02)

    def test_a_model_name_is_a_mixture_process_model(self) -> None:
        result = generate_design(list(self.FACTORS), budget=16)
        freedom = evaluate_design(result, model="main_effects", metric="degrees_of_freedom")["degrees_of_freedom"]
        assert freedom["model"] == 4  # five terms, which span the mean as the blending terms sum to 1
        assert freedom["residual"] == 16 - 5

    def test_table_6_2_worst_prediction_variance(self) -> None:
        """Goos and Jones (2011), p. 120: the worst relative variance of prediction is two-thirds."""
        region = DesignRegion([*MILLS, *COOLANT])
        result = evaluate_design(TABLE_6_2, model=TABLE_6_2_MODEL, region=region, metric="g_efficiency")
        assert result["max_prediction_variance"] == pytest.approx(2 / 3)
        assert result["g_efficiency"] == pytest.approx(100.0)

    def test_power_leaves_out_the_blending_terms_only(self) -> None:
        region = DesignRegion([*MILLS, *COOLANT])
        result = evaluate_design(TABLE_6_2, model=TABLE_6_2_MODEL, region=region, metric="power", effect_size=1.0)
        assert set(result["power"]) == {"mill1:mill2", "mill1:mill3", "mill2:mill3", "spray", "oil"}
        assert "not tested against zero" in result["notes"]["power"]


#: A response in the interactions model: blending, one nonlinear blend, two blend-by-process terms, T by S.
TRUE_COEFFICIENTS = {"x1": 10.0, "x2": 12.0, "x3": 8.0, "x1:x2": 6.0, "x1:T": 2.0, "x3:S": -1.5, "T:S": 0.8}
#: The standard deviation of the simulated noise.
NOISE_SD = 0.05


@pytest.fixture(scope="module")
def simulated_runs() -> tuple[pd.DataFrame, str]:
    """Return a 20-run interactions design with a simulated response, and the formula recorded for it."""
    result = generate_design([*BLENDS, *SETTINGS], budget=20, random_state=2)
    design = result.design.drop(columns="RunOrder")
    terms = {name: np.prod([design[n] for n in name.split(":")], axis=0) for name in TRUE_COEFFICIENTS}
    noise = np.random.default_rng(0).normal(0.0, NOISE_SD, len(design))
    y = sum(coefficient * terms[name] for name, coefficient in TRUE_COEFFICIENTS.items()) + noise
    return design.assign(y=y), result.metadata["formula"]


class TestAnalyzeExperiment:
    """The recorded formula fits mixture-process data, with coefficients on the proportions' scale."""

    def test_coefficients_are_recovered(self, simulated_runs: tuple[pd.DataFrame, str]) -> None:
        """Each estimate lies within five standard errors of its true value, on the proportions' scale.

        The standard errors use the simulated noise's known standard deviation, not the residual's,
        whose 7 degrees of freedom make it vary a lot from one design to another: the exchange may
        choose a different design on another platform.
        """
        data, formula = simulated_runs
        out = analyze_experiment(
            data, responses=data[["y"]], model=formula, analysis_type="coefficients", sigma=NOISE_SD
        )
        table = pd.DataFrame(out["coefficients"]).set_index("term")
        expected = pd.Series(TRUE_COEFFICIENTS).reindex(table.index, fill_value=0.0)
        assert (np.abs(table["coefficient"] - expected) < 5 * table["std_error"]).all()

    def test_coded_components_are_refused(self, simulated_runs: tuple[pd.DataFrame, str]) -> None:
        data, formula = simulated_runs
        with pytest.raises(ValueError, match=r"mixture components \['x1', 'x2', 'x3'\] have no coded scale"):
            analyze_experiment(data, responses=data[["y"]], model=formula, analysis_type="coefficients", coding="coded")


def _true_response(points: pd.DataFrame) -> np.ndarray:
    terms = {name: np.prod([points[n] for n in name.split(":")], axis=0) for name in TRUE_COEFFICIENTS}
    return sum(coefficient * terms[name] for name, coefficient in TRUE_COEFFICIENTS.items())


def test_optimum_is_a_blend_in_the_region_and_beats_a_dense_sample(simulated_runs: tuple[pd.DataFrame, str]) -> None:
    """optimize_responses searches the mixture-process region: blends that sum to 1, settings within it."""
    data, formula = simulated_runs
    fit = analyze_experiment(data, responses=data[["y"]], model=formula, analysis_type="coefficients")
    model = {"response_name": "y", "factor_names": ["x1", "x2", "x3", "T", "S"], **fit}
    region = DesignRegion([*BLENDS, *SETTINGS])
    goal = [{"response": "y", "goal": "maximize", "low": 5, "high": 20}]
    best = optimize_responses([model], goal, region=region)["desirability"]
    optimum = pd.DataFrame([best["optimal_coded"]])
    assert best["within_region"] is True
    assert optimum[["x1", "x2", "x3"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    sample = pd.DataFrame(region.sample(50_000, np.random.default_rng(1)), columns=region.names)
    assert _true_response(optimum)[0] >= _true_response(sample).max() - 0.05


class TestEdgeCases:
    FACTORS = (*BLENDS, *SETTINGS)

    def test_a_constraint_that_does_not_parse_gets_the_parser_message(self) -> None:
        with pytest.raises(ValueError, match="is not a valid expression"):
            DesignRegion(list(self.FACTORS), ["x1 + <= 2"])

    def test_too_many_candidates_are_sampled_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from process_improve.experiments import designs_mixture_process

        monkeypatch.setattr(designs_mixture_process, "MAX_CANDIDATES", 60)
        result = generate_design(list(self.FACTORS), design_type="d_optimal", model_type="main_effects", budget=8)
        assert result.metadata["n_candidates"] == 60

    def test_a_region_too_small_for_the_model(self) -> None:
        blends = [Factor(name="a", type="mixture"), Factor(name="b", type="mixture")]
        with pytest.raises(
            ValueError,
            match=r"1 candidate blends at 5 process settings\) cannot support the 'interactions' model with 5 terms",
        ):
            generate_design([*blends, SETTINGS[0]], design_type="d_optimal", budget=6, constraints=["0.5 <= a <= 0.5"])

    def test_space_filling_refuses_components_beside_process_factors(self) -> None:
        from process_improve.experiments.designs_space_filling import space_filling_design

        with pytest.raises(ValueError, match="fill a box or a simplex, not both"):
            space_filling_design(list(self.FACTORS), 10, "maximin")

    def test_evaluate_takes_an_explicit_formula_over_the_region(self) -> None:
        result = generate_design(list(self.FACTORS), budget=16)
        freedom = evaluate_design(result, model="-1 + x1 + x2 + x3 + T", metric="degrees_of_freedom")
        assert freedom["degrees_of_freedom"]["residual"] == 16 - 4
