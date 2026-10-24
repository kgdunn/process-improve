"""The full cubic Scheffé mixture model: its terms, its designs, its fit and its stepwise search (#788 A10)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from patsy import dmatrix

from process_improve.experiments import Factor, analyze_experiment, design_catalog, evaluate_design, generate_design
from process_improve.experiments.analysis import build_formula
from process_improve.experiments.designs_mixture import _simplex_centroid
from process_improve.experiments.designs_mixture_constrained import (
    SCHEFFE_MODELS,
    mixture_candidates,
    mixture_inequalities,
    scheffe_formula_rhs,
    scheffe_matrix,
    scheffe_model,
)
from process_improve.experiments.reference_catalogue import reference_designs, reference_experiments
from process_improve.experiments.strategy.engine import _n_parameters

NAMES = ["X1", "X2", "X3"]

#: A blending surface with every term of the full cubic in three components.
BETA = {
    "X1": 10.0,
    "X2": 12.0,
    "X3": 8.0,
    "X1:X2": 5.0,
    "X1:X3": -3.0,
    "X2:X3": 2.0,
    "X1:X2:X3": 20.0,
    "I(X1 * X2 * (X1 - X2))": 4.0,
    "I(X1 * X3 * (X1 - X3))": -6.0,
    "I(X2 * X3 * (X2 - X3))": 1.5,
}


def _components(q: int, low: float = 0.0, high: float = 1.0) -> list[Factor]:
    return [Factor(name=f"X{i + 1}", type="mixture", low=low, high=high) for i in range(q)]


def _surface(blends: pd.DataFrame) -> np.ndarray:
    """Evaluate :data:`BETA`'s surface: each term is a product of the blends, as patsy builds it."""
    x1, x2, x3 = (blends[n].to_numpy(dtype=float) for n in NAMES)
    columns = {
        "X1": x1,
        "X2": x2,
        "X3": x3,
        "X1:X2": x1 * x2,
        "X1:X3": x1 * x3,
        "X2:X3": x2 * x3,
        "X1:X2:X3": x1 * x2 * x3,
        "I(X1 * X2 * (X1 - X2))": x1 * x2 * (x1 - x2),
        "I(X1 * X3 * (X1 - X3))": x1 * x3 * (x1 - x3),
        "I(X2 * X3 * (X2 - X3))": x2 * x3 * (x2 - x3),
    }
    return sum(BETA[term] * column for term, column in columns.items())


def _lattice() -> pd.DataFrame:
    """Return the runs of the full cubic's default design in three components."""
    return generate_design(_components(3), "mixture", model_type="cubic").design_actual[NAMES].astype(float)


class TestTerms:
    def test_each_pair_adds_a_cubic_difference_term(self) -> None:
        """The full cubic is the special cubic plus ``delta_ij x_i x_j (x_i - x_j)`` for each pair."""
        rhs = "-1 + (A + B + C) ** 3 + I(A * B * (A - B)) + I(A * C * (A - C)) + I(B * C * (B - C))"
        assert scheffe_formula_rhs(["A", "B", "C"], "scheffe_cubic") == rhs
        assert build_formula("y", ["A", "B", "C"], "scheffe_cubic") == f"y ~ {rhs}"

    def test_two_components_have_no_ternary_term(self) -> None:
        assert scheffe_formula_rhs(["A", "B"], "cubic") == "-1 + (A + B) ** 2 + I(A * B * (A - B))"

    def test_the_matrix_holds_the_formulas_columns(self) -> None:
        """``scheffe_matrix`` (for designs) and the formula (for fits) are the same model, column by column."""
        blends = pd.DataFrame(np.random.default_rng(7).dirichlet(np.ones(3), size=12), columns=NAMES)
        by_formula = dmatrix(scheffe_formula_rhs(NAMES, "scheffe_cubic"), blends, return_type="dataframe")
        order = ["X1", "X2", "X3", "X1:X2", "X1:X3", "X2:X3", *list(BETA)[7:], "X1:X2:X3"]
        np.testing.assert_allclose(scheffe_matrix(blends.to_numpy(), "scheffe_cubic"), by_formula[order].to_numpy())

    @pytest.mark.parametrize("q", [2, 3, 4, 5, 6])
    def test_the_model_has_q_squared_plus_q_choose_3_terms(self, q: int) -> None:
        """``q`` linear, ``C(q, 2)`` binary, ``C(q, 2)`` cubic-difference and ``C(q, 3)`` ternary terms."""
        n_terms = scheffe_matrix(np.full((1, q), 1 / q), "scheffe_cubic").shape[1]
        assert n_terms == q * q + math.comb(q, 3)

    @pytest.mark.parametrize("model", SCHEFFE_MODELS)
    @pytest.mark.parametrize("q", [3, 5])
    def test_the_strategy_engine_counts_the_terms_of_every_model(self, model: str, q: int) -> None:
        assert _n_parameters(_components(q), model) == scheffe_matrix(np.full((1, q), 1 / q), model).shape[1]

    def test_cubic_names_the_full_cubic(self) -> None:
        assert scheffe_model("cubic") == "scheffe_cubic"
        assert generate_design(_components(3), "mixture", model_type="cubic").metadata["model_type"] == "scheffe_cubic"


class TestDesigns:
    @pytest.mark.dataset
    def test_three_components_give_nists_3_3_simplex_lattice(self) -> None:
        """NIST/SEMATECH e-Handbook, section 5.5.4.2, Table 5.3: ten blends in thirds."""
        result = generate_design(_components(3), "mixture", model_type="cubic")
        assert (result.metadata["method"], result.n_runs) == ("simplex_lattice_degree_3", 10)
        entries = [*reference_experiments("nist-ehandbook-ch5"), *reference_designs("nist-ehandbook-ch5")]
        printed = next(e for e in entries if e["name"] == "nist-5-5-4-2-simplex-lattice-3-3")["runs"]["rows"]
        runs = result.design_actual[NAMES].to_numpy(dtype=float)

        def ordered(rows: np.ndarray) -> np.ndarray:
            return rows[np.lexsort(np.round(rows, 3).T[::-1])]

        # The handbook prints the thirds as 0.333 and 0.667 (once 0.6667).
        np.testing.assert_allclose(ordered(runs), ordered(np.array(printed, dtype=float)), atol=1e-3)

    def test_four_components_add_the_centroid_to_the_lattice(self) -> None:
        """The {4, 3} lattice has no run at 1/4 of each component, so the centroid is added."""
        result = generate_design(_components(4), "mixture", model_type="scheffe_cubic")
        assert (result.metadata["method"], result.n_runs) == ("simplex_lattice_degree_3_plus_centroid", 21)
        runs = result.design_actual[["X1", "X2", "X3", "X4"]].to_numpy(dtype=float)
        assert np.linalg.matrix_rank(scheffe_matrix(runs, "scheffe_cubic")) == 20
        assert sum(np.allclose(run, 0.25) for run in runs) == 1

    def test_the_simplex_centroid_cannot_estimate_a_cubic_difference_term(self) -> None:
        """``x_i x_j (x_i - x_j)`` is zero at every simplex-centroid blend, so replicates do not help.

        At a pure blend one of the two proportions is zero; at a centroid of any face they are equal.
        """
        centroid = _simplex_centroid(3)
        assert np.allclose(scheffe_matrix(centroid, "scheffe_cubic")[:, 6:9], 0.0)
        assert np.linalg.matrix_rank(scheffe_matrix(np.vstack([centroid] * 3), "scheffe_cubic")) == 7
        assert np.linalg.matrix_rank(scheffe_matrix(_lattice().to_numpy(), "scheffe_cubic")) == 10

    def test_a_bounded_region_takes_the_points_a_third_along_each_edge(self) -> None:
        """Bounds of 0.1 to 0.7 cut the simplex to a hexagon: 6 vertices, 12 edge third-points, its centroid."""
        factors = _components(3, low=0.1, high=0.7)
        candidates = mixture_candidates(*mixture_inequalities(factors, None))
        assert len(candidates["vertex"]) == len(candidates["edge_midpoint"]) == 6
        assert len(candidates["edge_third"]) == 12
        result = generate_design(factors, "mixture", model_type="scheffe_cubic")
        runs = result.design_actual[NAMES].to_numpy(dtype=float)
        assert (result.metadata["method"], result.n_runs) == ("extreme_vertices", 19)
        assert all(np.isclose(runs, point).all(axis=1).any() for point in candidates["edge_third"])
        assert np.linalg.matrix_rank(scheffe_matrix(runs, "scheffe_cubic")) == 10

    def test_a_budget_below_the_number_of_terms_is_refused(self) -> None:
        with pytest.raises(
            ValueError, match=r"budget=8 run\(s\) cannot estimate the scheffe_cubic model, which has 10"
        ):
            generate_design(_components(3), "mixture", model_type="cubic", budget=8, random_state=1)

    def test_a_budget_of_one_run_per_term_fits_the_model(self) -> None:
        result = generate_design(_components(3), "mixture", model_type="cubic", budget=10, random_state=1)
        assert result.n_runs == 10
        assert np.linalg.matrix_rank(scheffe_matrix(result.design_actual[NAMES].to_numpy(dtype=float), "cubic")) == 10

    def test_the_design_catalog_lists_the_lattices(self) -> None:
        sizes = {
            design_catalog._spec_id(spec): design_catalog._generated_record(spec)["n_runs"]
            for spec in design_catalog.GENERATED
            if spec.get("options", {}).get("model_type") == "cubic"
        }
        assert sizes == {"lib-mixture-k3-cubic": 10, "lib-mixture-k4-cubic": 21, "lib-mixture-k5-cubic": 36}


class TestFit:
    def test_the_lattice_recovers_a_cubic_surface_exactly(self) -> None:
        blends = _lattice()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ten runs for ten terms: no residual degrees of freedom
            result = analyze_experiment(
                blends.assign(y=_surface(blends)),
                response_column="y",
                model="scheffe_cubic",
                analysis_type="coefficients",
            )
        fitted = {row["term"]: row["coefficient"] for row in result["coefficients"]}
        assert fitted == pytest.approx(BETA, abs=1e-9)

    def test_the_mixture_anova_tests_each_cubic_difference_term(self) -> None:
        blends = pd.concat([_lattice()] * 2, ignore_index=True)
        y = _surface(blends) + 0.05 * np.random.default_rng(1).standard_normal(len(blends))
        table = analyze_experiment(
            blends.assign(y=y), response_column="y", model="scheffe_cubic", analysis_type="anova"
        )["anova_table"]
        rows = {row["source"]: row for row in table}
        assert list(rows) == ["Linear mixture", *list(BETA)[3:], "Residual"]
        assert all(rows[term]["df"] == 1 and rows[term]["p_value"] < 0.05 for term in list(BETA)[7:])
        assert rows["Residual"]["df"] == 10

    def test_model_selection_keeps_the_cubic_difference_term_the_data_need(self) -> None:
        """Only ``x1 x2 (x1 - x2)`` curves this surface; the search drops the other six non-linear terms."""
        blends = pd.concat([_lattice()] * 2, ignore_index=True)
        x1, x2, x3 = (blends[n].to_numpy() for n in NAMES)
        y = 10 * x1 + 12 * x2 + 8 * x3 + 6 * x1 * x2 * (x1 - x2)
        y = y + 0.1 * np.random.default_rng(0).standard_normal(len(y))
        result = analyze_experiment(
            blends.assign(y=y), response_column="y", model="scheffe_cubic", analysis_type="model_selection"
        )["model_selection"]
        assert result["candidate_model"] == "scheffe_cubic"
        assert result["direction"] == "backward"
        assert result["selected_terms"] == [*NAMES, "I(X1 * X2 * (X1 - X2))"]
        assert result["selected_formula"] == "y ~ -1 + X1 + X2 + X3 + I(X1 * X2 * (X1 - X2))"


class TestEvaluate:
    def test_a_design_is_evaluated_for_the_cubic_it_was_built_for(self) -> None:
        result = generate_design(_components(3), "mixture", model_type="cubic")
        default = evaluate_design(result, metric="d_efficiency")["d_efficiency"]
        assert default == evaluate_design(result, model="scheffe_cubic", metric="d_efficiency")["d_efficiency"]
        assert default == evaluate_design(result, model="cubic", metric="d_efficiency")["d_efficiency"]
        special = evaluate_design(result, model="scheffe_special_cubic", metric="d_efficiency")["d_efficiency"]
        assert default != pytest.approx(special)
        df = evaluate_design(result, metric="degrees_of_freedom")["degrees_of_freedom"]
        assert (df["model"], df["residual"]) == (9, 0)
