"""Tests for the ``estimate_correlation`` and ``screening_power`` metrics of ``evaluate_design``.

The reference numbers are Jones and Nachtsheim's (2011) Tables 4 and 7, for the six-factor definitive
screening design of their Figure 1 and for its orthogonal two-level alternatives.
"""

from __future__ import annotations

import itertools
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import evaluate_design, evaluate_projections
from process_improve.experiments.designs import generate_design
from process_improve.experiments.evaluate import _second_order_factors
from process_improve.experiments.factor import Factor

_FIGURE_1_SIX = "0+---- 0-++++ +0-++- -0+--+ --0+-- ++0-++ -++0+- +--0-+ +-+-0- -+-+0+ ++++-0 ----+0 000000"


def _six_factor_dsd() -> pd.DataFrame:
    """Jones and Nachtsheim's (2011) six-factor definitive screening design, Figure 1."""
    rows = [[{"+": 1, "-": -1, "0": 0}[c] for c in run] for run in _FIGURE_1_SIX.split()]
    return pd.DataFrame(rows, columns=[f"x{i + 1}" for i in range(6)], dtype=float)


def _two_level(design_type: str, k: int, columns: int, **options: object) -> pd.DataFrame:
    """Return the first *columns* factors of a two-level design with one centre run, in coded units."""
    factors = [Factor(name=f"x{i + 1}", low=-1, high=1) for i in range(k)]
    made = generate_design(factors, design_type, n_center_points=1, **options)
    return pd.DataFrame(made.design)[[f"x{i + 1}" for i in range(columns)]]


def _hypotheses(design: pd.DataFrame, **kwargs: object) -> dict[int, dict]:
    found = evaluate_design(design, metric="screening_power", **kwargs)["screening_power"]
    return {h["hypothesis"]: h for h in found["hypotheses"]}


class TestSecondOrderFactors:
    """A column is a square or an interaction of named factors, or neither."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("I(A ** 2)", ("A",)),
            ("A:B", ("A", "B")),
            ("A:C[S.y]", ("A", "C")),
            ("x1:x10", ("x1", "x10")),
            ("I(x10 ** 2)", ("x10",)),
            ("I(A ** 3)", None),
            ("A:B:C", None),
            ("I(A ** 2):B", None),
            ("A", None),
        ],
    )
    def test_names(self, name: str, expected: tuple[str, ...] | None) -> None:
        assert _second_order_factors(name, ["A", "B", "C", "x1", "x10"]) == expected


class TestEstimateCorrelation:
    """Correlations between estimated second-order effects, from (X'X)^-1, by kind of pair."""

    def test_two_squares_in_the_model_of_every_main_effect_and_square_follow_table_4(self) -> None:
        design = _six_factor_dsd()
        model = " + ".join(design.columns) + " + " + " + ".join(f"I({c} ** 2)" for c in design.columns)
        found = evaluate_design(design, model=model, metric="estimate_correlation")["estimate_correlation"]
        assert found["mean_abs_r_two_squares"] == pytest.approx(0.087, abs=5e-4)
        assert found["max_abs_r_two_squares"] == pytest.approx(found["mean_abs_r_two_squares"])
        assert len(found["terms"]) == 6
        assert "max_abs_r_two_interactions" not in found  # no interaction in the model

    def test_three_factor_projections_follow_table_4(self) -> None:
        """Every projection of an orthogonal DSD gives the same correlations, so their average is Table 4's row."""
        found = evaluate_projections(_six_factor_dsd(), 3, model="quadratic", metric="estimate_correlation")
        averages = found["averages"]
        printed = {
            "square_and_own_interaction": 0.128,
            "square_and_other_interaction": 0.483,
            "two_interactions": 0.405,
        }
        for kind, value in printed.items():
            assert averages[f"mean_abs_r_{kind}"] == pytest.approx(value, abs=5e-4)
            assert averages[f"max_abs_r_{kind}"] == pytest.approx(value, abs=5e-4)

    def test_the_matrix_is_the_correlation_of_the_inverse_information(self) -> None:
        design = _six_factor_dsd()[["x1", "x2", "x3"]]
        found = evaluate_design(design, model="quadratic", metric="estimate_correlation")["estimate_correlation"]
        x = design.to_numpy()
        second = {"x1:x2": x[:, 0] * x[:, 1], "x1:x3": x[:, 0] * x[:, 2], "x2:x3": x[:, 1] * x[:, 2]}
        second |= {f"I(x{i + 1} ** 2)": x[:, i] ** 2 for i in range(3)}
        matrix = np.column_stack([np.ones(len(x)), x, *(second[t] for t in found["terms"])])
        inverse = np.linalg.inv(matrix.T @ matrix)[4:, 4:]
        expected = inverse / np.sqrt(np.outer(np.diag(inverse), np.diag(inverse)))
        np.testing.assert_allclose(found["matrix"], expected, atol=1e-12)

    def test_an_orthogonal_design_estimates_its_interactions_independently(self) -> None:
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=["A", "B", "C"])
        found = evaluate_design(design, model="interactions", metric="estimate_correlation")["estimate_correlation"]
        assert found["max_abs_r_two_interactions"] == pytest.approx(0.0, abs=1e-12)

    def test_a_categorical_factor_s_interaction_is_classed_by_its_factors(self) -> None:
        design = pd.DataFrame(
            [(a, c) for a in (-1.0, 0.0, 1.0) for c in ("lo", "hi")] + [(1.0, "lo"), (-1.0, "hi")], columns=["A", "C"]
        )
        found = evaluate_design(design, model="A + C + I(A ** 2) + A:C", metric="estimate_correlation")
        assert "max_abs_r_square_and_own_interaction" in found["estimate_correlation"]

    def test_the_columns_of_one_categorical_interaction_are_not_paired(self) -> None:
        levels = [(a, c) for a in (-1.0, 0.0, 1.0) for c in ("p", "q", "r")]
        design = pd.DataFrame([*levels, (1.0, "p"), (-1.0, "q"), (0.0, "r")], columns=["A", "C"])
        found = evaluate_design(design, model="A + C + I(A ** 2) + A:C", metric="estimate_correlation")
        result = found["estimate_correlation"]
        assert len(result["terms"]) == 3  # the square and the interaction's two columns
        assert "max_abs_r_two_interactions" not in result
        assert "max_abs_r_square_and_own_interaction" in result

    def test_a_rank_deficient_model_has_none_and_says_why(self) -> None:
        design = pd.DataFrame([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], columns=["A", "B", "C"], dtype=float)
        found = evaluate_design(design, model="interactions", metric="estimate_correlation")
        assert found["estimate_correlation"] is None
        assert "rank-deficient" in found["notes"]["estimate_correlation"]

    def test_fewer_than_two_second_order_terms_say_so(self) -> None:
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=2)), columns=["A", "B"])
        found = evaluate_design(design, model="A + B + A:B", metric="estimate_correlation")["estimate_correlation"]
        assert found["terms"] == ["A:B"]
        assert "Fewer than two" in found["note"]


class TestScreeningPower:
    """Jones and Nachtsheim's (2011) power study: the average power of each kind of test (Tables 6 and 7)."""

    #: Table 7, the six-factor design, |beta|/sigma = 1, 2 and 3. Two cells (hypothesis 1 at 2 and
    #: hypothesis 5 at 1) differ in the fifth decimal and so round the other way; abs=1e-3 covers it.
    TABLE_7_SIX: ClassVar[dict[int, tuple[float, float, float]]] = {
        1: (0.821, 0.999, 1),
        2: (0.236, 0.683, 0.949),
        3: (0.623, 0.993, 1),
        4: (0.774, 0.999, 1),
        5: (0.257, 0.734, 0.970),
        6: (0.681, 0.998, 1),
    }

    def test_the_six_factor_design_follows_table_7(self) -> None:
        found = _hypotheses(_six_factor_dsd())
        assert sorted(found) == [1, 2, 3, 4, 5, 6]
        for number, printed in self.TABLE_7_SIX.items():
            assert found[number]["status"] == "estimable"
            assert found[number]["average_power"] == pytest.approx(printed, abs=1e-3)
            lowest, average = found[number]["min_power"], found[number]["average_power"]
            assert all(low <= mean + 1e-12 for low, mean in zip(lowest, average, strict=True))
        assert [found[n]["n_tests"] for n in range(1, 7)] == [6, 6, 15, 30, 30, 15]

    def test_the_orthogonal_alternative_marks_what_table_7_cannot_test(self) -> None:
        """The 12-run Plackett-Burman design's first six columns and a centre run: Table 7's columns 6 to 8."""
        found = _hypotheses(_two_level("plackett_burman", 11, 6))
        assert found[1]["average_power"][0] == pytest.approx(0.883, abs=5e-4)
        assert found[3]["average_power"][0] == pytest.approx(0.548, abs=5e-4)
        assert found[2]["status"] == "aliased"  # every square has the same column
        assert found[2]["n_aliased"] == 6
        assert found[2]["average_power"] is None
        assert [found[n]["status"] for n in (4, 5, 6)] == ["not estimable"] * 3

    def test_a_resolution_iv_fraction_aliases_each_interaction(self) -> None:
        found = _hypotheses(_two_level("fractional_factorial", 8, 8, resolution=4))
        assert found[3]["status"] == "aliased"
        assert found[3]["n_aliased"] == 28

    def test_a_resolution_iii_fraction_aliases_each_main_effect_with_an_interaction(self) -> None:
        design = pd.DataFrame([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], columns=["A", "B", "C"], dtype=float)
        found = _hypotheses(design)
        assert found[1]["status"] == "aliased"  # A = BC, B = AC, C = AB
        assert found[1]["n_aliased"] == 3

    def test_an_effect_size_gives_one_power_and_scales_with_sigma(self) -> None:
        design = _six_factor_dsd()
        one = _hypotheses(design, effect_size=1.0)
        doubled = _hypotheses(design, effect_size=2.0, sigma=2.0)
        assert one[1]["average_power"] == pytest.approx([0.821], abs=5e-4)
        for number in one:
            assert doubled[number]["average_power"] == pytest.approx(one[number]["average_power"])

    def test_random_effects_of_ratio_zero_give_the_ordinary_least_squares_powers(self) -> None:
        dsd = _six_factor_dsd()
        design = dsd.assign(WholePlot=dsd["x1"])  # x1 set once per whole plot: hard to change
        ordinary = _hypotheses(dsd)
        shared = _hypotheses(design, random_effects={"WholePlot": 0.0})
        for number in ordinary:
            assert shared[number]["average_power"] == pytest.approx(ordinary[number]["average_power"])
        larger = _hypotheses(design, random_effects={"WholePlot": 1.0})
        assert larger[1]["average_power"][0] < ordinary[1]["average_power"][0]  # x1 is tested between plots

    def test_a_categorical_factor_is_refused_with_a_note(self) -> None:
        design = pd.DataFrame({"A": [-1.0, 0.0, 1.0, 1.0, -1.0, 0.0], "C": ["a", "b", "a", "b", "a", "b"]})
        found = evaluate_design(design, metric="screening_power")
        assert found["screening_power"] is None
        assert "numeric factors only" in found["notes"]["screening_power"]

    def test_both_metrics_are_part_of_all(self) -> None:
        found = evaluate_design(_six_factor_dsd(), model="main_effects", metric="all", n_samples=200)
        assert found["screening_power"]["hypotheses"]
        assert "estimate_correlation" in found
