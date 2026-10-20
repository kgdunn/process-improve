"""Tests for the second-order correlations by kind of pair, and the ``screening_power`` metric, of ``evaluate_design``.

The reference numbers are Jones and Nachtsheim's (2011) Tables 3, 4 and 7, for the six-factor definitive
screening design of their Figure 1 and for its orthogonal two-level alternatives.
"""

from __future__ import annotations

import itertools
import math
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from process_improve.experiments import evaluate_design, evaluate_projections
from process_improve.experiments.designs import generate_design
from process_improve.experiments.evaluate import _second_order_factors, _second_order_pair
from process_improve.experiments.factor import Factor

_FIGURE_1_SIX = "0+---- 0-++++ +0-++- -0+--+ --0+-- ++0-++ -++0+- +--0-+ +-+-0- -+-+0+ ++++-0 ----+0 000000"


def _six_factor_dsd() -> pd.DataFrame:
    """Return Jones and Nachtsheim's (2011) six-factor definitive screening design, Figure 1."""
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


def _by_pair(design: pd.DataFrame, model: str, metric: str) -> dict[str, dict]:
    return evaluate_design(design, model=model, metric=metric)[metric]["by_pair"]


class TestKindsOfPair:
    """A column is a square or an interaction of named factors, and two of them a kind of pair."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("I(A ** 2)", ("A", "A")),
            ("A:B", ("A", "B")),
            ("A:C[S.y]", ("A", "C")),
            ("x1:x10", ("x1", "x10")),
            ("I(A ** 3)", None),
            ("A:B:C", None),
            ("I(A ** 2):B", None),
            ("A", None),
            ("Intercept", None),
        ],
    )
    def test_the_factors_of_a_column(self, name: str, expected: tuple[str, str] | None) -> None:
        assert _second_order_factors(name) == expected

    @pytest.mark.parametrize(
        ("first", "second", "kind"),
        [
            (("A", "A"), ("B", "B"), "two_squares"),
            (("A", "A"), ("A", "B"), "square_and_own_interaction"),
            (("B", "C"), ("C", "C"), "square_and_own_interaction"),
            (("A", "A"), ("B", "C"), "square_and_other_interaction"),
            (("A", "B"), ("C", "D"), "two_interactions"),
            (("A", "B"), ("B", "C"), "two_interactions"),
        ],
    )
    def test_the_kind_of_a_pair(self, first: tuple[str, str], second: tuple[str, str], kind: str) -> None:
        assert _second_order_pair(first, second) == kind


class TestColumnCorrelationByPair:
    """``correlation`` summarises the second-order columns by kind of pair: Table 3 and equation (4)."""

    def test_the_six_factor_design_follows_table_3(self) -> None:
        found = _by_pair(_six_factor_dsd(), "quadratic", "correlation")
        assert found["square_and_own_interaction"]["max_abs_r"] == pytest.approx(0, abs=1e-12)
        other = found["square_and_other_interaction"]
        assert (other["mean_abs_r"], other["max_abs_r"]) == pytest.approx((0.465, 0.465), abs=5e-4)
        both = found["two_interactions"]
        assert (both["mean_abs_r"], both["max_abs_r"]) == pytest.approx((0.357, 0.5), abs=5e-4)
        assert found["two_squares"]["max_abs_r"] == pytest.approx(1 / 3 - 1 / 5)  # equation (4)
        assert [found[kind]["n_pairs"] for kind in found] == [15, 30, 60, 105]

    def test_fewer_than_two_second_order_terms_have_no_pairs(self) -> None:
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=2)), columns=["A", "B"])
        assert _by_pair(design, "A + B + A:B", "correlation") == {}


class TestEstimateCorrelation:
    """Correlations between estimated second-order effects, from (X'X)^-1, by kind of pair: Table 4."""

    def test_two_squares_in_the_model_of_every_main_effect_and_square_follow_table_4(self) -> None:
        design = _six_factor_dsd()
        model = " + ".join(design.columns) + " + " + " + ".join(f"I({c} ** 2)" for c in design.columns)
        found = evaluate_design(design, model=model, metric="estimate_correlation")["estimate_correlation"]
        assert list(found["by_pair"]) == ["two_squares"]  # no interaction in the model
        squares = found["by_pair"]["two_squares"]
        assert (squares["mean_abs_r"], squares["max_abs_r"]) == pytest.approx((0.087, 0.087), abs=5e-4)
        assert len(found["terms"]) == 6

    def test_every_three_factor_projection_follows_table_4(self) -> None:
        """Every projection of an orthogonal DSD gives Table 4's row (for any three factors)."""
        printed = {
            "square_and_own_interaction": 0.128,
            "square_and_other_interaction": 0.483,
            "two_interactions": 0.405,
        }
        found = evaluate_projections(_six_factor_dsd(), 3, model="quadratic", metric="estimate_correlation")
        for projection in found["projections"]:
            by_pair = projection["metrics"]["estimate_correlation"]["by_pair"]
            for kind, value in printed.items():
                assert by_pair[kind]["mean_abs_r"] == pytest.approx(value, abs=5e-4)
                assert by_pair[kind]["max_abs_r"] == pytest.approx(value, abs=5e-4)

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
        found = _by_pair(design, "interactions", "estimate_correlation")
        assert found["two_interactions"]["max_abs_r"] == pytest.approx(0.0, abs=1e-12)

    def test_the_columns_of_one_categorical_interaction_are_not_paired(self) -> None:
        levels = [(a, c) for a in (-1.0, 0.0, 1.0) for c in ("p", "q", "r")]
        design = pd.DataFrame([*levels, (1.0, "p"), (-1.0, "q"), (0.0, "r")], columns=["A", "C"])
        found = evaluate_design(design, model="A + C + I(A ** 2) + A:C", metric="estimate_correlation")
        result = found["estimate_correlation"]
        assert len(result["terms"]) == 3  # the square and the interaction's two columns
        assert list(result["by_pair"]) == ["square_and_own_interaction"]
        assert result["by_pair"]["square_and_own_interaction"]["n_pairs"] == 2

    def test_a_rank_deficient_model_has_none_and_says_why(self) -> None:
        design = pd.DataFrame([[-1, -1, 1], [1, -1, -1], [-1, 1, -1], [1, 1, 1]], columns=["A", "B", "C"], dtype=float)
        found = evaluate_design(design, model="interactions", metric="estimate_correlation")
        assert found["estimate_correlation"] is None
        assert "rank-deficient" in found["notes"]["estimate_correlation"]

    def test_fewer_than_two_second_order_terms_say_so(self) -> None:
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=2)), columns=["A", "B"])
        found = evaluate_design(design, model="A + B + A:B", metric="estimate_correlation")["estimate_correlation"]
        assert found["terms"] == ["A:B"]
        assert found["by_pair"] == {}
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

    def test_each_test_is_the_noncentral_t_power_of_its_coefficient(self) -> None:
        """In the 2^3 factorial an interaction is tested with variance 1/8 on 3 residual degrees of freedom."""
        design = pd.DataFrame(list(itertools.product([-1.0, 1.0], repeat=3)), columns=["a", "b", "c"])
        critical = stats.t.ppf(0.975, 3)
        delta = 1.5 * math.sqrt(8)
        expected = stats.nct.sf(critical, 3, delta) + stats.nct.cdf(-critical, 3, delta)
        assert _hypotheses(design, effect_size=1.5)[3]["average_power"] == pytest.approx([expected], rel=1e-6)

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

    def test_both_new_metrics_are_part_of_all(self) -> None:
        found = evaluate_design(_six_factor_dsd(), model="main_effects", metric="all", n_samples=200)
        assert found["screening_power"]["hypotheses"]
        assert "estimate_correlation" in found
