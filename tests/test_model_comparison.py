"""Extra-sum-of-squares F tests between nested models (#789 B2), against Montgomery's printed tables."""

import doctest
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS

from process_improve.experiments import compare_models, group_test, lm, model_comparison

SQRT2 = math.sqrt(2)


@pytest.fixture
def etch() -> pd.DataFrame:
    """Montgomery's plasma etch 2^3 (Table 6.4), the two replicates one after the other: 16 runs."""
    coded = [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)]
    runs = pd.DataFrame(coded * 2, columns=["A", "B", "C"])
    runs["y"] = [550, 669, 633, 642, 1037, 749, 1075, 729, 604, 650, 601, 635, 1052, 868, 1063, 860]
    return runs


@pytest.fixture
def filtration() -> pd.DataFrame:
    """Montgomery's filtration 2^4 (Table 6.10) with its four centre runs: 20 runs."""
    corners = [(a, b, c, d) for d in (-1, 1) for c in (-1, 1) for b in (-1, 1) for a in (-1, 1)]
    runs = pd.DataFrame([*corners, *[(0, 0, 0, 0)] * 4], columns=["A", "B", "C", "D"])
    runs["y"] = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96, 73, 75, 66, 69]
    return runs


@pytest.fixture
def paint() -> pd.DataFrame:
    """Montgomery's paint formulation (Table 11.20), percentages of three components that sum to 100."""
    rows = [
        (17.5, 32.5, 50.0, 29, 9.539),
        (10.0, 40.0, 50.0, 26, 27.33),
        (15.0, 25.0, 60.0, 17, 29.21),
        (25.0, 25.0, 50.0, 28, 30.46),
        (5.0, 25.0, 70.0, 35, 74.98),
        (5.0, 32.5, 62.5, 31, 31.5),
        (11.25, 32.5, 56.25, 21, 15.59),
        (5.0, 40.0, 55.0, 20, 19.2),
        (18.125, 28.75, 53.125, 29, 23.44),
        (8.125, 28.75, 63.125, 25, 32.49),
        (25.0, 25.0, 50.0, 19, 23.01),
        (15.0, 25.0, 60.0, 14, 41.46),
        (10.0, 40.0, 50.0, 30, 32.98),
        (5.0, 25.0, 70.0, 23, 70.95),
    ]
    return pd.DataFrame(rows, columns=["monomer", "crosslinker", "resin", "hardness", "solids"])


def _assert_printed(found: dict, printed: dict) -> None:
    """Compare each printed number with the test's, to the last digit printed."""
    for key, (value, digits) in printed.items():
        assert found[key] == pytest.approx(value, abs=0.51 * 10.0**-digits), key


class TestPrintedTables:
    def test_lack_of_fit_of_the_reduced_plasma_etch_model(self, etch: pd.DataFrame) -> None:
        """Table 6.7: the reduced model's lack of fit against pure error, the full model's residual."""
        found = compare_models(lm("y ~ A + C + A:C", etch), lm("y ~ A * B * C", etch))
        assert found["df"] == 4
        assert found["df_error"] == 8
        _assert_printed(
            found,
            {"ss": (2837.25, 2), "ms": (709.31, 2), "f": (0.31, 2), "p_value": (0.8604, 4), "ss_error": (18020.50, 2)},
        )

    def test_group_test_builds_the_reduced_model_by_leaving_the_terms_out(self, etch: pd.DataFrame) -> None:
        full = lm("y ~ A * B * C", etch)
        found = group_test(full, ["B", "A:B", "B:C", "A:B:C"])
        assert found["reduced"] == "y ~ A + C + A:C"
        assert found["terms"] == ["B", "A:B", "B:C", "A:B:C"]
        expected = compare_models(lm("y ~ A + C + A:C", etch), full)
        assert {key: found[key] for key in expected} == pytest.approx(expected)

    def test_filtration_model_and_lack_of_fit_beside_a_curvature_term(self, filtration: pd.DataFrame) -> None:
        """Table 6.22: the factorial terms together, and the reduced model's lack of fit, each on 3 df of pure error."""
        full = lm("y ~ A * B * C * D + I(A ** 2)", filtration)
        model = compare_models(lm("y ~ I(A ** 2)", filtration), full)
        assert (model["df"], model["df_error"]) == (15, 3)
        _assert_printed(model, {"ss": (5730.94, 2), "ms": (382.06, 2), "f": (23.51, 2), "p_value": (0.0121, 4)})
        dropped = ["B", "A:B", "B:C", "B:D", "C:D", "A:B:C", "A:B:D", "A:C:D", "B:C:D", "A:B:C:D"]
        lack_of_fit = group_test(full, dropped)
        assert lack_of_fit["df"] == 10
        _assert_printed(lack_of_fit, {"ss": (195.12, 2), "ms": (19.51, 2), "f": (1.20, 2), "p_value": (0.4942, 4)})

    @pytest.mark.parametrize(
        ("reduced", "full", "printed"),
        [
            ("1", "time + temp", {"ss": (10.04, 2), "ms": (5.02, 2), "f": (2.69, 2), "p_value": (0.1166, 4)}),
            ("time + temp", "time * temp", {"ss": (0.25, 2), "f": (0.12, 2), "p_value": (0.7350, 4)}),
            (
                "time * temp",
                "time * temp + I(time ** 2) + I(temp ** 2)",
                {"ss": (17.95, 2), "ms": (8.98, 2), "f": (126.88, 2)},
            ),
            (
                "time * temp + I(time ** 2) + I(temp ** 2)",
                "time * temp + I(time ** 2) + I(temp ** 2) + I(time ** 2 * temp) + I(time * temp ** 2)",
                {"ss": (2.042e-3, 6), "ms": (1.021e-3, 6), "f": (0.010, 3), "p_value": (0.9897, 4)},
            ),
        ],
        ids=["linear vs mean", "2FI vs linear", "quadratic vs 2FI", "cubic vs quadratic"],
    )
    def test_yield_ccd_sequential_model_sums_of_squares(self, reduced: str, full: str, printed: dict) -> None:
        """Table 11.7: Design-Expert tests each model against the one before it, on the residual of the larger."""
        runs = pd.DataFrame(
            {
                "time": [-1, -1, 1, 1, 0, 0, 0, 0, 0, SQRT2, -SQRT2, 0, 0],
                "temp": [-1, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, SQRT2, -SQRT2],
                "y": [76.5, 77.0, 78.0, 79.5, 79.9, 80.3, 80.0, 79.7, 79.8, 78.4, 75.6, 78.5, 77.0],
            }
        )
        found = compare_models(lm(f"y ~ {reduced}", runs), lm(f"y ~ {full}", runs))
        _assert_printed(found, printed)
        if "I(time ** 2)" in full and "I(time ** 2)" not in reduced:
            assert found["p_value"] < 0.0015  # Design-Expert prints 0.001, its floor

    def test_viscosity_extra_sum_of_squares_from_statsmodels_fits(self) -> None:
        """Example 10.6: SS_R(beta_2 | beta_0, beta_1) = 3316.3 and F = 12.39, fitted by the statsmodels formula API."""
        runs = pd.DataFrame(
            {
                "temperature": [80, 93, 100, 82, 90, 99, 81, 96, 94, 93, 97, 95, 100, 85, 86, 87],
                "feed": [8, 9, 10, 12, 11, 8, 8, 10, 12, 11, 13, 11, 8, 12, 9, 12],
                "viscosity": [
                    2256,
                    2340,
                    2426,
                    2293,
                    2330,
                    2368,
                    2250,
                    2409,
                    2364,
                    2379,
                    2440,
                    2364,
                    2404,
                    2317,
                    2309,
                    2328,
                ],
            }
        )
        reduced = smf.ols("viscosity ~ temperature", runs).fit()
        found = compare_models(reduced, smf.ols("viscosity ~ temperature + feed", runs).fit())
        assert found["df"] == 1
        assert found["ss"] == pytest.approx(3316.3, abs=0.1)
        assert found["f"] == pytest.approx(12.39, abs=0.005)

    @pytest.mark.parametrize(
        ("response", "printed"),
        [
            ("hardness", {"ss": (29.13, 2), "ms": (14.56, 2), "f": (0.62, 2), "p_value": (0.5630, 4)}),
            ("solids", {"ss": (2931.09, 2), "f": (43.95, 2)}),
        ],
    )
    def test_mixture_linear_blending_against_the_quadratic_residual(
        self, paint: pd.DataFrame, response: str, printed: dict
    ) -> None:
        """Tables 11.21 and 11.22: the linear blending tested against the residual of the quadratic model reported."""
        linear = f"{response} ~ 0 + monomer + crosslinker + resin"
        quadratic = f"{linear} + monomer:crosslinker + monomer:resin + crosslinker:resin"
        found = compare_models(lm(f"{response} ~ 1", paint), lm(linear, paint), error=lm(quadratic, paint))
        assert (found["df"], found["df_error"]) == (2, 8)
        _assert_printed(found, printed)

    def test_the_examples_in_the_docstrings_run(self) -> None:
        """No CI step runs doctests, so an example whose output drifted would go unnoticed."""
        result = doctest.testmod(model_comparison, report=False)
        assert (result.failed, result.attempted) == (0, 10)


class TestRefusals:
    def test_models_fitted_to_different_runs(self, etch: pd.DataFrame) -> None:
        """A missing value in a column only the full model reads drops that run from it alone."""
        gappy = etch.astype({"B": float})
        gappy.loc[3, "B"] = np.nan
        with pytest.raises(ValueError, match="same runs of the same response"):
            compare_models(lm("y ~ A + C", gappy), lm("y ~ A + B + C", gappy))

    def test_models_of_different_responses(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="same runs of the same response"):
            compare_models(lm("y ~ A", etch), lm("np.log(y) ~ A + B", etch))

    def test_a_reduced_model_that_is_not_nested(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not combinations of the full model's"):
            compare_models(lm("y ~ A", etch), lm("y ~ B + C", etch))

    def test_models_in_the_wrong_order(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="pass the smaller model first"):
            compare_models(lm("y ~ A * B", etch), lm("y ~ A", etch))

    def test_the_same_model_twice(self, etch: pd.DataFrame) -> None:
        """``A + B`` and ``B + A`` span the same columns, so there is nothing to test."""
        with pytest.raises(ValueError, match="nothing to test"):
            compare_models(lm("y ~ A + B", etch), lm("y ~ B + A", etch))

    def test_an_error_model_that_does_not_contain_the_full_model(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="must contain the full model"):
            compare_models(lm("y ~ A", etch), lm("y ~ A + B", etch), error=lm("y ~ A + C", etch))

    def test_a_denominator_with_no_residual(self, etch: pd.DataFrame) -> None:
        """One replicate of the 2^3 and the full model saturated: there is no pure error to test against."""
        single = etch.iloc[:8]
        with pytest.raises(ValueError, match="no residual degrees of freedom"):
            compare_models(lm("y ~ A + C + A:C", single), lm("y ~ A * B * C", single))


class TestGroupTest:
    def test_a_term_named_in_any_order_or_alone(self, etch: pd.DataFrame) -> None:
        """``"C:A"`` names the model's ``A:C``, and one term may be given as a string."""
        full = lm("y ~ A * C", etch)
        assert group_test(full, "C:A")["reduced"] == "y ~ A + C"
        assert group_test(full, ["C:A"])["f"] == pytest.approx(group_test(full, "A:C")["f"])

    def test_the_intercept_and_squares_are_terms_too(self, filtration: pd.DataFrame) -> None:
        full = lm("y ~ A + I(A ** 2)", filtration)
        assert group_test(full, "I(A**2)")["reduced"] == "y ~ A"
        assert group_test(full, "Intercept")["reduced"] == "y ~ 0 + A + I(A ** 2)"

    def test_the_reduced_model_is_fitted_to_the_runs_the_model_used(self, etch: pd.DataFrame) -> None:
        """Leaving out the one column with a gap would otherwise bring its run back into the reduced model."""
        gappy = etch.astype({"B": float})
        gappy.loc[3, "B"] = np.nan
        found = group_test(lm("y ~ A + B + C", gappy), ["B"])
        complete = gappy.dropna()
        expected = compare_models(lm("y ~ A + C", complete), lm("y ~ A + B + C", complete))
        assert {key: found[key] for key in expected} == pytest.approx(expected)

    def test_a_fit_from_the_statsmodels_formula_api(self, etch: pd.DataFrame) -> None:
        found = group_test(smf.ols("y ~ A * B * C", etch).fit(), ["B", "A:B", "B:C", "A:B:C"])
        assert found["ss"] == pytest.approx(2837.25)

    def test_a_categorical_main_effect_beside_its_interaction_tests_nothing(self, etch: pd.DataFrame) -> None:
        """Without ``C(gap)`` the interaction's columns span it, as R's ``update(m, . ~ . - gap)`` finds."""
        named = etch.rename(columns={"A": "gap", "B": "flow", "C": "power"})  # a column C hides patsy's C()
        with pytest.raises(ValueError, match="nothing to test"):
            group_test(lm("y ~ C(gap) * C(flow)", named), ["C(gap)"])

    def test_no_terms(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="at least one term"):
            group_test(lm("y ~ A * B", etch), [])

    def test_a_term_the_model_does_not_have(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=r"no term \['C'\]; its terms are \['Intercept', 'A', 'B', 'A:B'\]"):
            group_test(lm("y ~ A * B", etch), ["A:B", "C"])

    def test_several_terms_in_one_name(self, etch: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="is not one term"):
            group_test(lm("y ~ A * B", etch), ["A * B"])

    def test_a_fit_made_without_a_formula(self, etch: pd.DataFrame) -> None:
        exog = np.column_stack([np.ones(len(etch)), etch[["A", "B"]].to_numpy(dtype=float)])
        with pytest.raises(TypeError, match="fitted from a formula"):
            group_test(OLS(etch["y"], exog).fit(), ["A"])
