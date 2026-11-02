"""error_transmission: the variance a quantity worked out from measurements carries from their errors (#801 F4).

The local linearization gives sum_i (df/dx_i)^2 sigma_i^2. These tests hold it to what can be
worked out another way: the exact slope of a formula, the rules for a sum and a product, a large
simulation, and Montgomery's propagation of error for the filtration example (Example 12.1).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import analyze_experiment
from process_improve.experiments.reference_catalogue import reference_experiments
from process_improve.univariate import error_transmission


def test_the_volume_of_a_sphere_carries_its_diameter_s_error_through_the_slope():
    """V = pi d^3 / 6, so dV/dd = pi d^2 / 2: 6.283 at d = 2, and sigma_V = 6.283 sigma_d."""
    found = error_transmission(lambda d: math.pi * d**3 / 6, means={"d": 2.0}, std_devs={"d": 0.01})
    assert found["value"] == pytest.approx(math.pi * 8 / 6)
    assert found["slopes"]["d"] == pytest.approx(math.pi * 2, rel=1e-8)
    assert found["std_dev"] == pytest.approx(math.pi * 2 * 0.01, rel=1e-8)
    assert found["variance"] == pytest.approx(found["std_dev"] ** 2)
    assert found["shares"] == {"d": 1.0}


def test_a_sum_adds_the_variances_and_a_product_weights_each_by_the_other_input():
    total = error_transmission(lambda a, b: a + b, means={"a": 3.0, "b": 4.0}, std_devs={"a": 0.1, "b": 0.2})
    assert total["variance"] == pytest.approx(0.1**2 + 0.2**2)
    assert total["slopes"] == pytest.approx({"a": 1.0, "b": 1.0})
    product = error_transmission(lambda a, b: a * b, means={"a": 3.0, "b": 4.0}, std_devs={"a": 0.1, "b": 0.2})
    assert product["contributions"] == pytest.approx({"a": 4.0**2 * 0.1**2, "b": 3.0**2 * 0.2**2})
    assert product["shares"]["b"] == pytest.approx(0.36 / 0.52), "b's error matters most"


def test_an_input_known_exactly_transmits_nothing_and_has_no_linearity_check():
    found = error_transmission(
        lambda d, h: math.pi * d**2 * h / 4, means={"d": 2.0, "h": 5.0}, std_devs={"d": 0.01, "h": 0.0}
    )
    assert found["contributions"]["h"] == 0.0
    assert found["slopes"]["h"] == pytest.approx(math.pi, rel=1e-8), "its slope is still worked out: pi d^2 / 4"
    assert set(found["linearity"]) == {"d"}


def test_the_linearity_check_sets_the_changes_at_three_standard_deviations_beside_the_straight_line():
    straight = error_transmission(lambda x: 2 * x + 1, means={"x": 1.0}, std_devs={"x": 0.5})["linearity"]["x"]
    assert (straight["down"], straight["up"], straight["linear"]) == pytest.approx((-3.0, 3.0, 3.0))

    def exp(x: float) -> float:  # the inputs are passed by name, which math.exp does not take
        return math.exp(x)

    curved = error_transmission(exp, means={"x": 0.0}, std_devs={"x": 0.1})
    check = curved["linearity"]["x"]
    assert (check["low"], check["high"]) == pytest.approx((math.exp(-0.3), math.exp(0.3)))
    assert check["down"] == pytest.approx(math.exp(-0.3) - 1)
    assert check["up"] == pytest.approx(math.exp(0.3) - 1)
    assert check["linear"] == pytest.approx(0.3)
    assert check["up"] > check["linear"] > -check["down"], "the curve bends up: unequal changes either way"


@pytest.mark.parametrize(("sd_d", "sd_h"), [(0.01, 0.05), (0.02, 0.0)])
def test_the_variance_agrees_with_a_simulation_when_the_errors_are_small(sd_d, sd_h):
    """A cylinder's volume, pi d^2 h / 4: the linearization against 400 000 simulated measurements."""
    found = error_transmission(
        lambda d, h: math.pi * d**2 * h / 4, means={"d": 2.0, "h": 5.0}, std_devs={"d": sd_d, "h": sd_h}
    )
    rng = np.random.default_rng(7)
    d = rng.normal(2.0, sd_d, 400_000)
    h = rng.normal(5.0, sd_h, 400_000) if sd_h else 5.0
    simulated = np.std(np.pi * d**2 * h / 4)
    assert found["std_dev"] == pytest.approx(simulated, rel=0.01)


def test_montgomery_s_propagation_of_error_at_the_recommended_filtration_setting():
    """Example 12.1: temperature z is a noise factor of variance 1 (coded); concentration high, stirring rate middle.

    The response model's slope in z there is 10.8125 - 9.0625 x2 + 8.3125 x3 = 1.75, so z transmits
    1.75^2 = 3.0625; with the residual mean square of 19.51 the propagation of error is
    sqrt(3.0625 + 19.51) = 4.75, inside the 5.5 of the book's overlay (Figure 12.9).
    """
    [example] = [
        e for e in reference_experiments(source="montgomery-2013") if e["name"] == "montgomery-filtration-robust-design"
    ]
    runs = pd.DataFrame(example["runs"]["rows"], columns=example["runs"]["columns"])
    fit = analyze_experiment(
        runs[["A", "C", "D"]], runs["y"], model="A + C + D + A:C + A:D", analysis_type="coefficients"
    )
    b = {c["term"]: c["coefficient"] for c in fit["coefficients"]}

    def filtration_rate(z: float, x2: float = 1.0, x3: float = 0.0) -> float:
        return b["Intercept"] + b["A"] * z + b["C"] * x2 + b["D"] * x3 + b["A:C"] * z * x2 + b["A:D"] * z * x3

    found = error_transmission(filtration_rate, means={"z": 0.0}, std_devs={"z": 1.0})
    assert found["slopes"]["z"] == pytest.approx(1.75)
    assert found["variance"] == pytest.approx(3.0625)
    residual = fit["model_summary"]["mse_residual"]
    assert residual == pytest.approx(19.51, abs=0.005)
    assert math.sqrt(found["variance"] + residual) == pytest.approx(4.75, abs=0.005)
    assert found["value"] == pytest.approx(70.0625 + 4.9375), "the mean filtration rate there, about 75"


@pytest.mark.parametrize(
    ("means", "std_devs", "message"),
    [
        ({"d": 2.0}, {"h": 0.1}, "means and std_devs must name the same inputs"),
        ({"d": 2.0}, {"d": -0.1}, "Each standard deviation must be a finite number, 0 or more"),
        ({"d": 2.0}, {"d": math.inf}, "Each standard deviation must be a finite number, 0 or more"),
        ({"d": math.nan}, {"d": 0.1}, "Each mean must be a finite number"),
    ],
)
def test_inputs_that_are_not_measurements_are_refused(means, std_devs, message):
    with pytest.raises(ValueError, match=message):
        error_transmission(lambda d: d, means=means, std_devs=std_devs)


def test_a_function_that_is_not_finite_where_it_is_worked_out_is_refused():
    with pytest.raises(ValueError, match=r"The function is inf at \{'x': 0.0"):
        error_transmission(lambda x: 1 / x if x else math.inf, means={"x": 0.0}, std_devs={"x": 0.1})
