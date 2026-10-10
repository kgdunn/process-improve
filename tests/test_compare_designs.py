"""compare_designs and TrueProcess, each checked against an independent answer.

The nine measures follow *Process Improvement using Data*, "Nine measures down one
column" (omars-designs.rst), whose printed numbers are reproduced here digit for digit.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import (
    DesignComparison,
    Factor,
    TrueProcess,
    analyze_experiment,
    compare_designs,
    evaluate_design,
    generate_design,
)
from process_improve.experiments.compare import MEASURES, _model_exponents, _moments
from process_improve.experiments.visualization.plots import create_plot

#: The twelve design points behind the book's centre-run example: six half-rows and their negations.
HALF = np.array([[0, 1, -1], [0, 1, 1], [1, -1, 0], [1, 0, -1], [1, 0, 1], [1, 1, 0]], dtype=float)
POINTS = np.vstack([HALF, -HALF])


def with_centre_runs(n: int) -> np.ndarray:
    return np.vstack([POINTS, np.zeros((n, 3))])


def factors(names: str = "ABC") -> list[Factor]:
    return [Factor(name=n, low=-1, high=1) for n in names]


# --------------------------------------------------------------------------- the book's numbers
@pytest.mark.parametrize(
    ("n_centre", "printed"),
    [
        (1, {"a": "0.3839", "d": "5.384", "e": "0.5626", "i": "0.5125", "g": "1.000", "max_r": "0.3000"}),
        (3, {"a": "0.2173", "d": "6.298", "e": "1.635", "i": "0.3014", "g": "0.6458", "max_r": "0.07143"}),
        (5, {"a": "0.1839", "d": "6.775", "e": "2.635", "i": "0.2592", "g": "0.6125", "max_r": "0.05556"}),
    ],
)
def test_the_six_measures_reproduce_the_book(n_centre, printed):
    """The book prints each value with four significant figures (``f"{v:#.4g}"``)."""
    comparison = compare_designs({"BBD": with_centre_runs(n_centre)})
    values = comparison.values[f"BBD [n={12 + n_centre}]"]
    assert {key: f"{values[key]:#.4g}" for key in printed} == printed


def test_power_reproduces_the_book():
    """The fifteen-run design, full second-order model, |beta|/sigma = 1 and alpha = 0.05."""
    values = compare_designs({"BBD": with_centre_runs(3)}, "quadratic").values["BBD [n=15]"]
    powers = [f"{values[key]:#.4g}" for key in ("power_main", "power_interaction", "power_quadratic")]
    assert powers == ["0.6228", "0.3682", "0.3450"]


def test_the_moments_are_the_books_moment_matrix():
    """The book's moment_matrix(k) for the main-effects-and-quadratics model, written out by hand."""
    k = 4
    book = np.zeros((2 * k + 1, 2 * k + 1))
    book[0, 0] = 1.0
    for i in range(k):
        book[1 + i, 1 + i] = 1 / 3
        book[0, 1 + k + i] = book[1 + k + i, 0] = 1 / 3
        for j in range(k):
            book[1 + k + i, 1 + k + j] = 1 / 5 if i == j else 1 / 9
    np.testing.assert_allclose(_moments(_model_exponents(k, "main_quadratic")), book)


@pytest.mark.slow
def test_i_is_the_prediction_variance_averaged_over_the_cube():
    """Exact moments against a Monte Carlo average, for a full quadratic with interactions."""
    design = generate_design(factors(), "ccd", alpha="face_centered", n_center_points=2)
    exact = compare_designs({"CCD": design}, "quadratic").values["CCD [n=16]"]["i"]
    x = design.design[list("ABC")].to_numpy(dtype=float)

    def expand(points):
        pairs = list(itertools.combinations(range(3), 2))
        return np.column_stack(
            [np.ones(len(points)), points, *[points[:, i] * points[:, j] for i, j in pairs], points**2]
        )

    inverse = np.linalg.inv(expand(x).T @ expand(x))
    f = expand(np.random.default_rng(0).uniform(-1, 1, size=(400_000, 3)))
    sampled = np.einsum("ij,jk,ik->i", f, inverse, f).mean()
    assert exact == pytest.approx(sampled, rel=5e-3)


# --------------------------------------------------------------------------- against evaluate_design
@pytest.mark.parametrize(
    ("design_type", "options"),
    [
        ("box_behnken", {"n_center_points": 3}),
        ("ccd", {"alpha": "rotatable", "n_center_points": 2}),
        ("d_optimal", {"model_type": "quadratic", "budget": 14}),
    ],
)
def test_a_d_e_and_power_agree_with_evaluate_design(design_type, options):
    design = generate_design(factors(), design_type, random_state=1, **options)
    frame = design.design[list("ABC")]
    values = compare_designs({"d": design}, "quadratic").values[f"d [n={len(frame)}]"]
    p, n = 10, len(frame)
    metrics = evaluate_design(frame, model="quadratic", metric=["a_optimality", "d_efficiency", "e_optimality"])
    assert values["a"] * p == pytest.approx(metrics["a_optimality"])
    assert values["d"] == pytest.approx(metrics["d_efficiency"] * n / 100)
    assert values["e"] == pytest.approx(metrics["e_optimality"])
    power = evaluate_design(frame, model="quadratic", metric="power", effect_size=1.0, sigma=1.0)["power"]
    assert values["power_main"] == pytest.approx(min(power[f] for f in "ABC"))
    assert values["power_interaction"] == pytest.approx(min(power[t] for t in ("A:B", "A:C", "B:C")))
    assert values["power_quadratic"] == pytest.approx(min(power[f"I({f} ** 2)"] for f in "ABC"))


# --------------------------------------------------------------------------- feasibility
def test_a_two_level_design_cannot_fit_the_squares():
    """At two levels every square equals 1, and a centre run makes them equal to each other instead."""
    corners = generate_design(factors(), "full_factorial", n_center_points=0)
    centred = generate_design(factors(), "full_factorial", n_center_points=2)
    comparison = compare_designs({"2^3": corners, "2^3 + 2 centre": centred})
    bare, with_centre = (comparison.values[label] for label in comparison.designs)
    assert (bare["fits"], bare["rank"], bare["residual_df"]) == (False, 4, 4)
    assert (with_centre["fits"], with_centre["rank"], with_centre["residual_df"]) == (False, 5, 5)
    assert all(bare[m.key] is None for m in MEASURES if m.group not in {"Feasibility", "Neither"})
    bare_notes, centre_notes = (comparison.notes[label] for label in comparison.designs)
    assert "same value in every run" in bare_notes[0]
    assert "I(A ** 2) and I(B ** 2) and I(C ** 2) cannot be told apart" in centre_notes[0]
    assert any("squares do not vary" in note for note in bare_notes)
    assert comparison.table.loc["Fits the model?"].tolist() == ["", "no (rank 4)", "no (rank 5)"]


def test_a_saturated_design_has_no_power():
    """A three-factor DSD has nine runs; its main-effects-and-quadratics model has seven terms."""
    dsd = generate_design(factors(), "dsd")
    values = compare_designs({"DSD": dsd}, "quadratic").values["DSD [n=9]"]
    assert not values["fits"]
    seven = compare_designs({"DSD": dsd}, "main_quadratic")
    assert seven.values["DSD [n=9]"]["residual_df"] == 2
    saturated = compare_designs(
        {"PB": generate_design(factors(), "plackett_burman", n_center_points=0)}, "main_effects"
    )
    label = saturated.designs[0]
    assert saturated.values[label]["residual_df"] == 0
    assert saturated.values[label]["power_main"] is None
    assert any("No degrees of freedom" in note for note in saturated.notes[label])


def test_power_is_none_for_a_term_type_the_model_lacks():
    values = compare_designs({"BBD": with_centre_runs(3)}).values["BBD [n=15]"]
    assert values["power_interaction"] is None
    assert values["power_main"] is not None


def test_runs_outside_the_cube_are_noted():
    rotatable = generate_design(factors(), "ccd", alpha="rotatable", n_center_points=3)
    comparison = compare_designs({"CCD": rotatable})
    assert any("outside the cube" in note and "1.68" in note for note in comparison.notes[comparison.designs[0]])


# --------------------------------------------------------------------------- the grid for G
@pytest.mark.parametrize(
    ("k", "described"),
    [
        (3, "a 7-level grid of 343 points"),
        (6, "a 5-level grid of 15625 points"),
        (8, "a 3-level grid of 6561 points"),
        (11, "the 2048 corners and 20000 points drawn uniformly from the cube"),
    ],
)
def test_g_is_searched_over_a_stated_grid(k, described):
    names = "ABCDEFGHJKL"[:k]
    dsd = generate_design(factors(names), "dsd", n_center_points=3)
    assert compare_designs({"DSD": dsd}, "main_effects").g_points == described


def test_g_on_a_coarser_grid_is_the_maximum_over_its_points():
    """At six factors the 5-level grid holds every 3-level point, so G is at least the 3-level maximum."""
    design = generate_design(factors("ABCDEF"), "box_behnken", n_center_points=3)
    values = compare_designs({"BBD": design}, "main_quadratic").values[f"BBD [n={design.n_runs}]"]
    x = design.design[list("ABCDEF")].to_numpy(dtype=float)
    expand = lambda p: np.column_stack([np.ones(len(p)), p, p**2])  # noqa: E731
    inverse = np.linalg.inv(expand(x).T @ expand(x))
    grid = expand(np.array(list(itertools.product([-1.0, 0.0, 1.0], repeat=6))))
    assert values["g"] >= np.einsum("ij,jk,ik->i", grid, inverse, grid).max() - 1e-12


# --------------------------------------------------------------------------- the result object
def test_designs_are_labelled_with_their_size_in_the_order_given():
    results = [generate_design(factors(), t, n_center_points=3) for t in ("box_behnken", "box_behnken", "ccd")]
    comparison = compare_designs(results)
    assert comparison.designs == ("box_behnken [n=15]", "box_behnken [n=15] (2)", "ccd [n=17]")
    unnamed = compare_designs([with_centre_runs(1)])
    assert unnamed.designs == ("Design 1 [n=13]",)
    assert unnamed.terms[:4] == ("Intercept", "A", "B", "C")


def test_the_table_and_long_form_hold_every_measure():
    comparison = compare_designs({"BBD": with_centre_runs(3), "DSD": generate_design(factors(), "dsd")})
    table = comparison.table
    assert list(table.index) == [m.label for m in MEASURES]
    assert list(table.columns) == ["Better", "BBD [n=15]", "DSD [n=9]"]
    assert table.loc["G, largest prediction variance", "Better"] == "↓"
    assert table.loc["D, |M|^(1/p)", "Better"] == "↑"
    assert table.loc["Power, interaction", "DSD [n=9]"] == "", "the model has no interactions"
    assert table.loc["A/p, average coefficient variance", "BBD [n=15]"] == "0.2173", "as the book prints it"
    assert table.loc["Runs", "DSD [n=9]"] == "9"
    assert len(comparison.long) == 2 * len(MEASURES)


def test_to_dict_is_json_ready():
    comparison = compare_designs({"2^3": generate_design(factors(), "full_factorial", n_center_points=0)})
    payload = comparison.to_dict()
    text = json.dumps(payload, allow_nan=False)
    assert payload["book"].startswith("https://learnche.org/pid/")
    assert payload["values"]["2^3 [n=8]"]["fits"] is False
    matrix = payload["correlations"]["2^3 [n=8]"]["matrix"]
    assert matrix[3][3] is None, "a square that does not vary has no correlation"
    assert "NaN" not in text


def test_the_result_is_read_only():
    comparison = compare_designs({"BBD": with_centre_runs(3)})
    assert isinstance(comparison, DesignComparison)
    with pytest.raises(dataclasses.FrozenInstanceError):
        comparison.model = "quadratic"  # type: ignore[misc]
    with pytest.raises(TypeError):
        comparison.values["BBD [n=15]"]["a"] = 0.0  # type: ignore[index]


def test_correlations_cover_main_effects_squares_and_interactions():
    frame = compare_designs({"BBD": with_centre_runs(3)}).correlations["BBD [n=15]"]
    assert list(frame.columns) == ["A", "B", "C", "I(A ** 2)", "I(B ** 2)", "I(C ** 2)", "A:B", "A:C", "B:C"]
    np.testing.assert_allclose(frame.iloc[:3, :3], np.eye(3), atol=1e-12)
    assert frame.loc["I(A ** 2)", "I(B ** 2)"] == pytest.approx(1 / 14)


def test_one_factor_has_no_pair_of_second_order_terms():
    values = compare_designs({"line": np.array([[-1.0], [0.0], [1.0], [0.0]])}).values["line [n=4]"]
    assert values["max_r"] is None
    assert values["fits"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "cubic"}, "model must be one of"),
        ({"effect_size": 0.0}, "effect_size must be a positive"),
        ({"effect_size": math.inf}, "effect_size must be a positive"),
        ({"significance_level": 1.0}, "significance_level must lie between"),
    ],
)
def test_impossible_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        compare_designs({"BBD": with_centre_runs(3)}, **kwargs)


def test_designs_that_cannot_be_compared_are_refused():
    with pytest.raises(TypeError, match="mapping"):
        compare_designs(pd.DataFrame(with_centre_runs(3), columns=list("ABC")))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one design"):
        compare_designs({})
    with pytest.raises(ValueError, match="same factors"):
        compare_designs({"three": with_centre_runs(3), "two": np.zeros((4, 2))})
    with pytest.raises(ValueError, match="name their factors differently"):
        compare_designs(
            {
                "ABC": pd.DataFrame(with_centre_runs(3), columns=list("ABC")),
                "XYZ": pd.DataFrame(with_centre_runs(3), columns=list("XYZ")),
            }
        )
    with pytest.raises(ValueError, match="not numeric"):
        compare_designs({"labels": pd.DataFrame({"A": ["lo", "hi"], "B": [-1.0, 1.0]})})
    with pytest.raises(ValueError, match="finite numbers"):
        compare_designs({"gap": np.array([[0.0, np.nan], [1.0, 1.0]])})


def test_run_bookkeeping_columns_are_not_factors():
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    frame = design.design[list("ABC")].assign(RunOrder=range(1, design.n_runs + 1))
    from_frame = compare_designs({"BBD": frame}).values["BBD [n=15]"]
    from_result = compare_designs({"BBD": design}).values["BBD [n=15]"]
    assert from_frame["a"] == pytest.approx(from_result["a"])


# --------------------------------------------------------------------------- the correlation map
def test_the_correlation_map_draws_the_comparisons_matrix():
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    rows = design.design[list("ABC")].to_dict("records")
    spec = create_plot("correlation_map", design_data=rows).to_spec()
    layer = spec.panels[0].layers[0]
    assert layer.style["x_grid"] == ["A", "B", "C", "A²", "B²", "C²", "A·B", "A·C", "B·C"]
    assert layer.style["y_grid"] == layer.style["x_grid"][::-1], "the first term reads at the top"
    max_r = compare_designs({"BBD": design}).values["BBD [n=15]"]["max_r"]
    assert spec.metadata["max_abs_r_second_order"] == pytest.approx(max_r)
    dividers = sorted((a.axis, a.value) for a in spec.panels[0].annotations)
    assert dividers == [("x", 2.5), ("x", 5.5), ("y", 2.5), ("y", 5.5)]


def test_the_correlation_map_renders_in_both_backends():
    rows = generate_design(factors(), "full_factorial", n_center_points=0).design[list("ABC")].to_dict("records")
    plot = create_plot("correlation_map", design_data=rows)
    trace = plot.to_plotly()["data"][0]
    assert (trace["type"], trace["zmin"], trace["zmax"]) == ("heatmap", 0.0, 1.0)
    assert trace["z"][-4][3] is None, "the square of a two-level factor does not vary"
    option = plot.to_echarts()
    json.dumps(option, allow_nan=False)
    assert option["xAxis"]["type"] == option["yAxis"]["type"] == "category"
    assert (option["visualMap"]["min"], option["visualMap"]["max"]) == (0.0, 1.0)


def test_the_correlation_map_without_data_says_so():
    assert "no design data" in create_plot("correlation_map").to_spec().title
    only_response = create_plot("correlation_map", design_data=[{"y": 1.0}, {"y": 2.0}], response_column="y")
    assert "no factor columns" in only_response.to_spec().title


# --------------------------------------------------------------------------- TrueProcess
def test_spellings_of_a_term_agree():
    runs = pd.DataFrame({"A": [-1.0, 1.0, 0.5], "B": [1.0, 1.0, -0.5]})
    means = [
        TrueProcess({"Intercept": 1, "A": 2, product: 3, square: -1}, sigma=1).mean(runs)
        for product, square in [("A:B", "I(B ** 2)"), ("A*B", "B**2"), ("B:A", "B^2"), ("B * A", "B:B")]
    ]
    for mean in means[1:]:
        np.testing.assert_allclose(mean, means[0])
    np.testing.assert_allclose(means[0], 1 + 2 * runs["A"] + 3 * runs["A"] * runs["B"] - runs["B"] ** 2)


def test_the_mean_is_the_model_evaluated_at_each_run():
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    process = TrueProcess({"Intercept": 62, "A": 4, "B": -3.1, "C": 1.75, "A:B": 1.5, "I(C ** 2)": -2.25}, sigma=0.5)
    x = design.design
    expected = 62 + 4 * x["A"] - 3.1 * x["B"] + 1.75 * x["C"] + 1.5 * x["A"] * x["B"] - 2.25 * x["C"] ** 2
    np.testing.assert_allclose(process.mean(design), expected)
    assert process.factors == ("A", "B", "C")


def test_simulate_is_reproducible_and_noisy():
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    process = TrueProcess({"Intercept": 10, "A": 1}, sigma=2.0)
    first, again, other = (process.simulate(design, random_state=s) for s in (7, 7, 8))
    pd.testing.assert_series_equal(first, again)
    assert not first.equals(other)
    assert first.name == "y"
    assert first.index.equals(design.design.index)
    rng = np.random.default_rng(3)
    drawn = process.simulate(design, random_state=rng)
    expected = process.mean(design) + np.random.default_rng(3).normal(0.0, 2.0, design.n_runs)
    np.testing.assert_allclose(drawn, expected)


def test_an_analysis_recovers_the_simulated_truth():
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    truth = {"Intercept": 62.0, "A": 4.0, "B": -3.1, "C": 1.75, "A:B": 1.5, "I(C ** 2)": -2.25}
    y = TrueProcess(truth, sigma=0.01).simulate(design, random_state=0)
    fit = analyze_experiment(design.design[list("ABC")], y, model="quadratic", analysis_type="coefficients")
    estimates = {c["term"]: c["coefficient"] for c in fit["coefficients"]}
    for term, value in truth.items():
        assert estimates[term] == pytest.approx(value, abs=0.05)
    assert estimates["A:C"] == pytest.approx(0.0, abs=0.05), "a term the process lacks is estimated near zero"


@pytest.mark.parametrize(
    ("coefficients", "match"),
    [
        ({"A:B:C": 1.0}, "only second-order models"),
        ({"A:B": 1.0, "B*A": 2.0}, "are the same term"),
        ({"A^2": 1.0, "I(A ** 2)": 2.0}, "are the same term"),
        ({"2A": 1.0}, "is not a factor name"),
        ({"A+B": 1.0}, "is not a factor name"),
        ({"": 1.0}, "is not a factor name"),
        ({"A": math.nan}, "finite number"),
    ],
)
def test_terms_that_cannot_be_simulated_are_refused(coefficients, match):
    with pytest.raises(ValueError, match=match):
        TrueProcess(coefficients, sigma=1.0)


@pytest.mark.parametrize("sigma", [0.0, -1.0, math.nan, math.inf])
def test_the_noise_must_be_positive(sigma):
    with pytest.raises(ValueError, match="sigma must be a positive"):
        TrueProcess({"A": 1.0}, sigma=sigma)


def test_a_process_needs_the_designs_factors():
    process = TrueProcess({"Intercept": 1.0, "Z": 2.0}, sigma=1.0)
    with pytest.raises(ValueError, match=r"\['Z'\]"):
        process.mean(pd.DataFrame({"A": [0.0], "B": [1.0]}))
    with pytest.raises(TypeError, match="DesignResult or a DataFrame"):
        process.mean(np.zeros((2, 2)))  # type: ignore[arg-type]


def test_the_coefficients_are_read_only():
    process = TrueProcess({"A*B": 1.0}, sigma=1.0)
    assert dict(process.coefficients) == {"A:B": 1.0}
    with pytest.raises(TypeError):
        process.coefficients["A:B"] = 2.0  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        process.sigma = 2.0  # type: ignore[misc]
