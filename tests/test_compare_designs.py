"""compare_designs and TrueProcess, each checked against an independent answer.

The nine measures follow *Process Improvement using Data*, "Nine measures down one
column" (omars-designs.rst), whose printed numbers are reproduced here digit for digit.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import (
    DesignComparison,
    DesignRegion,
    Factor,
    TrueProcess,
    analyze_experiment,
    compare_designs,
    evaluate_design,
    generate_design,
)
from process_improve.experiments import _region_geometry as geometry
from process_improve.experiments.compare import MEASURES, _curved_boundary, _model_exponents, _moments
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


# --------------------------------------------------------------------------- without a region: as before
GOLDEN = json.loads((pathlib.Path(__file__).parent / "fixtures" / "compare_designs_golden" / "golden.json").read_text())


def _assert_same(got, recorded, path="result"):
    """Compare JSON values: numbers to 1e-12 (a last-bit platform difference), everything else exactly."""
    if isinstance(recorded, dict):
        assert isinstance(got, dict), path
        assert list(got) == list(recorded), path
        for key, value in recorded.items():
            _assert_same(got[key], value, f"{path}[{key!r}]")
    elif isinstance(recorded, list):
        assert isinstance(got, list), path
        assert len(got) == len(recorded), path
        for i, (a, b) in enumerate(zip(got, recorded, strict=True)):
            _assert_same(a, b, f"{path}[{i}]")
    elif isinstance(recorded, float) and not isinstance(got, bool):
        assert got == pytest.approx(recorded, rel=1e-12, abs=1e-12), path
    else:
        assert got == recorded, path
        assert type(got) is type(recorded), path


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda case: case["name"])
def test_without_a_region_the_comparison_is_what_it_was(case):
    """Recorded before ``region`` was added: every value, note and definition is unchanged.

    See tests/fixtures/compare_designs_golden/README.md; on the platform that recorded the
    fixture the result is bit-identical.
    """
    designs = {
        label: pd.DataFrame(design["runs"], columns=design["columns"]) for label, design in case["designs"].items()
    }
    comparison = compare_designs(designs, case["model"], **case["kwargs"])
    assert comparison.region is None
    assert comparison.measures is MEASURES
    result = comparison.to_dict()
    region = result.pop("region")
    assert region is None
    if not case["correlations"]:
        result.pop("correlations")
    _assert_same(result, case["result"])
    explicit = compare_designs(designs, case["model"], region=None, **case["kwargs"]).to_dict()
    assert json.dumps(explicit) == json.dumps(comparison.to_dict()), "region=None is the default"


# --------------------------------------------------------------------------- over a region
HEAT = [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60)]
HEAT_LIMIT = "3*T + 5*D <= 600"


def heat_designs() -> dict[str, object]:
    """Two quadratic designs inside the heat limit, and a face-centred CCD with three corners outside it."""
    inside = {
        name: generate_design(HEAT, name, constraints=[HEAT_LIMIT], budget=9, model_type="quadratic", random_state=1)
        for name in ("d_optimal", "i_optimal")
    }
    return {**inside, "CCD": generate_design(HEAT, "ccd", alpha="face_centered", n_center_points=1)}


def prediction_variance(design, points: np.ndarray, model: str = "quadratic") -> np.ndarray:
    """Return f(x)' M^-1 f(x) at coded points, for a two-factor design, written out independently."""

    def expand(p):
        columns = [np.ones(len(p)), p[:, 0], p[:, 1]]
        if model == "quadratic":
            columns += [p[:, 0] * p[:, 1], p[:, 0] ** 2, p[:, 1] ** 2]
        return np.column_stack(columns)

    x = expand(design.design[["T", "D"]].to_numpy(dtype=float))
    f = expand(points)
    return np.einsum("ij,jk,ik->i", f, np.linalg.inv(x.T @ x), f)


def test_i_over_the_region_is_the_average_over_an_independent_sample():
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    designs = heat_designs()
    comparison = compare_designs(designs, "quadratic", region=region)
    sample = region.sample(200_000, np.random.default_rng(12345))
    for (name, design), label in zip(designs.items(), comparison.designs, strict=True):
        independent = prediction_variance(design, sample).mean()
        assert comparison.values[label]["i"] == pytest.approx(independent, rel=0.01), name


def test_g_over_the_region_is_its_largest_prediction_variance():
    """At least the variance at every vertex and nearly the largest over a dense independent sample."""
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    designs = heat_designs()
    comparison = compare_designs(designs, "quadratic", region=region)
    vertices = (region.vertices() - [125.0, 40.0]) / [25.0, 20.0]
    sample = region.sample(100_000, np.random.default_rng(7))
    for (name, design), label in zip(designs.items(), comparison.designs, strict=True):
        g = comparison.values[label]["g"]
        assert g >= prediction_variance(design, vertices).max() - 1e-12, name
        assert g == pytest.approx(max(prediction_variance(design, sample).max(), g), rel=0.01), name


def many_limits() -> tuple[DesignRegion, np.ndarray, np.ndarray]:
    """Return a region of 8 factors and 6 limits with 758 vertices, 41 runs in it, and their variance at each vertex.

    The variance is for the model with every two-factor interaction, written out independently.
    """
    names = [f"X{j}" for j in range(8)]
    rng = np.random.default_rng(0)
    limits = []
    for _ in range(6):
        coefficients = rng.integers(-2, 3, size=8)
        lhs = " + ".join(f"{c}*{n}" for c, n in zip(coefficients, names, strict=True))
        limits.append(f"{lhs} <= {rng.integers(1, 4)}")
    region = DesignRegion([Factor(name=n, low=-1, high=1) for n in names], limits)
    runs = region.sample(41, np.random.default_rng(1))

    def expand(x):
        pairs = [x[:, i] * x[:, j] for i, j in itertools.combinations(range(8), 2)]
        return np.column_stack([np.ones(len(x)), x, *pairs])

    vertices = (region.vertices() - region._centre) / region._half
    f, design = expand(vertices), expand(runs)
    return region, runs, np.einsum("ij,jk,ik->i", f, np.linalg.inv(design.T @ design), f)


def test_g_is_at_least_the_variance_at_every_vertex_of_a_region_with_many_limits():
    """With 8 factors and 6 limits there are too many ways of choosing 8 boundaries for the support points to try.

    They then held 236 of the region's 758 vertices, and G, usually at a vertex, came out 7% low.
    """
    region, runs, variance = many_limits()
    assert len(variance) == 758
    comparison = compare_designs({"inside": pd.DataFrame(runs, columns=region.names)}, "interactions", region=region)
    assert comparison.values["inside [n=41]"]["g"] >= variance.max() * (1 - 1e-12)
    assert "its 758 vertices" in comparison.g_points


def test_g_says_when_the_vertices_are_not_listed(monkeypatch):
    """Past 65536 vertices none are listed, and G, usually at a vertex, can come out low: 3.6% for 14 factors."""
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    assert "not listed" not in compare_designs(heat_designs(), "quadratic", region=region).g_points
    monkeypatch.setattr(geometry, "MAX_VERTICES", 2)  # the heat region has 4
    g_points = compare_designs(heat_designs(), "quadratic", region=DesignRegion(HEAT, [HEAT_LIMIT])).g_points
    assert "vertices" not in g_points.split(";")[0]
    assert g_points.endswith(
        "; the region's vertices are not listed (there can be more than 65536), so G can come out below its "
        "largest variance, which usually sits at a vertex"
    )


def test_evaluate_design_reaches_every_vertex_of_a_region_with_many_limits():
    """evaluate_design(region=) takes its worst case over the region's support points, which now list every vertex.

    They listed the vertices by solving every choice of 8 boundaries, which was given up here, so they
    held 236 of the 758, and evaluate_design's largest variance came out 7% below compare_designs' G.
    """
    region, runs, variance = many_limits()
    support = {tuple(point) for point in region.support_points().round(9)}
    vertices = (region.vertices() - region._centre) / region._half
    assert all(tuple(vertex) in support for vertex in vertices.round(9))
    result = evaluate_design(
        pd.DataFrame(runs, columns=region.names), "interactions", "g_efficiency", region=region, random_state=1
    )
    assert result["max_prediction_variance"] == pytest.approx(variance.max(), rel=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_g_reaches_a_curved_boundary(seed):
    """G usually sits on the boundary, which a uniform sample rarely reaches and a 3-level grid touches at a few points.

    Over the cylinder A**2 + B**2 <= 1 G used to come out up to 12% below the variance at a
    point of its surface. With the points on the boundary it came within 1% of a dense search
    of that surface and its caps, and with the local searches from the best of them it is at
    least the search's largest value and within 0.1% of it.
    """
    region = DesignRegion(factors("ABC"), ["A**2 + B**2 <= 1"])
    runs = region.sample(14, np.random.default_rng(seed))
    comparison = compare_designs({"inside": pd.DataFrame(runs, columns=list("ABC"))}, "quadratic", region=region)
    g = comparison.values["inside [n=14]"]["g"]
    angle, height = np.meshgrid(np.linspace(0, 2 * np.pi, 721), np.linspace(-1, 1, 81))
    side = np.column_stack([np.cos(angle.ravel()), np.sin(angle.ravel()), height.ravel()]) * [1 - 1e-12, 1 - 1e-12, 1]
    radius, turn = np.meshgrid(np.linspace(0, 1 - 1e-12, 41), np.linspace(0, 2 * np.pi, 181))
    disc = np.column_stack([radius.ravel() * np.cos(turn.ravel()), radius.ravel() * np.sin(turn.ravel())])
    caps = np.vstack([np.column_stack([disc, np.full(len(disc), z)]) for z in (-1.0, 1.0)])
    surface = np.vstack([side, caps])
    assert region.feasible(surface).all()

    def expand(x):
        a, b, c = x.T
        return np.column_stack([np.ones(len(x)), a, b, c, a * b, a * c, b * c, a**2, b**2, c**2])

    f, design = expand(surface), expand(runs)
    densest = np.einsum("ij,jk,ik->i", f, np.linalg.inv(design.T @ design), f).max()
    assert densest * (1 - 1e-12) <= g <= densest * (1 + 1e-3)
    assert "points on its boundary" in comparison.g_points


def test_the_boundary_points_lie_on_a_curved_limit():
    """A ray the cylinder's surface stops is bisected onto it; one the box stops ends on a cap.

    G reached the cylinder's surface through the local searches whatever these points were,
    and with no bisection every one of them was the point inside the region its ray started
    from, so G alone could not show that the bisection works.
    """
    region = DesignRegion(factors("ABC"), ["A**2 + B**2 <= 1"])
    sample = region.sample(2000, np.random.default_rng(0))
    boundary = _curved_boundary(region, sample)
    assert len(boundary) == len(sample)
    assert region.feasible(boundary).all()
    radius = np.hypot(boundary[:, 0], boundary[:, 1])
    on_a_cap = np.abs(np.abs(boundary[:, 2]) - 1.0) <= 1e-12
    assert np.all(np.abs(radius[~on_a_cap] - 1.0) <= 1e-9)
    assert (~on_a_cap).sum() > len(boundary) / 2, "most rays meet the curved surface"
    assert _curved_boundary(DesignRegion(factors("ABC"), ["A + B <= 1"]), sample).shape == (0, 3)


def sphere_meets_faces(free_levels: int) -> np.ndarray:
    """Return points where the sphere A**2 + B**2 + C**2 = 1.5 meets faces of the box of A to E.

    These are the sphere's surface inside the box with D and E at their bounds, and the arcs
    where it meets a face of A, B or C (the other two on a circle of radius sqrt(0.5)) with D at
    ``free_levels`` settings across its range and E at its bounds. Each point is moved inside
    the sphere by a relative 1e-12, so that every one of them can be run.
    """
    radius = math.sqrt(1.5) * (1 - 1e-12)
    polar, turn = (a.ravel() for a in np.meshgrid(np.linspace(0, np.pi, 181), np.linspace(0, 2 * np.pi, 361)))
    surface = radius * np.column_stack([np.sin(polar) * np.cos(turn), np.sin(polar) * np.sin(turn), np.cos(polar)])
    surface = surface[np.all(np.abs(surface) <= 1, axis=1)]
    parts = [
        np.column_stack([surface, np.tile(corner, (len(surface), 1))]) for corner in itertools.product([-1, 1], [-1, 1])
    ]
    angle = np.linspace(0, 2 * np.pi, 361)
    circle = math.sqrt(0.5) * (1 - 1e-12) * np.column_stack([np.cos(angle), np.sin(angle)])
    for face, sign, d, e in itertools.product(range(3), [-1, 1], np.linspace(-1, 1, free_levels), [-1, 1]):
        arc = np.empty((len(circle), 5))
        arc[:, face], arc[:, [j for j in range(3) if j != face]], arc[:, 3], arc[:, 4] = sign, circle, d, e
        parts.append(arc)
    return np.vstack(parts)


@pytest.mark.parametrize(("model", "n_runs"), [("quadratic", 30), ("main_quadratic", 16)])
def test_g_reaches_where_a_curved_limit_meets_faces_of_the_box(model, n_runs):
    """With a limit on three of five factors, G sits where the curved boundary meets faces of the box.

    For these designs that is on the sphere with D and E at their bounds (quadratic model), and
    on an arc where it meets the face C = 1 with D inside its range (main effects and squares).
    The boundary points, each on a ray from a point inside the region, reach the sphere with D
    and E inside their ranges, and G came out 12% and 5% below the largest variance there; the
    local searches from the points of largest variance now reach it.
    """
    names = list("ABCDE")
    region = DesignRegion(factors(names), ["A**2 + B**2 + C**2 <= 1.5"])
    runs = region.sample(n_runs, np.random.default_rng(1))
    comparison = compare_designs({"inside": pd.DataFrame(runs, columns=names)}, model, region=region)
    g = comparison.values[f"inside [n={n_runs}]"]["g"]
    points = sphere_meets_faces(free_levels=41)
    assert region.feasible(points).all()

    def expand(x):
        pairs = [x[:, i] * x[:, j] for i, j in itertools.combinations(range(5), 2)] if model == "quadratic" else []
        return np.column_stack([np.ones(len(x)), x, *pairs, x**2])

    f, design = expand(points), expand(runs)
    densest = ((f @ np.linalg.inv(design.T @ design)) * f).sum(axis=1).max()
    assert densest * (1 - 1e-12) <= g <= densest * (1 + 1e-3), "at least the scan's largest, and within 0.1% of it"
    assert "local searches" in comparison.g_points
    again = compare_designs({"inside": pd.DataFrame(runs, columns=names)}, model, region=region)
    assert again.values[f"inside [n={n_runs}]"]["g"] == g, "the same on every run"


def test_a_region_changes_only_i_and_g():
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    designs = heat_designs()
    over_cube = compare_designs(designs, "quadratic")
    over_region = compare_designs(designs, "quadratic", region=region)
    for label in over_cube.designs:
        cube, limited = over_cube.values[label], over_region.values[label]
        assert {k: v for k, v in cube.items() if k not in "ig"} == {k: v for k, v in limited.items() if k not in "ig"}
    pd.testing.assert_frame_equal(over_cube.correlations["CCD [n=9]"], over_region.correlations["CCD [n=9]"])
    d_optimal = "d_optimal [n=9]"
    assert over_region.values[d_optimal]["g"] < over_cube.values[d_optimal]["g"], "the cube's far corner is excluded"


def test_a_region_that_removes_nothing_gives_the_cubes_numbers():
    designs = heat_designs()
    over_cube = compare_designs(designs, "quadratic")
    loose = compare_designs(designs, "quadratic", region=DesignRegion(HEAT, ["T + D <= 1000"]))
    assert loose.g_points.startswith("the 49 points of a 7-level grid of 49 points that lie in the region")
    for label in over_cube.designs:
        assert loose.values[label]["i"] == pytest.approx(over_cube.values[label]["i"], rel=0.02)
        assert loose.values[label]["g"] == pytest.approx(over_cube.values[label]["g"], rel=1e-12)
        assert loose.notes[label] == over_cube.notes[label] == ()


def test_a_region_without_constraints_is_the_cube_exactly():
    designs = heat_designs()
    over_cube = compare_designs(designs, "quadratic").to_dict()
    unlimited = compare_designs(designs, "quadratic", region=DesignRegion(HEAT))
    result = unlimited.to_dict()
    region = result.pop("region")
    assert region == DesignRegion(HEAT).to_dict()
    over_cube.pop("region")
    assert result == over_cube, "the definitions of I and G too: they are the cube's exact moments and grid"
    assert "uniform sample" not in {m.key: m.description for m in unlimited.measures}["i"]


def test_the_comparison_records_its_region_and_says_what_i_and_g_are_over():
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    comparison = compare_designs(heat_designs(), "quadratic", region=region)
    assert comparison.region is region
    payload = comparison.to_dict()
    json.dumps(payload, allow_nan=False)
    assert payload["region"] == region.to_dict()
    described = {m["key"]: m["description"] for m in payload["measures"]}
    assert "averaged over the region" in described["i"]
    assert "worst point of the region" in described["g"]
    assert [m["key"] for m in payload["measures"]] == [m.key for m in MEASURES]
    assert "the cube" not in described["i"] + described["g"]
    assert comparison.g_points.endswith("support points and 20000 points drawn uniformly from it")


def test_runs_outside_the_region_are_noted():
    comparison = compare_designs(heat_designs(), "quadratic", region=DesignRegion(HEAT, [HEAT_LIMIT]))
    assert comparison.notes["d_optimal [n=9]"] == comparison.notes["i_optimal [n=9]"] == ()
    assert comparison.notes["CCD [n=9]"] == (
        "3 of its 9 runs lie outside the region, while I and G are computed over the region.",
    )
    one_out = compare_designs(
        {"one": np.array([[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])},
        "main_effects",
        region=DesignRegion([Factor(name=n, low=-1, high=1) for n in "AB"], ["A + B <= 1.5"]),
    )
    assert one_out.notes["one [n=5]"] == (
        "1 of its 5 runs lies outside the region, while I and G are computed over the region.",
    )


@pytest.mark.parametrize(
    ("factors", "limit"),
    [
        (
            [Factor(name="T", low=100, high=150), Factor(name="D", low=20, high=60), Factor(name="F", low=1, high=7)],
            "120*T + 33*D + 450*F <= 17000",
        ),
        ([Factor(name="P", low=100000, high=100100), Factor(name="T", low=0, high=1)], "3*P + 7*T <= 300251"),
        (
            [Factor(name="t", low=1.7e9, high=1.7e9 + 3600), Factor(name="F", low=0, high=10)],
            "t + 77*F <= 1700000908.5892",
        ),
    ],
)
def test_a_design_of_the_regions_own_vertices_lies_inside_it(factors, limit):
    """A reported corner lies within about 1e-10 (coded) of its boundary; these used to be noted as outside.

    The first, a steep limit, failed the tolerance in the limit's own units; the others, factors
    whose range is narrow next to their values (a 100 Pa window at 1 bar, an hour of a time
    stamp), failed after their corners were rounded to 12 significant figures.
    """
    region = DesignRegion(factors, [limit])
    design = pd.DataFrame((region.vertices() - region._centre) / region._half, columns=region.names)
    notes = compare_designs({"corners": design}, "main_effects", region=region).notes
    assert not any("outside" in note for note in notes[f"corners [n={len(design)}]"])


def test_the_region_can_be_given_as_its_dict_and_is_deterministic():
    region = DesignRegion(HEAT, [HEAT_LIMIT])
    designs = heat_designs()
    first = compare_designs(designs, "quadratic", region=region)
    again = compare_designs(designs, "quadratic", region=region.to_dict())
    assert json.dumps(first.to_dict()) == json.dumps(again.to_dict())
    assert isinstance(again.region, DesignRegion)
    written = {"factors": [f.model_dump(mode="json") for f in HEAT], "constraints": [HEAT_LIMIT]}
    by_hand = compare_designs(designs, "quadratic", region=written)
    assert json.dumps(by_hand.to_dict()) == json.dumps(first.to_dict()), "a bare expression in the dict form"


def test_a_factor_the_region_does_not_name_is_free_over_its_range():
    """Limits on A and B leave C free: the region over A, B and C is the A-B region times C's range.

    The region over B, A and C lists its factors in another order than the design's columns,
    and the limit is not symmetric in A and B, so reading the design's A column as the
    region's B would change the numbers.
    """
    design = generate_design(factors(), "box_behnken", n_center_points=3)
    partial = compare_designs({"BBD": design}, region=DesignRegion(factors("AB"), ["A + 0.5*B <= 0.5"]))
    full = compare_designs({"BBD": design}, region=DesignRegion(factors("BAC"), ["A + 0.5*B <= 0.5"]))
    assert partial.values["BBD [n=15]"] == full.values["BBD [n=15]"]
    assert partial.notes["BBD [n=15]"] == full.notes["BBD [n=15]"]
    assert partial.region.names == ["A", "B"]
    swapped = compare_designs({"BBD": design}, region=DesignRegion(factors("AB"), ["B + 0.5*A <= 0.5"]))
    assert swapped.values["BBD [n=15]"]["i"] != partial.values["BBD [n=15]"]["i"], "the test can tell A from B"


@pytest.mark.parametrize(
    ("region", "error", "match"),
    [
        (DesignRegion([Factor(name=n, type="mixture", low=0, high=1) for n in "ABC"]), ValueError, "mixture region"),
        (DesignRegion(factors("ABZ"), ["A + Z <= 1"]), ValueError, r"names \['Z'\], which the designs do not have"),
        (DesignRegion(factors("AB"), ["A + B >= 3"]), ValueError, "nothing to average"),
        ("A + B <= 1", TypeError, "DesignRegion or its dict form"),
    ],
)
def test_regions_that_cannot_be_compared_over_are_refused(region, error, match):
    with pytest.raises(error, match=match):
        compare_designs({"BBD": with_centre_runs(3)}, region=region)
