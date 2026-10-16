"""Mixture designs: ``api_make_design`` for a blend's components, and the analysis, simulation and pictures after it.

In a mixture the factors are the proportions of a blend's components: each lies between its
lower and upper bounds, 0 and 1 unless the reader sets others, and the proportions of every run
sum to 1. The page makes three kinds of mixture design (a simplex lattice, a simplex centroid,
and extreme vertices for a mixture with bounds) and analyses their results with a Scheffé model,
which has no intercept. A design's run sheet lists the proportions as they are weighed out.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from process_improve.experiments import Factor, generate_design
from tests.web._support import ALL_MIXTURES, held_of, run_sheet, sheets_of


def components(*names: str, **bounds: tuple[float, float]) -> list[dict]:
    """Return a mixture's components as the brief sends them: blank bounds, 0 and 1, unless ``bounds`` gives some."""
    return [
        {"name": n, "low": bounds.get(n, (None, None))[0], "high": bounds.get(n, (None, None))[1], "units": ""}
        for n in names
    ]


#: Bounds that cut the triangle of three components down to a hexagon, and the four components' tetrahedron
#: down to a smaller solid.
BOUNDED = {
    3: components("A", "B", "C", A=(0.2, 0.7), B=(0.1, 0.6), C=(0.0, 0.5)),
    4: components("A", "B", "C", "D", A=(0.1, 0.5), B=(0.1, 0.5), C=(0.0, 0.4), D=(0.05, 0.4)),
}

#: Each kind of design, as the brief asks for it, and the runs it makes in three and in four components.
KINDS = {
    "simplex lattice, degree 1": ({"design_type": "simplex_lattice", "lattice_degree": 1}, {3: 4, 4: 5}),
    "simplex lattice, degree 2": ({"design_type": "simplex_lattice", "lattice_degree": 2}, {3: 7, 4: 11}),
    "simplex centroid": ({"design_type": "simplex_centroid"}, {3: 7, 4: 15}),
    "extreme vertices, linear": ({"design_type": "extreme_vertices", "mixture_model": "scheffe_linear"}, {}),
    "extreme vertices, quadratic": ({"design_type": "extreme_vertices", "mixture_model": "scheffe_quadratic"}, {}),
}


def make(api, kind: str, q: int, **change: object) -> dict:
    """Make one kind of mixture design in ``q`` components, the bounded components for extreme vertices."""
    request, _ = KINDS[kind]
    names = "ABCDEF"[:q]
    factors = BOUNDED[q] if request["design_type"] == "extreme_vertices" else components(*names)
    return api.ok("api_make_design", {**request, "factors": factors, **change})


def proportions(design: dict) -> pd.DataFrame:
    """Return a design's runs as their proportions, a column per component."""
    return run_sheet(design)[[f["name"] for f in design["spec"]["factors"]]].astype(float)


def test_every_mixture_the_page_offers_is_tested_here(bootstrap):
    assert list(bootstrap.MIXTURES) == list(ALL_MIXTURES)
    assert {request["design_type"] for request, _ in KINDS.values()} == set(ALL_MIXTURES)
    assert not set(bootstrap.MIXTURES) & set(bootstrap.DESIGNS), "the comparison never tries a mixture"


# --------------------------------------------------------------------------- the designs
@pytest.mark.parametrize("q", [3, 4])
@pytest.mark.parametrize("kind", KINDS)
def test_every_run_is_a_blend_within_its_bounds(api, kind, q):
    design = make(api, kind, q)
    blends = proportions(design)
    np.testing.assert_allclose(blends.sum(axis=1), 1.0, atol=1e-9)
    for f in design["spec"]["factors"]:
        assert (blends[f["name"]] >= f["low"] - 1e-9).all(), f["name"]
        assert (blends[f["name"]] <= f["high"] + 1e-9).all(), f["name"]
    expected = KINDS[kind][1].get(q)
    if expected is not None:
        assert design["n_runs"] == expected
    assert design["columns"] == ["Std order", "Run order", *"ABCD"[:q], "y"]
    assert design["facts"]["notes"][0] == "Each run is a blend: its proportions sum to 1."
    assert design["outside_range"] == []


@pytest.mark.parametrize("q", [3, 4])
def test_a_simplex_lattice_holds_the_pure_components_and_the_50_50_blends(api, q):
    blends = proportions(make(api, "simplex lattice, degree 2", q)).to_numpy()
    expected = [np.eye(q)[i] for i in range(q)]
    expected += [(np.eye(q)[i] + np.eye(q)[j]) / 2 for i, j in itertools.combinations(range(q), 2)]
    expected.append(np.full(q, 1 / q))  # the overall centroid, to check the fit
    assert sorted(map(tuple, blends.round(9))) == sorted(map(tuple, np.array(expected).round(9)))


@pytest.mark.parametrize("q", [3, 4])
def test_a_simplex_centroid_is_every_blend_of_equal_parts(api, q):
    blends = proportions(make(api, "simplex centroid", q)).to_numpy()
    expected = [
        np.isin(np.arange(q), subset) / len(subset)
        for r in range(1, q + 1)
        for subset in itertools.combinations(range(q), r)
    ]
    assert len(blends) == 2**q - 1
    assert sorted(map(tuple, blends.round(9))) == sorted(map(tuple, np.array(expected, dtype=float).round(9)))


def test_extreme_vertices_place_runs_at_the_corners_of_the_bounded_region(api):
    """The hexagon the bounds leave of the triangle: its six corners, the midpoints of its edges, and its centroid."""
    design = make(api, "extreme vertices, quadratic", 3)
    blends = proportions(design).round(9)
    corners = {(0.2, 0.6, 0.2), (0.4, 0.6, 0.0), (0.7, 0.3, 0.0), (0.7, 0.1, 0.2), (0.4, 0.1, 0.5), (0.2, 0.3, 0.5)}
    assert corners <= set(map(tuple, blends.to_numpy()))
    assert design["n_runs"] == 13
    assert design["model"] == "scheffe_quadratic"
    assert "The 6 extreme vertices of the blends your bounds allow" in design["facts"]["notes"][1]


def test_the_run_sheet_is_the_library_s_design_in_its_randomised_order(api):
    """The design the page makes is ``generate_design(design_type="mixture")``, with the seed it stores."""
    design = make(api, "simplex lattice, degree 2", 4)
    factors = [Factor(name=n, type="mixture", low=0, high=1) for n in "ABCD"]
    made = generate_design(factors, design_type="mixture", **design["spec"]["options"])
    np.testing.assert_allclose(proportions(design), made.design_actual[list("ABCD")].to_numpy(float), atol=1e-11)
    assert run_sheet(design)["Std order"].tolist() == made.run_order
    assert design["spec"]["options"] == {"random_state": 42, "model_type": "scheffe_quadratic"}


@pytest.mark.parametrize(
    ("kind", "model"),
    [
        ("simplex lattice, degree 1", "scheffe_linear"),
        ("simplex lattice, degree 2", "scheffe_quadratic"),
        ("simplex centroid", "scheffe_quadratic"),
        ("extreme vertices, linear", "scheffe_linear"),
    ],
)
def test_each_design_is_analysed_with_the_model_it_supports(api, kind, model):
    """A lattice of degree 2 or a centroid is analysed with the quadratic model, which leaves runs for the error."""
    assert make(api, kind, 3)["model"] == model


def test_a_replicated_mixture_makes_each_blend_again(api):
    once, twice = make(api, "simplex centroid", 3), make(api, "simplex centroid", 3, n_replicates=2)
    assert twice["n_runs"] == 2 * once["n_runs"]
    assert "Each blend appears 2 times, so the repeats estimate the pure error." in twice["facts"]["notes"]
    assert sorted(map(tuple, proportions(twice).to_numpy())) == sorted(
        2 * list(map(tuple, proportions(once).to_numpy()))
    )


def test_a_component_s_bounds_are_kept_as_proportions(api):
    design = make(api, "extreme vertices, linear", 3)
    assert [(f["low"], f["high"]) for f in design["spec"]["factors"]] == [(0.2, 0.7), (0.1, 0.6), (0.0, 0.5)]
    blank = make(api, "simplex centroid", 3)
    assert [(f["low"], f["high"]) for f in blank["spec"]["factors"]] == [(0.0, 1.0)] * 3, "blank boxes are 0 and 1"


# --------------------------------------------------------------------------- the analysis
#: A blend's true response: each pure component's, a synergy between A and B, and an antagonism between B and C.
TRUE_BLEND = {"A": 30.0, "B": 25.0, "C": 20.0, "D": 22.0, "A:B": 15.0, "A:C": 0.0, "B:C": -12.0}

#: The same blend in three components.
TRUE_THREE = {term: value for term, value in TRUE_BLEND.items() if "D" not in term}


def blended(blends: pd.DataFrame, terms: dict[str, float]) -> np.ndarray:
    """Return a Scheffé model's response at each blend."""
    return sum(value * blends[term.split(":")].prod(axis=1) for term, value in terms.items()).to_numpy()


@pytest.mark.parametrize(("kind", "q"), [("simplex lattice, degree 2", 3), ("simplex lattice, degree 2", 4)])
def test_the_scheffe_fit_recovers_a_known_blend_exactly(api, kind, q):
    design = make(api, kind, q)
    truth = {t: v for t, v in TRUE_BLEND.items() if all(n in "ABCD"[:q] for n in t.split(":"))}
    y = blended(proportions(design), truth)
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y.tolist()})
    fitted = {c["term"]: c["coefficient"] for c in analysis["coefficients"]}
    assert "Intercept" not in fitted, "a Scheffé model has no intercept"
    assert len(fitted) == q + math.comb(q, 2)
    for term, value in fitted.items():
        assert value == pytest.approx(truth.get(term, 0.0), abs=1e-9), term
    assert analysis["model"] == analysis["formula"] == "scheffe_quadratic"
    assert analysis["lenth"] is None
    assert analysis["equation"].startswith("y = 30·A + 25·B + 20·C")
    assert "A·B" in analysis["equation"]


def test_a_mixture_analysis_tests_the_linear_blending_together(api):
    design = make(api, "simplex centroid", 3, n_replicates=2)
    noise = np.random.default_rng(1).normal(0, 0.5, design["n_runs"])
    y = blended(proportions(design), TRUE_THREE) + noise
    analysis = api.ok("api_analyze_runs", {**held_of(design), "responses": y.tolist()})
    sources = [row["source"] for row in analysis["anova"]]
    assert sources == ["Linear blending", "A·B", "A·C", "B·C", "Residual"]
    assert analysis["anova"][0]["df"] == 2
    assert analysis["lack_of_fit"]["df_pure_error"] == 7, "each of the 7 blends made twice"
    assert any(
        note.startswith("Lenth's method and the effects of two-level factors do not apply")
        for note in analysis["notes"]
    )
    effects = {e["component"]: e for e in analysis["effects"]}
    assert set(effects) == {"A", "B", "C"}
    assert effects["A"]["reference"] == pytest.approx(1 / 3)
    assert (effects["A"]["low"], effects["A"]["high"]) == pytest.approx((0.0, 1.0))


@pytest.mark.parametrize(
    ("kind", "q", "model", "fits"),
    [
        ("simplex centroid", 3, "scheffe_special_cubic", True),
        ("simplex centroid", 4, "scheffe_special_cubic", True),
        ("simplex lattice, degree 2", 4, "scheffe_special_cubic", False),
        ("simplex lattice, degree 1", 3, "scheffe_quadratic", False),
        ("simplex lattice, degree 2", 3, "scheffe_linear", True),
    ],
)
def test_the_reader_chooses_any_model_the_design_can_estimate(api, kind, q, model, fits):
    design = make(api, kind, q)
    y = blended(proportions(design), {n: 10.0 + i for i, n in enumerate("ABCD"[:q])}).tolist()
    reply = api.reply("api_analyze_runs", {**held_of(design), "responses": y, "model": model})
    assert reply["ok"] == fits, reply.get("error")
    if not fits:
        assert reply["error"].startswith(f"This design cannot fit the {model[8:].replace('_', ' ')} blending model")


def test_the_true_values_beside_a_simulated_mixture_s_coefficients(api):
    design = make(api, "simplex lattice, degree 2", 3)
    simulation = api.ok("api_simulate", {**held_of(design), "seed": 3})
    analysis = api.ok(
        "api_analyze_runs", {**held_of(design), "responses": simulation["responses"], "truth": simulation["truth"]}
    )
    truth = {t["term"]: t["coefficient"] for t in simulation["truth"]}
    for c in analysis["coefficients"]:
        assert c["true_value"] == pytest.approx(truth.get(c["term"], 0.0)), c["term"]


# --------------------------------------------------------------------------- the simulation
def test_the_blend_process_has_a_synergy_and_an_antagonism(api, bootstrap):
    design = make(api, "simplex lattice, degree 2", 4)
    simulation = api.ok("api_simulate", {**held_of(design), "seed": 1})
    truth = {t["term"]: t["coefficient"] for t in simulation["truth"]}
    assert simulation["scenario"] == "blend"
    assert simulation["label"] == bootstrap.MIXTURE_SCENARIOS["blend"]["label"]
    assert truth == {"A": 54.0, "B": 48.0, "C": 51.0, "D": 50.0, "A:B": 12.0, "B:C": -10.0}
    assert simulation["equation"] == "y = 54·A + 48·B + 51·C + 50·D + 12·A·B - 10·B·C"
    assert simulation["inactive"] == []


def test_a_mixture_s_process_follows_the_reader_s_typical_value_and_noise(api):
    info = {"label": "Hardness", "typical": 80.0, "noise": 2.0}
    design = make(api, "simplex lattice, degree 2", 3, response_info=info)
    truth = {t["term"]: t["coefficient"] for t in api.ok("api_simulate", {**held_of(design)})["truth"]}
    assert truth == {"A": 88.0, "B": 76.0, "C": 82.0, "A:B": 24.0, "B:C": -20.0}


def test_a_mixture_is_simulated_in_its_proportions(api):
    design = make(api, "simplex centroid", 3)
    simulation = api.ok("api_simulate", {**held_of(design), "sigma": 0.01, "seed": 4})
    truth = {t["term"]: t["coefficient"] for t in simulation["truth"]}
    np.testing.assert_allclose(simulation["responses"], blended(proportions(design), truth), atol=0.05)


# --------------------------------------------------------------------------- the pictures
@pytest.fixture
def plotly():
    return pytest.importorskip("plotly", reason="the page installs plotly for its first chart")


def test_three_components_are_drawn_on_a_triangle(api, plotly):
    design = make(api, "simplex lattice, degree 2", 3, n_replicates=2)
    simulation = api.ok("api_simulate", {**held_of(design), "seed": 1})
    request = {**held_of(design), "responses": simulation["responses"], "truth": simulation["truth"]}
    charts = api.ok("api_plots", request)
    ternary = charts["ternary"]
    assert ternary["components"] == ["A", "B", "C"]
    grid = pd.DataFrame({n: ternary["grid"][key] for n, key in zip("ABC", "abc", strict=True)})
    assert len(grid) == 41 * 42 // 2
    np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-9)
    fitted = {c["term"]: c["coefficient"] for c in api.ok("api_analyze_runs", request)["coefficients"]}
    np.testing.assert_allclose(ternary["grid"]["fitted"], blended(grid, fitted), rtol=1e-5)
    truth = {t["term"]: t["coefficient"] for t in simulation["truth"]}
    np.testing.assert_allclose(ternary["grid"]["true"], blended(grid, truth), rtol=1e-5)
    assert ternary["runs"]["y"] == simulation["responses"]
    assert set(charts["figures"]) == {"residuals"}
    assert "cox" not in charts


def test_a_bounded_triangle_is_drawn_only_where_the_bounds_allow(api, plotly):
    design = make(api, "extreme vertices, quadratic", 3)
    y = blended(proportions(design), TRUE_THREE).tolist()
    grid = api.ok("api_plots", {**held_of(design), "responses": y})["ternary"]["grid"]
    for key, (low, high) in zip("abc", [(0.2, 0.7), (0.1, 0.6), (0.0, 0.5)], strict=True):
        assert min(grid[key]) >= low - 1e-9, key
        assert max(grid[key]) <= high + 1e-9, key


def test_four_components_are_traced_along_their_cox_directions(api, plotly):
    design = make(api, "simplex lattice, degree 2", 4, n_replicates=2)
    y = blended(proportions(design), TRUE_BLEND).tolist()
    cox = api.ok("api_plots", {**held_of(design), "responses": y})["cox"]
    assert [t["component"] for t in cox["traces"]] == ["A", "B", "C", "D"]
    assert cox["reference"] == pytest.approx(dict.fromkeys("ABCD", 0.25))
    at_reference = blended(pd.DataFrame([cox["reference"]]), TRUE_BLEND)[0]
    for trace in cox["traces"]:
        assert [trace["change"][0], trace["change"][-1]] == pytest.approx([-0.25, 0.75]), "from 0 to 1"
        middle = int(np.argmin(np.abs(trace["change"])))
        assert trace["change"][middle] == pytest.approx(0.0, abs=1e-9)
        assert trace["fitted"][middle] == pytest.approx(at_reference, rel=1e-5), "every trace crosses the others there"
        assert trace["true"] is None


# --------------------------------------------------------------------------- the workbook and renaming
def test_a_mixture_comes_back_from_its_workbook_as_a_mixture(api):
    design = make(api, "extreme vertices, quadratic", 3)
    held = held_of(design)
    simulation = api.ok("api_simulate", {**held, "seed": 2})
    responses = simulation["responses"]
    sources = ["simulated"] * len(responses)
    book = api.ok("api_workbook", {**held, "responses": responses, "sources": sources, "model": "scheffe_linear"})
    stored = sheets_of(book["xlsx_base64"])["Runs"]
    np.testing.assert_allclose(stored[["A", "B", "C"]].sum(axis=1), 1.0, atol=1e-9)
    back = api.ok("api_read_workbook", {"xlsx_base64": book["xlsx_base64"]})
    assert back["spec"]["design_type"] == "extreme_vertices"
    assert back["spec"]["factors"] == design["spec"]["factors"]
    assert back["spec"]["analysis_model"] == "scheffe_linear", "the model the reader analysed with"
    assert back["responses"] == responses
    assert back["sources"] == sources
    np.testing.assert_allclose(proportions(back), proportions(design))
    whole = api.ok("api_analyze", {"xlsx_base64": book["xlsx_base64"]})
    runs = api.ok("api_analyze_runs", {**held, "responses": responses, "model": "scheffe_linear"})
    assert [c["coefficient"] for c in whole["coefficients"]] == pytest.approx(
        [c["coefficient"] for c in runs["coefficients"]]
    )


def test_renaming_a_mixture_s_components_keeps_its_runs_and_its_notes(api):
    design = make(api, "simplex lattice, degree 2", 3)
    held = held_of(design)
    simulation = api.ok("api_simulate", {**held, "seed": 1})
    names = {"A": "Water", "B": "Oil", "C": "Wax"}
    factors = [{**f, "name": names[f["name"]]} for f in design["spec"]["factors"]]
    sent = {**held, "facts": design["facts"], "outside_range": []}
    renamed = api.ok("api_relabel", {"design": sent, "factors": factors, "simulation": simulation})
    assert renamed["spec"]["design_type"] == "simplex_lattice"
    assert renamed["columns"] == ["Std order", "Run order", "Water", "Oil", "Wax", "y"]
    assert renamed["facts"] == design["facts"], "no note names a component"
    assert renamed["simulation"]["equation"] == "y = 54·Water + 48·Oil + 51·Wax + 12·Water·Oil - 10·Oil·Wax"
    made = api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": factors})
    assert made["rows"] == held["rows"], "renaming gives what making the design under the new names gives"
    analysed = api.ok(
        "api_analyze_runs",
        {
            "spec": renamed["spec"],
            "columns": renamed["columns"],
            "rows": held["rows"],
            "responses": simulation["responses"],
        },
    )
    assert [c["label"] for c in analysed["coefficients"]] == [
        "Water",
        "Oil",
        "Wax",
        "Water·Oil",
        "Water·Wax",
        "Oil·Wax",
    ]
