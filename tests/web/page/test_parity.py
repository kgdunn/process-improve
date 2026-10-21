"""The page against the Python it mirrors.

``web/index.html`` and ``web/app.js`` repeat some of ``web/bootstrap.py``'s choices (the design
list and what each design takes, the processes, the aims and goals, the defaults) so the form is
right from the first paint, before Python has loaded. Each copy must say what the Python says,
or a reader would see one design's settings and generate another's.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from tests.web._support import (
    ENTRY_POINT,
    FACTORS,
    NO_BOOK_PAGE,
    correlated_study,
    factors_named,
    held_of,
    javascript,
    page,
    source,
)


# --------------------------------------------------------------------------- the form's choices
def test_page_design_list_matches_bootstrap(bootstrap):
    """The page lists the designs in HTML so the form is right before Python loads."""
    in_page = {o["value"]: o for o in page().options["design-type"]}
    given = in_page.pop(bootstrap.GIVEN)
    mixtures = {key: in_page.pop(key) for key in list(in_page) if key in bootstrap.MIXTURES}
    assert list(in_page) == list(bootstrap.DESIGNS)
    assert list(mixtures) == list(bootstrap.MIXTURES), "the mixtures follow the designs for factors with ranges"
    assert "hidden" in given, "the list offers runs as given only while a study of given runs is open"
    assert given["label"] == bootstrap.GIVEN_INFO["label"]
    assert given["data-hint"] == bootstrap.GIVEN_INFO["hint"]
    assert (given["data-options"], int(given["data-min-factors"]), int(given["data-max-factors"])) == (
        "",
        bootstrap.GIVEN_INFO["min_factors"],
        bootstrap.GIVEN_INFO["max_factors"],
    )
    for key, info in [*bootstrap.DESIGNS.items(), *bootstrap.MIXTURES.items()]:
        option = in_page.get(key) or mixtures[key]
        assert option["label"] == info["label"], key
        assert option["data-hint"] == info["hint"], key
        assert option["data-options"].split() == info["options"], key
        assert int(option["data-min-factors"]) == info["min_factors"], key
        assert int(option["data-max-factors"]) == info["max_factors"], key
    assert page().selected("design-type") == bootstrap.DEMO["design"]["design_type"], "until the brief picks one"


def test_the_page_names_the_mixtures_as_bootstrap_does(bootstrap, node):
    """logic.js and python.js each know which designs are a mixture's: the factors are then a blend's components."""
    assert javascript(node, source("logic.js"), [("MIXTURES", []), ("MIXTURE_MODELS", [])]) == [
        list(bootstrap.MIXTURES),
        list(bootstrap.MIXTURE_MODELS),
    ]
    assert javascript(node, source("python.js"), [("MIXTURES", [])]) == [list(bootstrap.MIXTURES)]


def test_a_mixture_s_settings_are_bootstrap_s(api, bootstrap):
    """A lattice's degrees and the models extreme vertices are built for, each opening on bootstrap's default."""
    assert page().values("lattice_degree") == [str(d) for d in sorted(bootstrap.LATTICE_DEGREES, reverse=True)]
    assert page().selected("lattice_degree") == str(bootstrap.DEFAULT_LATTICE_DEGREE)
    assert page().values("mixture_model") == list(bootstrap.VERTICES_MODELS)
    blank = [{"name": n, "low": None, "high": None} for n in "ABC"]
    lattice = api.ok("api_make_design", {"design_type": "simplex_lattice", "factors": blank})
    assert lattice["spec"]["options"]["model_type"] == bootstrap.LATTICE_DEGREES[int(page().selected("lattice_degree"))]
    bounded = [{**blank[0], "low": 0.2}, *blank[1:]]
    vertices = api.ok("api_make_design", {"design_type": "extreme_vertices", "factors": bounded})
    assert vertices["spec"]["options"]["model_type"] == page().selected("mixture_model")


@pytest.mark.parametrize(
    "case",
    [
        ("simplex_lattice", [("", ""), ("", ""), ("", "")], False),
        ("simplex_lattice", [("0.2", ""), ("", ""), ("", "")], True),
        ("extreme_vertices", [("0.2", ""), ("", ""), ("", "")], False),
        ("extreme_vertices", [("", ""), ("", ""), ("", "")], True),
        ("extreme_vertices", [("0.5", ""), ("0.4", ""), ("0.3", "")], True),
        ("extreme_vertices", [("", "0.3"), ("", "0.3"), ("", "0.3")], True),
        ("simplex_centroid", [("150", "200"), ("", ""), ("", "")], False),
        ("simplex_centroid", [("", "most"), ("", ""), ("", "")], False),
        ("extreme_vertices", [("0.6", "0.4"), ("", ""), ("", "")], False),
    ],
)
def test_the_page_reads_a_mixture_s_bounds_as_bootstrap_does(api, bootstrap, node, case):
    """A blank bound is 0 or 1 on both sides; a component the page marks Python refuses, in the same words for the
    bounds taken together. Each case is the design type, each component's bounds as typed, and whether the page
    finds a problem only in the bounds taken together.
    """
    design_type, typed, together = case
    names = ["A", "B", "C"]
    rows = [{"name": n, "low": low, "high": high} for n, (low, high) in zip(names, typed, strict=True)]
    calls = [("proportionValue", [r["low"], 0]) for r in rows] + [("proportionValue", [r["high"], 1]) for r in rows]
    calls += [("componentProblem", [r]) for r in rows]
    asked = javascript(node, source("logic.js"), calls)
    sent = [{"name": n, "low": low, "high": high} for n, low, high in zip(names, asked[:3], asked[3:6], strict=True)]
    reply = api.reply("api_make_design", {"design_type": design_type, "factors": sent})
    each = [problem for problem in asked[6:] if problem]
    label = bootstrap.MIXTURES[design_type]["label"]
    # The page says what the bounds need together once each component is ready on its own.
    whole = "" if each else javascript(node, source("logic.js"), [("mixtureProblem", [sent, design_type, label])])[0]
    assert bool(each or whole) == (not reply["ok"]), reply.get("error")
    assert bool(whole) == together
    if together:
        assert whole == reply["error"]


def test_the_page_lists_the_scenarios_bootstrap_simulates(bootstrap):
    """The process list is in the HTML, so it reads right before Python loads; it says what SCENARIOS says.

    A mixture's processes come last, marked, and hidden until a mixture's run sheet is on the page.
    """
    in_page = {o["value"]: o for o in page().options["scenario"]}
    assert list(in_page) == [*bootstrap.SCENARIOS, *bootstrap.MIXTURE_SCENARIOS]
    for key, info in [*bootstrap.SCENARIOS.items(), *bootstrap.MIXTURE_SCENARIOS.items()]:
        assert in_page[key]["label"] == info["label"], key
        assert in_page[key]["data-hint"] == info["hint"], key
        assert ("data-mixture" in in_page[key]) == (key in bootstrap.MIXTURE_SCENARIOS), key
        assert ("hidden" in in_page[key]) == (key in bootstrap.MIXTURE_SCENARIOS), key
    assert page().selected("scenario") == bootstrap.DEMO["scenario"]


# --------------------------------------------------------------------------- links to the book
def test_every_link_to_the_book_is_extensionless_and_one_the_page_checked():
    """The book is served without ``.html``; every link the page makes names a page and anchor that exist."""
    checked = {
        "omars-designs#doe-omars-metric-choice",
        "comparing-design-families#doe-omnibus-comparison",
        "full-factorial-designs/using-two-levels-for-two-or-more-factors#doe-two-level-factorials",
        "fractional-factorial-designs/design-resolution#doe-design-resolution",
        "fractional-factorial-designs/saturated-designs-for-screening#doe-saturated-screening-designs",
        "definitive-screening-designs#doe-definitive-screening-designs",
        "response-surface-methods#doe-box-behnken-designs",
        "response-surface-methods#doe-central-composite-designs",
        "optimal-designs#doe-optimal-designs",
        "full-factorial-designs/assessing-significance-of-main-effects-and-interactions#doe-lenth-method",
        "judging-and-comparing-designs#doe-fds-plot",
        "mixture-experiments#doe-mixture-designs",
    }
    text = source("index.html") + source("app.js")
    links = set(re.findall(r"learnche\.org/pid/design-analysis-experiments/([\w/#.-]+)", text))
    links |= set(re.findall(r"\$\{BOOK\}/([\w/#.-]+)", text))
    assert links, "the page links to the book"
    assert not any(".html" in link for link in links)
    assert links <= checked


def test_every_design_links_to_the_book(bootstrap):
    block = re.search(r"const BOOK_PAGES = \{(.*?)\};", source("app.js"), re.DOTALL).group(1)
    linked = set(re.findall(r"^\s*(\w+):", block, re.MULTILINE))
    assert {*bootstrap.DESIGNS, *bootstrap.MIXTURES} - NO_BOOK_PAGE <= linked
    assert not NO_BOOK_PAGE & linked, "a design the book now covers links to its page"


# --------------------------------------------------------------------------- the design settings
def test_every_setting_a_design_takes_has_one_control(bootstrap):
    """The page shows the controls a design names in its ``options``; each must exist, once."""
    designs = [*bootstrap.DESIGNS.values(), *bootstrap.MIXTURES.values()]
    named = {option for info in designs for option in info["options"]}
    assert sorted(page().data_options) == sorted(named)


def test_the_axial_choices_are_bootstrap_s(api, bootstrap, node):
    """The named distances are bootstrap's, then the choice that sends the number typed beside it."""
    (typed,) = javascript(node, source("logic.js"), [("AXIAL_DISTANCE", [])])
    assert page().values("alpha") == [*bootstrap.ALPHAS, typed]
    made = api.ok("api_make_design", {"design_type": "ccd", "factors": bootstrap.DEMO["design"]["factors"]})
    assert page().selected("alpha") == made["spec"]["options"]["alpha"], "the page opens on bootstrap's default"
    asked = javascript(node, source("logic.js"), [("axialDistance", [typed, " 1.4 "]), ("axialDistance", [typed, ""])])
    assert asked == [1.4, None], "a typed distance is sent as a number, and a blank box as null for Python to refuse"


@pytest.mark.parametrize(("box", "most"), [("n_replicates", "MAX_REPLICATES"), ("n_blocks", "MAX_BLOCKS")])
def test_the_replicates_and_blocks_boxes_take_what_bootstrap_takes(api, bootstrap, box, most):
    shown = page().ids[box]
    assert (int(shown["min"]), int(shown["max"])) == (1, getattr(bootstrap, most))
    assert int(shown["value"]) == 1, "a design is made once, in one block, unless the reader asks for more"
    factors = [{"name": n, "low": -1, "high": 1} for n in "ABCDEF"]  # 64 runs, enough for eight blocks
    for n in (1, getattr(bootstrap, most)):
        api.ok("api_make_design", {"design_type": "full_factorial", "factors": factors, box: n})


def test_the_resolutions_are_bootstrap_s(bootstrap):
    assert page().values("resolution") == [str(r) for r in bootstrap.RESOLUTIONS]
    assert page().selected("resolution") == str(bootstrap.DEFAULT_RESOLUTION)


def test_the_levels_of_a_taguchi_array_are_bootstrap_s(api, bootstrap):
    """The page offers the levels bootstrap takes, and opens on the one a design made without the option has."""
    assert page().values("n_levels") == [str(n) for n in bootstrap.TAGUCHI_LEVELS]
    made = api.ok("api_make_design", {"design_type": "taguchi", "factors": FACTORS})
    assert "n_levels" not in made["spec"]["options"]
    assert page().selected("n_levels") == str(bootstrap.TAGUCHI_LEVELS[0])


def test_the_models_to_design_for_are_bootstrap_s(api, bootstrap):
    assert sorted(page().values("model_type")) == sorted(bootstrap.MODELS)
    made = api.ok("api_make_design", {"design_type": "d_optimal", "factors": bootstrap.DEMO["design"]["factors"]})
    assert page().selected("model_type") == made["spec"]["options"]["model_type"]


def test_the_models_to_analyse_with_are_bootstrap_s_after_as_designed(bootstrap, node):
    (chosen,) = javascript(node, source("logic.js"), [("CHOSEN_TERMS", [])])
    models = ["", *bootstrap.MODELS, chosen, *bootstrap.MIXTURE_MODELS]
    assert page().values("model") == models, "then the terms the reader ticks, then a mixture's models"
    assert page().selected("model") == bootstrap.DEMO["model"] == ""
    marked = [o["value"] for o in page().options["model"] if "data-mixture" in o and "hidden" in o]
    assert marked == list(bootstrap.MIXTURE_MODELS), "offered for a mixture's runs only"
    (kept,) = javascript(node, source("campaign.js"), [("MODELS", [])])
    assert kept == ["", *bootstrap.MODELS, *bootstrap.MIXTURE_MODELS], "a saved study's model is one of these"


@pytest.mark.parametrize("categorical", [(), ("S",)], ids=["numeric factors", "a categorical factor"])
def test_the_terms_to_tick_are_the_terms_bootstrap_fits(bootstrap, node, categorical):
    """Each term the page lists is one bootstrap reads, labelled as its fit labels it; a named model's are its own."""
    factors = [
        {"name": n, **({"low": "A", "high": "B"} if n in categorical else {"low": -1, "high": 1})}
        for n in ["T", "P", "S", "pH"]
    ]
    names = [f["name"] for f in factors]
    listed, *named = javascript(
        node,
        source("logic.js"),
        [("candidateTerms", [factors]), *[("namedModelTerms", [model, factors]) for model in bootstrap.MODELS]],
    )
    fitted = bootstrap._chosen_terms([t["term"] for t in listed], names, categorical)
    assert len(fitted) == len(listed) == 4 + 6 + 4 + 4 - len(categorical), "no square of a categorical factor"
    assert [t["label"] for t in listed] == [bootstrap._pretty(term) for term in fitted]
    for model, terms in zip(bootstrap.MODELS, named, strict=True):
        found = bootstrap._chosen_terms(terms, names, categorical)
        assert found == bootstrap._model_terms(model, names, categorical), model


def test_the_designs_that_take_a_categorical_factor_are_bootstrap_s(bootstrap, node):
    assert javascript(node, source("logic.js"), [("CATEGORICAL_DESIGNS", [])]) == [list(bootstrap.CATEGORICAL_DESIGNS)]


def test_a_categorical_factor_s_levels_are_held_to_bootstrap_s_limits(bootstrap, node):
    """The designs that take three levels or more, the most levels, and the longest label, as the page checks them."""
    held = javascript(node, source("logic.js"), [("SEVERAL_LEVEL_DESIGNS", []), ("MAX_LEVELS", []), ("LABEL_MOST", [])])
    assert held == [list(bootstrap.SEVERAL_LEVEL_DESIGNS), bootstrap.MAX_LEVELS, bootstrap.LABEL_MOST]
    stored = javascript(node, source("campaign.js"), [("MAX_LEVELS", []), ("LABEL_MOST", [])])
    assert stored == [bootstrap.MAX_LEVELS, bootstrap.LABEL_MOST], "a study file holds what the page takes"
    joined = javascript(node, source("python.js"), [("LEVEL_JOIN", [])])
    assert joined == [bootstrap.LEVEL_JOIN], "the code shown names a level's contrast column as the analysis does"


def test_the_page_names_runs_as_given_as_bootstrap_does(bootstrap, node):
    assert javascript(node, source("logic.js"), [("GIVEN", [])]) == [bootstrap.GIVEN]
    assert bootstrap.GIVEN not in bootstrap.DESIGNS, "the page holds runs as given, and never makes them from the form"


def test_the_page_reads_a_label_where_bootstrap_does(bootstrap, node):
    """What the reader types as a level is a label to the page exactly when it is one to Python."""
    typed = ["A", " old ", "Supplier 2", "150", "-1.5", "1e3", "", " ", "inf", "-Infinity", "nan", "NaN"]
    asked = javascript(node, source("logic.js"), [("isLabel", [text]) for text in typed])
    assert asked == [bootstrap._label(text) is not None for text in typed]


def test_the_aims_are_the_models_designs_are_compared_on(api, bootstrap, node):
    """Each aim the brief offers is a model the comparison scores; a study of few factors opens on the comparison's."""
    from process_improve.experiments.compare import MODELS as COMPARISON_MODELS

    assert [r["value"] for r in page().radios["aim"]] == list(COMPARISON_MODELS)
    plan = api.ok("api_compare_plan", {"factors": bootstrap.DEMO["design"]["factors"]})
    (opened,) = javascript(node, source("logic.js"), [("defaultAim", [3])])
    assert page().checked("aim") == opened == plan["model"]


def test_an_optimal_design_is_built_for_the_aim_as_the_comparison_builds_it(bootstrap, node):
    (built,) = javascript(node, source("logic.js"), [("OPTIMAL_MODEL", [])])
    assert built == bootstrap.OPTIMAL_MODEL


def test_the_goals_are_bootstrap_s_and_a_new_study_just_learns(bootstrap):
    assert page().values("response-goal") == list(bootstrap.GOALS)
    assert page().selected("response-goal") == "learn"


def test_the_most_responses_a_study_measures_is_bootstrap_s(bootstrap, node):
    """The page, the campaigns it reads and its words all hold as many responses as bootstrap.py takes."""
    (page_most,) = javascript(node, source("logic.js"), [("MAX_RESPONSES", [])])
    (campaign_most,) = javascript(node, source("campaign.js"), [("RESPONSES_HELD", [])])
    assert page_most == campaign_most == bootstrap.MAX_RESPONSES
    assert page().ids["responses-full"]["tag"] == "span"
    assert f"A study measures at most {bootstrap.MAX_RESPONSES} responses." in source("index.html")


@pytest.mark.parametrize(
    "labels",
    [["Yield", "Cost", "Purity"], ["Yield", "Yield", "yield"], ["", "Cost"], ["T", "Cost ($/kg)", "Cost"]],
)
def test_the_page_names_each_response_s_column_as_bootstrap_takes_them(api, node, labels):
    """The columns the page derives for several responses are names bootstrap.py takes for them, all different."""
    (names,) = javascript(node, source("logic.js"), [("responseColumns", [labels, ["T", "P"]])])
    assert len(set(names)) == len(names)
    made = api.ok(
        "api_make_design",
        {"design_type": "full_factorial", "factors": FACTORS[:2], "responses": [{"name": n} for n in names]},
    )
    assert made["columns"][-len(names) :] == names


def test_the_form_opens_on_a_new_study_with_bootstrap_s_defaults(api, bootstrap):
    """Before Python loads, the response is blank and every setting is the one bootstrap.py would use."""
    demo = bootstrap.DEMO
    ids = page().ids
    assert ids["response"].get("value") is None, "a new study's response is the reader's to name"
    assert float(ids["sigma"]["value"]) == demo["sigma"] == bootstrap.SCENARIOS[demo["scenario"]]["sigma"]
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": demo["design"]["factors"]})
    assert int(ids["n_center_points"]["value"]) == made["spec"]["options"]["n_center_points"], "bootstrap's default"


# --------------------------------------------------------------------------- rules web/logic.js repeats
def test_the_page_gives_an_optimal_design_bootstrap_s_spare_runs(bootstrap, node):
    assert javascript(node, source("logic.js"), [("SPARE_RUNS", [])]) == [bootstrap.SPARE_RUNS]


def test_design_marks_too_few_spare_runs_where_learn_offers_to_make_them_again(bootstrap, node):
    """Design marks a design that leaves fewer runs for the noise; Learn offers such a study its runs again."""
    assert javascript(node, source("logic.js"), [("SPARE_FOR_NOISE", [])]) == [bootstrap.SPARE_FOR_NOISE]


def test_a_design_s_runs_open_where_bootstrap_makes_them(api, bootstrap, node):
    """A design that takes a number of runs opens on bootstrap's own: ten a factor for a space-filling design."""
    spread, runs = javascript(node, source("logic.js"), [("SPACE_FILLING", []), ("SPACE_FILLING_RUNS", [])])
    assert spread == list(bootstrap.SPACE_FILLING)
    assert runs == bootstrap.SPACE_FILLING_RUNS
    assert javascript(node, source("python.js"), [("SPACE_FILLING", [])]) == [list(bootstrap.SPACE_FILLING)]
    counted = [d for d, info in bootstrap.DESIGNS.items() if "budget" in info["options"]]
    assert set(bootstrap.SPACE_FILLING) < set(counted)
    for design_type in counted:
        made = api.ok("api_make_design", {"design_type": design_type, "factors": FACTORS, "model_type": "interactions"})
        opened = javascript(node, source("logic.js"), [("defaultBudget", [design_type, len(FACTORS), "interactions"])])
        assert opened == [made["spec"]["options"]["budget"]], design_type


def test_a_report_names_the_seed_bootstrap_puts_the_runs_in_order_with(bootstrap, api):
    """report.js imports logic.js, which a module given as text cannot, so its constant is read from its source."""
    (seed,) = re.findall(r"^export const RUN_ORDER_SEED = (\d+);$", source("report.js"), flags=re.MULTILINE)
    assert int(seed) == bootstrap.RUN_ORDER_SEED
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS})
    again = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS, "random_state": int(seed)})
    assert made["rows"] == again["rows"], "a design's run order is drawn with that seed"


def test_the_page_shows_the_vif_column_where_bootstrap_writes_the_note(api, bootstrap, node):
    """The coefficients table shows each term's VIF only when the analysis names a correlated term, and then always."""
    assert javascript(node, source("logic.js"), [("VIF_HIGH", [])]) == [bootstrap.VIF_HIGH]
    design = api.ok("api_make_design", correlated_study())
    held = held_of(design)
    responses = [float(i % 3) + 0.5 * i for i in range(design["n_runs"])]
    for model in ("main_effects", {"terms": [["A"], ["C"]]}):
        analysis = api.ok("api_analyze_runs", {**held, "responses": responses, "model": model})
        (shown,) = javascript(node, source("logic.js"), [("showsVif", [analysis["coefficients"]])])
        noted = any("variance inflation factor" in note for note in analysis["notes"])
        assert shown is noted is (model == "main_effects"), model


def test_a_blank_response_on_the_page_is_the_one_bootstrap_names(api, node):
    """The page reads a blank response box as the name Python gives a design made with one."""
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS[:2], "response": ""})
    assert javascript(node, source("logic.js"), [("DEFAULT_RESPONSE", [])]) == [made["spec"]["response"]]


@pytest.mark.parametrize(
    ("label", "factor_names"),
    [
        ("Yield", ["T", "P"]),
        ("Tensile strength (MPa)", ["T", "P"]),
        ("2nd pass yield", ["T", "P"]),
        ("yield", ["T", "P"]),
        ("T", ["T", "T_2"]),
        ("Block", ["T", "P"]),
        ("Viscosité, Pa·s", ["T", "P"]),
        ("%", ["y", "P"]),
        ("", ["y", "y_2"]),
    ],
)
def test_the_page_names_the_results_column_as_bootstrap_takes_it(bootstrap, node, label, factor_names):
    """The column the page derives from the reader's words is one bootstrap.py accepts as a response's name."""
    (name,) = javascript(node, source("logic.js"), [("responseName", [label, factor_names])])
    factors = bootstrap._factors({"factors": [{"name": n, "low": -1, "high": 1} for n in factor_names]})
    assert bootstrap._response_name(name, factors) == name


def test_the_page_s_block_column_and_goals_are_bootstrap_s(bootstrap, node):
    block, goals = javascript(node, source("logic.js"), [("BLOCK", []), ("GOALS", [])])
    assert block == bootstrap.BLOCK
    assert list(goals) == list(bootstrap.GOALS)


def test_the_aims_a_study_keeps_are_the_comparison_s_models(node):
    from process_improve.experiments.compare import MODELS as COMPARISON_MODELS

    assert javascript(node, source("campaign.js"), [("AIMS", [])]) == [list(COMPARISON_MODELS)]


def test_the_page_counts_a_model_s_coefficients_as_bootstrap_does(bootstrap, node):
    """The page sets an optimal design's runs from its own count of the model's coefficients."""
    cases = [(k, model) for k in range(2, 13) for model in bootstrap.MODELS]
    counted = javascript(node, source("logic.js"), [("nTerms", [k, model]) for k, model in cases])
    assert counted == [bootstrap._n_terms(k, model) for k, model in cases]


def test_the_page_opens_a_fraction_at_bootstrap_s_resolution(bootstrap, node):
    """The page picks the resolution for the factor count until the reader chooses one; bootstrap picks the same."""
    counts = list(range(3, 12))
    picked = javascript(node, source("logic.js"), [("defaultResolution", [k]) for k in counts])
    assert picked == [bootstrap._resolution({}, k) for k in counts]


def test_the_page_asks_for_a_center_run_where_bootstrap_needs_one(bootstrap, node):
    """Box-Behnken designs and rotatable CCDs need a center run, as does a CCD whose typed axial distance is the
    corners' own; a DSD counts its own center run, so the page asks for one too.
    """
    alphas = [*bootstrap.ALPHAS, 1.4, 2**0.5, 3**0.5, 2.0]
    cases = [(design, alpha, k) for design in bootstrap.DESIGNS for alpha in alphas for k in (2, 3, 4)]
    asked = javascript(node, source("logic.js"), [("needsCentre", list(case)) for case in cases])
    needed = [design == "dsd" or bootstrap._needs_center_run(design, alpha, k) for design, alpha, k in cases]
    assert asked == needed
    assert sum(needed) > len(bootstrap.DESIGNS) * 3, "the cases include typed distances that need a center run"


# --------------------------------------------------------------------------- the trade-off grid
#: Factor names for the grid's widest column, eleven factors; ``I`` is left out, as a defining relation uses it.
GRID_NAMES = "ABCDEFGHJKL"

#: The grid's columns, by number of factors: the library's searches take several seconds in the last two.
GRID_COLUMNS = [pytest.param(k, marks=pytest.mark.slow) if k >= 10 else k for k in range(3, 12)]


def trade_off_grid(node: str) -> tuple[list[int], list[int], list[list], str]:
    """Return the grid ``web/logic.js`` holds: its run counts, its numbers of factors, its cells, and its Full."""
    names = ["TRADE_OFF_RUNS", "TRADE_OFF_FACTORS", "TRADE_OFF", "FULL"]
    runs, factors, cells, full = javascript(node, source("logic.js"), [(name, []) for name in names])
    return runs, factors, cells, full


def library_cell(label: str, full: str) -> int | str | None:
    """Read a cell of ``trade_off_table`` as the grid holds it: a resolution, Full, or None.

    The library writes a fraction ``2^(5-2) III``, the full factorial ``2^5 (full)``, and a run count
    that repeats it ``2^5 (twice)``; the grid leaves those blank, as it does a design that does not exist.
    """
    from process_improve.experiments.trade_off import _ROMAN

    if fraction := re.fullmatch(r"2\^\(\d+-\d+\) ([IVX]+)", label):
        return next(r for r, roman in _ROMAN.items() if roman == fraction.group(1))
    if re.fullmatch(r"2\^\d+ \(full\)", label):
        return full
    assert label == "" or re.fullmatch(r"2\^\d+ \((twice|\d+ times)\)", label), f"a cell the grid cannot show: {label}"
    return None


def test_the_trade_off_grid_spans_the_fractions_the_page_makes(bootstrap, node):
    runs, factors, _cells, _full = trade_off_grid(node)
    fraction = bootstrap.DESIGNS["fractional_factorial"]
    assert factors == list(range(fraction["min_factors"], fraction["max_factors"] + 1))
    assert runs == [4, 8, 16, 32, 64, 128]
    assert javascript(node, source("logic.js"), [("RESOLUTIONS", [])]) == [list(bootstrap.RESOLUTIONS)]


@pytest.mark.parametrize("k", GRID_COLUMNS)
def test_the_trade_off_grid_is_the_library_s_trade_off_table(node, k):
    """Each cell of the grid's column for k factors is what ``trade_off_table`` finds, so the two cannot drift."""
    from process_improve.experiments.trade_off import trade_off_table

    runs, factors, cells, full = trade_off_grid(node)
    table = trade_off_table(runs=runs, factors=(k,), display=False)
    column = factors.index(k)
    assert [row[column] for row in cells] == [library_cell(table.loc[n, k], full) for n in runs]


def test_each_cell_a_reader_can_pick_makes_exactly_its_runs_and_resolution(api, node):
    """A pick sets a resolution or the full factorial; the design made from it has the cell's runs and resolution."""
    runs, factors, cells, full = trade_off_grid(node)
    places = [(n, k) for k in factors for n in runs]
    picks = javascript(node, source("logic.js"), [("cellDesign", [n, k]) for n, k in places])
    picked = 0
    for (n, k), design in zip(places, picks, strict=True):
        if design is None:
            continue
        cell = cells[runs.index(n)][factors.index(k)]
        spec = {**design, "factors": factors_named(*GRID_NAMES[:k]), "n_center_points": 0}
        made = api.ok("api_make_design", spec)
        assert (made["n_runs"], made["facts"]["resolution"]) == (n, None if cell == full else cell), (n, k)
        picked += 1
    assert picked == 26, "every column offers the cells the page makes, and only those"


def test_a_column_offers_every_design_the_page_makes_there_and_marks_it(api, bootstrap, node):
    """The cells offered in a column are the page's fraction at each resolution it offers, and the full factorial.

    A cell the page cannot make in its runs (resolution IV in 32 runs in seven factors, which the page makes
    in 16) is not offered, and the cell the form's resolution makes is the one the grid marks.
    """
    runs, factors, _cells, _full = trade_off_grid(node)
    places = [(n, k) for k in factors for n in runs]
    offered = javascript(node, source("logic.js"), [("cellDesign", [n, k]) for n, k in places])
    for k in factors:
        resolutions = [r for r in bootstrap.RESOLUTIONS if r <= k]
        made = [
            api.ok(
                "api_make_design",
                {"design_type": "fractional_factorial", "factors": factors_named(*GRID_NAMES[:k]), "resolution": r},
            )
            for r in resolutions
        ]
        fractions = [m["n_runs"] - m["facts"]["center_runs"] for m in made]
        marked = javascript(node, source("logic.js"), [("fractionRuns", [k, r]) for r in resolutions])
        assert marked == fractions, f"{k} factors"
        full_factorial = {2**k} if k <= bootstrap.DESIGNS["full_factorial"]["max_factors"] and 2**k in runs else set()
        column = {n for (n, j), design in zip(places, offered, strict=True) if j == k and design is not None}
        assert column == set(fractions) | full_factorial, f"{k} factors"


# --------------------------------------------------------------------------- what the worker answers
def test_the_worker_answers_every_entry_point_bootstrap_defines_and_nothing_else(bootstrap):
    """worker.js refuses a name its pattern does not match; every api_* function of bootstrap.py matches it."""
    worker = re.search(r"const ENTRY_POINT = /(.+)/;", source("worker.js")).group(1)
    assert worker == ENTRY_POINT.pattern, "the e2e tier's native transport refuses the same names"
    entry_points = [name for name in dir(bootstrap) if name.startswith("api_") and callable(getattr(bootstrap, name))]
    assert len(entry_points) >= 10
    assert all(ENTRY_POINT.match(name) for name in entry_points)
    for private in ("_reply", "make_design", "json", "api_", "api_plots.__call__", "__import__"):
        assert not ENTRY_POINT.match(private), private


# --------------------------------------------------------------------------- the page's own elements
def test_every_element_app_js_looks_up_is_on_the_page():
    """A renamed or removed element would leave app.js reaching for nothing."""
    text = source("app.js")
    ids = {name for name in re.findall(r"[\"'`]#([\w-]+)", text) if not name.endswith("-")}
    ids |= {f"py-{key}" for key in re.findall(r'showPython\(\s*"(\w+)"', text)}
    ids |= {f"{part}-{tab}" for tab in re.findall(r'selectTab\("(\w+)"\)', text) for part in ("tab", "panel")}
    assert len(ids) > 40, "the search found app.js's lookups"
    assert ids - set(page().ids) == set()


def test_every_tab_controls_a_panel_on_the_page():
    controlled = {attrs["aria-controls"] for attrs in page().ids.values() if attrs.get("aria-controls")}
    assert {"step-plan", "step-run", "step-analyse", "panel-real", "panel-simulate"} <= controlled
    assert controlled - set(page().ids) == set()


def test_every_marked_word_has_a_definition_and_every_definition_a_word(node):
    """A word marked on the page names a glossary entry, and each entry is a word the page or app.js marks.

    Each definition is one or two short sentences, as a pop-up holds: the page says what a word means
    once, there, rather than in a sentence under every box.
    """
    glossary, marked = javascript(node, source("glossary.js"), [("GLOSSARY", []), ("MARKED", [])])
    assert set(page().terms) <= set(glossary), "a word on the page with no definition"
    assert set(marked.values()) <= set(glossary), "a phrase app.js marks with no definition"
    named = set(page().terms) | set(marked.values()) | set(re.findall(r'"([\w-]+)"', source("app.js")))
    assert set(glossary) <= named, "a definition no word opens"
    for key, text in glossary.items():
        assert text.endswith("."), key
        assert len(text.split()) <= 40, key
        assert "\u2014" not in text, key


def test_confirmation_runs_are_marked_and_counted_as_bootstrap_marks_and_counts_them(bootstrap, node):
    """The run sheet's column, its mark, how many runs the page adds, and how far each factor moves for the chart."""
    asked = ["CONFIRMATION", "CONFIRMS", "CONFIRMATION_RUNS", "NUDGE"]
    assert javascript(node, source("best.js"), [(name, []) for name in asked]) == [
        bootstrap.CONFIRMATION,
        bootstrap.CONFIRMS,
        bootstrap.CONFIRMATION_RUNS,
        bootstrap.NUDGE,
    ]
    assert javascript(node, source("python.js"), [("CONFIRMATION", [])]) == [bootstrap.CONFIRMATION]
    assert f"Add {bootstrap.CONFIRMATION_RUNS} confirmation runs" in source("index.html"), "before app.js writes it"


def test_the_limits_on_the_settings_are_held_as_bootstrap_holds_them(bootstrap, node):
    """The most limits, the longest note and limit, the designs that keep to them, and the signs read as operators."""
    asked = ["MAX_CONSTRAINTS", "NOTE_LENGTH", "EXPRESSION_LENGTH", "REGION_DESIGNS", "LIMIT_SYMBOLS"]
    got = javascript(node, source("logic.js"), [(name, []) for name in asked])
    assert got == [
        bootstrap.MAX_CONSTRAINTS,
        bootstrap.NOTE_LENGTH,
        bootstrap.MAX_EXPRESSION_LENGTH,
        list(bootstrap.REGION_DESIGNS),
        {chr(code): op for code, op in bootstrap._SYMBOLS.items()},
    ]
    held = {
        "count": bootstrap.MAX_CONSTRAINTS,
        "expression": bootstrap.MAX_EXPRESSION_LENGTH,
        "note": bootstrap.NOTE_LENGTH,
    }
    assert javascript(node, source("campaign.js"), [("LIMITS_HELD", [])]) == [held], "a saved study is checked alike"


def test_a_limit_s_key_is_the_same_as_typed_and_as_python_writes_it(api, node):
    """A run sheet is keyed to its limits as typed; the text Python gives back for each must key the same."""
    typed = [
        "T+25*P<=250",
        "T \u2264 190",
        "T^2/100 + P <= 400",
        "250>=T+25*P",
        "400<=3*T+5*P<=600",
        "sqrt(T)+P<=15",
        "T*F <= 3600",
        "0.04*T + F/2 <= 14.5",
    ]
    reply = api.ok("api_limits", {"factors": FACTORS, "constraints": [{"expression": text} for text in typed]})
    names = [f["name"] for f in FACTORS]
    stored = [entry["expression"] for entry in reply["constraints"]]
    keys = javascript(node, source("logic.js"), [("positional", [text, names]) for text in [*typed, *stored]])
    assert keys[: len(typed)] == keys[len(typed) :]


# --------------------------------------------------------------------------- the balance of the responses
def test_the_balance_offers_bootstrap_s_shapes_and_importances(bootstrap, node):
    shapes, importances, importance, shape = javascript(
        node, source("best.js"), [("GOOD_END", []), ("IMPORTANCES", []), ("IMPORTANCE", []), ("SHAPE", [])]
    )
    assert list(shapes) == list(bootstrap.SHAPES)
    assert importances == list(bootstrap.IMPORTANCES)
    assert importance == bootstrap.IMPORTANCE
    assert shape == bootstrap.SHAPE
    (kept,) = javascript(node, source("campaign.js"), [("GOAL_SHAPES", [])])
    assert kept == list(bootstrap.SHAPES), "a study keeps the shapes the balance takes"


@pytest.mark.parametrize(
    ("goal", "low", "high"),
    [
        ({"goal": "maximize"}, 60.0, 50.0),
        ({"goal": "minimize"}, 20.0, 20.0),
        ({"goal": "target", "target": 98.5}, 99.0, 100.0),
        ({"goal": "target", "target": 98.5}, 90.0, 98.5),
    ],
)
def test_the_page_says_a_goal_s_problem_in_bootstrap_s_words(bootstrap, node, goal, low, high):
    """The balance's table says what a range typed still needs before Python is asked, as Python would refuse it."""
    response = {"name": "Purity", "label": "Purity", **goal}
    results = pd.Series([97.0, 98.0, 99.5])
    with pytest.raises(ValueError, match="Purity") as refused:
        bootstrap._balance_goal(response, results, {"low": low, "high": high})
    (said,) = javascript(node, source("best.js"), [("goalProblem", [response, low, high])])
    assert said == str(refused.value)


def test_one_factor_s_runs_and_study_format_are_held_as_bootstrap_and_the_build_hold_them(
    bootstrap, build_script, node
):
    """The replicates the page offers, more for one factor's levels, and the format a one-factor study is saved in."""
    held = javascript(node, source("logic.js"), [("MAX_REPLICATES", []), ("MAX_LEVEL_RUNS", [])])
    assert held == [bootstrap.MAX_REPLICATES, bootstrap.MAX_LEVEL_RUNS]
    names = [("CAMPAIGN_VERSION", []), ("ONE_FACTOR_FORMAT", []), ("LEVELS_FORMAT", [])]
    formats = javascript(node, source("campaign.js"), names)
    assert formats == [build_script.CAMPAIGN_VERSION, build_script.ONE_FACTOR_FORMAT, build_script.LEVELS_FORMAT]
    assert build_script.study_format({"factors": [{"name": "S"}]}) == build_script.ONE_FACTOR_FORMAT
    assert build_script.study_format({"factors": [{"name": "S"}, {"name": "T"}]}) == 1
    lot = {"name": "Lot", "type": "categorical", "levels": ["1", "2", "3"]}
    assert build_script.study_format({"factors": [{"name": "S"}, lot]}) == build_script.LEVELS_FORMAT
