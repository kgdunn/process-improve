"""The page against the Python it mirrors.

``web/index.html`` and ``web/app.js`` repeat some of ``web/bootstrap.py``'s choices (the design
list and what each design takes, the processes, the worked example's inputs) so the form is
right from the first paint, before Python has loaded. Each copy must say what the Python says,
or a reader would see one design's settings and generate another's.
"""

from __future__ import annotations

import re

from tests.web._support import FACTORS, javascript, page, source


# --------------------------------------------------------------------------- the form's choices
def test_page_design_list_matches_bootstrap(bootstrap):
    """The page lists the designs in HTML so the form is right before Python loads."""
    in_page = {o["value"]: o for o in page().options["design-type"]}
    assert list(in_page) == list(bootstrap.DESIGNS)
    for key, info in bootstrap.DESIGNS.items():
        option = in_page[key]
        assert option["label"] == info["label"], key
        assert option["data-hint"] == info["hint"], key
        assert option["data-options"].split() == info["options"], key
        assert int(option["data-min-factors"]) == info["min_factors"], key
        assert int(option["data-max-factors"]) == info["max_factors"], key
    assert page().selected("design-type") == bootstrap.DEMO["design"]["design_type"], "it opens on the worked example"


def test_the_page_lists_the_scenarios_bootstrap_simulates(bootstrap):
    """The process list is in the HTML, so it reads right before Python loads; it says what SCENARIOS says."""
    in_page = {o["value"]: o for o in page().options["scenario"]}
    assert list(in_page) == list(bootstrap.SCENARIOS)
    for key, info in bootstrap.SCENARIOS.items():
        assert in_page[key]["label"] == info["label"], key
        assert in_page[key]["data-hint"] == info["hint"], key
    assert page().selected("scenario") == bootstrap.DEMO["scenario"]


def test_the_page_s_default_factors_are_the_worked_example_s(bootstrap):
    block = re.search(r"const DEFAULT_FACTORS = \[(.*?)\];", source("app.js"), re.DOTALL).group(1)
    in_page = [
        {"name": n, "low": float(lo), "high": float(hi), "units": u}
        for n, lo, hi, u in re.findall(r'name: "(\w+)", low: ([\d.]+), high: ([\d.]+), units: "([^"]*)"', block)
    ]
    demo = [{**f, "low": float(f["low"]), "high": float(f["high"])} for f in bootstrap.DEMO["design"]["factors"]]
    assert in_page == demo


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
    }
    text = source("index.html") + source("app.js")
    links = set(re.findall(r"learnche\.org/pid/design-analysis-experiments/([\w/#.-]+)", text))
    links |= set(re.findall(r"\$\{BOOK\}/([\w/#.-]+)", text))
    assert links, "the page links to the book"
    assert not any(".html" in link for link in links)
    assert links <= checked


def test_every_design_links_to_the_book(bootstrap):
    block = re.search(r"const BOOK_PAGES = \{(.*?)\};", source("app.js"), re.DOTALL).group(1)
    assert set(bootstrap.DESIGNS) <= set(re.findall(r"^\s*(\w+):", block, re.MULTILINE))


# --------------------------------------------------------------------------- the design settings
def test_every_setting_a_design_takes_has_one_control(bootstrap):
    """The page shows the controls a design names in its ``options``; each must exist, once."""
    named = {option for info in bootstrap.DESIGNS.values() for option in info["options"]}
    assert sorted(page().data_options) == sorted(named)


def test_the_axial_choices_are_bootstrap_s(api, bootstrap):
    assert page().values("alpha") == list(bootstrap.ALPHAS)
    made = api.ok("api_make_design", {"design_type": "ccd", "factors": bootstrap.DEMO["design"]["factors"]})
    assert page().selected("alpha") == made["spec"]["options"]["alpha"], "the page opens on bootstrap's default"


def test_the_resolutions_are_bootstrap_s(bootstrap):
    assert page().values("resolution") == [str(r) for r in bootstrap.RESOLUTIONS]
    assert page().selected("resolution") == str(bootstrap.DEFAULT_RESOLUTION)


def test_the_models_to_design_for_are_bootstrap_s(api, bootstrap):
    assert sorted(page().values("model_type")) == sorted(bootstrap.MODELS)
    made = api.ok("api_make_design", {"design_type": "d_optimal", "factors": bootstrap.DEMO["design"]["factors"]})
    assert page().selected("model_type") == made["spec"]["options"]["model_type"]


def test_the_models_to_analyse_with_are_bootstrap_s_after_as_designed(bootstrap):
    assert page().values("model") == ["", *bootstrap.MODELS]
    assert page().selected("model") == bootstrap.DEMO["model"] == ""


def test_the_models_to_compare_on_are_the_comparison_s(api, bootstrap):
    from process_improve.experiments.compare import MODELS as COMPARISON_MODELS

    assert sorted(page().values("compare-model")) == sorted(COMPARISON_MODELS)
    plan = api.ok("api_compare_plan", {"factors": bootstrap.DEMO["design"]["factors"]})
    assert page().selected("compare-model") == plan["model"]


def test_the_form_opens_on_the_worked_example(api, bootstrap):
    """Before Python loads, the form already holds what the worked example used and bootstrap's defaults."""
    demo = bootstrap.DEMO
    ids = page().ids
    assert ids["response"]["value"] == demo["design"]["response"]
    assert float(ids["sigma"]["value"]) == demo["sigma"] == bootstrap.SCENARIOS[demo["scenario"]]["sigma"]
    assert int(ids["n_center_points"]["value"]) == demo["design"]["n_center_points"]
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": demo["design"]["factors"]})
    assert int(ids["n_center_points"]["value"]) == made["spec"]["options"]["n_center_points"], "bootstrap's default"


# --------------------------------------------------------------------------- rules web/logic.js repeats
def test_the_page_gives_an_optimal_design_bootstrap_s_spare_runs(bootstrap, node):
    assert javascript(node, source("logic.js"), [("SPARE_RUNS", [])]) == [bootstrap.SPARE_RUNS]


def test_a_blank_response_on_the_page_is_the_one_bootstrap_names(api, node):
    """The page reads a blank response box as the name Python gives a design made with one."""
    made = api.ok("api_make_design", {"design_type": "full_factorial", "factors": FACTORS[:2], "response": ""})
    assert javascript(node, source("logic.js"), [("DEFAULT_RESPONSE", [])]) == [made["spec"]["response"]]


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
    """Box-Behnken designs and rotatable CCDs need a center run; a DSD counts its own, so the page asks for one too."""
    cases = [(design, alpha) for design in bootstrap.DESIGNS for alpha in bootstrap.ALPHAS]
    asked = javascript(node, source("logic.js"), [("needsCentre", [design, alpha]) for design, alpha in cases])

    def needed(design: str, alpha: str) -> bool:
        if design == "dsd":
            return True
        return design in bootstrap.NEEDS_CENTER_RUN and bootstrap.NEEDS_CENTER_RUN[design] in (None, alpha)

    assert asked == [needed(design, alpha) for design, alpha in cases]


# --------------------------------------------------------------------------- the page's own elements
def test_every_element_app_js_looks_up_is_on_the_page():
    """A renamed or removed element would leave app.js reaching for nothing."""
    text = source("app.js")
    ids = {name for name in re.findall(r"[\"'`]#([\w-]+)", text) if not name.endswith("-")}
    ids |= {f"py-{key}" for key in re.findall(r'showPython\(\s*"(\w+)"', text)}
    ids |= {f"{part}-{tab}" for tab in re.findall(r'selectTab\("(\w+)"\)', text) for part in ("tab", "panel")}
    assert len(ids) > 40, "the search found app.js's lookups"
    assert ids - set(page().ids) == set()


def test_every_link_within_the_page_leads_somewhere():
    targets = set(re.findall(r'href="#([\w-]+)"', source("index.html")))
    assert targets, "the stepper links to each step"
    assert targets - set(page().ids) == set()
