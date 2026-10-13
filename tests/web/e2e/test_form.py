"""Step 1's form: it keeps its settings in line with the factors before anything is sent to Python.

The design list, which settings each design shows, the resolution, the center runs and an
optimal design's runs all follow the factor table as the reader edits it, and the page says
when the factors no longer describe the design it holds.
"""

from __future__ import annotations

import pytest

from tests.web._support import ALL_DESIGNS, factors_named
from tests.web.e2e._page import choose, factor_table, generate, set_factors, settled

NAMES = "ABCDEFGHJKLM"


def disabled_designs(page) -> list[str]:
    return page.evaluate(
        "[...document.querySelectorAll('#design-type option')].filter((o) => o.disabled).map((o) => o.value)"
    )


def shown_settings(page) -> list[str]:
    return page.evaluate(
        "[...document.querySelectorAll('[data-option]')].filter((e) => !e.hidden).map((e) => e.dataset.option)"
    )


def resolution(page) -> tuple[str, list[str]]:
    """Return the resolution chosen, and the ones the factor count rules out."""
    ruled_out = page.evaluate(
        "[...document.querySelectorAll('#resolution option')].filter((o) => o.disabled).map((o) => o.value)"
    )
    return page.input_value("#resolution"), ruled_out


def runs(page) -> tuple[str, str]:
    return page.input_value("#budget"), page.text_content("#budget-hint")


@pytest.mark.parametrize("k", [2, 7, 12])
def test_designs_the_factor_count_rules_out_cannot_be_chosen(page, bootstrap, k):
    set_factors(page, factors_named(*NAMES[:k]))
    expected = [d for d, info in bootstrap.DESIGNS.items() if not info["min_factors"] <= k <= info["max_factors"]]
    assert disabled_designs(page) == expected


def test_a_design_the_factors_rule_out_says_why_and_cannot_be_generated(page):
    set_factors(page, factors_named("A", "B"))  # the worked example's Box-Behnken design takes 3 to 7
    assert page.text_content("#design-hint") == "Box-Behnken takes 3 to 7 factors on this page; you have 2."
    assert "error" in page.get_attribute("#design-hint", "class").split()
    assert page.is_disabled("#generate")


@pytest.mark.parametrize("design_type", ALL_DESIGNS)
def test_each_design_shows_its_settings_and_where_the_book_explains_it(page, bootstrap, design_type):
    choose(page, design_type=design_type)
    info = bootstrap.DESIGNS[design_type]
    assert sorted(shown_settings(page)) == sorted(info["options"])
    assert page.text_content("#design-hint") == f"{info['hint']} More in the book"
    assert page.get_attribute("#design-hint a", "href").startswith("https://learnche.org/pid/")


def test_the_resolution_follows_the_factors_until_the_reader_chooses_one(page):
    choose(page, design_type="fractional_factorial")
    assert resolution(page) == ("3", ["4", "5"]), "three factors allow only resolution III"
    set_factors(page, factors_named(*"ABCD"))
    assert resolution(page) == ("4", ["5"])
    set_factors(page, factors_named(*"ABCDE"))
    assert resolution(page) == ("4", [])
    choose(page, resolution=5)
    set_factors(page, factors_named(*"ABCDEF"))
    assert resolution(page) == ("5", []), "the reader's choice is kept"
    set_factors(page, factors_named(*"ABCD"))
    assert resolution(page) == ("4", ["5"]), "until the factors rule it out"


def test_an_optimal_design_s_runs_follow_its_model_until_the_reader_sets_them(page, bootstrap):
    def automatic(model: str) -> tuple[str, str]:
        terms = bootstrap._n_terms(3, model)
        return str(terms + bootstrap.SPARE_RUNS), f"The model has {terms} coefficients."

    choose(page, design_type="d_optimal")
    assert runs(page) == automatic("quadratic")
    choose(page, model_type="main_effects")
    assert runs(page) == automatic("main_effects")
    choose(page, budget=20)
    choose(page, model_type="interactions")
    assert runs(page) == ("20", automatic("interactions")[1]), "the reader's number is kept"
    choose(page, budget="")
    choose(page, model_type="quadratic")
    assert runs(page) == automatic("quadratic"), "a cleared box follows the model again"


@pytest.mark.parametrize(
    ("design_type", "alpha", "least"),
    [("box_behnken", None, 1), ("dsd", None, 1), ("ccd", "rotatable", 1), ("ccd", "face_centered", 0)],
)
def test_designs_that_need_a_center_run_keep_one(page, design_type, alpha, least):
    choose(page, design_type="full_factorial", n_center_points=0)
    choose(page, design_type=design_type, **({"alpha": alpha} if alpha else {}))
    assert page.get_attribute("#n_center_points", "min") == str(least)
    assert page.input_value("#n_center_points") == str(least)


def test_the_page_says_when_the_factors_no_longer_describe_the_design(page):
    name = page.locator("#factors tbody tr").first.locator("[name=name]")
    assert page.is_hidden("#design-stale")
    name.fill("Temp")
    assert page.is_visible("#design-stale")
    name.fill("T")
    assert page.is_hidden("#design-stale"), "back as the design has it"
    page.fill("#response", "Yield")
    assert page.is_visible("#design-stale"), "the response name counts too"


def test_a_blank_response_is_the_y_the_design_is_made_with(blank_page):
    page = blank_page
    page.fill("#response", "")
    generate(page)
    assert page.is_hidden("#design-stale")
    page.fill("#response", "y")
    assert page.is_hidden("#design-stale"), "typing the name it stands for changes nothing"


def test_a_new_factor_starts_from_minus_one_to_one_and_can_be_removed(page):
    page.click("#add-factor")
    assert factor_table(page)[-1] == {"name": "", "low": "-1", "high": "1", "units": ""}
    page.locator("#factors tbody tr").last.get_by_role("button", name="Remove factor").click()
    assert len(factor_table(page)) == 3


def test_a_typed_axial_distance_has_a_box_only_while_it_is_chosen(page):
    choose(page, design_type="ccd")
    assert page.is_hidden("#alpha-distance")
    choose(page, alpha="distance")
    assert page.is_visible("#alpha-distance")
    choose(page, design_type="box_behnken")
    assert page.is_hidden("#alpha-distance"), "a design without axial runs has no distance to type"


def test_a_typed_distance_at_the_corners_radius_keeps_a_center_run(page):
    set_factors(page, factors_named("A", "B"))
    choose(page, design_type="ccd", alpha="face_centered", n_center_points=0)
    choose(page, alpha="distance", alpha_value="1.4")
    assert page.get_attribute("#n_center_points", "min") == "0"
    choose(page, alpha_value="1.4142135623730951")  # the square root of 2: every other run on one circle
    assert page.get_attribute("#n_center_points", "min") == "1"
    assert page.input_value("#n_center_points") == "1"


def test_replicates_chosen_for_one_design_do_not_carry_over_to_the_next_one_loaded(page):
    choose(page, n_replicates=3)
    page.click("#run-example")
    settled(page)
    assert page.input_value("#n_replicates") == "1", "the worked example is made once"


def test_two_labels_make_a_factor_categorical_and_leave_the_two_level_designs(page):
    set_factors(page, [*factors_named("A", "B"), {"name": "Supplier", "low": "old", "high": "new", "units": ""}])
    assert disabled_designs(page) == ["dsd", "box_behnken", "ccd", "d_optimal", "i_optimal"]
    assert page.text_content("#design-hint") == (
        "Box-Behnken needs numeric factors, and Supplier is categorical: choose a full factorial, fractional "
        "factorial or Plackett-Burman design."
    )
    assert page.is_disabled("#generate")
    choose(page, design_type="full_factorial")
    assert not page.is_disabled("#generate")


# --------------------------------------------------------------------------- numeric and categorical factors
def types(page) -> list[str]:
    """Read each factor's type, as its Type box shows it."""
    return page.eval_on_selector_all("#factors tbody [name=type]", "boxes => boxes.map((b) => b.value)")


def row(page, i: int):
    return page.locator("#factors tbody tr").nth(i)


def problems(page) -> str | None:
    return None if page.is_hidden("#factor-problems") else page.text_content("#factor-problems")


def test_every_factor_says_whether_it_is_numeric_or_categorical(page):
    assert types(page) == ["numeric", "numeric", "numeric"]
    assert page.locator("#factors thead th").all_text_contents()[:5] == ["Factor", "Type", "Low", "High", "Units"]
    assert row(page, 0).locator("[name=type] option").all_text_contents() == ["Numeric", "Categorical"]


def test_choosing_categorical_gives_two_labels_and_takes_the_units_away(page):
    row(page, 1).locator("[name=type]").select_option("categorical")
    assert factor_table(page)[1] == {"name": "P", "low": "A", "high": "B", "units": ""}
    assert row(page, 1).locator("[name=units]").is_disabled()
    assert row(page, 1).locator("[name=low]").get_attribute("aria-label") == "First label"
    assert page.evaluate("document.activeElement.name") == "low", "the first label is ready to be typed over"
    assert "box_behnken" in disabled_designs(page), "a categorical factor leaves only the two-level designs"

    row(page, 1).locator("[name=low]").fill("old")
    row(page, 1).locator("[name=type]").select_option("numeric")
    assert factor_table(page)[1] == {"name": "P", "low": "1", "high": "3", "units": "bar"}, "its numbers come back"
    row(page, 1).locator("[name=type]").select_option("categorical")
    assert factor_table(page)[1]["low"] == "old", "and so do its labels"


def test_typing_two_labels_makes_a_factor_categorical_and_two_numbers_numeric(page):
    factor = row(page, 0)
    factor.locator("[name=low]").fill("old")
    assert types(page)[0] == "numeric", "one label beside a number keeps the type chosen"
    assert problems(page) == (
        "T: give a number for both the low and the high value, or make it categorical for two labels."
    )
    assert page.is_disabled("#generate")
    factor.locator("[name=high]").fill("new")
    assert types(page)[0] == "categorical"
    assert factor.locator("[name=units]").input_value() == ""
    assert problems(page) is None
    factor.locator("[name=low]").fill("150")
    factor.locator("[name=high]").fill("200")
    assert types(page)[0] == "numeric"
    assert factor.locator("[name=units]").input_value() == "degC", "the units put aside come back"


def test_a_categorical_factor_is_added_with_two_labels(page):
    page.click("#add-categorical")
    assert types(page)[-1] == "categorical"
    assert factor_table(page)[-1] == {"name": "", "low": "A", "high": "B", "units": ""}
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "Factor name"


def test_what_a_factor_s_labels_still_need_is_said_before_python_is_asked(page):
    choose(page, design_type="full_factorial")
    page.click("#add-categorical")
    added = row(page, 3)
    added.locator("[name=name]").fill("Catalyst")
    added.locator("[name=low]").fill("1")
    assert problems(page) == "Catalyst: a label cannot be a number; write C1 rather than 1."
    assert added.locator("[name=low]").get_attribute("aria-invalid") is not None
    assert page.is_disabled("#generate")
    added.locator("[name=low]").fill("C1")
    added.locator("[name=high]").fill("C1")
    assert problems(page) == "Catalyst: its two labels must differ."
    added.locator("[name=high]").fill("C2")
    assert problems(page) is None
    assert not page.is_disabled("#generate")
    generate(page)
    assert factor_table(page)[3] == {"name": "Catalyst", "low": "C1", "high": "C2", "units": ""}


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_each_factor_shows_its_headings_and_its_type_in_full(page):
    row(page, 1).locator("[name=type]").select_option("categorical")
    captions = row(page, 1).evaluate(
        "tr => [...tr.cells].map((td) => getComputedStyle(td, '::before').content).filter((c) => c !== 'none')"
    )
    assert captions == ['"Factor"', '"Type"', '"First label"', '"Second label"', '"Units"']
    select = row(page, 1).locator("[name=type]")
    assert select.evaluate("s => s.scrollWidth <= s.clientWidth"), "Categorical is not cut short"
