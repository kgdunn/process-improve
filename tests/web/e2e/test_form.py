"""The plan's form: it keeps its settings in line with the factors before anything is sent to Python.

The design list, which settings each design shows, the resolution, the center runs and an
optimal design's runs all follow the factor table as the reader edits it, a factor's name and
levels are checked as they are typed, and the page says when the design no longer describes the
run sheet it holds.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from tests.web._support import ALL_DESIGNS, NO_BOOK_PAGE, factors_named
from tests.web.e2e._page import choose, factor_table, generate, reveal, run_sheet, set_factors, settled

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
    designs = [*bootstrap.DESIGNS.items(), *bootstrap.MIXTURES.items()]
    expected = [d for d, info in designs if not info["min_factors"] <= k <= info["max_factors"]]
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
    if design_type in NO_BOOK_PAGE:
        assert page.text_content("#design-hint") == info["hint"], "the hint alone, with no link"
        assert page.locator("#design-hint a").count() == 0
        return
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


def stale(page) -> bool:
    """Whether Run says the design has changed since its run sheet was made, and Design's tab says so too."""
    shown = page.get_attribute("#design-stale", "hidden") is None
    assert shown == (page.text_content("#badge-plan") == "changed")
    return shown


def test_the_page_says_when_the_plan_no_longer_describes_the_run_sheet(page):
    high = page.locator("#factors tbody tr").first.locator("[name=high]")
    assert not stale(page)
    high.fill("210")
    assert stale(page)
    high.fill("200")
    assert not stale(page), "back as the run sheet has it"
    choose(page, n_center_points=5)
    assert stale(page), "the design's settings count too"
    choose(page, n_center_points=3)
    assert not stale(page)
    choose(page, design_type="ccd")
    assert stale(page)
    reveal(page, "#design-stale")
    assert page.is_visible("#design-stale")
    page.click("#restore-plan")
    assert not stale(page), "the plan put back as the run sheet was made"
    assert page.input_value("#design-type") == "box_behnken"


def test_renaming_the_response_leaves_the_run_sheet_as_it_is(page):
    page.fill("#response", "Yield of the reactor")
    page.fill("#response-units", "mol%")
    assert not stale(page), "the response is the reader's to rename: its runs and its column stay"
    assert page.text_content("#study-title") == "Yield of the reactor (mol%), maximise"


def test_the_new_run_sheet_is_made_from_the_stale_notice(page):
    page.locator("#factors tbody tr").first.locator("[name=high]").fill("210")
    reveal(page, "#remake")
    page.click("#remake")
    settled(page)
    assert not stale(page)
    assert max(float(row[2]) for row in run_sheet(page)) == 210, "T's runs reach its new high value"


def test_a_blank_response_is_the_y_the_design_is_made_with(blank_page):
    page = blank_page
    page.fill("#response", "")
    generate(page)
    assert not stale(page)
    assert page.locator("#design-table thead th button.sort").last.text_content() == "y"
    page.fill("#response", "y")
    assert not stale(page), "typing the name it stands for changes nothing"


def test_a_new_factor_starts_blank_and_can_be_removed(page):
    page.click("#add-factor")
    assert factor_table(page)[-1] == {"name": "", "low": "", "high": "", "units": ""}
    assert page.evaluate("document.activeElement.getAttribute('aria-label')") == "Factor name"
    assert page.is_hidden("#factor-problems"), "a blank row is no factor yet"
    page.locator("#factors tbody tr").last.get_by_role("button", name="Remove factor").click()
    assert len(factor_table(page)) == 3


def test_a_factor_s_name_is_checked_as_it_is_typed(page):
    """Python takes a variable name, unlike every other factor's: the page says so before Python is asked."""
    name = page.locator("#factors tbody tr").first.locator("[name=name]")
    name.fill("a b")
    assert problems(page) == "a b: a name is letters, digits and _, and starts with a letter, such as Temp_2."
    assert name.get_attribute("aria-invalid") is not None
    assert page.is_disabled("#generate")
    name.fill("P")
    assert problems(page) == "P: two factors cannot share a name."
    name.fill("Temp_2")
    assert problems(page) is None
    assert page.is_enabled("#generate")


def test_a_named_factor_s_empty_levels_are_pointed_out_once_the_reader_has_been_to_them(page):
    page.click("#add-factor")
    added = row(page, 3)
    added.locator("[name=name]").fill("Salt")
    assert problems(page) is None, "not before the levels are reached"
    assert page.is_disabled("#generate"), "but the run sheet waits for them"
    added.locator("[name=low]").focus()
    added.locator("[name=units]").focus()
    assert (
        problems(page)
        == "Salt: give a number for both the low and the high value, or make it categorical for two labels."
    )


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
    page.click("#show-example")
    settled(page)
    assert page.input_value("#n_replicates") == "1", "the worked example is made once"


def test_two_labels_make_a_factor_categorical_and_leave_the_two_level_designs(page):
    set_factors(page, [*factors_named("A", "B"), {"name": "Supplier", "low": "old", "high": "new", "units": ""}])
    numeric = ["dsd", "box_behnken", "ccd", "d_optimal", "i_optimal", "maximin_lhs"]
    assert disabled_designs(page) == [*numeric, "simplex_lattice", "simplex_centroid", "extreme_vertices"]
    assert page.text_content("#design-hint") == (
        "Box-Behnken needs numeric factors, and Supplier is categorical: choose a full factorial, fractional "
        "factorial, Plackett-Burman or Taguchi design."
    )
    assert page.is_disabled("#generate")
    choose(page, design_type="full_factorial")
    assert not page.is_disabled("#generate")


# --------------------------------------------------------------------------- numeric and categorical factors
def types(page) -> list[str]:
    """Read each factor's type, as its switch shows it."""
    return page.eval_on_selector_all(
        "#factors tbody [name=categorical]", "boxes => boxes.map((b) => (b.checked ? 'categorical' : 'numeric'))"
    )


def row(page, i: int):
    return page.locator("#factors tbody tr").nth(i)


def flip(factor) -> None:
    """Flip a factor's type with one tap on its switch, as a reader would."""
    factor.locator(".switch").click()


def problems(page) -> str | None:
    return None if page.is_hidden("#factor-problems") else page.text_content("#factor-problems")


def test_every_factor_says_whether_it_is_numeric_or_categorical(page):
    assert types(page) == ["numeric", "numeric", "numeric"]
    headings = ["Factor name", "Type", "Low", "High", "Units"]
    assert page.locator("#factors thead th").all_text_contents()[:5] == headings
    assert row(page, 0).locator(".switch__text").text_content() == "Numeric"
    assert row(page, 0).locator("[name=categorical]").get_attribute("role") == "switch"


def test_one_tap_on_the_switch_flips_the_type_and_the_word_beside_it(page):
    flip(row(page, 1))
    assert types(page)[1] == "categorical"
    assert row(page, 1).locator(".switch__text").text_content() == "Categorical"
    flip(row(page, 1))
    assert types(page)[1] == "numeric"
    assert row(page, 1).locator(".switch__text").text_content() == "Numeric"


def test_the_switch_answers_the_keyboard(page):
    row(page, 1).locator("[name=categorical]").focus()
    page.keyboard.press("Space")
    assert types(page)[1] == "categorical"


def test_choosing_categorical_gives_two_labels_and_takes_the_units_away(page):
    flip(row(page, 1))
    assert factor_table(page)[1] == {"name": "P", "low": "A", "high": "B", "units": ""}
    assert row(page, 1).locator("[name=units]").is_disabled()
    assert row(page, 1).locator("[name=low]").get_attribute("aria-label") == "First label"
    assert page.evaluate("document.activeElement.name") == "low", "the first label is ready to be typed over"
    assert "box_behnken" in disabled_designs(page), "a categorical factor leaves only the two-level designs"

    row(page, 1).locator("[name=low]").fill("old")
    flip(row(page, 1))
    assert factor_table(page)[1] == {"name": "P", "low": "1", "high": "3", "units": "bar"}, "its numbers come back"
    flip(row(page, 1))
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
def test_on_a_phone_each_factor_is_a_numbered_card_with_its_headings(page):
    reveal(page, "#factors")
    flip(row(page, 1))
    captions = row(page, 1).evaluate(
        "tr => [...tr.cells].map((td) => getComputedStyle(td, '::before').content).filter((c) => c !== 'none')"
    )
    assert captions == ['"Factor name"', '"Type"', '"First label"', '"Second label"', '"Units"']
    cards = page.eval_on_selector_all("#factors tbody tr", "rows => rows.map((tr) => tr.getBoundingClientRect())")
    gaps = [b["top"] - (a["top"] + a["height"]) for a, b in pairwise(cards)]
    assert all(gap >= 8 for gap in gaps), "the cards stand apart"
    border = row(page, 1).evaluate("tr => getComputedStyle(tr).borderTopStyle")
    assert border == "solid", "each factor's boxes sit inside a card of their own"
    text = row(page, 1).locator(".switch__text")
    assert text.evaluate("t => t.scrollWidth <= t.clientWidth + 1"), "Categorical is not cut short"
