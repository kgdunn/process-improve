"""The model as terms to press, in Learn, under "For the statistician".

The terms a model of the study's factors may hold are always there, with the model's own ticked, whatever model is
fitted. One press takes a term out or puts it in, and the model is fitted again: the answer, its checks and the
findings above follow. A term put in brings the main effects it is made of. Beside the terms, "As designed" goes
back to the design's own model and "Keep the clear terms" fits the terms the results support. The model list no
longer offers terms to choose: it says "Your terms" while the model is one the reader built.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web._support import FACTORS
from tests.web.e2e._page import (
    calls,
    choose,
    generate,
    open_question,
    open_statistician,
    practise,
    reveal,
    set_factors,
    settled,
)
from tests.web.e2e.test_analysis import PICKED, terms_fitted, ticked

MODEL_SHOWN = "document.querySelector('#model').selectedOptions[0].textContent"


def press(page: Any, term: str) -> None:  # noqa: ANN401 - a Playwright Page
    """Press a term, as a reader would, and wait for the model fitted with the change."""
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.get_by_label(term, exact=True).click()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)


def model_asked(page: Any) -> Any:  # noqa: ANN401
    """Return the model the last analysis asked Python for."""
    return calls(page, "api_analyze_runs")[-1]["payload"]["model"]


@pytest.fixture
def terms(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with the fold of the model open, its terms on show."""
    open_statistician(page)
    page.wait_for_selector("#statistician #terms:not([hidden])")
    return page


def test_the_terms_stand_with_the_model_and_one_press_takes_a_term_out_or_puts_it_in(terms):
    page = terms
    assert page.locator("#statistician > #terms").count() == 1, "under For the statistician, with the equation"
    assert page.evaluate(MODEL_SHOWN) == "As designed"
    assert "Your terms" not in page.locator("#model option").all_text_contents(), "no model to choose from the list"
    assert "Temperature·Flow" in ticked(page)
    assert "Temperature·Flow" in terms_fitted(page)
    said = page.locator("#findings > li").all_text_contents()

    press(page, "Temperature·Flow")
    assert "Temperature·Flow" not in ticked(page)
    assert "Temperature·Flow" not in terms_fitted(page), "the tables follow"
    assert page.locator("#findings > li").all_text_contents() != said, "and so do the findings above them"
    assert page.evaluate(MODEL_SHOWN) == "Your terms"
    assert ["Temperature", "Flow"] not in model_asked(page)["terms"]
    assert "Temperature:Flow" not in page.text_content("#py-analyse code"), "the Python describes the model built"

    press(page, "Temperature·Flow")
    assert "Temperature·Flow" in terms_fitted(page), "restored in one press"


def test_a_term_is_pressed_from_the_keyboard_and_keeps_the_keyboard_through_the_fit(terms):
    page = terms
    box = page.get_by_label("Pressure²", exact=True)
    box.focus()
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.keyboard.press("Space")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert "Pressure²" not in terms_fitted(page)
    assert page.evaluate("document.activeElement.closest('label').textContent.trim()") == "Pressure²"
    assert not box.is_checked(), "and says whether it is in the model"


def test_a_term_put_in_brings_the_main_effects_it_is_made_of(blank_page):
    page = blank_page
    generate(page, design_type="full_factorial", n_center_points=2)
    practise(page)
    open_statistician(page)
    choose(page, model="main_effects")
    settled(page)
    press(page, "P")
    assert ticked(page) == ["T", "F"], "a term taken out takes nothing with it"
    assert page.text_content("#terms-said") == ""

    press(page, "T·P")
    assert ticked(page) == ["T", "P", "F", "T·P"]
    assert page.text_content("#terms-said") == "T·P brought P with it."
    assert model_asked(page) == {"terms": [["T"], ["P"], ["F"], ["T", "P"]]}

    # Taken out again, the main effect leaves its interaction: a factor nested in another is written this way.
    press(page, "P")
    assert ticked(page) == ["T", "F", "T·P"]
    assert page.text_content("#terms-said") == ""


def test_as_designed_goes_back_to_the_design_s_own_model(terms):
    page = terms
    designed = ticked(page)
    assert page.is_disabled("#terms-designed"), "the model is the design's own already"
    press(page, "Flow²")
    assert page.is_enabled("#terms-designed")
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#terms-designed")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert ticked(page) == designed
    assert page.evaluate(MODEL_SHOWN) == "As designed"
    assert model_asked(page) == ""


def test_keep_the_clear_terms_fits_the_terms_the_results_support_at_once(terms):
    page = terms
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.click("#terms-clear")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert ticked(page) == ["Temperature", "Pressure", "Flow", "Temperature·Pressure", "Flow²"]
    assert page.text_content("#terms-said").startswith(
        "The results support Temperature, Pressure, Flow, Temperature·Pressure and Flow², without "
    )
    assert page.locator("#findings > li").last.text_content().startswith(PICKED), "the page says how they were chosen"
    assert model_asked(page)["terms"][-1] == ["Flow", "Flow"]

    press(page, "Flow²")
    assert page.text_content("#terms-said") == "", "what the search said was of the model before"
    assert page.locator("#findings li.picked").count() == 0, "the reader chose these terms"


def test_terms_the_design_cannot_tell_apart_are_one_term(blank_page):
    """In a half fraction of four factors each two-factor interaction is aliased with another: three terms, not six."""
    page = blank_page
    open_question(page)
    set_factors(page, [*FACTORS, {"name": "C", "low": 1, "high": 2, "units": "g"}])
    generate(page, design_type="fractional_factorial", n_center_points=0)
    practise(page)
    open_statistician(page)
    labels = [label.strip() for label in page.locator("#term-list label").all_text_contents()]
    pairs = [label for label in labels if " = " in label]
    assert pairs == ["T·P = F·C", "T·F = P·C", "T·C = P·F"]
    assert not {"F·C", "P·C", "P·F"} & set(labels), "an aliased term is not listed apart"


def test_a_model_that_cannot_be_fitted_leaves_its_terms_in_reach(blank_page):
    """With its last term taken out a model has nothing to fit: the terms stay on show, to put one back."""
    page = blank_page
    set_factors(page, FACTORS[:2])
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    reveal(page, "#model")
    open_statistician(page)
    assert ticked(page) == ["T", "P", "T·P"]
    for term in ("T·P", "P"):
        press(page, term)
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    page.get_by_label("T", exact=True).click()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)
    assert page.text_content("#analysis p.error") == "Choose at least one term for the model."
    assert page.locator("#statistician").count() == 0, "no model, and no fold of it"
    assert page.is_visible("#terms"), "the terms stand above what the page says"
    press(page, "P")
    assert page.locator("#statistician > #terms").count() == 1, "back with the model once one is fitted"
    assert terms_fitted(page) == sorted(["Intercept", "P"])


def test_a_mixture_s_model_is_chosen_by_name_and_has_no_terms_to_press(blank_page):
    page = blank_page
    set_factors(page, [{"name": name, "low": "", "high": "", "units": ""} for name in ("Oil", "Water", "Wax")])
    choose(page, design_type="simplex_lattice")
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.wait_for_selector("#analysis .equation", state="attached")
    open_statistician(page)
    assert page.is_hidden("#terms")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_terms_wrap_and_fit_the_screen(terms):
    page = terms
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    right = page.locator("#term-list label").evaluate_all(
        "(labels) => Math.max(...labels.map((one) => one.getBoundingClientRect().right))"
    )
    assert right <= 390
