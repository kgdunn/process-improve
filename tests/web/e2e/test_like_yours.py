"""Experiments like yours: beside the plan, what others in the reader's field have run.

The panel reads ``catalog/fields.json``, which the build writes from the catalog: here the catalog of a few designs,
whose one field is chemical reactions, with the pilot plant of Box, Hunter and Hunter (3 factors in 8 runs). A field
chosen shows its sentence, its experiments on the chart with the plan marked, and how the plan compares; the choice is
kept in the browser, and a field's page in the catalog chooses one through ``?field=``.
"""

from __future__ import annotations

from tests.web._support import factors_named
from tests.web.e2e._page import generate, reveal, set_factors

CHEMICAL = "chemical-processes"


def test_a_field_chosen_shows_its_experiments_beside_the_plan(fresh_page):
    page = fresh_page
    set_factors(page, factors_named("A", "B", "C"))
    reveal(page, "#like-yours")
    assert page.input_value("#field") == "", "no field until the reader chooses one"
    assert page.is_hidden("#like-yours-body")

    page.select_option("#field", CHEMICAL)
    assert page.text_content("#like-yours-sentence").startswith("Experiments here usually have about 3 factors")
    assert page.locator("#like-yours-chart .field-chart__mark").count() == 1
    assert "Your factors" in page.text_content("#like-yours-chart"), "before a design, the plan is its factors"
    assert page.text_content("#like-yours-verdict") == "Experiments here often have as many factors as yours."
    assert page.get_attribute("#like-yours-link", "href") == f"catalog/fields/{CHEMICAL}/"

    generate(page, design_type="full_factorial")
    assert "Your plan" in page.text_content("#like-yours-chart"), "with a design, the plan is its factors and runs"
    assert page.text_content("#like-yours-verdict") == "Your plan is larger than most here."


def test_the_field_is_kept_and_a_field_s_page_chooses_one(context, site):
    page = context.new_page()
    page.goto(f"{site}?field={CHEMICAL}")
    page.wait_for_function(f"document.querySelector('#field').value === '{CHEMICAL}'")
    page.goto(site)
    page.wait_for_function("!document.querySelector('#like-yours').hidden")
    assert page.input_value("#field") == CHEMICAL, "the browser keeps the field chosen"

    page.select_option("#field", "")
    page.goto(f"{site}?field=no-such-field")
    page.wait_for_function("!document.querySelector('#like-yours').hidden")
    assert page.input_value("#field") == "", "a field the catalog does not have chooses none"
