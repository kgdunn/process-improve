"""The ladder: for two to five numeric factors, the designs as steps that each add runs to the one before.

A step is a point on one line with its runs in all, the runs it adds, and what they let the reader learn. A press
puts the step's design and its aim in the form, and the run sheet it makes has the runs the step says.
"""

from __future__ import annotations

import pytest

from tests.web._support import factors_named
from tests.web.e2e._page import generate, run_sheet, set_factors, settled

STEPS = "#ladder-steps .ladder__step"


def steps(page) -> list[list[str]]:
    """Return each step as its key, its runs and its name."""
    return page.locator(STEPS).evaluate_all(
        """(all) => all.map((one) => [one.dataset.step, one.querySelector('.ladder__runs').textContent,
            one.querySelector('strong').textContent])"""
    )


def test_three_factors_have_four_steps_each_with_its_runs_and_what_it_adds(blank_page):
    assert steps(blank_page) == [
        ["half", "4 runs", "Half the corners"],
        ["corners", "8 runs, 4 more", "Every corner"],
        ["center", "11 runs, 3 more", "Center points"],
        ["axial", "17 runs, 6 more", "Axial points"],
    ]
    top = blank_page.locator("#ladder").bounding_box()["y"]
    assert top < blank_page.locator("#design-card").bounding_box()["y"], "over the design it sets"


@pytest.mark.parametrize(
    ("step", "runs", "design", "aim"),
    [
        ("half", 4, "fractional_factorial", "main_effects"),
        ("corners", 8, "full_factorial", "interactions"),
        ("center", 11, "full_factorial", "interactions"),
        ("axial", 17, "ccd", "quadratic"),
    ],
)
def test_a_step_pressed_sets_the_design_and_the_aim_and_makes_the_runs_it_says(blank_page, step, runs, design, aim):
    page = blank_page
    page.click(f'{STEPS}[data-step="{step}"]')
    settled(page)
    assert page.input_value("#design-type") == design
    assert page.locator('input[name="aim"]:checked').get_attribute("value") == aim
    pressed = page.locator(f'{STEPS}[aria-pressed="true"]')
    assert pressed.evaluate_all("(all) => all.map((one) => one.dataset.step)") == [step], "the step the form stands on"
    assert page.evaluate("document.activeElement.dataset.step") == step, "the focus stays on the step pressed"
    generate(page)
    assert len(run_sheet(page)) == runs


def test_the_step_follows_the_form_and_no_step_is_pressed_off_the_ladder(blank_page):
    page = blank_page
    page.select_option("#design-type", "box_behnken")
    assert page.locator(f'{STEPS}[aria-pressed="true"]').count() == 0, "a Box-Behnken design is no step of it"
    page.select_option("#design-type", "full_factorial")
    page.fill("#n_center_points", "0")
    page.dispatch_event("#n_center_points", "input")
    page.dispatch_event("#n_center_points", "change")
    pressed = page.locator(f'{STEPS}[aria-pressed="true"]')
    assert pressed.evaluate_all("(all) => all.map((one) => one.dataset.step)") == ["corners"]


@pytest.mark.parametrize(("names", "count"), [("AB", 3), ("ABCDE", 4), ("ABCDEF", 0), ("A", 0)])
def test_the_ladder_is_for_two_to_five_numeric_factors(blank_page, names, count):
    set_factors(blank_page, factors_named(*names))
    assert blank_page.locator(STEPS).count() == count
    assert blank_page.is_visible("#ladder") == (count > 0)


def test_the_table_s_row_for_the_step_chosen_shows_the_step_s_runs(blank_page):
    """The table lists each design with its center points. A step without them said 4 runs over a row of 7."""
    page = blank_page
    set_factors(page, factors_named("Temp", "Solvent"))
    page.click('#ladder-steps button[data-step="corners"]')
    settled(page)
    step = page.text_content('#ladder-steps button[data-step="corners"] .ladder__runs')
    row = page.locator('#candidates tr[data-design="full_factorial"] td')
    assert step == "4 runs"
    assert row.nth(1).text_content() == "4", "the runs of the step chosen"
    assert row.nth(2).text_content() == "0", "and its spare runs"
    assert "shows the 4 runs of the step you chose" in page.text_content("#candidates .stepped")
    page.click('#ladder-steps button[data-step="center"]')
    settled(page)
    assert row.nth(1).text_content() == "7", "the table's own row: the step is the design as first made"
    assert page.locator("#candidates .stepped").count() == 0
