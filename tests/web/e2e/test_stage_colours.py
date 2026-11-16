"""A colour for each stage, on every screen: the bar, the rule over a card, the button to press, and the checks.

Design is blue, Run is ochre and Learn is green. The stage on show gives its colour to the rule over each of its
cards and to the button a reader presses there, and a button that leads to the next stage wears that stage's
colour. Under Learn's answer a check is tinted by what it found, and its mark still says it: nothing rests on colour
alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.e2e._page import reveal

COLOUR = "(one) => getComputedStyle(one).{}"
THEMES = pytest.mark.parametrize("context_options", [{"color_scheme": "light"}, {"color_scheme": "dark"}])


def stage_colour(page: Any, stage: str) -> str:  # noqa: ANN401 - a Playwright Page
    """Return the colour the bar gives a stage, as its number wears it."""
    return page.locator(f"#go-{stage} .step").evaluate(COLOUR.format("backgroundColor"))


def rules(page: Any, panels: str) -> set[str]:  # noqa: ANN401
    """Return the colours of the rule over each of the panels on show that has one: a part inside a panel has none."""
    return set(
        page.locator(panels).evaluate_all(
            "(all) => all.filter((one) => one.checkVisibility()).map((one) => getComputedStyle(one))"
            ".filter((style) => style.borderTopWidth !== '0px').map((style) => style.borderTopColor)"
        )
    )


@THEMES
def test_the_stage_on_show_gives_its_colour_to_its_panels_and_its_answer(page):
    colours = {stage: stage_colour(page, stage) for stage in ("plan", "run", "learn")}
    assert len(set(colours.values())) == 3, "a colour a stage"
    assert rules(page, "#step-analyse, #step-analyse .best") == {colours["learn"]}
    assert page.locator("#answer").evaluate(COLOUR.format("borderLeftColor")) == colours["learn"]
    page.click("#go-run")
    assert rules(page, "#step-run") == {colours["run"]}
    chosen = page.locator('#step-run .tabs [aria-selected="true"]')
    assert chosen.evaluate(COLOUR.format("borderBottomColor")) == colours["run"], "the tab chosen inside the stage"
    page.click("#go-plan")
    assert rules(page, "#step-plan, .brief .card") == {colours["plan"]}


def test_a_button_that_leads_on_wears_the_colour_of_the_stage_it_leads_to(page):
    colours = {stage: stage_colour(page, stage) for stage in ("plan", "run", "learn")}
    background = COLOUR.format("backgroundColor")
    page.click("#go-plan")
    assert page.locator("#generate").evaluate(background) == colours["run"], "Make the run sheet leads to Run"
    page.click("#go-run")
    assert page.locator("#to-learn").evaluate(background) == colours["learn"], "and what the results say is in Learn"
    assert page.locator("#simulate").evaluate(background) == colours["run"], "a button that stays wears its stage's"


@THEMES
def test_a_check_is_tinted_by_what_it_found_and_its_mark_still_says_it(page):
    reveal(page, "#checks")
    found = page.locator("#checks .learn-check").evaluate_all(
        """(checks) => checks.map((check) => ({
          state: check.dataset.state,
          tint: getComputedStyle(check).backgroundColor,
          mark: getComputedStyle(check.querySelector('summary'), '::after').content,
        }))"""
    )
    states = {check["state"] for check in found}
    assert {"ok", "look"} <= states, "the worked example has a check that passes and one to look at"
    tints = {state: {check["tint"] for check in found if check["state"] == state} for state in states}
    assert all(len(tint) == 1 for tint in tints.values()), "one tint a finding"
    assert tints["ok"] != tints["look"]
    panel = page.locator("#answer").evaluate(COLOUR.format("backgroundColor"))
    assert panel not in tints["ok"] | tints["look"], "neither is the plain panel"
    marks = {check["state"]: check["mark"] for check in found}
    assert (marks["ok"], marks["look"]) == ('"✓"', '"!"'), "the mark says it without the colour"
