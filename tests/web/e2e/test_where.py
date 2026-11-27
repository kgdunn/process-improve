"""Where to run, in Learn: one card. The dials set the factors, and under them a goal is chosen.

Four cards used to answer the one question: the best settings, What if, the settings that give a target, and those
that meet a specification. They are one card now. The dials and the map of What if stand first; under them the
reader chooses what the settings are for, and that goal's panel finds them: the best for the response's goal, a
target to hit, or a window to stay in. Every control of the four cards is in the one, and a link to a part of it
chooses its goal.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.e2e._page import (
    choose,
    generate,
    open_question,
    practise,
    reveal,
    saved_studies,
    set_factors,
    settled,
)

GOALS = ["best", "target", "window"]
TABS = "#goals [role=tab]:not([hidden])"


def panels(page: Any) -> list[bool]:  # noqa: ANN401 - a Playwright Page
    """Return whether each goal's panel is on show: the best, the target, the window."""
    return [page.is_visible(f"#{goal}") for goal in GOALS]


@pytest.fixture
def where(page: Any) -> Any:  # noqa: ANN401
    """Return the worked example with the card that answers where to run on show."""
    reveal(page, "#where")
    page.wait_for_selector("#where:not([hidden]) #what-if:not([hidden])")
    page.wait_for_selector("#best:not([hidden])")
    return page


def test_one_card_holds_the_dials_and_a_goal_to_choose(where):
    page = where
    assert page.text_content("#h-where") == "Where to run"
    inside = page.locator("#where").evaluate(
        "(card) => ['what-if', 'best', 'target', 'window'].every((id) => card.contains(document.getElementById(id)))"
    )
    assert inside, "the four cards are parts of the one"
    assert page.locator(TABS).all_text_contents() == ["The best for your goal", "Hit a target", "Stay in a window"]
    assert page.locator(f"{TABS}[aria-selected=true]").all_text_contents() == ["The best for your goal"]
    assert panels(page) == [True, False, False], "one goal's panel at a time, under the dials"
    dials, tabs = page.locator("#what-if-settings").bounding_box(), page.locator("#goals").bounding_box()
    assert dials["y"] + dials["height"] <= tabs["y"], "the dials first, then the goal"


def test_a_goal_is_chosen_by_a_press_or_by_the_arrow_keys(where):
    page = where
    page.click("#goal-target")
    assert panels(page) == [False, True, False]
    assert page.get_attribute("#goal-target", "aria-selected") == "true"
    assert page.get_attribute("#goal-best", "aria-selected") == "false"
    assert page.is_visible("#target-goals"), "what each response should be"
    page.keyboard.press("ArrowRight")
    assert panels(page) == [False, False, True]
    assert page.evaluate("document.activeElement.id") == "goal-window"
    page.keyboard.press("ArrowRight")
    assert panels(page) == [True, False, False], "round to the first"
    page.keyboard.press("End")
    assert page.evaluate("document.activeElement.id") == "goal-window"
    # Each tab says which panel it shows, and each panel which tab names it.
    for goal in GOALS:
        assert page.get_attribute(f"#goal-{goal}", "aria-controls") == f"goal-panel-{goal}"
        assert page.get_attribute(f"#goal-panel-{goal}", "aria-labelledby") == f"goal-{goal}"


def test_the_goal_chosen_is_kept_through_a_new_fit(where):
    page = where
    page.click("#goal-window")
    page.select_option("#model", "main_effects")
    settled(page)
    page.wait_for_selector("#window:not([hidden])")
    assert panels(page) == [False, False, True]


def test_a_goal_the_study_cannot_be_asked_is_not_offered_and_one_goal_is_no_choice(blank_page):
    """With no factor to move over a range, no target or window can be searched: the best settings stand alone."""
    page = blank_page
    open_question(page)
    page.fill("#response", "Strength")
    page.select_option("#response-goal", "maximize")
    glue = {"name": "Glue", "type": "categorical", "levels": ["Old", "New"]}
    cure = {"name": "Cure", "type": "categorical", "levels": ["Air", "Oven"]}
    set_factors(page, [glue, cure])
    generate(page, design_type="full_factorial", n_center_points=0, n_replicates=2)
    practise(page)
    page.click("#go-learn")
    page.wait_for_selector("#where:not([hidden]) #best:not([hidden])")
    settled(page)
    assert panels(page) == [True, False, False]
    assert page.is_hidden("#goals"), "one goal is no choice: its panel stands alone"


def test_the_card_opens_on_what_the_study_asks(blank_page, context, site):
    """A study with a target typed opens on that goal the next time; until then, on the first goal offered."""
    page = blank_page
    generate(page)
    practise(page)
    reveal(page, "#where")
    page.wait_for_selector("#where:not([hidden]) #what-if-out table", state="attached")
    assert page.get_attribute("#goal-target", "aria-selected") == "false"
    predicted = page.locator("#what-if-out tbody td").nth(1).text_content().split()[0].rstrip("%")
    page.click("#goal-target")
    page.fill("#target-goals input", predicted)
    page.keyboard.press("Tab")
    settled(page)
    page.wait_for_selector("#target-out .lead")
    saved_studies(page, lambda saved: bool(saved) and bool(saved[0].get("targets")))
    again = context.new_page()
    again.goto(site)
    again.wait_for_selector("#target-out .lead")
    assert again.get_attribute("#goal-target", "aria-selected") == "true"
    assert panels(again)[1], "the target's panel is the one on show"


def test_the_answer_leads_to_the_best_settings_whatever_goal_is_chosen(where):
    page = where
    page.click("#goal-target")
    page.click("#answer-more")
    assert panels(page) == [True, False, False]
    assert page.evaluate("document.activeElement.id") == "best"
    top = page.locator("#best").bounding_box()["y"]
    assert 0 <= top < 400, "under the bar"


@pytest.mark.parametrize(("part", "shown"), [("target", [False, True, False]), ("window", [False, False, True])])
def test_a_link_to_a_part_of_a_goal_chooses_that_goal(context, site, part, shown):
    page = context.new_page()
    page.goto(f"{site}?example=worked&show={part}")
    page.wait_for_selector(f"#{part}.shown-part", timeout=60_000)
    assert panels(page) == shown
    assert page.evaluate("document.activeElement.id") == part


def test_the_report_holds_every_goal_s_part_whichever_is_on_show(where):
    """The settings that give a target, typed under their goal, are in the report made while another goal shows."""
    page = where
    page.click("#goal-target")
    page.fill("#target-goals input", "62")
    page.keyboard.press("Tab")
    settled(page)
    page.wait_for_selector("#target-out table")
    page.click("#goal-best")
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    text = report.text_content("body")
    assert "The best settings for your goal" in text
    assert "Settings that give a target" in text


def test_a_mixture_s_card_has_its_goals_and_no_dials(blank_page):
    """A mixture has no What if panel: its best blend, a target and a window are the card."""
    page = blank_page
    open_question(page)
    page.fill("#response", "Strength")
    page.select_option("#response-goal", "maximize")
    set_factors(page, [{"name": name, "low": "", "high": "", "units": ""} for name in ("Oil", "Water", "Wax")])
    choose(page, design_type="simplex_lattice")
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.wait_for_selector("#where:not([hidden]) #best:not([hidden])")
    assert page.is_hidden("#what-if")
    assert page.locator(TABS).first.text_content() == "The best for your goal"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_goals_fit_and_the_page_scrolls_only_downwards(where):
    page = where
    tabs = page.locator("#goals").bounding_box()
    assert tabs["x"] >= 0
    assert tabs["x"] + tabs["width"] <= 390
    for goal in GOALS:
        page.click(f"#goal-{goal}")
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), goal
