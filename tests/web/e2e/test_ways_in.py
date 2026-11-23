"""The ways into a study: where a first visit lands, and the menu on the study's name in the bar.

A first visit, and any visit with nothing kept in the browser, opens on a blank question under a word of welcome:
what Factorial is, and the other ways in (the worked example, a table of the reader's own, the catalog). The study's
name in the bar opens a list that holds every way in and what is done with the study on the page. The list is
buttons under a button: Tab walks it, and a choice, Escape, a press elsewhere and Tab out of it each close it.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import generate, open_library, saved_studies, settled, study_menu

MENU = "#study-menu"
ENTRIES = [
    "New study",
    "Your studies 0",
    "Open the worked example",
    "Analyze data you have",
    "Design catalog",
    "Share or download this study",
    "View its report",
    "Settings",
]


def entries(page) -> list[str]:
    """Return what the study's menu offers, in its order."""
    return [" ".join(text.split()) for text in page.locator(f"{MENU} button, {MENU} a").all_text_contents()]


# --------------------------------------------------------------------------- a first visit
def test_a_first_visit_is_welcomed_over_a_blank_question(fresh_page):
    page = fresh_page
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    assert page.text_content("#study-title") == "A new study"
    assert page.is_visible("#welcome")
    assert "Sense it. Test it. Improve it." in page.text_content("#welcome .tagline")
    assert page.text_content("#h-welcome") == "Find out what really makes the difference."
    assert page.locator("#welcome button").all_text_contents() == [
        "Start my first experiment",
        "See a finished one",
        "Analyze data you have",
    ]
    assert page.locator("#welcome .welcome__promise li").all_text_contents() == [
        "Free, with nothing to install",
        "Your numbers stay on this device",
        "Works without internet",
    ]
    welcome, question = page.locator("#welcome").bounding_box(), page.locator("#response-card").bounding_box()
    assert welcome["y"] + welcome["height"] <= question["y"], "over the question it says to start with"
    assert page.evaluate("document.activeElement.id") != "response", "the welcome has the first screen, not a box"


def test_the_welcome_shows_a_study_s_four_steps_each_in_its_stage_s_colour(fresh_page):
    page = fresh_page
    steps = page.locator("#welcome .welcome__steps li")
    assert [text.split(" ", 1)[0] for text in steps.all_text_contents()] == ["Design", "Run", "Learn", "Next"]
    colours = steps.evaluate_all("(all) => all.map((one) => getComputedStyle(one, '::before').backgroundColor)")
    assert len(set(colours)) == 4, "a color a step"
    # The bar's stages wear the same colors, on show or not, and the mark is drawn in them.
    bar = page.locator("#studybar [role=tab] .step").evaluate_all(
        "(all) => all.map((one) => getComputedStyle(one).backgroundColor)"
    )
    assert bar == colours[: len(bar)], "Design, Run and Learn as the welcome names them"
    mark = page.locator("#studybar .mark__runs circle").evaluate_all(
        "(all) => all.map((one) => getComputedStyle(one).fill)"
    )
    assert mark == colours[:3] * 2, "opposite corners alike"
    assert page.locator("#studybar .mark__centre").evaluate("(one) => getComputedStyle(one).fill") == colours[3]


def test_start_my_first_experiment_puts_the_reader_at_the_question(fresh_page):
    page = fresh_page
    page.click("#welcome-start")
    assert page.is_hidden("#welcome")
    assert page.evaluate("document.activeElement.id") == "response", "what the study measures comes first"
    top = page.locator("#response-card").bounding_box()["y"]
    assert 0 <= top < 400, "and the question is on the screen"


def test_the_welcome_offers_the_worked_example(fresh_page):
    page = fresh_page
    page.click("#welcome-example")
    page.wait_for_function("document.querySelector('#study-title').textContent === 'Yield (%), maximize'")
    assert page.get_attribute("#workspace", "data-phase") == "learn", "at what the example found"
    assert page.is_hidden("#welcome")
    assert page.is_visible("#demo-banner")


def test_the_welcome_offers_to_analyse_data_the_reader_has(fresh_page):
    page = fresh_page
    page.click("#welcome-import")
    page.wait_for_selector("#import-dialog[open]")


def test_the_welcome_goes_once_a_new_study_is_asked_for_by_name(fresh_page):
    page = fresh_page
    study_menu(page, "#new-study")
    assert page.is_hidden("#welcome")
    assert page.evaluate("document.activeElement.id") == "response", "the keyboard is at the first question"


def test_a_question_begun_under_the_welcome_is_kept_and_the_next_visit_is_not_a_first_one(fresh_page, context, site):
    page = fresh_page
    page.fill("#response", "Purity")
    page.wait_for_function("localStorage.getItem('factorial-draft') !== null")
    again = context.new_page()
    again.goto(site)
    again.wait_for_function("document.querySelector('#response').value === 'Purity'")
    assert again.is_hidden("#welcome")


def test_the_worked_example_from_the_menu_comes_without_the_welcome(fresh_page):
    page = fresh_page
    study_menu(page, "#show-example")
    page.wait_for_function("document.querySelector('#study-title').textContent === 'Yield (%), maximize'")
    assert page.is_visible("#demo-banner")
    assert page.is_hidden("#welcome"), "the welcome is a first visit's"


def test_a_reader_with_a_study_is_picked_up_where_they_left_off_and_not_welcomed(blank_page, context, site):
    generate(blank_page)
    saved_studies(blank_page, lambda saved: len(saved) == 1)
    again = context.new_page()
    again.goto(site)
    again.wait_for_function("document.querySelector('#campaign-banner').textContent.startsWith('Picked up')")
    assert again.is_hidden("#welcome")
    assert again.is_hidden("#demo-banner")


# --------------------------------------------------------------------------- the study's menu
def test_the_study_s_name_opens_every_way_in_and_what_is_done_with_the_study(fresh_page):
    page = fresh_page
    assert page.is_hidden(MENU)
    assert page.get_attribute("#study-button", "aria-expanded") == "false"
    page.click("#study-button")
    assert page.is_visible(MENU)
    assert page.get_attribute("#study-button", "aria-expanded") == "true"
    assert entries(page) == ENTRIES
    assert page.get_attribute("#catalog-link", "href") == "catalog/"
    # A question with no run sheet has nothing to share and no report.
    assert page.is_disabled("#menu-share")
    assert page.is_disabled("#menu-report")
    page.click("#study-button")
    assert page.is_hidden(MENU), "a second press closes it"


def test_the_menu_closes_on_escape_on_a_press_elsewhere_and_on_a_choice(fresh_page):
    page = fresh_page
    page.click("#study-button")
    page.keyboard.press("Escape")
    assert page.is_hidden(MENU)
    assert page.evaluate("document.activeElement.id") == "study-button", "the keyboard is back where it was"

    page.click("#study-button")
    page.click("#h-plan")
    assert page.is_hidden(MENU)

    study_menu(page, "#new-study")
    assert page.is_hidden(MENU)
    assert page.get_attribute("#study-button", "aria-expanded") == "false"


def test_the_keyboard_opens_the_menu_walks_it_and_leaves_it(fresh_page):
    page = fresh_page
    page.focus("#study-button")
    page.keyboard.press("Enter")
    assert page.is_visible(MENU)
    walked = []
    for _ in range(5):
        page.keyboard.press("Tab")
        walked.append(page.evaluate("document.activeElement.id"))
    assert walked == ["new-study", "open-library", "show-example", "import-data", "catalog-link"]
    page.keyboard.press("Tab")  # past the two that are off while there is no run sheet, to the settings
    assert page.evaluate("document.activeElement.id") == "menu-settings"
    page.keyboard.press("Tab")  # and out of the list
    assert page.is_hidden(MENU), "Tab out of the list closes it"


def test_the_menu_shares_the_study_and_shows_its_report_once_there_is_a_run_sheet(blank_page):
    page = blank_page
    generate(page)
    settled(page)
    page.click("#study-button")
    assert page.is_enabled("#menu-share")
    assert page.is_enabled("#menu-report")
    page.click("#menu-share")
    assert page.get_attribute("#workspace", "data-phase") == "run", "where the study is downloaded and shared"
    assert page.evaluate("document.activeElement.id") == "download"
    assert page.is_visible("#share")

    with page.expect_popup() as popup:
        study_menu(page, "#menu-report")
    popup.value.wait_for_url("blob:**")
    assert popup.value.title().endswith(": report")


def test_your_studies_opens_over_the_page_and_says_when_it_keeps_none(fresh_page):
    page = fresh_page
    open_library(page)
    assert page.locator("#library").evaluate("(box) => box.open && box.matches(':modal')")
    assert page.text_content("#h-library") == "Your studies"
    assert page.is_visible("#library-none")
    assert page.is_disabled("#download-all")
    assert page.is_visible("#study-drop"), "a study file is opened here"
    page.click("#library-close")
    assert page.is_hidden("#library")
    assert page.evaluate("document.activeElement.id") == "study-button", "the keyboard is back at the study's name"
    open_library(page)
    page.keyboard.press("Escape")
    assert page.is_hidden("#library")


def test_your_studies_lists_a_kept_study_and_closes_when_one_is_opened(blank_page):
    page = blank_page
    generate(page)
    saved_studies(page, lambda saved: len(saved) == 1)
    study_menu(page, "#new-study")
    page.click("#study-button")
    assert "Your studies 1" in entries(page)
    page.click("#open-library")
    assert page.is_hidden("#library-none")
    page.locator("#library-list li .library__study button").click()
    page.wait_for_function("document.querySelector('#campaign-banner').textContent.startsWith('Opened')")
    assert page.is_hidden("#library"), "the study is on the page, so the list has gone"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 360, "height": 740}, "is_mobile": True}])
def test_on_a_phone_the_menu_and_your_studies_fit_the_screen(fresh_page):
    page = fresh_page
    page.click("#study-button")
    menu = page.locator(MENU).bounding_box()
    assert menu["x"] >= 0
    assert menu["x"] + menu["width"] <= 360
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    page.click("#open-library")
    sheet = page.locator("#library").bounding_box()
    assert sheet["x"] >= 0
    assert sheet["x"] + sheet["width"] <= 360
