"""Design opens with the question read back as one sentence: what Learn's answer will answer."""

from __future__ import annotations


def test_design_reads_the_question_back_and_follows_the_form(page):
    page.click("#go-plan")
    asked = page.locator("#asked")
    assert asked.text_content() == "Find the settings of Temperature, Pressure and Flow that give the highest Yield."
    assert asked.bounding_box()["y"] < page.locator("#response-card").bounding_box()["y"], "over the form"

    page.select_option("#response-goal", "minimize")
    assert asked.text_content().endswith("that give the lowest Yield.")
    page.select_option("#response-goal", "learn")
    assert asked.text_content() == "Learn how Temperature, Pressure and Flow change Yield."

    # Over Run and Learn the question is its one line, and the sentence is not repeated.
    page.click("#go-run")
    assert asked.is_hidden()


def test_nothing_is_read_back_before_there_is_a_question(blank_page):
    page = blank_page
    assert page.locator("#asked").is_hidden()
    page.fill("#response", "Strength")
    assert page.locator("#asked").is_visible(), "once a response is named: the factors are typed already"
