"""Runs from a model of every run, in Next (#1374).

The worked example's goal is the largest yield. A press fits a Gaussian process of every run with a result and lists
the runs where it expects the most to gain, each with why; another adds them to the run sheet, as runs of the study.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, reveal, settled

PART = "#next-out .next__suggested"


def test_the_runs_suggested_are_listed_with_why_and_added_to_the_run_sheet(page):
    reveal(page, "#next")
    assert page.text_content(f"{PART} h4") == "Runs from a model of every run"
    page.select_option("#suggested-count", "2")
    page.click("#suggest-runs")
    page.wait_for_selector(f"{PART} tbody tr")
    heads = [head.strip() for head in page.locator(f"{PART} thead th").all_text_contents()]
    assert heads == ["Run", "Temperature", "Pressure", "Flow", "The model predicts", "Why here"]
    rows = page.locator(f"{PART} tbody tr").evaluate_all(
        "(rows) => rows.map((row) => [...row.cells].map((cell) => cell.textContent.trim()))"
    )
    assert [row[0] for row in rows] == ["1", "2"]
    assert all(row[5] in ("it looks better than the best so far", "little is known there") for row in rows)
    before = page.locator("#design-table tbody tr").count()
    page.click("#add-suggested")
    settled(page)
    assert page.locator("#design-table tbody tr").count() == before + 2
    assert calls(page, "api_add_runs")[-1]["payload"]["kind"] == "suggested"


def set_goal(page, goal: str, target: object = None) -> None:
    """Choose the response's goal in the brief, and wait for the analysis it asks for to be on the page."""
    before = page.evaluate("window.__pi.count('api_best_settings')")
    reveal(page, "#response-goal")
    page.select_option("#response-goal", goal)
    if target is not None:
        page.fill("#response-target", str(target))
    page.wait_for_function(f"window.__pi.count('api_best_settings') > {before}")
    settled(page)
    reveal(page, "#next")


def test_a_target_is_a_goal_runs_are_suggested_for_once_it_is_typed(page):
    set_goal(page, "target")
    assert not page.is_visible("#suggest-runs"), "a target not yet typed is no goal to suggest runs for"
    set_goal(page, "target", 60)
    page.click("#suggest-runs")
    page.wait_for_selector(f"{PART} tbody tr")
    sent = calls(page, "api_suggest_runs")[-1]["payload"]["spec"]["response_info"]
    assert (sent["goal"], sent["target"]) == ("target", 60)
