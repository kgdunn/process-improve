"""The settings that balance the responses, in Learn: with two responses or more given a goal.

A study of a yield to maximise, a cost to minimise and a purity with a target is made, practised
and analysed. Below the analysis of the response on show, Learn lists each response's goal in the
balance, to change, says how good the balance is, lists the settings and each response there with
its desirability, names the response that holds the overall desirability down most, and draws the
overlay of the acceptable ranges with a star at the balanced settings. Confirmation runs there are
judged response by response.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.web.e2e._page import calls, last_reply, practise, reveal, run_sheet, saved_studies, settled
from tests.web.e2e.test_layout import overflowing
from tests.web.e2e.test_responses import PHONE, three_responses


def balance_rows(page: Any) -> list[list[str]]:  # noqa: ANN401
    """Read the table of each response at the balanced settings, its foot too, as the reader sees it."""
    return page.evaluate(
        "() => [...document.querySelectorAll('#balance-out table.balance-table tr')]"
        ".map((tr) => [...tr.cells].map((c) => c.textContent))"
    )


def balanced(page: Any) -> dict:  # noqa: ANN401
    """Return the last balance the page found."""
    return last_reply(page, "api_balance")["result"]


def rebalanced(page: Any, change: Any) -> dict:  # noqa: ANN401
    """Make a change in the balance's table, wait for the balance it asks for, and return it."""
    before = page.evaluate("window.__pi.count('api_balance')")
    change()
    page.wait_for_function(f"window.__pi.count('api_balance') > {before}")
    settled(page)
    return balanced(page)


def d_of(balance: dict, name: str) -> float:
    return next(r["d"] for r in balance["responses"] if r["name"] == name)


@pytest.fixture
def practised(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of the three responses, its results simulated, in Learn at the balance."""
    three_responses(fresh_page)
    practise(fresh_page)
    reveal(fresh_page, "#balance")
    fresh_page.wait_for_selector("#balance-chart .js-plotly-plot")
    return fresh_page


def test_the_balance_says_how_good_it_is_and_where_to_run(practised):
    page = practised
    b = balanced(page)
    assert page.text_content("#h-balance") == "Balance the responses"
    goals = page.evaluate(
        "() => [...document.querySelectorAll('#balance-goals tbody tr')].map((tr) => [...tr.cells].map((c) => "
        "c.querySelector('input, select')?.value ?? c.textContent))"
    )
    assert [row[:2] for row in goals] == [
        ["Yield (%)", "Maximise"],
        ["Cost ($/kg)", "Minimise"],
        ["Purity (%)", "Target 63%"],
    ]
    assert [row[4:] for row in goals] == [["3", "even"]] * 3, "each counts alike, and evenly, until the reader says"
    assert page.text_content("#balance-out .lead") == (
        f"These settings balance the responses best: an overall desirability of {b['composite']:.2f}."
    )
    rows = balance_rows(page)
    assert rows[0] == ["Response", "Predicted", "95% prediction interval", "Desirability, d"]
    assert [row[0] for row in rows[1:4]] == ["Yield", "Cost", "Purity"]
    assert [row[3] for row in rows[1:4]] == [f"{r['d']:.2f}" for r in b["responses"]]
    assert rows[4] == ["Overall desirability, D", f"{b['composite']:.2f}"]
    limiting = next(r for r in b["responses"] if r["name"] == b["limiting"])
    assert (
        f"{limiting['label']} holds the overall desirability down most: its desirability is {limiting['d']:.2f}."
        in (page.text_content("#balance-out"))
    )
    sent = calls(page, "api_balance")[-1]["payload"]
    assert set(sent["models"]) == {"Yield", "Cost", "Purity"}, "each response with its own model"


def test_the_overlay_shades_where_every_response_is_acceptable_and_stars_the_balanced_settings(practised):
    page = practised
    b = balanced(page)
    traces = page.evaluate(
        "() => document.querySelector('#balance-chart .js-plotly-plot').data.map((t) => ({name: t.name, type: t.type, "
        "contours: t.contours ?? null, x: t.x, y: t.y, legend: t.showlegend !== false}))"
    )
    shade = traces[0]
    assert shade["contours"]["type"] == "constraint"
    boundaries = [t for t in traces if t["type"] == "contour" and t["contours"].get("coloring") == "none"]
    assert len(boundaries) == 3, "a line at each response's acceptable boundary"
    star = traces[-1]
    assert star["name"] == "Balanced settings"
    assert [star["x"][0], star["y"][0]] == b["overlay"]["at"]
    legend = [t["name"] for t in traces if t["legend"]]
    assert legend[0] == "Every response acceptable"
    assert legend[-1] == "Balanced settings"
    caption = page.text_content("#balance-chart figcaption")
    assert caption.startswith(f"Drawn over {b['overlay']['x']} and {b['overlay']['y']}, the two factors that move")
    drawn = page.evaluate(
        "() => [...document.querySelectorAll('#balance-chart g.contour')].map((g) => g.querySelectorAll('path').length)"
    )
    assert drawn[0] > 0, "the shading is drawn"
    for surface, paths, line in zip(b["overlay"]["responses"], drawn[1:], boundaries, strict=True):
        z = np.array(surface["z"])
        crosses = any(z.min() < level < z.max() for level in surface["levels"])
        assert (paths > 0) == crosses, "a line where the boundary crosses the settings drawn"
        assert (line["name"] in legend) == crosses, "and a key for it, or the caption says why there is none"
    shown = page.evaluate("() => document.querySelector('#balance-chart .js-plotly-plot').data[0].z")
    assert np.array(shown)[1:-1, 1:-1].tolist() == b["overlay"]["acceptable"]


def test_a_boundary_drawn_bounds_the_shading(practised):
    page = practised
    b = balanced(page)
    cost = next(s for s in b["overlay"]["responses"] if s["name"] == "Cost")
    middle = round(float(np.median(cost["z"])), 2)
    high = page.locator('#balance-goals [aria-label="Cost: acceptable to"]')
    after = rebalanced(page, lambda: high.fill(str(middle)))
    surface = next(s for s in after["overlay"]["responses"] if s["name"] == "Cost")
    acceptable = np.array(after["overlay"]["acceptable"], dtype=bool)
    assert not (acceptable & (np.array(surface["z"]) >= middle)).any(), "no shading where the cost is above its range"
    legend = page.evaluate(
        "() => document.querySelector('#balance-chart .js-plotly-plot').data"
        ".filter((t) => t.showlegend !== false).map((t) => t.name)"
    )
    assert f"Cost: {middle:g} $/kg and below" in legend
    assert "No line for Cost" not in page.text_content("#balance-chart figcaption")


def test_a_response_counted_as_more_important_moves_the_balance_towards_it(practised):
    page = practised
    before = balanced(page)
    after = rebalanced(page, lambda: page.select_option('#balance-goals [aria-label="Cost: importance"]', "5"))
    assert calls(page, "api_balance")[-1]["payload"]["goals"] == {"Cost": {"importance": 5}}
    assert after["coded"] != before["coded"], "the balanced settings move"
    assert d_of(after, "Cost") > d_of(before, "Cost"), "towards the cost's own best"
    assert "at importance 5" in page.text_content("#balance-out") or after["limiting"] != "Cost"
    assert page.is_visible("#balance-reset")
    back = rebalanced(page, lambda: page.click("#balance-reset"))
    assert back["coded"] == before["coded"]
    saved = saved_studies(page, lambda s: bool(s) and "goals" not in s[0])
    assert "goals" not in saved[0]


def test_a_narrower_range_typed_in_moves_the_balance_and_one_upside_down_is_said(practised):
    page = practised
    before = balanced(page)
    cost = next(r for r in before["responses"] if r["name"] == "Cost")
    low = page.locator('#balance-goals [aria-label="Cost: acceptable from"]')
    high = page.locator('#balance-goals [aria-label="Cost: acceptable to"]')
    count = page.evaluate("window.__pi.count('api_balance')")
    low.fill(str(round(cost["high"] + 5, 2)))
    page.wait_for_timeout(700)  # the balance waits for the typing to rest
    assert (
        page.text_content("#balance-problems")
        == "The acceptable range of Cost must run from a lower value to a higher one."
    )
    assert low.get_attribute("aria-invalid") is not None
    assert page.evaluate("window.__pi.count('api_balance')") == count, "nothing Python would refuse is sent"
    target = round(cost["low"] + 0.25 * (cost["high"] - cost["low"]), 3)
    after = rebalanced(page, lambda: (low.fill(""), high.fill(str(target))))
    assert page.is_hidden("#balance-problems")
    assert next(r for r in after["responses"] if r["name"] == "Cost")["high"] == target
    cost_of = lambda b: next(r["predicted"] for r in b["responses"] if r["name"] == "Cost")  # noqa: E731
    assert cost_of(after) < cost_of(before), "a cost acceptable only near its lowest: the balance costs less"
    saved = saved_studies(page, lambda s: bool(s) and s[0].get("goals", {}).get("Cost", {}).get("high") == target)
    assert saved[0]["goals"] == {"Cost": {"high": target}}


def test_confirmation_runs_at_the_balanced_settings_are_judged_response_by_response(practised):
    page = practised
    b = balanced(page)
    rows = len(run_sheet(page))
    before = page.evaluate("window.__pi.count('api_balance')")
    page.click("#add-balance-confirmation")
    page.wait_for_function(f"window.__pi.count('api_balance') > {before}")
    settled(page)
    assert len(run_sheet(page)) == rows + 3
    (group,) = balanced(page)["confirmation"]
    assert group["at"] == b["confirm_at"]
    assert [v["name"] for v in group["responses"]] == ["Yield", "Cost", "Purity"]
    reveal(page, "#balance-confirm")
    verdict = page.locator("#balance-confirm-out .verdict")
    assert verdict.count() == 1
    lines = verdict.locator("p").all_text_contents()
    assert lines[0].startswith("3 confirmation runs at T = ")
    assert [line.split(":")[0] for line in lines[1:]] == ["Yield", "Cost", "Purity"]
    assert all("95% prediction interval" in line for line in lines[1:]), "their results simulated, each judged"
    assert page.is_hidden("#add-balance-confirmation"), "runs there already: their results are what is wanted"
    assert balanced(page)["coded"] == b["coded"], "the confirmation runs are not fitted"


def test_the_balance_shows_its_python(practised):
    page = practised
    reveal(page, "#py-balance")
    page.click("#py-balance summary")
    code = page.text_content("#py-balance code")
    assert "optimize_responses(" in code
    assert "balanced = optimize_responses(" in code
    assert "analyze_experiment(X[done], runs[name][done]" in code
    assert "analysis = analyze_experiment" not in code, "the balance fits each response itself"


def test_one_response_with_a_goal_has_nothing_to_balance(practised):
    page = practised
    reveal(page, "#responses")
    rows = page.locator("#responses tbody tr")
    rows.nth(1).locator("[name=goal]").select_option("learn")
    before = page.evaluate("window.__pi.count('api_balance')")
    page.wait_for_function(f"window.__pi.count('api_balance') > {before}")  # the yield and the purity: still two
    settled(page)
    assert [r["name"] for r in balanced(page)["responses"]] == ["Yield", "Purity"]
    analysed = page.evaluate("window.__pi.count('api_analyze_runs')")
    rows.nth(2).locator("[name=goal]").select_option("learn")
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {analysed}")
    settled(page)
    page.wait_for_function("document.querySelector('#balance').hidden")
    assert page.evaluate("window.__pi.count('api_balance')") == before + 1, "one response with a goal: nothing to ask"


@pytest.mark.parametrize("context_options", [{**PHONE, "color_scheme": "dark"}])
def test_the_balance_fits_a_phone_in_dark_mode(practised):
    page = practised
    page.wait_for_function("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    assert overflowing(page) == []
    chart = page.locator("#balance-chart .js-plotly-plot").bounding_box()
    assert chart["x"] + chart["width"] <= 390
