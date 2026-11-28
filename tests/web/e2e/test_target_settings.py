"""The settings that give a target, in Learn: each response's model run backwards.

The reader types what a response should be, or several, or picks a run to match. The page finds the settings at
which the models predict that, says whether they do, lists each factor's setting with where it lies in its range and
each response's target against its prediction, and draws each response's line of settings over two factors with the
runs made. The settings can be confirmed with runs or taken to the what-if panel, and one factor is solved for from
the others where that panel holds them.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.web.e2e._page import copy_link, last_reply, practise, reveal, run_sheet, settled
from tests.web.e2e.test_rename import rename
from tests.web.e2e.test_report import download_report, headings, opened
from tests.web.e2e.test_responses import three_responses
from tests.web.e2e.test_specification import window

PLOT = "document.querySelector('#target-chart .plot')"
YIELD = '#target-goals tr[data-response="Yield"]'


@pytest.fixture
def studied(fresh_page: Any) -> Any:  # noqa: ANN401 - a Playwright Page
    """Return a study of three responses, its results simulated, in Learn at the section of the targets."""
    three_responses(fresh_page)
    practise(fresh_page)
    reveal(fresh_page, "#target")
    settled(fresh_page)
    return fresh_page


def aim(page: Any, response: str, value: str) -> dict:  # noqa: ANN401
    """Type a response's target, wait for the settings, and return what Python answered."""
    answered = "window.__pi.replies.filter((r) => r.fn === 'api_target_settings').length"
    before = page.evaluate(answered)
    page.fill(f'#target-goals tr[data-response="{response}"] [name="target"]', value)
    page.wait_for_function(f"{answered} > {before}")
    page.wait_for_selector("#target-out .lead")
    return last_reply(page, "api_target_settings")["result"]


def names(page: Any) -> list[str]:  # noqa: ANN401
    """Return the names of the chart's traces that its legend lists."""
    return page.evaluate(f"{PLOT}.data.filter((trace) => trace.showlegend !== false).map((trace) => trace.name)")


def test_a_target_typed_gives_the_settings_each_response_there_and_the_line_of_them(studied):
    page = studied
    assert page.locator("#target-out .lead").count() == 0, "nothing is found before a target is typed"
    found = aim(page, "Yield", "62")
    assert page.text_content("#target-out .lead") == "These settings give the target, by the models."
    settings, responses = page.locator("#target-out table").all()
    assert [row.locator("td").first.text_content() for row in settings.locator("tbody tr").all()] == ["T", "P", "F"]
    meters = settings.locator("meter").all()
    assert [float(m.get_attribute("value")) for m in meters] == pytest.approx([s["value"] for s in found["settings"]])
    assert meters[0].get_attribute("aria-label").endswith(", in its range of 150 to 200 degC")
    cells = [cell.text_content() for cell in responses.locator("tbody tr").first.locator("th, td").all()]
    assert cells[:3] == ["Yield", "62%", "62%"]
    assert "of the 20,000 settings tried give the target" in page.text_content("#target-out")
    page.wait_for_selector("#target-chart .js-plotly-plot")
    assert names(page) == ["Yield at 62%", "Runs you made", "Settings found"]
    assert "The line is where the response is predicted at its target" in page.text_content("#target-chart figcaption")
    code = page.text_content("#py-target code")
    assert 'targets = {"Yield": 62}' in code
    assert "least_squares(" in code


def test_two_targets_are_met_together_and_the_slider_sets_one_too(studied):
    page = studied
    aim(page, "Yield", "62")
    found = aim(page, "Cost", "60")
    assert page.text_content("#target-out .lead") == "These settings give every target, by the models."
    assert [r["name"] for r in found["responses"]] == ["Yield", "Cost"]
    assert names(page) == ["Yield at 62%", "Cost at 60 $/kg", "Runs you made", "Settings found"]
    slider = page.locator(f'{YIELD} [name="slider"]')
    assert float(slider.input_value()) == pytest.approx(62, abs=0.5), "the slider follows the box"
    slider.focus()
    page.keyboard.press("ArrowRight")
    page.wait_for_function(f"document.querySelector('{YIELD} [name=\"target\"]').value !== '62'")
    assert float(page.input_value(f'{YIELD} [name="target"]')) > 62, "and the box follows the slider"


def test_a_run_is_matched_and_the_targets_are_cleared(studied):
    page = studied
    page.select_option("#target-match", "2")
    page.wait_for_selector("#target-out .lead")
    sent = last_reply(page, "api_target_settings")["result"]["search"]["targets"]
    results = [float(v) for v in run_sheet(page)[2][-3:]]
    assert [sent[name] for name in ("Yield", "Cost", "Purity")] == pytest.approx(results, rel=1e-3)
    assert float(page.input_value(f'{YIELD} [name="target"]')) == pytest.approx(results[0], rel=1e-3)
    page.click("#target-clear")
    page.wait_for_function("document.querySelector('#target-out').childElementCount === 0")
    assert page.input_value(f'{YIELD} [name="target"]') == ""
    assert page.locator("#target-chart").is_hidden()
    assert page.locator("#py-target").is_hidden()


def test_a_target_out_of_reach_is_said_with_the_settings_that_come_closest(studied):
    page = studied
    found = aim(page, "Yield", "500")
    assert not found["reached"]
    assert page.text_content("#target-out .lead") == (
        "No settings inside your ranges give the target, by the models. These come closest."
    )
    assert "of the 20,000 settings tried" not in page.text_content("#target-out")


def test_the_settings_found_go_to_what_if_are_confirmed_and_one_factor_is_solved_for(studied):
    page = studied
    found = aim(page, "Yield", "62")
    page.wait_for_function("document.querySelector('#solve-out').textContent.includes('by the model')")
    assert page.input_value("#solve-factor") == "T"
    assert page.input_value("#solve-response") == "Yield"
    assert " gives Yield 62%, by the model." in page.text_content("#solve-out")
    page.click("#target-what-if")
    what_if = page.evaluate(
        "[...document.querySelectorAll('#what-if-settings input[type=number]')].map((box) => Number(box.value))"
    )
    assert what_if == pytest.approx([s["value"] for s in found["settings"]], rel=1e-3)
    assert page.get_attribute("#workspace", "data-phase") == "run", "the What-if row is over the runs, in Run"
    reveal(page, "#solve-factor")
    page.select_option("#solve-factor", "P")
    page.wait_for_function("document.querySelector('#solve-out').textContent.includes('P = ')")
    solved = last_reply(page, "api_solve_for")["result"]
    assert solved["factor"] == "P"
    # At the settings found the response is on its target already, so the factor is solved to where it stands.
    at = next(s["value"] for s in found["settings"] if s["name"] == "P")
    assert min(abs(one["value"] - at) for one in solved["solutions"]) < 0.02
    page.select_option("#target-up", "F")
    page.wait_for_function(f"{PLOT}.layout.yaxis.title.text.startsWith('F')")
    assert [page.input_value("#target-across"), page.input_value("#target-up")] == ["T", "F"]
    before = len(run_sheet(page))
    page.click("#target-confirm")
    page.wait_for_function(f"document.querySelectorAll('#design-table tr[data-run]').length === {before + 3}")
    settled(page)
    assert page.locator("#target-confirm").is_hidden(), "the runs are there: their results are what is wanted"
    assert page.text_content("#target-out .lead") == "These settings give the target, by the models."


def test_the_settings_found_are_found_again_under_a_factor_s_new_name(studied):
    """A factor renamed: what was found under the old name goes at once, and comes back under the new one."""
    page = studied
    aim(page, "Yield", "62")
    answered = "window.__pi.replies.filter((r) => r.fn === 'api_target_settings').length"
    before = page.evaluate(answered)
    rename(page, 0, "Temperature")
    page.wait_for_function(f"{answered} > {before}")
    settled(page)
    reveal(page, "#target")
    settings = page.locator("#target-out table").first
    assert [row.locator("td").first.text_content() for row in settings.locator("tbody tr").all()] == [
        "Temperature",
        "P",
        "F",
    ]
    assert page.input_value(f'{YIELD} [name="target"]') == "62", "the target typed stays"


def test_a_target_and_a_window_go_with_the_study_by_link_and_in_this_browser(open_context, context, studied):
    """What Learn was asked is kept, not its answers: opened again, the boxes hold it and it is answered again."""
    page = studied
    found = aim(page, "Yield", "62")
    reveal(page, "#window")
    judged = window(page, "Yield", "low", "60")
    copy_link(page)
    by_link, in_this_browser = open_context().new_page(), context.new_page()
    for again, address in ((by_link, page.input_value("#share-url")), (in_this_browser, page.url.split("#")[0])):
        again.goto(address)
        again.wait_for_selector("#campaign-banner", state="visible")
        again.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
        settled(again)
        again.click("#go-learn")  # a study this browser keeps opens on the stage it was left on, which was Run
        again.wait_for_selector("#target-out .lead")  # the card opens on the goal the study asks: its target
        reveal(again, "#window")
        again.wait_for_selector("#window-out .lead")
        assert again.input_value(f'{YIELD} [name="target"]') == "62"
        yields = '#window-limits tr[data-response="Yield"]'
        ends = [again.input_value(f'{yields} [name="{end}"]') for end in ("low", "high")]
        assert ends == ["60", ""], "an end left open stays open"
        assert last_reply(again, "api_target_settings")["result"]["settings"] == found["settings"]
        assert last_reply(again, "api_specification")["result"]["share"] == judged["share"]
        assert again.locator("#window-clear").is_visible()
        reveal(again, "#target")
        assert again.locator("#target-clear").is_visible()


def test_a_factor_solved_for_is_set_under_what_if_at_one_press(studied):
    """The sentence says what the factor must be; its button puts the factor there, the others where they were."""
    page = studied
    aim(page, "Yield", "62")
    boxes = "[...document.querySelectorAll('#what-if-settings input[type=number]')].map((box) => Number(box.value))"
    before = page.evaluate(boxes)
    page.wait_for_selector("#solve-out button[data-value]")
    solved = last_reply(page, "api_solve_for")["result"]["solutions"][0]["value"]
    use = page.locator("#solve-out button[data-value]").first
    assert use.text_content().startswith("Set T to ")
    assert use.text_content().endswith(" degC under What if")
    use.click()
    page.wait_for_function(f"{boxes}[0] !== {before[0]}")
    after = page.evaluate(boxes)
    assert after[0] == pytest.approx(solved, rel=1e-3)
    assert after[1:] == before[1:], "the other factors stay where What if had them"


def test_the_report_holds_what_the_targets_and_the_windows_gave(studied):
    page = studied
    aim(page, "Yield", "62")
    page.wait_for_selector("#solve-out button[data-value]")
    reveal(page, "#window")
    window(page, "Yield", "low", "60")
    page.wait_for_selector("#window-chart .js-plotly-plot")
    _, html = download_report(page)
    report = opened(page, html)
    asked = ["Settings that give a target", "Settings that meet a specification"]
    assert [heading for heading in headings(report) if heading in asked] == asked
    assert headings(report)[-3:] == ["The next step", "The Python code", "Terms used"], "after them, as on the page"
    target, specification = (report.locator(f"section:has(> h2:text-is('{heading}'))") for heading in asked)
    assert target.locator("p").first.text_content() == "These settings give the target, by the models."
    assert target.locator("table").count() == 2, "each factor's setting, and each response there beside its target"
    assert target.locator("h3").all_text_contents()[-1] == "One factor from the others"
    assert " gives Yield 62%, by the model." in target.text_content()
    assert specification.locator("p").first.text_content() == page.text_content("#window-out .lead")
    for section in (target, specification):
        assert section.locator("figure img").count() == 1, "its chart, as an image"
    assert report.locator("button, input, select, script").count() == 0, "nothing to use, only to read"
