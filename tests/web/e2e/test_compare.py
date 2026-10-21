"""The designs that can learn the brief: made and scored as it is typed, picked from a list, compared in detail.

Design asks Python, once the brief has two factors and the typing pauses, for every design the
page offers for these factors, each made as Design makes it and all scored on the model the study
aims to learn, with power for the response's smallest change worth finding. The designs that fit
are listed by their runs, one picked for a reader who has not chosen; the detailed comparison
shows the nine measures, the FDS curves and a correlation map per design, and any design can be
taken from it into the plan.
"""

from __future__ import annotations

import re

import pytest

from tests.web._support import FACTORS, factors_named
from tests.web.e2e._page import (
    calls,
    candidates_done,
    definition,
    generate,
    last_reply,
    name_response,
    practise,
    reveal,
    set_aim,
    set_factors,
    settled,
)


def listed(page) -> list[str]:
    """Return the designs listed as able to learn the brief, in the order shown."""
    return page.locator("#candidates tbody tr").evaluate_all("rows => rows.map((tr) => tr.dataset.design)")


def picked(page) -> str:
    return page.get_attribute('#candidates tbody tr[aria-selected="true"]', "data-design")


def design_columns(page) -> list[str]:
    """Return the designs' names as the detailed comparison's columns run, left to right."""
    heads = page.locator("#comparison thead tr").first.locator("th").all_text_contents()
    assert heads[:2] == ["Measure", "Better"]
    return heads[2:]


def flip(page, design: str) -> None:
    """Tap the switch that shows or hides a design in the detailed comparison, and let its column settle."""
    page.get_by_role("switch", name=f"Show {design}").locator("xpath=..").click()
    page.wait_for_function("document.getAnimations().length === 0")


def test_the_designs_that_can_learn_the_brief_are_made_and_scored_as_it_is_typed(fresh_page):
    page = fresh_page
    name_response(page, "Yield", "%", noise=1, change=2)
    assert calls(page, "api_candidates") == [], "nothing to learn about before two factors"
    set_factors(page, FACTORS)
    candidates_done(page)
    (request,) = [c["payload"] for c in calls(page, "api_candidates")]
    assert request["model"] == "main_quadratic", "each factor's curvature, for a study of three factors"
    assert request["factors"] == FACTORS
    assert request["response_info"]["smallest_change"] == 2
    reply = last_reply(page, "api_candidates")["result"]
    fitting = [
        m["design_type"]
        for m, label in zip(reply["made"], reply["result"]["designs"], strict=True)
        if reply["result"]["values"][label]["fits"]
    ]
    assert sorted(listed(page)) == sorted(fitting)
    runs = page.locator("#candidates tbody tr td:nth-child(2)").all_text_contents()
    assert [int(r) for r in runs] == sorted(int(r) for r in runs), "sorted by their runs"
    assert picked(page) == "dsd", "the fewest runs that leave three or more to estimate the noise"
    assert page.input_value("#design-type") == "dsd"
    assert "cannot fit this model" in page.text_content("#candidates")
    assert page.text_content("#candidates thead th[colspan]") == "Power to find a change of 2 %"


def test_another_aim_lists_the_designs_that_can_learn_it(blank_page):
    page = blank_page
    set_aim(page, "main_effects")
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["model"] == "main_effects"
    assert "fractional_factorial" in listed(page), "a two-level fraction can learn the main effects"
    assert picked(page) == "fractional_factorial"
    headings = page.locator("#candidates thead th").all_text_contents()
    power = "Power to find a change of two noise standard deviations"  # no smallest change given
    assert headings == ["Design", "Runs", "Spare runs", power, "Stands out for", "Main effect"]


def stands_out(page) -> dict[str, str]:
    """Return what each listed design does best of those listed, by its design type."""
    return page.locator("#candidates tbody tr").evaluate_all(
        "rows => Object.fromEntries(rows.map((tr) => [tr.dataset.design, tr.querySelector('td.stands').textContent]))"
    )


def test_each_design_says_what_it_does_best_of_those_listed(blank_page):
    page = blank_page
    assert "Stands out for" in page.locator("#candidates thead th").all_text_contents()
    best = stands_out(page)
    assert best[listed(page)[0]].startswith("fewest runs"), "the first design listed has the fewest runs"
    assert sum("fewest runs" in words for words in best.values()) == 1
    reply = last_reply(page, "api_candidates")["result"]
    values = {
        m["design_type"]: reply["result"]["values"][sized]
        for m, sized in zip(reply["made"], reply["result"]["designs"], strict=True)
    }
    curvature = {t: v["power_quadratic"] for t, v in values.items() if v["fits"]}
    most = max(curvature.values())
    assert {t for t, words in best.items() if "power for curvature" in words} == {
        t for t, power in curvature.items() if power >= 0.99 * most
    }, "the designs with the most power for curvature, or within 1% of it"
    assert "within 1% of the best" in definition(page, "#candidates thead tr:first-child th:last-child .term")


def test_design_says_why_it_picked_its_design_until_the_reader_chooses(blank_page):
    page = blank_page
    assert page.text_content("#candidates .picked") == "Picked for you: DSD, the fewest runs with 3 or more spare."
    page.click('#candidates tbody tr[data-design="box_behnken"]')
    assert page.locator("#candidates .picked").is_hidden(), "the reader's choice needs no reason from Design"
    set_aim(page, "quadratic")
    candidates_done(page)
    assert page.locator("#candidates .picked").count() == 0


def test_the_designs_not_listed_say_why(blank_page):
    page = blank_page
    assert page.locator("#candidates .left-out").count() == 0, "three numeric factors: every design is offered"
    set_factors(page, FACTORS[:2])
    candidates_done(page)
    left_out = page.locator("#candidates .left-out").all_text_contents()
    assert left_out[0].endswith("are not listed: they take 3 factors or more, and the brief has 2.")
    assert "Box-Behnken" in left_out[0]
    set_factors(page, [{"name": "Catalyst", "low": "Pd", "high": "Pt"}, *FACTORS[1:]])
    candidates_done(page)
    left_out = page.locator("#candidates .left-out").all_text_contents()
    assert any(s.endswith("not listed: they need numeric factors, and Catalyst is categorical.") for s in left_out)
    assert "box_behnken" not in listed(page)


def test_power_is_for_the_smallest_change_worth_finding(blank_page):
    page = blank_page
    name_response(page, "Purity", "%", noise=1.5, change=6)
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    candidates_done(page)
    reply = last_reply(page, "api_candidates")["result"]
    assert reply["result"]["effect_size"] == 2.0, "a change of 6 is a coefficient of 3, two noise standard deviations"
    assert page.text_content("#candidates thead th[colspan]") == "Power to find a change of 6 %"


def test_picking_a_design_puts_it_in_the_plan_and_the_reader_s_choice_is_kept(blank_page):
    page = blank_page
    page.click('#candidates tbody tr[data-design="box_behnken"]')
    assert page.input_value("#design-type") == "box_behnken"
    assert picked(page) == "box_behnken"
    set_aim(page, "quadratic")
    candidates_done(page)
    assert picked(page) == "box_behnken", "a design the reader picked stays picked for another aim"


def test_the_designs_are_made_only_while_plan_is_on_show(blank_page):
    page = blank_page
    page.click("#go-run")
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    page.wait_for_timeout(700)
    assert len(calls(page, "api_candidates")) == 1, "nothing is made for a list nobody is looking at"
    page.click("#go-plan")
    candidates_done(page)
    assert len(calls(page, "api_candidates")) == 2
    assert calls(page, "api_candidates")[-1]["payload"]["factors"][0]["name"] == "Temp"


def test_a_brief_changed_while_its_designs_are_made_is_listed_again(blank_page):
    page = blank_page
    page.evaluate("window.__pi.hold('api_candidates')")
    name = page.locator("#factors tbody tr").first.locator("[name=name]")
    name.fill("Temp")
    name.press("Tab")  # a name is taken as the reader leaves its box
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    name.fill("Heat")
    name.press("Tab")
    assert page.get_attribute("#candidates", "data-stale") == "true", "the last list stays, dimmed"
    page.evaluate("window.__pi.release('api_candidates')")
    page.wait_for_function("window.__pi.count('api_candidates') === 3")
    page.evaluate("window.__pi.release('api_candidates')")
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["factors"][0]["name"] == "Heat"
    assert page.get_attribute("#candidates", "data-stale") == "false"


def test_when_the_designs_cannot_be_made_the_page_says_why(blank_page):
    page = blank_page
    page.evaluate("window.__pi.fake('api_candidates', {ok: false, error: 'the solver gave up', warnings: []})")
    page.locator("#factors tbody tr").first.locator("[name=name]").fill("Temp")
    page.keyboard.press("Tab")
    page.wait_for_function("window.__pi.count('api_candidates') === 2")
    candidates_done(page)
    assert page.text_content("#candidates") == "the solver gave up"
    assert page.is_disabled("#compare")


@pytest.mark.slow
def test_with_five_factors_the_optimal_designs_wait_to_be_searched_for(fresh_page):
    page = fresh_page
    set_factors(page, factors_named(*"ABCDE"))
    candidates_done(page)
    assert page.is_checked('input[name="aim"][value="main_effects"]'), "a screening aim for five factors"
    assert "d_optimal" in listed(page), "an optimal design for main effects is quick to search for"
    set_aim(page, "main_quadratic")
    candidates_done(page)
    assert "d_optimal" not in listed(page)
    assert "designs take a while to search for with 5 factors" in page.text_content("#candidates")
    page.click("#search")
    page.wait_for_function("window.__pi.count('api_candidates') === 3", timeout=10_000)
    candidates_done(page)
    assert calls(page, "api_candidates")[-1]["payload"]["search"] is True
    assert {"d_optimal", "i_optimal"} <= set(listed(page))
    assert page.is_hidden("#search")


def order_of(compared: dict) -> list[int]:
    """Return the order of the comparison's columns: the designs that fit the model, then those that cannot."""
    fits = [compared["values"][d]["fits"] for d in compared["designs"]]
    return [i for i, ok in enumerate(fits) if ok] + [i for i, ok in enumerate(fits) if not ok]


def test_the_comparison_in_detail_has_a_column_per_design_and_a_row_per_measure(page, first_paint):
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_function("document.querySelectorAll('#compare-charts .js-plotly-plot').length > 1")
    compared = first_paint["comparison"]["result"]
    assert design_columns(page) == [compared["designs"][i] for i in order_of(compared)]
    shown = [
        m for row, m in enumerate(compared["measures"]) if any(compared["display"][d][row] for d in compared["designs"])
    ]
    groups = page.locator("#comparison tbody tr.group").all_text_contents()
    assert groups == list(dict.fromkeys(m["group"] for m in shown))
    assert page.locator("#comparison tbody tr:not(.group)").count() == len(shown)
    charts = 1 + len(compared["designs"])  # the FDS plot and a correlation map per design
    page.wait_for_function(f"document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}")
    assert [c["fn"] for c in calls(page)] == ["api_catalogue"], "the build compared them"


def test_the_designs_that_cannot_fit_the_model_are_compared_at_the_right(page, first_paint):
    compared = first_paint["comparison"]["result"]
    fitting = [d for d in compared["designs"] if compared["values"][d]["fits"]]
    unfit = [d for d in compared["designs"] if not compared["values"][d]["fits"]]
    assert fitting, "the worked example compares designs that fit its model"
    assert unfit, "and designs that cannot"
    reveal(page, "#compare")
    page.click("#compare")
    assert design_columns(page) == fitting + unfit
    names = page.locator("#comparison thead tr").first
    assert names.locator("th.unfit").all_text_contents() == unfit
    assert names.locator("th.unfit.first").count() == 1, "one line sets them apart"
    fit_row = page.locator("#comparison tbody tr", has_text="Fits the model?").locator("td").all_text_contents()
    assert all(cell.startswith("no") for cell in fit_row[-len(unfit) :])


def test_the_detailed_comparison_shades_each_row_towards_its_best(page):
    """Among the designs that fit the model, a row's best is bold and the most shaded, its worst not shaded at all.

    The designs that cannot fit the model are not shaded. A row prints as many digits as tell its designs apart.
    """
    reveal(page, "#compare")
    page.click("#compare")
    rows = page.evaluate(
        """() => [...document.querySelectorAll('#comparison tbody tr:not(.group)')].map((tr) => ({
            measure: tr.cells[0].textContent,
            cells: [...tr.cells].slice(2).map((td) => ({
                text: td.textContent,
                unfit: td.classList.contains('unfit'),
                good: td.classList.contains('scored') ? Number(td.style.getPropertyValue('--good')) : null,
                best: td.classList.contains('row-best'),
                bold: Number(getComputedStyle(td).fontWeight) >= 700,
                shade: getComputedStyle(td).backgroundColor,
            })),
        }))"""
    )
    scored = [row for row in rows if any(c["good"] is not None for c in row["cells"])]
    assert len(scored) >= 6, "the averages, the worst cases, D, max |r| and power"
    for row in scored:
        fitting = [c for c in row["cells"] if not c["unfit"] and c["good"] is not None]
        assert not any(c["good"] is not None for c in row["cells"] if c["unfit"]), row["measure"]
        best = [c for c in fitting if c["best"]]
        assert best, row["measure"]
        assert all(c["good"] == 1 and c["bold"] for c in best), row["measure"]
        worst = [c for c in fitting if c["good"] == 0]
        assert worst, row["measure"]
        assert best[0]["shade"] != worst[0]["shade"], row["measure"]
    averages = next(row for row in rows if row["measure"].startswith("A/p"))
    assert all(re.fullmatch(r"0\.\d\d", c["text"]) for c in averages["cells"] if c["text"]), "two significant figures"
    runs = next(row for row in rows if row["measure"] == "Runs")
    assert all(c["good"] is None and c["text"].isdigit() for c in runs["cells"]), "a count, whole and unshaded"


def test_the_designs_heads_are_a_switch_to_show_each_then_the_button_to_use_it(page):
    reveal(page, "#compare")
    page.click("#compare")
    designs = design_columns(page)
    heads = page.locator("#comparison thead tr")
    assert heads.count() == 3, "the names, a row of switches, a row of buttons"
    assert heads.nth(1).locator("th").first.text_content() == "Show"
    assert all(page.get_by_role("switch", name=f"Show {d}").is_checked() for d in designs), "every design shown"
    uses = heads.nth(2).get_by_role("button").all()
    assert [b.get_attribute("aria-label") for b in uses] == [f"Use this design: {d}" for d in designs]


def test_a_design_hidden_moves_past_a_thick_line_in_grey_and_leaves_the_charts(page, first_paint):
    """Hiding a design sets it aside: the shading re-ranks without it, and showing it puts it back where it was."""
    reveal(page, "#compare")
    page.click("#compare")
    charts = 1 + len(first_paint["comparison"]["result"]["designs"])
    page.wait_for_function(f"document.querySelectorAll('#compare-charts .js-plotly-plot').length === {charts}")
    before = design_columns(page)
    best = page.locator("#comparison tbody tr", has_text="A/p, average").locator("td.row-best")
    leader = best.first.get_attribute("data-design")  # the design with the best A/p
    flip(page, leader)
    after = design_columns(page)
    assert after == [d for d in before if d != leader] + [leader], "to the far right"
    hidden = page.locator(f'#comparison [data-design="{leader}"]')
    assert all("set-aside" in (c.get_attribute("class") or "") for c in hidden.all()), "every cell of it, in grey"
    assert page.locator("#comparison thead tr").first.locator("th.parted").get_attribute("data-design") == leader
    groups = page.locator("#comparison tbody tr.group")
    assert groups.locator("td.parted").count() == groups.count(), "the line runs through the group rows"
    assert best.count() == 1, "the next design leads the row"
    assert best.get_attribute("data-design") != leader
    fds = "document.querySelector('#compare-charts figure.fds .js-plotly-plot')"
    shown = page.evaluate(f"{fds}._fullData.filter((t) => t.visible === true).map((t) => t.name)")
    assert leader not in shown
    assert page.locator(f'#compare-charts figure[data-design="{leader}"]').is_hidden()

    flip(page, leader)
    assert design_columns(page) == before, "back where it was"
    assert page.locator("#comparison .set-aside, #comparison .parted").count() == 0
    assert page.locator(f'#compare-charts figure[data-design="{leader}"]').is_visible()


def test_the_arrows_stay_beside_the_measures_as_the_designs_scroll(page):
    page.set_viewport_size({"width": 600, "height": 900})
    reveal(page, "#compare")
    page.click("#compare")
    box = page.locator("#comparison .scroll")
    row = page.locator("#comparison tbody tr:not(.group)").first
    cells = row.locator("td")

    def lefts() -> list[float]:
        return [cells.nth(i).evaluate("(td) => td.getBoundingClientRect().left") for i in range(3)]

    def right_of_name() -> float:
        return cells.nth(0).evaluate("(td) => td.getBoundingClientRect().right")

    before = lefts()
    assert before[1] == pytest.approx(right_of_name(), abs=1), "the arrows start where the names end"
    assert box.evaluate("(div) => div.scrollWidth > div.clientWidth"), "the designs do not all fit at this width"
    box.evaluate("(div) => { div.scrollLeft = 200; }")
    page.wait_for_function("document.querySelector('#comparison .scroll').scrollLeft > 0")
    after = lefts()
    assert after[:2] == pytest.approx(before[:2], abs=1), "the names and the arrows stay in view"
    assert after[2] < before[2] - 50, "the designs scroll past them"


def test_a_click_outside_the_detailed_comparison_closes_it(page):
    reveal(page, "#compare")
    page.click("#compare")
    page.wait_for_selector("#compare-dialog[open]")
    box = page.locator("#compare-dialog").bounding_box()
    page.mouse.click(box["x"] + 20, box["y"] + 20)
    assert page.is_visible("#compare-dialog"), "a click inside it leaves it open"
    page.mouse.click(box["x"] / 2, box["y"] + 20)
    page.wait_for_selector("#compare-dialog", state="hidden")


def test_a_design_taken_from_the_detailed_comparison_reaches_the_run_sheet_as_made(page, first_paint):
    chosen = next(m for m in first_paint["comparison"]["made"] if m["design_type"] == "d_optimal")
    reveal(page, "#compare")
    page.click("#compare")
    page.get_by_role("button", name=f"Use this design: {chosen['label']}").click()
    assert page.is_hidden("#compare-dialog")
    assert page.input_value("#design-type") == "d_optimal"
    assert page.text_content("#badge-plan") == "changed", "the run sheet on the page is the Box-Behnken design's"
    page.click("#generate")
    settled(page)
    assert calls(page, "api_make_design") == [], "the comparison made it already"
    assert page.locator("#design-table tbody tr").count() == chosen["design"]["n_runs"]
    assert calls(page, "api_simulate"), "practising: the same process, noise and seed, on the new runs"
    assert page.locator("#design-table input.sim").count() == chosen["design"]["n_runs"]


def test_a_design_made_for_the_brief_keeps_the_response_as_the_reader_named_it(blank_page):
    page = blank_page
    name_response(page, "Tensile strength", "MPa")
    candidates_done(page)
    generate(page)
    assert page.locator("#design-table thead th button.sort").last.text_content() == "Tensile strength"
    practise(page)
    assert page.text_content("#truth-equation").startswith("Tensile strength = ")
