"""The design catalog in the browser: search and filter the designs, read one, and open it in Factorial.

The catalog served here is the build's, for the few designs of ``CATALOG_SAMPLE``: a published
experiment the page regenerates, with its results; a comparison; a generated design the page makes;
and one it does not. The tests of a book design's place, and of the applications, use ``book_context``,
whose catalog also holds ``BOOK_DESIGN``, a case study the book prints.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.web._support import BLOCKED_DESIGN, BOOK_DESIGN, CATALOG_SAMPLE, PAIRED_DESIGN
from tests.web.e2e._page import headers, run_sheet, settled

PILOT, COMPARISON, CCD, OMARS, CUBE = CATALOG_SAMPLE


def open_catalog(context, site: str, fragment: str = "") -> object:
    """Open the catalog, and wait until it lists its designs."""
    page = context.new_page()
    page.goto(f"{site}catalog/{fragment}")
    page.wait_for_selector(".result")
    return page


def shown(page) -> list[str]:
    """Return the designs the list shows, in order, by their ids."""
    return page.eval_on_selector_all(".result__title", "titles => titles.map((t) => t.dataset.id)")


def counted(page, text: str) -> None:
    page.wait_for_function(f"document.querySelector('#count').textContent.startsWith({text!r})")


def test_the_app_s_header_opens_the_catalog(fresh_page):
    fresh_page.click("#catalog-link")
    fresh_page.wait_for_url("**/catalog/")
    fresh_page.wait_for_selector(".result")
    assert fresh_page.text_content("#count") == "5 of 5 designs: 2 published, 3 generated"
    assert set(shown(fresh_page)) == {PILOT, COMPARISON, CCD, OMARS, CUBE}
    assert shown(fresh_page)[-1] == COMPARISON, "a comparison is listed after the designs"


def test_its_address_without_the_slash_leads_to_it(context, site):
    """factori.al/catalog is the folder's address: the server sends it on to factori.al/catalog/."""
    page = context.new_page()
    page.goto(f"{site}catalog")
    page.wait_for_selector(".result")
    assert page.url == f"{site}catalog/"


def test_the_toggle_and_the_search_narrow_the_list_and_the_address_keeps_them(context, site):
    page = open_catalog(context, site)
    page.click("#collection [data-value=generated]")
    counted(page, "3 of 5")
    assert set(shown(page)) == {CCD, OMARS, CUBE}
    page.fill("#query", "quadratic")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "col=generated" in page.url
    assert "q=quadratic" in page.url


def test_a_filtered_address_opens_with_its_filters(context, site):
    page = open_catalog(context, site, "#q=pilot&col=published")
    assert shown(page) == [PILOT]
    assert page.input_value("#query") == "pilot"
    assert page.get_attribute("#collection [data-value=published]", "aria-checked") == "true"


def test_a_family_s_count_is_what_ticking_it_shows(context, site):
    page = open_catalog(context, site)
    family = page.locator("#families label", has=page.locator("input[value=full_factorial]"))
    assert family.locator(".count").text_content() == "2"
    family.locator("input").check()
    counted(page, "2 of 5")
    assert set(shown(page)) == {PILOT, CUBE}
    assert "fam=full_factorial" in page.url


def test_a_design_s_detail_opens_on_the_design_its_source_and_its_use(context, site):
    """The same three cells for every design, each about this design; its family's history folds away."""
    page = open_catalog(context, site)
    page.click(f".result__title[data-id='{PILOT}']")
    page.wait_for_selector("#detail[open]")
    cells = page.locator("#detail-body .detail__head .detail__cell h3").all_text_contents()
    assert cells == ["This design", "Where it comes from", "Use it"]
    facts = page.locator("#detail-body .detail__facts li").all_text_contents()
    assert facts == ["3 factors (1 categorical)", "8 runs", "Estimates two-factor interactions"]
    text = " ".join(page.text_content("#detail-body").split())
    assert "Box, G. E. P., Hunter, W. G. and Hunter, J. S. (1978)" in text
    assert "recomputed" not in text, "how the catalogue checks an entry is not the reader's concern"
    assert "The library makes this design" not in text
    family = page.locator("#detail-body details.detail__family")
    assert family.locator("summary").text_content() == "About this design type: Full factorial"
    assert family.get_attribute("open") is None
    assert "Full factorial designs were first published by Fisher, R. A. (1926)" in family.text_content()
    assert page.get_attribute("#detail-body .detail__cite a", "target") == "_blank", "the source opens in a new tab"
    stray = "a => [...a.childNodes].filter((n) => n.nodeType === Node.TEXT_NODE).map((n) => n.textContent).join('')"
    assert page.eval_on_selector("#detail-body .actions", stray).strip() == "", "no text between the buttons"
    assert "d=bhh-pilot-plant-2-3" in page.url
    page.keyboard.press("Escape")
    page.wait_for_selector("#detail:not([open])", state="attached")
    assert "d=" not in page.url


def test_a_click_outside_a_design_s_detail_closes_it_but_a_selection_dragged_out_does_not(context, site):
    page = open_catalog(context, site, f"#d={PILOT}")
    page.wait_for_selector("#detail[open]")
    box = page.locator("#detail").bounding_box()
    line = page.locator("#detail-body .detail__application").bounding_box()
    y = line["y"] + 8
    page.mouse.move(line["x"] + 1, y)
    page.mouse.down()
    for x in range(int(line["x"]) + 20, int(box["x"] + box["width"]) + 20, 20):  # out past the dialog's right edge
        page.mouse.move(x, y)
    page.mouse.up()
    assert page.evaluate("document.querySelector('#detail').open"), "text selected to copy keeps the detail open"
    assert page.evaluate("getSelection().toString()") != ""
    page.mouse.click(box["x"] + box["width"] + 8, box["y"] + 8)
    page.wait_for_selector("#detail:not([open])", state="attached")
    assert "d=" not in page.url


def test_a_book_design_s_detail_links_to_its_section_and_names_its_figures(book_context, site):
    """The place links to the book's section, at its anchor; the figures are named, without their files."""
    page = open_catalog(book_context, site, f"#d={BOOK_DESIGN}")
    page.wait_for_selector("#detail[open] .detail__where a")
    place = page.locator("#detail-body .detail__where a")
    assert place.get_attribute("href") == (
        "https://learnche.org/pid/design-analysis-experiments/omars-worked-study#doe-omars-worked-study"
    )
    assert (place.get_attribute("target"), place.get_attribute("rel")) == ("_blank", "noopener")
    assert place.text_content() == "section A worked OMARS study: four factors on a fed-batch bioreactor"
    assert page.get_attribute("#detail-body .detail__cite a", "href") == "https://learnche.org/pid", (
        "the citation still links to the book"
    )
    where = " ".join(page.text_content("#detail-body .detail__where").split())
    assert where.startswith("Printed in section A worked OMARS study")
    assert "; Factor ranges; A table: Recipe / True titer, g/L / Gain over current;" in where, "tables by name"
    assert where.endswith("; and 7 more tables and figures."), "four places, then a count: the link leads to them"
    text = page.text_content("#detail-body") + page.text_content("#results")
    assert ".png" not in text, "no file path is shown"
    assert "../" not in text
    assert "list-table" not in text, "nor a note of the lines that print a table"
    notes = page.locator("#detail-body details.detail__notes")
    assert notes.count() == 1, "the long notes of how the entry was checked fold under a summary"
    assert notes.get_attribute("open") is None
    assert "numpy default_rng(7)" in notes.text_content()
    assert "Application: Bioprocessing and fermentation." in page.text_content("#detail-body .detail__application")


def test_a_design_printed_in_a_book_names_its_place_without_a_link(context, site):
    page = open_catalog(context, site, f"#d={PILOT}")
    page.wait_for_selector("#detail[open] .detail__where")
    assert page.locator("#detail-body .detail__where a").count() == 0, "a printed book has no address for its pages"
    assert page.text_content("#detail-body .detail__where").startswith("Printed in section 10.2 to 10.7")
    assert page.text_content("#detail-body .detail__application") == "Application: Chemical reactions and processes."


def test_an_application_narrows_the_list_and_the_address_keeps_it(book_context, site):
    page = open_catalog(book_context, site)
    listed = page.eval_on_selector_all("#topics input", "boxes => boxes.map((b) => b.value)")
    assert listed == ["chemical-processes", "bioprocessing"], "the most designs first"
    chemical = page.locator("#topics label", has=page.locator("input[value='chemical-processes']"))
    assert chemical.locator(".count").text_content() == "2"
    chemical.locator("input").check()
    counted(page, "2 of 6")
    assert set(shown(page)) == {PILOT, COMPARISON}, "a generated design has no application"
    assert "app=chemical-processes" in page.url
    badges = page.locator(".result", has=page.locator(f".result__title[data-id='{PILOT}']")).locator(".badge--topic")
    assert badges.all_text_contents() == ["Chemical reactions and processes"]
    assert "keeps the published designs only" in " ".join(page.text_content("#topics-hint").split())
    page.locator("#topics input[value='bioprocessing']").check()
    counted(page, "3 of 6")
    assert "app=chemical-processes%2Cbioprocessing" in page.url
    again = open_catalog(book_context, site, "#app=bioprocessing")
    counted(again, "1 of 6")
    assert shown(again) == [BOOK_DESIGN]
    assert again.is_checked("#topics input[value='bioprocessing']")


def test_the_applications_of_the_fewest_designs_are_a_click_away(context, site, catalog_files):
    """With ten applications, the filter lists eight; the rest open on a click, or when the address chose one."""
    index = json.loads((catalog_files / "index.json").read_text())
    ten = [topic for topic in index["topics"] if topic != "general"][:10]
    for design in index["designs"]:
        design["topics"] = {PILOT: ten, COMPARISON: ten[:3]}.get(design["id"], [])
    context.route(f"{site}catalog/index.json", lambda route: route.fulfill(json=index))
    page = open_catalog(context, site)
    assert page.locator("#topics > label").count() == 8
    assert page.text_content("#topics-more summary") == "2 more applications"
    assert not page.is_visible(f"#topics-more input[value='{ten[-1]}']")
    page.click("#topics-more summary")
    page.locator(f"#topics-more input[value='{ten[-1]}']").check()
    counted(page, "1 of 5")
    chosen = open_catalog(context, site, f"#app={ten[-1]}")
    assert chosen.is_visible(f"#topics-more input[value='{ten[-1]}']"), "the application chosen is in view"
    assert chosen.is_checked(f"#topics-more input[value='{ten[-1]}']")


def test_a_word_of_an_application_s_name_finds_its_designs(book_context, site):
    page = open_catalog(book_context, site)
    page.fill("#query", "fermentation")
    counted(page, "1 of 6")
    assert shown(page) == [BOOK_DESIGN]


def test_a_design_whose_runs_differ_in_leverage_lists_it_beside_each_run(context, site):
    """The face-centred CCD leans on its corners more than on its centre: a leverage column, and the heaviest named."""
    page = open_catalog(context, site, f"#d={CCD}")
    page.wait_for_selector("#detail[open] table.runs-table")
    words = page.text_content("#detail-body .detail__leverage")
    assert re.match(r"\d+ runs carry the most weight, with leverage 0\.\d\d; the least is 0\.\d\d\. ", words), words
    assert page.locator("#detail-body canvas").count() == 0, "colours and leverage sit in the table, not beside it"
    heads = page.locator("table.runs-table thead th").all_text_contents()
    assert (heads[0], heads[-1]) == ("Run", "Leverage")
    cells = page.locator("table.runs-table tbody tr").first.locator("td")
    tinted = "(td) => td.classList.contains('setting') && td.style.getPropertyValue('--tint') !== ''"
    assert cells.nth(1).evaluate(tinted), "a coded setting is tinted in its own cell"
    assert re.fullmatch(r"0\.\d\d", cells.last.text_content()), "the first run's leverage, beside its bar"
    assert "how much the model leans on a run" in page.text_content("#detail-body")
    projectivity = next(line for line in page.locator("#detail-body p").all_text_contents() if "two-level" in line)
    assert projectivity == "Its 8 two-level runs, of 17, hold every combination of its 3 factors' levels."


def test_a_design_s_detail_says_what_it_withstands_and_where_else_it_is(context, site):
    """The pilot plant is the library's 2^3 factorial: each lists the other as the same design."""
    page = open_catalog(context, site)
    badge = page.locator(".result", has=page.locator(f".result__title[data-id='{PILOT}']")).locator(".badge")
    assert "same design in 1 more place" in badge.all_text_contents()
    page.click(f".result__title[data-id='{PILOT}']")
    page.wait_for_selector("#detail[open]")
    text = " ".join(page.text_content("#detail-body").split())
    assert "If any one run is lost, the design still estimates two-factor interactions" in text
    assert "Its two-level runs hold every combination of its 3 factors' levels." in text
    assert page.text_content("#detail-body .detail__leverage").startswith("Every run has the same leverage, 0.88:")
    heads = page.locator("table.runs-table thead th").all_text_contents()
    assert "Leverage" not in heads, "a column of one value would add nothing to the sentence"
    page.click(f"#detail-body .detail__twins [data-id='{CUBE}']")
    page.wait_for_function(
        "document.querySelector('#detail-title').textContent.startsWith('Full factorial, 3 factors')"
    )
    assert "d=lib-full-factorial-k3" in page.url


def test_the_robustness_filters_keep_the_designs_that_withstand_more(context, site):
    page = open_catalog(context, site)
    page.select_option("#projection", "3")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "qp=3" in page.url
    page.select_option("#projection", "0")
    page.select_option("#projectivity", "3")
    counted(page, "3 of 5")
    assert set(shown(page)) == {PILOT, CUBE, CCD}
    survives = page.locator("label", has=page.locator("[data-switch=survives]"))
    assert survives.locator(".count").text_content() == "3", "each of the three survives losing a run"


def test_a_design_the_page_cannot_take_says_why_it_does_not_open(context, site, catalog_files):
    """The comparison's detail, as the build writes a design it cannot open: no campaign, and the reason."""
    index = json.loads((catalog_files / "index.json").read_text())
    name = next(d["detail"] for d in index["designs"] if d["id"] == COMPARISON)
    shard = json.loads((catalog_files / "detail" / f"{name}.json").read_text())
    closed = {"campaign": None, "opens_as": None, "closed": "the source prints no complete table of runs"}
    shard["designs"][COMPARISON] |= closed
    context.route(f"{site}catalog/detail/{name}.json", lambda route: route.fulfill(json=shard))
    page = open_catalog(context, site, f"#d={COMPARISON}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial does not open this design: the source prints no complete table of runs." in page.text_content(
        "#detail-body"
    )
    assert page.locator(f"#detail-body [data-open='{COMPARISON}']").count() == 0


def test_a_comparison_of_two_methods_opens_in_factorial_with_its_results(context, site):
    """Box, Hunter and Hunter's ten batches by method A, then ten by method B: one factor, its runs as printed."""
    page = open_catalog(context, site, f"#d={COMPARISON}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial opens the source's runs as it prints them, in their order" in page.text_content("#detail-body")
    page.click(f"#detail-body [data-open='{COMPARISON}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    column = headers(page).index("method")
    assert [run[column] for run in run_sheet(page)] == ["A"] * 10 + ["B"] * 10
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analysed"


def test_a_design_the_page_does_not_make_opens_with_its_runs_as_given(context, site):
    """The page makes no OMARS design: Factorial opens the library's runs as they are, and lists them in Design."""
    page = open_catalog(context, site, f"#d={OMARS}")
    page.wait_for_selector("#detail[open]")
    assert "Factorial opens these runs as the library made them, in their order" in page.text_content("#detail-body")
    page.click(f"#detail-body [data-open='{OMARS}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 9
    assert page.input_value("#design-type") == "given"
    shown = page.eval_on_selector("#design-type option[value=given]", "o => !o.hidden")
    assert shown, "the list offers runs as given while they are open"


def test_a_blocked_design_opens_in_factorial_in_its_blocks(blocked_context, site):
    """Montgomery's filtration 2^4 in two blocks of eight: each run's block in the table, then fitted in Learn."""
    page = open_catalog(blocked_context, site, f"#d={BLOCKED_DESIGN}")
    page.wait_for_selector("#detail[open] table.runs-table")
    heads = page.locator("table.runs-table thead th").all_text_contents()
    assert heads[:6] == ["Run", "Block", "A", "B", "C", "D"], "the block is each run's, not a factor"
    assert "Block is the block each run is made in" in page.text_content("#detail-body")
    page.click(f"#detail-body [data-open='{BLOCKED_DESIGN}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    column = headers(page).index("Block")
    assert sorted({run[column] for run in run_sheet(page)}) == ["1", "2"]
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analysed"
    anova = page.locator("#analysis table").last.locator("tbody tr td:first-child").all_text_contents()
    assert "Block" in anova
    assert "The blocks are fitted as a fixed effect" in page.text_content("#analysis")


def test_a_comparison_in_pairs_opens_and_learn_compares_within_each_pair(blocked_context, site):
    """Box, Hunter and Hunter's boys' shoes, ten pairs: Learn tests the difference within the pairs, three ways."""
    page = open_catalog(blocked_context, site, f"#d={PAIRED_DESIGN}")
    page.wait_for_selector("#detail[open]")
    page.click(f"#detail-body [data-open='{PAIRED_DESIGN}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "learn"
    text = " ".join(page.text_content("#analysis").split())
    assert "Within the 10 pairs, y is on average 0.41 higher at B than at A." in text
    assert "within each of the 10 pairs, tested three ways" in text
    assert "Paired t-test: the difference within each pair" in text
    assert page.locator("#analysis [data-term='signed-rank']").count() == 1


def test_a_published_design_opens_in_factorial_with_the_source_s_results(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-open='{PILOT}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 8
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analysed"


def test_a_generated_design_opens_in_factorial_ready_to_run(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-open='{CCD}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 17


def test_the_map_draws_a_mark_for_every_design_listed(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#map .scatterlayer")
    marks = page.evaluate(
        "document.querySelector('#map').data.filter((t) => t.customdata).flatMap((t) => t.customdata).sort()"
    )
    assert marks == sorted(CATALOG_SAMPLE)


def _drag_a_box(page, factors: tuple[float, float], runs: tuple[float, float]) -> None:
    """Drag a box on the map over a span of factors and runs, as a reader does with the mouse."""
    corners = page.evaluate(
        """([factors, runs]) => {
          const div = document.querySelector('#map');
          const { xaxis, yaxis } = div._fullLayout;
          const area = div.querySelector('.nsewdrag').getBoundingClientRect();
          const at = (k, n) => [area.left + xaxis.l2p(k), area.top + yaxis.l2p(Math.log10(n))];
          return [at(factors[0], runs[0]), at(factors[1], runs[1])];
        }""",
        [list(factors), list(runs)],
    )
    page.mouse.move(*corners[0])
    page.mouse.down()
    page.mouse.move(*corners[1], steps=8)
    page.mouse.up()


BOXES = "document.querySelector('#map')._fullLayout.selections?.length ?? 0"


def test_a_box_on_the_map_filters_and_the_reset_puts_the_map_back(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#map .scatterlayer")
    _drag_a_box(page, (2.5, 3.5), (4, 64))
    counted(page, "3 of 5")
    assert set(shown(page)) == {PILOT, CCD, CUBE}, "the designs of 3 factors"
    page.wait_for_timeout(500)
    assert page.text_content("#count").startswith("3 of 5"), "the map's redraw keeps the filter"
    page.click("#map-reset")
    counted(page, "5 of 5")
    assert "runs=" not in page.url, "the address loses the box's filter"
    assert "k=" not in page.url


def test_a_box_around_no_design_stays_drawn_until_the_map_is_reset(context, site):
    """A box that catches no design filters nothing, and Plotly leaves it on the map: the reset takes it away."""
    page = open_catalog(context, site)
    page.wait_for_selector("#map .scatterlayer")
    _drag_a_box(page, (5.5, 6.5), (100, 400))
    page.wait_for_function(f"({BOXES}) === 1")
    assert page.text_content("#count").startswith("5 of 5")
    page.click("#map-reset")
    page.wait_for_function(f"({BOXES}) === 0")
    assert page.text_content("#count").startswith("5 of 5")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_it_reads_on_a_phone_without_scrolling_sideways(context, site):
    page = open_catalog(context, site)
    wider = page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('.js-plotly-plot'))
             .map((e) => e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''))"""
    )
    assert wider == []


# --------------------------------------------------------------------------- phase 3: the views over the list
def row(page, label: str) -> list[str]:
    """Return a row of the comparison: each design's value of the measure with this label."""
    cells = page.locator("table.compare tr", has=page.locator("th", has_text=label)).locator("td")
    return cells.all_text_contents()[1:]


def toast_text(page) -> str:
    page.wait_for_selector("#toast.shown")
    return page.text_content("#toast")


def test_two_designs_of_one_size_are_compared_on_the_model_both_fit(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-compare='{CCD}']")
    page.wait_for_selector("#tray:not([hidden])")
    assert page.is_disabled("#tray-open"), "one design is nothing to compare"
    page.click(f"[data-compare='{CUBE}']")
    assert page.text_content("#tray-text") == "2 designs of 3 factors"
    page.click("#tray-open")
    page.wait_for_selector("#compare[open] table.compare")
    assert page.input_value("#compare-model") == "interactions", "the 2^3 cannot fit the quadratic"
    heads = page.eval_on_selector_all("table.compare thead th", "cells => cells.map((c) => c.textContent)")
    assert heads[2:] == ["Central composite, 3 factors, face-centered [n=17]", "Full factorial, 3 factors [n=8]"]
    assert row(page, "Runs") == ["17", "8"]
    assert "cmp=" in page.url
    assert "view=compare" in page.url
    page.wait_for_selector("#compare .compare__fds .scatterlayer")
    page.select_option("#compare-model", "quadratic")
    assert row(page, "Fits the model?") == ["yes", "no (rank 7)"], "the squares of a 2^3 are its intercept"


def test_a_design_of_another_size_does_not_join_the_comparison(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-compare='{CCD}']")
    page.click(f"[data-compare='{OMARS}']")
    assert "the designs compared have 3 factors, and this one has 4" in toast_text(page)
    assert page.locator("#tray-list li").count() == 1


def test_a_design_s_projections_say_what_any_two_or_three_factors_estimate(context, site):
    page = open_catalog(context, site, f"#d={CCD}")
    page.wait_for_selector("#detail[open] .explorer")
    text = page.locator(".explorer__text")
    assert text.text_content() == (
        "In A, B and C, the runs sit at 15 distinct settings and estimate the full quadratic in them (10 terms). "
        "They hold all 8 corners: a full 2^3 factorial in these factors. "
        "A label such as 3\u00d7 counts the runs at a point; a point with no label holds one run."
    )
    assert page.locator(".projection__point").count() == 15
    assert page.locator(".projection__count").all_text_contents() == ["3\u00d7"], "the centre's runs, as a count"
    page.click(".explorer__picks [data-column='2']")
    assert text.text_content().startswith(
        "In A and B, the runs sit at 9 distinct settings and estimate the full quadratic"
    )
    assert "They hold all 4 corners" in text.text_content()
    assert "The largest |r| between two second-order terms is 0.51." in page.text_content("#detail-body")


def test_a_design_leads_to_the_families_it_can_be_augmented_to(context, site):
    page = open_catalog(context, site, f"#d={CUBE}")
    page.wait_for_selector("#detail[open] .detail__steps")
    page.click(".detail__steps a:text('Central composite')")
    page.wait_for_selector("#detail:not([open])", state="attached")
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "fam=ccd" in page.url


def test_the_errata_list_every_printed_number_that_does_not_reproduce(context, site):
    page = open_catalog(context, site)
    page.click("#errata-open")
    page.wait_for_selector("#errata[open] .errata__table")
    total = page.text_content("#errata-count")
    assert total.startswith(total.split(" of ")[1].split()[0]), "every printed number, before a filter"
    page.select_option("#errata-source", "box-hunter-hunter-1978")
    page.fill("#errata-words", "already rounded 3.65")
    page.wait_for_function("document.querySelector('#errata-count').textContent.startsWith('1 of')")
    assert "view=errata" in page.url
    page.click(f"#errata [data-id='{COMPARISON}']")
    page.wait_for_selector("#detail[open]")
    assert page.text_content("#detail-title").startswith("An industrial experiment")


def test_the_gaps_are_the_generated_designs_no_source_prints(context, site):
    page = open_catalog(context, site)
    page.click(".try:text('gaps')")
    counted(page, "2 of 5")
    assert set(shown(page)) == {CCD, OMARS}, "the library's 2^3 is Box, Hunter and Hunter's pilot plant"
    page.fill("#query", "")
    page.select_option("#source", "box-hunter-hunter-1978")
    counted(page, "2 of 5")
    assert set(shown(page)) == {PILOT, COMPARISON}
    assert "src=box-hunter-hunter-1978" in page.url


def test_a_bar_of_the_timeline_keeps_its_years(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#year-bars .bin")
    page.locator("#year-bars .bin[data-from='1975']").click()
    counted(page, "2 of 5")
    assert (page.input_value("#year-min"), page.input_value("#year-max")) == ("1975", "1979")
    assert "year=1975-1979" in page.url


# --------------------------------------------------------------------------- phase 4: limits on the settings
def test_a_limit_keeps_the_designs_whose_every_run_meets_it(context, site):
    """A + B <= 2 holds for every coded run of a cube; the comparison has one factor, so the limit names no B of its."""
    page = open_catalog(context, site)
    page.fill("#limit", "A + B <= 2")
    counted(page, "4 of 5")
    assert COMPARISON not in shown(page)
    page.fill("#limit", "A + B <= 1")
    counted(page, "0 of 5")
    page.check("#limit-all")
    counted(page, "4 of 5")
    badge = page.locator(".result", has=page.locator(f".result__title[data-id='{CCD}']")).locator(".badge--warn")
    assert badge.text_content() == "2 of 17 runs outside the limit"
    assert "lim=A+%2B+B+%3C%3D+1" in page.url
    page.fill("#limit", "A + B")
    assert page.text_content("#limit-error").startswith("The limit cannot be read: write <= for at most")
    counted(page, "5 of 5")


def test_a_design_s_runs_outside_the_limit_are_marked_and_it_opens_with_the_limit(context, site):
    page = open_catalog(context, site, f"#lim=A+%2B+B+%3C%3D+1&lima=1&d={CCD}")
    page.wait_for_selector("#detail[open] .detail__warn")
    line = page.text_content("#detail-body .detail__warn")
    assert line.endswith("are outside the limit A + B <= 1, of its 17 runs.")
    assert page.locator("#detail-body tr.row--outside").count() == 2
    page.click(f"[data-open-limited='{CCD}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 17
    assert page.locator("#limit-list .limit [name=expression]").first.input_value() == "A + B <= 1", (
        "a CCD from -1 to +1: the limit in its own units is the limit in coded units"
    )
    assert "outside your limits on the settings: choose the D-optimal or I-optimal design" in page.text_content(
        "#design-summary"
    )


def test_a_design_opened_with_a_limit_waits_for_the_other_designs_runs_to_be_judged(context, site):
    # The central composite design is a generated one: its file arrives at once, the published sources' late, so
    # the detail is ready before every design's runs are judged against the limit.
    context.add_init_script(
        """const fetchNow = window.fetch;
        window.fetch = (url, ...rest) => /catalog\\/detail\\/(?!generated\\.json)/.test(String(url))
          ? new Promise((done) => setTimeout(done, 1000)).then(() => fetchNow(url, ...rest))
          : fetchNow(url, ...rest);"""
    )
    page = open_catalog(context, site, f"#lim=A+%2B+B+%3C%3D+1&lima=1&d={CCD}")
    page.wait_for_selector("#detail[open] .detail__warn")
    assert page.text_content("#detail-body .detail__warn").endswith("are outside the limit A + B <= 1, of its 17 runs.")


def test_a_link_that_names_a_design_in_its_query_opens_it(context, site):
    """The book links to a design as catalog/?d=<id>: its theme reads a fragment as a CSS selector."""
    page = context.new_page()
    page.goto(f"{site}catalog/?d={CCD}")
    page.wait_for_selector("#detail[open]")
    assert page.text_content("#detail-title").startswith("Central composite, 3 factors")
    assert page.url == f"{site}catalog/#d={CCD}", "the address keeps it in the fragment, which no server sees"
