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

from tests.web._support import BLOCKED_DESIGN, BOOK_DESIGN, CATALOG_SAMPLE, PAIRED_DESIGN, source
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


def unfolded(page, selector: str) -> object:
    """Return what ``selector`` finds in the list, with its line's fold open if it is under one.

    The list has a line for each design. Where several entries hold the same runs, as the pilot plant and the
    library's own 2^3 factorial do, they are under the line's fold, "printed in 1 place".
    """
    found = page.locator(f"#results {selector}")
    if not found.is_visible():
        page.locator("#results .result", has=page.locator(selector)).locator(".result__fold").click()
    return found


def press_title(page, design: str) -> None:
    """Press an entry's title in the list, which opens its details."""
    unfolded(page, f".result__title[data-id='{design}']").click()


def press_open(page, design: str) -> None:
    """Press the button that opens an entry of the list in Factorial."""
    unfolded(page, f"[data-open='{design}']").click()


def test_a_goal_writes_a_search_and_the_ways_in_by_book_and_by_field_are_at_the_foot(context, site):
    """A goal fills the search box, for the reader to edit; the lists of books and of fields are a link each.

    The page opened on a question about the reader's field and three doors, over the search: some twenty things to
    press before the first design. The top is now the search and the goals, and the table follows.
    """
    page = open_catalog(context, site)
    assert page.locator("#field-form, .door").count() == 0
    first = page.locator("main > :is(form, nav, div, h2, p):not(.about__back, .site-top, .lede)").first
    assert first.get_attribute("class") == "catalog__search", (
        "the search is the first thing under the page's first line"
    )
    assert page.text_content("#goals-title") == "Try:"
    page.click("text=Blend ingredients")
    assert page.input_value("#query") == "mixture"
    counted(page, "0 of 5")
    foot = page.locator(".catalog__views a, .catalog__views button").all_text_contents()
    assert [" ".join(words.split()) for words in foot] == [
        "Browse by book",
        "Browse by field",
        "Errata: every printed number that does not reproduce",
        "Propose a design for the catalog",
        "Share a case study of your own",
    ]
    assert page.evaluate(
        "document.querySelector('.catalog__views').compareDocumentPosition(document.querySelector('#results')) & 2"
    ), "after the list"
    # The build writes the list of fields (tests/web/page/test_field_pages.py): this site stands in for it.
    page.route("**/catalog/fields/**", lambda route: route.fulfill(body="<h1>Fields</h1>", content_type="text/html"))
    page.click("text=Browse by field")
    page.wait_for_url("**/catalog/fields/")


def test_where_a_design_is_from_is_a_filter_that_says_what_the_two_kinds_are(context, site):
    """Published, Generated and Both stood beside the search with no word on what they meant, or why to choose."""
    page = open_catalog(context, site)
    box = page.locator("#facets fieldset", has=page.locator("#collection"))
    assert box.locator("legend").text_content() == "Where it is from"
    assert page.locator("#collection [role=radio]").all_text_contents() == [
        "Books and papers",
        "Made by Factorial",
        "Both",
    ]
    said = " ".join(box.locator(".hint").text_content().split())
    assert said.startswith("A design from a book or paper comes with the experiment it was run in")
    assert "Factorial makes the others itself" in said


def test_a_family_s_row_has_one_number_the_designs_it_holds(context, site):
    """A row had the year the family was first published and its count: two numbers on some rows, one on others."""
    page = open_catalog(context, site)
    rows = page.locator("#families label").evaluate_all(
        """(rows) => rows.map((row) => [...row.querySelectorAll('span')]
             .map((part) => part.textContent).filter((text) => /^\\d+$/.test(text)))"""
    )
    assert rows, "the families are listed"
    assert all(len(numbers) == 1 for numbers in rows), rows


def test_the_app_s_bar_opens_the_catalog(fresh_page):
    fresh_page.click(".studybar__link[href='catalog/']")
    fresh_page.wait_for_url("**/catalog/")
    fresh_page.wait_for_selector(".result")
    assert fresh_page.text_content("#count") == "5 designs, 4 different", "two of them are the same runs"
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
    press_title(page, PILOT)
    page.wait_for_selector("#detail[open]")
    cells = page.locator("#detail-body .detail__head .detail__cell h3").all_text_contents()
    assert cells == ["This design", "Where it comes from", "Use it"]
    facts = page.locator("#detail-body .detail__facts li").all_text_contents()
    assert facts == ["3 factors (1 categorical)", "8 runs", "Estimates two-factor interactions"]
    text = " ".join(page.text_content("#detail-body").split())
    assert "Box, G. E. P., Hunter, W. G. and Hunter, J. S. (1978)" in text
    assert "recomputed" not in text, "how the catalog checks an entry is not the reader's concern"
    assert "The library makes this design" not in text
    family = page.locator("#detail-body details.detail__family")
    assert family.locator("summary").text_content() == "About this design type: Full factorial"
    assert family.get_attribute("open") is None
    assert "Full factorial designs were first published by Fisher, R. A. (1926)" in family.text_content()
    assert page.get_attribute("#detail-body .detail__cite a", "target") == "_blank", "the source opens in a new tab"
    stray = "a => [...a.childNodes].filter((n) => n.nodeType === Node.TEXT_NODE).map((n) => n.textContent).join('')"
    assert page.eval_on_selector("#detail-body .actions", stray).strip() == "", "no text between the buttons"
    assert "d=bhh-pilot-plant-2-3" in page.url
    on_the_map = page.locator("#detail-body .detail__map a")
    assert on_the_map.text_content() == "Full factorial on the concept map"
    assert on_the_map.get_attribute("href").endswith("map/#full-factorial")
    page.keyboard.press("Escape")
    page.wait_for_selector("#detail:not([open])", state="attached")
    # The dialog's close event, which takes the design out of the address, fires after its open attribute goes.
    page.wait_for_function("!location.hash.includes('d=')")


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
    # The dialog's close event, which takes the design out of the address, fires after its open attribute goes.
    page.wait_for_function("!location.hash.includes('d=')")


def test_a_book_design_s_detail_links_to_its_section_and_names_its_figures(book_context, site):
    """The place links to the book's section, at its anchor; the figures are named, without their files."""
    page = open_catalog(book_context, site, f"#d={BOOK_DESIGN}")
    page.wait_for_selector("#detail[open] .detail__where a")
    place = page.locator("#detail-body .detail__where a")
    assert place.get_attribute("href") == (
        "https://learnche.org/pid/design-analysis-experiments/omars-worked-study#doe-omars-worked-study"
    )
    assert (place.get_attribute("target"), place.get_attribute("rel")) == ("pid-book", "opener"), (
        "the book's one tab, which the app's other links into the book go to as well"
    )
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
    index = json.loads((catalog_files / "index.json").read_text(encoding="utf-8"))
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


def test_a_catalog_of_few_applications_lists_them_and_nothing_in_place_of_the_rest(context, site):
    """With no more applications than the list shows there is no fold of the rest, and no stray word where it goes."""
    page = open_catalog(context, site)
    assert page.locator("#topics-more").count() == 0
    assert "null" not in page.text_content("#topics")


def test_a_word_of_an_application_s_name_finds_its_designs(book_context, site):
    page = open_catalog(book_context, site)
    page.fill("#query", "fermentation")
    counted(page, "1 of 6")
    assert shown(page) == [BOOK_DESIGN]


def test_a_design_whose_runs_differ_in_leverage_lists_it_beside_each_run(context, site):
    """The face-centered CCD leans on its corners more than on its center: a leverage column, and the heaviest named."""
    page = open_catalog(context, site, f"#d={CCD}")
    page.wait_for_selector("#detail[open] table.runs-table")
    words = page.text_content("#detail-body .detail__leverage")
    assert re.match(r"\d+ runs carry the most weight, with leverage 0\.\d\d; the least is 0\.\d\d\. ", words), words
    assert page.locator("#detail-body canvas").count() == 0, "colors and leverage sit in the table, not beside it"
    heads = page.locator("table.runs-table thead th").all_text_contents()
    assert (heads[0], heads[-1]) == ("Run", "Leverage")
    cells = page.locator("table.runs-table tbody tr").first.locator("td")
    tinted = "(td) => td.classList.contains('setting') && td.style.getPropertyValue('--tint') !== ''"
    assert cells.nth(1).evaluate(tinted), "a coded setting is tinted in its own cell"
    assert re.fullmatch(r"0\.\d\d", cells.last.text_content()), "the first run's leverage, beside its bar"
    assert "how much the model leans on a run" in page.text_content("#detail-body")
    projectivity = next(line for line in page.locator("#detail-body p").all_text_contents() if "two-level" in line)
    assert projectivity == "Its 8 two-level runs, of 17, hold every combination of its 3 factors' levels."


def test_the_list_has_a_short_line_for_each_design_with_its_picture_upright_and_a_key(context, site):
    """A card a design held three or four designs to a screen, under a wide picture nothing explained.

    A line names the design by its family and size, marks what it estimates in its cell's color, and ends in a
    small button. Its picture has a run a row, so it is narrow, and a key over the list says what it shows.
    """
    page = open_catalog(context, site)
    lines = page.locator("#results .result")
    assert lines.count() == 4, "five entries, two of them the same runs"
    key = " ".join(page.text_content(".results__key").split())
    assert key == "A picture is the design: a row is a run, a column a factor. low middle high"
    swatches = page.locator(".results__key i").evaluate_all("(marks) => marks.map((mark) => mark.style.background)")
    assert len(set(swatches)) == 3, "the pictures' three colors"
    assert all(swatches), swatches
    heights = lines.evaluate_all("(rows) => rows.map((row) => row.getBoundingClientRect().height)")
    assert max(heights) < 80, f"two rows of text a line: {heights}"
    composite = page.locator("#results .result", has=page.locator(f".result__title[data-id='{CCD}']"))
    picture = composite.locator("canvas.glyph").bounding_box()
    assert picture["height"] > picture["width"], "17 runs down, 3 factors across"
    assert composite.locator(".result__size").text_content() == "Central composite · 3 factors · 17 runs"
    mark = composite.locator(".result__step")
    assert mark.text_content() == "+ quadratic", "the words of the table's key"
    colour = mark.locator("i").evaluate("(dot) => getComputedStyle(dot).backgroundColor")
    assert colour == "rgb(253, 180, 182)", "and the color of its cell"
    button = composite.locator(".result__actions [data-open]")
    assert button.text_content() == "Open"
    assert button.get_attribute("aria-label") == "Open in Factorial"
    assert button.bounding_box()["width"] < 90, "a small button, at the line's end"
    assert composite.locator(".result__source").text_content() == "Made by Factorial"


def test_a_design_in_several_entries_folds_them_under_its_line(context, site):
    """The pilot plant and the library's own 2^3 factorial are the same runs: one line, with both under its fold."""
    page = open_catalog(context, site)
    line = page.locator("#results .result", has=page.locator(f".result__title[data-id='{PILOT}']"))
    fold = line.locator(".result__fold")
    assert fold.text_content() == "printed in 1 place"
    assert fold.get_attribute("aria-expanded") == "false"
    assert line.locator(".result__uses").is_hidden()
    assert line.locator(".result__actions [data-open]").get_attribute("data-open") == CUBE, (
        "the design alone opens first"
    )
    fold.click()
    assert fold.get_attribute("aria-expanded") == "true"
    entries = line.locator(".result__uses li", has=page.locator(".result__title"))
    assert entries.locator(".result__title").evaluate_all("(titles) => titles.map((t) => t.dataset.id)") == [
        PILOT,
        CUBE,
    ]
    assert entries.locator(".result__source").all_text_contents() == [
        "Box, Hunter and Hunter (1978)",
        "Made by Factorial",
    ]
    assert entries.locator("[data-open]").all_text_contents() == ["Open", "Open"], "each opens in Factorial"
    # The list is drawn again when its order changes: the line stays open.
    page.select_option("#sort", "runs")
    fold = page.locator("#results .result", has=page.locator(f".result__title[data-id='{PILOT}']")).locator(
        ".result__fold"
    )
    assert fold.get_attribute("aria-expanded") == "true"
    fold.click()
    assert page.locator(f"#results .result__title[data-id='{PILOT}']").is_hidden()


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_a_line_of_the_list_keeps_inside_the_screen(context, site):
    page = open_catalog(context, site)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    lines = page.locator("#results .result")
    heights = lines.evaluate_all("(rows) => rows.map((row) => row.getBoundingClientRect().height)")
    assert max(heights) < 150, f"a line is a few rows of text: {heights}"
    for box in lines.locator(".result__actions").evaluate_all(
        "(all) => all.map((one) => one.getBoundingClientRect().right)"
    ):
        assert box <= 390


def test_a_design_s_detail_says_what_it_withstands_and_where_else_it_is(context, site):
    """The pilot plant is the library's 2^3 factorial: each lists the other as the same design."""
    page = open_catalog(context, site)
    # The two are one line of the list, named for the design, with both entries under its fold.
    line = page.locator(".result", has=page.locator(f".result__title[data-id='{PILOT}']"))
    assert line.locator(f".result__title[data-id='{CUBE}']").count() == 1
    assert line.locator(".result__size").text_content() == "Full factorial · 3 factors · 8 runs"
    assert line.locator(".result__fold").text_content() == "printed in 1 place"
    press_title(page, PILOT)
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


def test_a_design_the_library_makes_says_what_its_kind_was_checked_against(context, site):
    """The library's 2^3 factorial names the sources that print one and the packages that build it the same way."""
    checked = json.loads(source("checked.json"))
    found = checked["designs"]["full_factorial"]["factors"]["3"]
    page = open_catalog(context, site)
    badges = page.locator(".result", has=page.locator(f".result__title[data-id='{CUBE}']")).locator(".badge")
    assert f"checked: {len(found['sources'])} sources, {len(found['packages'])} packages" in badges.all_text_contents()
    omars = page.locator(".result", has=page.locator(f".result__title[data-id='{OMARS}']"))
    assert omars.locator(".badge--checked").count() == 0, "no other package builds an OMARS design: it has no note"
    press_title(page, CUBE)
    page.wait_for_selector("#detail[open]")
    note = page.locator("#detail-body .detail__checked")
    assert note.locator("h3").text_content() == "Checked"
    lists = note.locator(".detail__against > div")
    assert lists.locator("h4").all_text_contents() == ["In print", "In other software"]
    in_print = lists.nth(0).locator("li").all_text_contents()
    assert len(in_print) == 5, "four sources by name, then how many more"
    assert in_print[-1] == f"and {len(found['sources']) - 4} more"
    software = lists.nth(1).locator("li").all_text_contents()
    assert software == [f"{name} ({checked['packages'][name]})" for name in found["packages"]]
    link = note.locator("a")
    assert link.text_content() == "How Factorial is checked"
    assert link.get_attribute("href") == f"{site}validation.html"
    # No source prints the face-centered composite design: it counts and lists the packages alone.
    page.keyboard.press("Escape")
    composite = checked["designs"]["ccd"]["factors"]["3"]["packages"]
    badges = page.locator(".result", has=page.locator(f".result__title[data-id='{CCD}']")).locator(".badge")
    assert {"no source prints it", f"checked: {len(composite)} package"} <= set(badges.all_text_contents())
    press_title(page, CCD)
    page.wait_for_function("document.querySelector('#detail-title').textContent.includes('composite')")
    assert page.locator("#detail-body .detail__against h4").all_text_contents() == ["In other software"]
    assert "as the books print it" not in page.text_content("#detail-body .detail__checked")
    page.keyboard.press("Escape")
    press_title(page, OMARS)
    page.wait_for_function("document.querySelector('#detail-title').textContent.includes('OMARS')")
    assert page.locator("#detail-body .detail__checked").count() == 0


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
    index = json.loads((catalog_files / "index.json").read_text(encoding="utf-8"))
    name = next(d["detail"] for d in index["designs"] if d["id"] == COMPARISON)
    shard = json.loads((catalog_files / "detail" / f"{name}.json").read_text(encoding="utf-8"))
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
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analyzed"


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
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analyzed"
    anova = page.locator("#statistician > .scroll table").last.locator("tbody tr td:first-child").all_text_contents()
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
    press_open(page, PILOT)
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 8
    assert page.get_attribute("#workspace", "data-phase") == "learn", "the source's results, analyzed"
    title = page.text_content("#study-title")
    assert title != "y", "a response known only by its column name does not name the study"
    assert page.text_content("#campaign-banner").startswith(f"Opened from the catalog: {title}, ")


def test_a_published_study_names_its_source_and_learn_shows_what_it_prints_beside_its_own(context, site):
    """The pilot plant's banner cites Box, Hunter and Hunter; Learn offers their model, then lists their effects."""
    page = open_catalog(context, site)
    press_open(page, PILOT)
    page.wait_for_selector("#campaign-banner", state="visible")
    banner = page.text_content("#campaign-banner")
    assert banner.startswith("Opened from the catalog: ")
    assert "Box" in banner
    assert page.locator("#campaign-banner a", has_text="Read the source").count() == 1
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    page.wait_for_selector("#printed:not([hidden]) .printed-model")
    assert "prints no number for the model fitted here" in page.text_content("#printed")
    offer = page.locator("#printed .printed-model").first
    assert "prints 21 numbers for the model T + C + T·C + K + T·K + C·K + T·C·K" in offer.text_content()
    offer.locator("button").click()
    page.wait_for_selector("#printed:not([hidden]) table")
    lead = page.text_content("#printed .lead")
    assert lead.startswith("Factorial's analysis gives 21 of the 21 numbers the source prints for this model")
    first = page.locator("#printed tbody tr").first.locator("td").all_text_contents()
    assert first == ["Effect, T", "p. 309", "23", "23.0", "agrees"]
    assert page.locator("#term-list input:checked").count() == 7, "their model is the one ticked now"


def test_a_generated_design_opens_in_factorial_ready_to_run(context, site):
    page = open_catalog(context, site)
    page.click(f"[data-open='{CCD}']")
    page.wait_for_selector("#campaign-banner", state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    assert len(run_sheet(page)) == 17


#: What each cell of the map says of itself: its factors, its band of runs, how many designs, what most estimate.
CELLS = "(cells) => cells.map((cell) => cell.getAttribute('aria-label'))"


def open_map(context, site: str) -> object:
    """Open the catalog, and wait until its map is drawn."""
    page = open_catalog(context, site)
    page.wait_for_selector("#map .sizemap__cell")
    return page


def cell(row: int, band: int) -> str:
    """Return the selector of the map's cell in a row of factors and a band of runs, both counted as the map does."""
    return f"#map .sizemap__cell[data-row='{row}'][data-band='{band}']"


def test_the_map_counts_every_design_in_the_cell_of_its_factors_and_runs(context, site):
    page = open_map(context, site)
    assert page.locator("#map .sizemap__cell").evaluate_all(CELLS) == [
        "1 factor, 17 to 24 runs: 1 design. It estimates the main effects. Press to list it.",
        (
            "3 factors, 5 to 8 runs: 2 designs. At most, they estimate the main effects and the two-factor "
            "interactions. Press to list them."
        ),
        "3 factors, 17 to 24 runs: 1 design. It estimates the full quadratic. Press to list it.",
        "4 factors, 9 to 12 runs: 1 design. It estimates the main effects. Press to list it.",
    ]
    assert page.locator("#map .sizemap__count").all_text_contents() == ["1", "2", "1", "1"]
    assert page.locator("#map [data-unfold]").count() == 13, "a row for 1 to 12 factors, and one for more"
    lines = page.locator("#map path.sizemap__line:not(.sizemap__line--under)")
    assert lines.count() == 2, "the fewest runs for the main effects, and for the full quadratic"
    assert page.locator("#map .sizemap__name").all_text_contents() == ["main effects", "full quadratic"]


def test_a_cell_of_the_map_lists_its_designs_and_a_second_press_or_the_reset_lists_them_all(context, site):
    page = open_map(context, site)
    page.click(cell(3, 1))
    counted(page, "2 of 5")
    assert set(shown(page)) == {PILOT, CUBE}, "the designs of 3 factors in 5 to 8 runs"
    assert "runs=5-8" in page.url, "the address keeps the cell"
    assert "k=3-3" in page.url
    assert [page.input_value(f"#{box}") for box in ("factors-min", "factors-max", "runs-min", "runs-max")] == [
        "3",
        "3",
        "5",
        "8",
    ], "and the filters beside the list say it"
    assert page.locator("#map .sizemap__cell").count() == 4, "the other cells are still counted"
    assert page.locator("#map .sizemap__cell.out").count() == 3, "and pale: none of theirs is in the list"
    page.click(cell(3, 1))
    counted(page, "5 designs")
    assert page.locator("#map .sizemap__cell.out").count() == 0

    page.click(cell(4, 2))
    counted(page, "1 of 5")
    assert shown(page) == [OMARS]
    page.click("#map-reset")
    counted(page, "5 designs")
    assert "runs=" not in page.url, "the address loses the cell"
    assert "k=" not in page.url


def test_a_row_of_the_map_opens_into_its_families_and_a_mark_lists_its_size(context, site):
    page = open_map(context, site)
    row = page.locator("#map [data-unfold='3']")
    assert row.get_attribute("aria-expanded") == "false", "every row is shut at first"
    assert page.locator("#map .sizemap__mark").count() == 0
    row.click()
    assert page.locator("#map [data-unfold='3']").get_attribute("aria-expanded") == "true"
    assert page.locator("#map .sizemap__family").all_text_contents() == ["Full factorial · 2", "Central composite · 1"]
    marks = page.locator("#map .sizemap__mark").evaluate_all(CELLS)
    assert marks == [
        (
            "8 runs, Full factorial: 2 designs of 3 factors. At most, they estimate the main effects and the "
            "two-factor interactions. Press to list them."
        ),
        "17 runs, Central composite: 1 design of 3 factors. It estimates the full quadratic. Press to list it.",
    ]
    assert page.locator("#map .sizemap__fewest").all_text_contents() == ["4", "10"], "the fewest runs, by the lines"
    assert page.text_content("#count") == "5 designs, 4 different", "opening a row filters nothing"

    page.click("#map .sizemap__mark[data-runs='17']")
    counted(page, "1 of 5")
    assert shown(page) == [CCD], "the design under the mark"
    assert "runs=17-17" in page.url
    assert "fam=ccd" in page.url, "its family too: a mark is one family's designs of one size"
    assert page.locator("#map [data-unfold='3']").get_attribute("aria-expanded") == "true", "the row stays open"
    assert page.locator("#map .sizemap__family").count() == 2, "and keeps the lanes of its other families"
    assert page.locator("#map .sizemap__mark.out").count() == 1, "whose marks are pale"
    # A press beside a mark, in its lane, is one on the nearest: marks of neighboring sizes are a few pixels apart.
    page.locator("#map .sizemap__lane[data-row='3'][data-lane='0']").click(position={"x": 4, "y": 4})
    counted(page, "2 of 5")
    assert "runs=8-8" in page.url
    assert "fam=full_factorial" in page.url
    # A cell after a mark lists every family of the cell: the family went with the mark.
    page.click(cell(3, 4))
    counted(page, "1 of 5")
    assert "fam=" not in page.url

    page.click("#map [data-unfold='3']")
    assert page.locator("#map .sizemap__mark").count() == 0, "a second press shuts the row"
    page.click("#map [data-unfold='3']")
    page.click("#map-reset")
    counted(page, "5 designs")
    assert page.locator("#map [aria-expanded=true]").count() == 0, "the reset shuts every row"


def test_the_map_says_what_a_press_listed_and_leads_to_the_list_where_a_design_opens(context, site):
    """The list is a screen or more below the map: the map says what it kept there, with the way down to it."""
    page = open_map(context, site)
    assert page.is_hidden("#map-said"), "nothing kept, nothing said"
    page.click("#map [data-unfold='3']")
    page.click("#map .sizemap__mark[data-runs='8']")
    counted(page, "2 of 5")
    assert page.text_content("#map-said-text") == "2 listed: Full factorial, 3 factors, 8 runs."
    page.click("#map-list")
    # The list ends the page, so it rises as far as the page can scroll: into the window, clear of its foot.
    head = "document.querySelector('.catalog__listhead').getBoundingClientRect().top"
    page.wait_for_function(f"{head} < 0.6 * innerHeight")
    # The two are the same runs, so they are one line: its fold has the focus, and Enter opens it onto both.
    assert page.locator("#results .result").count() == 1
    fold = page.locator("#results .result__fold")
    assert fold.evaluate("(button) => button === document.activeElement"), "the first line listed has the focus"
    assert set(shown(page)) == {PILOT, CUBE}
    assert page.locator("#results .result__actions [data-open]").count() == 1, "the line's button opens the design"
    page.keyboard.press("Enter")
    assert fold.get_attribute("aria-expanded") == "true"
    assert page.locator("#results .result__uses [data-open]:visible").count() == 2, "each entry opens in Factorial too"
    page.locator(f"#results .result__title[data-id='{PILOT}']").press("Enter")
    page.wait_for_selector("#detail[open]")
    page.keyboard.press("Escape")
    page.click("#map-reset")
    counted(page, "5 designs")
    assert page.is_hidden("#map-said")
    assert "fam=" not in page.url, "the reset takes the mark's family with its size"


def test_a_list_of_one_family_has_a_card_that_says_what_the_family_is(context, site):
    """A reader who meets a family for the first time reads what it is, over its designs."""
    page = open_catalog(context, site)
    assert page.is_hidden("#family"), "every family is listed: there is no one family to describe"

    page = open_catalog(context, site, "#fam=full_factorial")
    card = page.locator("#family")
    assert card.locator("h2").text_content() == "Full factorial"
    assert card.locator(".famcard__first").text_content().startswith("First published: Fisher, R. A. (1926), ")
    assert card.locator("dt").all_text_contents() == ["What it is", "Use it when", "Watch for", "Where next"]
    assert card.locator("dd").first.text_content().startswith("A design that runs every combination of the levels")
    assert card.locator("dd").nth(1).locator("li").count() == 3, "when it is used: a line each"
    assert card.locator(".famcard__more a").get_attribute("href").endswith("map/#full-factorial")
    assert card.locator("#family-all").count() == 0, "no size is kept: the list is the whole family already"
    assert page.get_attribute("#family", "aria-labelledby") == "family-name"
    assert card.locator("h2").get_attribute("id") == "family-name"

    # "Where next" leads to a family this one can be augmented to, whose card takes this one's place.
    card.locator("dd a", has_text="Central composite").click()
    counted(page, "1 of 5")
    page.wait_for_function("document.querySelector('#family h2').textContent === 'Central composite'")
    assert card.locator("dt").all_text_contents() == ["What it is", "Use it when", "Watch for"], "it leads nowhere on"

    # Two families ticked: no card, as neither is the one the list is of.
    page.check("#families input[value='full_factorial']")
    counted(page, "3 of 5")
    assert page.is_hidden("#family")
    assert page.locator("#family").text_content() == ""


def test_a_family_no_one_work_first_published_has_a_card_in_its_own_words(context, site):
    """Nine families had no description. One with no first work and no place on the concept map says the rest."""
    page = open_catalog(context, site, "#fam=comparison")
    card = page.locator("#family")
    assert card.locator("h2").text_content() == "Comparison or regression"
    assert card.locator(".famcard__first").count() == 0
    assert "null" not in card.text_content(), "a part that is not there is left out, and not written as a word"
    assert card.locator("dt").all_text_contents() == ["What it is", "Use it when", "Watch for"]
    assert card.locator("dd").first.text_content().startswith("A study of one factor, or of measured data")
    assert card.locator(".famcard__more").is_hidden(), "nothing to press under it"


def test_a_family_s_name_in_an_open_row_lists_its_designs_of_the_row_under_its_card(context, site):
    page = open_map(context, site)
    page.click("#map [data-unfold='3']")
    name = page.locator("#map .sizemap__family", has_text="Central composite")
    assert name.get_attribute("role") == "button"
    assert name.get_attribute("aria-label") == (
        "Central composite: 1 design of 3 factors. Press to read what it is, and to list it."
    )
    name.click()
    counted(page, "1 of 5")
    assert shown(page) == [CCD]
    assert "fam=ccd" in page.url
    assert "k=3-3" in page.url, "the family's designs of this row"
    assert "runs=" not in page.url, "of every run size"
    assert page.text_content("#map-said-text") == "1 listed: Central composite, 3 factors."
    assert page.text_content("#family h2") == "Central composite"

    # The way down from the table is to the card, which is over the list.
    page.click("#map-list")
    page.wait_for_function("document.querySelector('#family').getBoundingClientRect().top < 0.6 * innerHeight")
    every = page.locator("#family-all")
    assert every.text_content() == "Show every Central composite design", "the row's size is kept: the way out of it"
    every.click()
    page.wait_for_function("!location.hash.includes('k=')")
    assert "fam=ccd" in page.url
    assert page.locator("#family-all").count() == 0
    assert page.text_content("#family h2") == "Central composite"

    # From the keyboard: down from the row's first cell is its first family's name, and Enter presses it.
    page.click("#map-reset")
    counted(page, "5 designs")
    assert page.is_hidden("#family"), "the reset takes the family with it"
    page.click("#map [data-unfold='3']")
    page.wait_for_function("document.getAnimations().length === 0")
    page.focus(cell(3, 1))
    page.keyboard.press("ArrowDown")
    focused = "JSON.stringify({ ...document.activeElement.dataset })"
    assert json.loads(page.evaluate(focused)) == {"row": "3", "family": "full_factorial", "all": "1"}
    page.keyboard.press("Enter")
    counted(page, "2 of 5")
    assert json.loads(page.evaluate(focused)) == {"row": "3", "family": "full_factorial", "all": "1"}, (
        "the name keeps the focus when the table is redrawn"
    )
    assert page.text_content("#family h2") == "Full factorial"
    page.keyboard.press("Enter")
    counted(page, "5 designs")
    assert page.is_hidden("#family"), "a second press on the name shows every design again"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_a_family_s_card_keeps_inside_the_screen(context, site):
    page = open_catalog(context, site, "#fam=full_factorial")
    page.wait_for_selector("#family h2")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "nothing to scroll sideways"
    # Its labels are over their text, which has the card's width.
    tops = page.eval_on_selector_all(
        "#family :is(dt, dd)", "(parts) => parts.map((p) => p.getBoundingClientRect().top)"
    )
    assert tops == sorted(tops)
    assert len(set(tops)) == len(tops), "one under another"


def test_the_table_s_cells_are_coloured_by_the_most_a_design_of_that_size_estimates(context, site):
    """One coloring, in three steps a reader can tell apart, with a key of a few words a label.

    The cells were four shades of one blue, by what most designs of a cell estimate or by how many it holds: both
    follow the sources the catalog happens to hold. The most that one of them estimates says what the size can do.
    """
    page = open_map(context, site)
    fills = "(cells) => cells.map((cell) => [...cell.classList].find((name) => name.startsWith('fill-')))"
    assert page.locator("#map .sizemap__cell").evaluate_all(fills) == ["fill-2", "fill-3", "fill-4", "fill-2"]
    assert page.locator("[data-shade]").count() == 0, "no second coloring to choose"
    key = [" ".join(words.split()) for words in page.locator(".sizemap__key li").all_text_contents()]
    assert key == ["main effects", "+ interactions", "+ quadratic", "none stated", "fewest needed"]
    assert all(len(label.split()) <= 5 for label in key), "a few words a label"
    tops = page.locator(".sizemap__key li").evaluate_all(
        "(items) => items.map((item) => item.getBoundingClientRect().top)"
    )
    assert len(set(tops)) == 1, "the key is one line at a laptop's width"
    fill = "(marks) => marks.map((mark) => getComputedStyle(mark).fill)"
    colours = page.locator(".sizemap__key rect").evaluate_all(fill)
    assert colours[:3] == ["rgb(34, 96, 97)", "rgb(164, 138, 51)", "rgb(253, 180, 182)"], "three steps of batlow"
    cells = page.locator("#map .sizemap__cell rect").evaluate_all(fill)
    assert cells == [colours[0], colours[1], colours[2], colours[0]], "a cell has its key's color"


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 656, "height": 1000}}])
def test_on_a_tablet_the_table_s_key_is_one_line(context, site):
    """The key took three lines of a tablet's screen over the table, with seven long labels."""
    page = open_map(context, site)
    # In a wide font, as a device without the page's own may show it: the key was one line in a narrow font only.
    tops = page.locator(".sizemap__key").evaluate(
        """(key) => { key.style.fontFamily = '"DejaVu Sans", Verdana, sans-serif';
             return [...key.querySelectorAll('li')].map((item) => Math.round(item.getBoundingClientRect().top)); }"""
    )
    assert len(tops) == 5
    assert len(set(tops)) == 1, "one line"
    room = page.locator(".sizemap__key").evaluate(
        """(key) => { const last = key.lastElementChild.getBoundingClientRect();
             return key.getBoundingClientRect().right - last.right; }"""
    )
    assert room > 24, f"with room to spare for a wider font still: {room} px"


def test_the_table_names_its_two_sides_and_a_row_s_number_looks_like_the_button_it_is(context, site):
    """The numbers down the side were plain circles: nothing said they open, or what the two sides count."""
    page = open_map(context, site)
    assert page.text_content("#map-title") == "Start with your number of factors"
    sides = [" ".join(words.split()) for words in page.locator(".sizemap__axis").all_text_contents()]
    assert sides == ["Factors: press a number to open its design families", "Number of experiments (scales with cost)"]
    row = page.locator("#map [data-unfold='3']")
    assert row.locator(".sizemap__arrow").count() == 1, "an arrow says the number opens"
    ring = row.locator("rect").evaluate("(mark) => getComputedStyle(mark).stroke")
    accent = page.evaluate("getComputedStyle(document.querySelector('.catalog__views a')).color")
    assert ring == accent, "ringed in the color of a link"
    shut = row.locator(".sizemap__arrow").get_attribute("d")
    row.click()
    row = page.locator("#map [data-unfold='3']")
    assert row.get_attribute("aria-expanded") == "true"
    assert row.locator(".sizemap__arrow").get_attribute("d") != shut, "the arrow turns down on an open row"
    how = page.locator(".sizemap__how")
    assert how.get_attribute("open") is None, "how to read the table is shut until asked for"
    how.locator("summary").click()
    said = " ".join(how.text_content().split())
    assert "The number in a cell is how many designs it holds" in said
    assert "the main effects, which are each factor's own effect" in said, "the key's words, at length"


def test_the_map_is_used_from_the_keyboard(context, site):
    """The arrow keys move between the rows' buttons, the cells and the marks; Enter and Space press one."""
    page = open_map(context, site)
    focused = "JSON.stringify({ ...document.activeElement.dataset })"
    page.focus("#map [data-unfold='3']")
    page.keyboard.press("ArrowRight")
    assert json.loads(page.evaluate(focused)) == {"row": "3", "band": "1"}, "the row's first cell"
    page.keyboard.press("Enter")
    counted(page, "2 of 5")
    assert json.loads(page.evaluate(focused)) == {"row": "3", "band": "1"}, (
        "the cell keeps the focus when it is redrawn"
    )
    page.keyboard.press("ArrowRight")
    assert json.loads(page.evaluate(focused)) == {"row": "3", "band": "4"}
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    assert json.loads(page.evaluate(focused)) == {"unfold": "3"}
    page.keyboard.press("ArrowDown")
    assert json.loads(page.evaluate(focused)) == {"unfold": "4"}
    page.keyboard.press("Space")
    assert page.get_attribute("#map [data-unfold='4']", "aria-expanded") == "true"
    assert json.loads(page.evaluate(focused)) == {"unfold": "4"}, "the button keeps the focus as its row opens"
    assert page.locator("#map .sizemap__family").all_text_contents() == ["OMARS · 1"]
    assert page.locator("#map [data-unfold][tabindex='0']").count() == 13, "Tab stops at each row's button"
    assert page.locator("#map .sizemap__cell[tabindex='0']").count() == 0, "and the arrows reach the cells"


def test_a_design_of_more_factors_than_the_map_has_rows_is_in_its_last_row(many_context, site):
    page = open_map(many_context, site)
    labels = page.locator("#map .sizemap__cell[data-row='13']").evaluate_all(CELLS)
    assert labels == [
        "13 factors or more, 25 to 32 runs: 1 design. It estimates the main effects. Press to list it.",
        "13 factors or more, 65 to 128 runs: 1 design. It estimates the main effects. Press to list it.",
    ]
    page.click(cell(13, 5))
    counted(page, "1 of 7")
    assert "k=13-" in page.url, "13 factors and more"


PHONE = {"viewport": {"width": 390, "height": 844}, "has_touch": True, "is_mobile": True}


@pytest.mark.parametrize("context_options", [PHONE])
def test_on_a_phone_the_map_fits_its_card_and_a_row_opens_under_a_finger(context, site):
    page = open_map(context, site)
    fits = """() => {
      const [map, drawn] = [document.querySelector('#map'), document.querySelector('#map svg')];
      const widths = [...map.querySelectorAll('.sizemap__cell rect')].map((cell) => cell.getBoundingClientRect().width);
      return [drawn.getBoundingClientRect().width <= map.clientWidth + 0.5, Math.min(...widths)];
    }"""
    fitted, narrowest = page.evaluate(fits)
    assert fitted, "the map is drawn at the width of its card"
    assert narrowest >= 24, "and each cell is wide enough for a finger"
    page.tap("#map [data-unfold='3']")
    page.wait_for_selector("#map .sizemap__mark")
    page.tap(cell(3, 4))
    counted(page, "1 of 5")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")


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
    assert page.locator(".projection__count").all_text_contents() == ["3\u00d7"], "the center's runs, as a count"
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
    page.fill("#query", "gaps")
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
    assert page.locator("#year-bars .bin--out").count() == page.locator("#year-bars .bin").count() - 1


def test_the_years_are_a_strip_in_the_filter_panel_above_their_boxes(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#year-bars .bin")
    assert page.locator("#facets #year-bars").count() == 1, "with the filter it sets, not above the results"
    strip, boxes = page.locator("#year-bars").bounding_box(), page.locator("#year-min").bounding_box()
    assert strip["height"] == pytest.approx(28, abs=1)
    assert strip["y"] + strip["height"] <= boxes["y"]
    ends = (page.text_content("#year-first"), page.text_content("#year-last"))
    bins = page.locator("#year-bars .bin")
    assert ends == (bins.first.get_attribute("data-from"), bins.last.get_attribute("data-to"))
    assert not page.locator("#timeline-marks, .timeline__mark").count(), "no diamonds and no list of them"


def test_typing_the_years_shades_the_strip(context, site):
    page = open_catalog(context, site)
    page.wait_for_selector("#year-bars .bin")
    page.fill("#year-min", "1975")
    page.fill("#year-max", "1979")
    page.wait_for_function("() => document.querySelectorAll('#year-bars .bin:not(.bin--out)').length === 1")
    assert page.locator("#year-bars .bin:not(.bin--out)").get_attribute("data-from") == "1975"


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
    counted(page, "5 designs")


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


def test_a_design_s_family_opens_on_the_concept_map(context, site):
    """The link under a design's type opens the map on that term, with its definition and the book's line."""
    page = open_catalog(context, site)
    press_title(page, PILOT)
    page.wait_for_selector("#detail[open]")
    page.click("#detail-body .detail__map a")
    page.wait_for_url("**/map/#full-factorial")
    page.wait_for_function("() => document.querySelector('#card h2')?.textContent === 'Full factorial'")
