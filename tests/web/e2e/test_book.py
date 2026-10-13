"""The book's examples, opened by name: ``?book=<name>`` opens the example with the book's runs and results.

The book prints these links beside its worked examples and exercises. Each opens the factors,
their units and the run sheet as the book's table has them, analysed once Python has started,
and says where in the book it comes from.
"""

from __future__ import annotations

from tests.web.e2e._page import factor_table, first_paint_done, run_sheet, settled, status, type_results

BANNER = "#campaign-banner"


def open_example(context, site: str, name: str) -> object:
    """Open the page at ``?book=<name>``, and wait until it has analysed the example's results."""
    page = context.new_page()
    page.goto(f"{site}?book={name}")
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    return page


def coefficients(page) -> dict[str, str]:
    """Read the coefficient table: each term's label and its coefficient, as the page shows them."""
    rows = page.locator("#analysis table").first.locator("tbody tr").all()
    return {cells[0]: cells[1] for cells in (row.locator("td").all_text_contents() for row in rows)}


def saved(page, count: int) -> None:
    page.wait_for_function(f"document.querySelector('#library-count').textContent === '{count}'")


def test_a_book_example_opens_with_its_factors_runs_and_results(context, site, book_files):
    page = open_example(context, site, "pollutant")
    example = book_files["pollutant"]
    banner = page.text_content(BANNER)
    assert banner.startswith(f"Opened the book's example: {example['title']}, with the book's results.")
    assert page.get_attribute(f"{BANNER} a", "href") == example["book"]
    assert page.get_attribute(f"{BANNER} a", "target") == "pid-book", (
        "one tab for the book, so the page keeps its state"
    )
    assert factor_table(page) == [
        {"name": "C", "low": "P", "high": "Q", "units": ""},
        {"name": "T", "low": "72", "high": "100", "units": "°F"},
        {"name": "S", "low": "200", "high": "400", "units": "rpm"},
    ]
    assert page.input_value("#response") == "Pollutant"
    categorical = page.eval_on_selector_all("#factors tbody [name=categorical]", "boxes => boxes.map((b) => b.checked)")
    assert categorical == [True, False, False], "the chemical, P or Q, is categorical"
    # The run sheet is the book's table, row by row: its experiment number, and the order it was run in.
    sheet = run_sheet(page)
    assert [row[:2] for row in sheet] == [[str(i), str(r)] for i, r in enumerate([5, 6, 1, 4, 2, 7, 3, 8], start=1)]
    by_experiment = {int(row[0]): row[2:] for row in sheet}
    assert by_experiment[2] == ["Q", "72", "200", "30"]
    assert by_experiment[7] == ["P", "100", "400", "5"]
    assert float(coefficients(page)["C·S"]) == -6.75, "the book's coefficient, from its own runs"
    assert page.url == site, "the address loses the name, so a reload opens the saved copy"
    saved(page, 1)


def test_opening_the_same_example_again_reopens_the_copy_kept_here(context, site):
    open_example(context, site, "conversion")
    again = open_example(context, site, "conversion")
    saved(again, 1)
    assert again.text_content(BANNER).startswith("Opened the book's example: Conversion against temperature")


def test_a_reader_s_changes_are_kept_and_the_book_s_numbers_come_back(context, site):
    """A copy with a result typed over the book's stays as the reader left it, beside a fresh copy."""
    page = open_example(context, site, "conversion")
    type_results(page, {0: 99})
    saved(page, 1)
    page.close()
    again = open_example(context, site, "conversion")
    saved(again, 2)
    assert sorted(row[-1] for row in run_sheet(again)) == ["53", "60", "64", "69"]


def test_an_example_of_another_published_source_opens_with_its_citation(context, site, book_files):
    """``?example=<name>`` opens it from the same index, and the banner says where it came from."""
    page = context.new_page()
    page.goto(f"{site}?example=ullmann-agitator")
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    settled(page)
    source = book_files["ullmann-agitator"]["source"]
    assert page.text_content(BANNER).startswith(
        'Opened an example from Soravia and Orth (2012), "Design of Experiments", Ullmann\'s Encyclopedia of '
        "Industrial Chemistry, Section 3.3, Figure 8 (p. 177): Yield against catalyst A, temperature B and "
        "agitator type C, with its published results."
    )
    assert page.get_attribute(f"{BANNER} a", "href") == source["url"] == "https://doi.org/10.1002/14356007.e08_e01.pub3"
    assert page.get_attribute(f"{BANNER} a", "target") == "source", "a tab of its own, not the book's"
    assert factor_table(page)[2] == {"name": "C", "low": "current", "high": "new", "units": ""}
    assert [row[-1] for row in run_sheet(page)] == ["60", "62", "74", "92", "57", "59", "67", "82"]
    assert float(coefficients(page)["C"]) == -2.875, "half the article's main effect of C, -5.75"
    assert page.url == site
    saved(page, 1)


def test_a_name_the_book_does_not_have_is_refused_and_the_worked_example_shown(context, site, first_paint):
    """The name is looked up in the index first: no file is asked for that the index does not list."""
    page = context.new_page()
    fetched = []
    page.on("request", lambda request: fetched.append(request.url) if "/book/" in request.url else None)
    page.goto(f"{site}?book=no-such-example")
    first_paint_done(page, first_paint)
    assert page.text_content(BANNER) == (
        'This design could not be opened: the book has no example named "no-such-example". '
        "Here is the worked example instead."
    )
    assert page.get_attribute(BANNER, "class") == "banner error"
    assert page.is_visible("#demo-banner")
    assert status(page)[1] == "ok"
    assert fetched == [f"{site}book/index.json"]


def test_a_name_no_book_could_print_is_refused_before_anything_is_fetched(context, site, first_paint):
    page = context.new_page()
    fetched = []
    page.on("request", lambda request: fetched.append(request.url) if "/book/" in request.url else None)
    page.goto(f"{site}?book=../demo")
    first_paint_done(page, first_paint)
    assert page.text_content(BANNER).startswith(
        'This design could not be opened: "../demo" is not the name of an example in the book.'
    )
    assert fetched == []
