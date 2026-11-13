"""A table of the reader's own in the page: pasted or chosen as a file, its columns checked, and opened as a study.

The study is one of runs as given, so Run lists the table's runs with their results and Learn analyses them.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, settled

DESIGNED = "Temp (degC),Pressure (bar),Catalyst,Yield (%)\n" + "".join(
    f"{t},{p},{c},{60 + 0.1 * (t - 175) - 3 * (p - 2) + (2 if c == 'B' else 0) + 0.3 * (i % 5)}\n"
    for i, (t, p, c) in enumerate([(t, p, c) for t in (150, 200) for p in (1, 3) for c in "AB"] * 2)
)


def paste(page, text: str) -> None:
    """Open the dialog, paste a table and have it read."""
    page.click("#import-data")
    page.fill("#import-text", text)
    page.click("#import-read")
    page.wait_for_function(
        "!document.querySelector('#import-columns').hidden"
        " || document.querySelector('#import-status').classList.contains('error')"
    )


def column(page, heading: str):
    """Return the line of the list for one column of the table."""
    return page.locator(f'#import-rows tr[data-column="{heading}"]')


def test_a_pasted_table_is_read_and_its_columns_guessed(fresh_page):
    page = fresh_page
    paste(page, DESIGNED)
    assert page.text_content("#import-status") == (
        "Read 16 runs in 4 columns. Check what each column is, then open the study."
    )
    roles = page.locator("#import-rows select").evaluate_all("(boxes) => boxes.map((b) => b.value)")
    assert roles == ["factor", "factor", "factor", "response"]
    temp = column(page, "Temp (degC)")
    assert temp.locator("th .hint").text_content() == "numbers, 150 to 200, at 2 settings"
    assert temp.locator('input[type="text"]').first.input_value() == "Temp"
    assert column(page, "Catalyst").locator(".import-range").is_hidden(), "a categorical factor's levels are its labels"
    assert page.is_enabled("#import-open")


def test_the_table_opens_as_a_study_with_its_results_and_learn_analyses_it(fresh_page):
    page = fresh_page
    paste(page, DESIGNED)
    page.fill("#import-title", "Trials in March")
    page.click("#import-open")
    page.wait_for_function("document.querySelector('#workspace').dataset.phase === 'learn'")
    settled(page)
    assert not page.locator("#import-dialog").evaluate("(d) => d.open")
    assert page.text_content("#campaign-banner").startswith(
        "Opened your table as a study of 16 runs, taken as they are."
    )
    made = calls(page, "api_import_study")[-1]["payload"]
    assert made["title"] == "Trials in March"
    assert page.input_value("#design-type") == "given"
    assert page.locator("#findings li").first.text_content().startswith("Pressure matters most: ")
    assert "Trials in March" in page.text_content("#library"), "kept under Your studies"


def test_a_column_s_name_is_mended_before_the_study_opens(fresh_page):
    page = fresh_page
    paste(page, DESIGNED)
    name = column(page, "Temp (degC)").locator('input[type="text"]').first
    name.fill("Temp C")
    assert page.text_content("#import-problems").startswith("Temp C: a name is letters, digits and _")
    assert page.is_disabled("#import-open")
    name.fill("T")
    assert page.is_hidden("#import-problems")
    column(page, "Yield (%)").locator("select").select_option("ignore")
    assert page.text_content("#import-problems") == "Choose at least one column as a response."


def test_a_table_python_cannot_read_is_refused_in_a_sentence(fresh_page):
    page = fresh_page
    paste(page, "A,B,y\n1,2,3\n")
    assert page.text_content("#import-status") == "The table needs a line of headings and at least two runs below it."
    assert page.is_hidden("#import-columns")


def test_a_csv_file_is_read_as_a_pasted_table_is(fresh_page):
    page = fresh_page
    page.click("#import-data")
    page.set_input_files(
        "#import-file", {"name": "march trials.csv", "mimeType": "text/csv", "buffer": DESIGNED.encode()}
    )
    page.wait_for_function("!document.querySelector('#import-columns').hidden")
    assert page.input_value("#import-title") == "march trials", "the file's name names the study"
    assert page.locator("#import-rows tr").count() == 4
