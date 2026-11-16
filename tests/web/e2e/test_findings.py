"""What the results say, each sentence beside its picture, with the numbers a press away.

A finding of Python's names the fitted terms it is about and the picture it takes; the page draws the picture from
those terms' coefficients and writes the sentence, so the two cannot disagree. A slope is a factor's own effect, a
curve a square, two lines a pair of factors that act together, and a level line terms that show no sign of an
effect. The search for a simpler model stands with those terms, where a reader would leave them out.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import choose, generate, open_statistician, practise, set_factors, settled

FINDINGS = "#findings > li"
PICKED = "The terms of this model were picked from these same results"

#: Each item of the list: its words, what each of its pictures says, and the words that follow each picture.
READ = """(items) => items.map((item) => ({ words: item.textContent,
  pictures: [...item.querySelectorAll('svg.spark')].map((one) => one.getAttribute('aria-label')),
  after: [...item.querySelectorAll('svg.spark')].map((one) => one.nextSibling.textContent) }))"""

#: How far each picture's first line rises from its low end to its high end, in the picture's own units.
RISES = """(pictures) => pictures.map((one) => {
  const [first, last] = one.querySelector('.spark__line').getAttribute('points').split(' ').filter((_, i, all) =>
    i === 0 || i === all.length - 1).map((point) => Number(point.split(',')[1]));
  return first - last; })"""


def coefficients(page) -> dict[str, dict[str, str]]:
    """Return the table of coefficients by term: each row's cells under the table's own headings."""
    open_statistician(page)
    table = page.locator("#statistician table").first
    heads = table.locator("thead th").all_text_contents()
    rows = [row.locator("td").all_text_contents() for row in table.locator("tbody tr").all()]
    return {cells[0]: dict(zip(heads, cells, strict=True)) for cells in rows}


def test_each_finding_of_the_worked_example_stands_beside_its_picture(page):
    read = page.locator(FINDINGS).evaluate_all(READ)
    assert [item["pictures"] for item in read] == [
        ["A line that rises."],
        [],
        ["A line that falls.", "A line that rises."],
        ["A line that is level."],
        ["Two lines: the response rises at both ends of Pressure, faster at its high end."],
        ["A curve that peaks inside the range."],
        [],
    ]
    # The words are the sentences alone: a picture adds none, and the list reads the same without it.
    assert read[0]["words"] == (
        "Temperature matters most: raising Temperature from 150 to 200 degC raises Yield by 7.9 %, on average."
    )
    assert read[2]["words"] == "Also real: Pressure (lowers it), Flow (raises it)."
    assert read[3]["words"].startswith(
        "No sign of an effect in these runs: Temperature·Flow, Pressure·Flow, Temperature², Pressure²."
    )
    # A picture stands before the words it shows: each of the other real factors has its own.
    assert read[0]["after"][0].startswith("Temperature matters most")
    assert [words.split(",")[0] for words in read[2]["after"]] == ["Pressure (lowers it)", "Flow (raises it)."]


def test_the_pictures_share_one_scale_so_a_steeper_line_is_a_larger_effect(page):
    """Temperature, Pressure and Flow: each line's rise is its coefficient's share of the largest drawn."""
    slopes = page.locator(
        f"{FINDINGS} svg.spark[aria-label^='A line that r'], {FINDINGS} svg.spark[aria-label$='falls.']"
    )
    rises = slopes.evaluate_all(RISES)
    table = coefficients(page)
    fitted = [float(table[term]["Coefficient"]) for term in ("Temperature", "Pressure", "Flow")]
    assert len(rises) == len(fitted)
    assert [rise > 0 for rise in rises] == [True, False, True]
    for rise, coefficient in zip(rises[1:], fitted[1:], strict=True):
        assert rise / rises[0] == pytest.approx(coefficient / fitted[0], abs=0.06), "to the picture's tenth of a pixel"


def test_a_switch_shows_the_numbers_behind_each_finding_and_takes_them_away(page):
    assert page.text_content("#findings-numbers") == "Show the numbers"
    assert page.get_attribute("#findings-numbers", "aria-pressed") == "false"
    assert page.locator("#findings .finding__numbers").count() == 0
    page.click("#findings-numbers")
    assert page.text_content("#findings-numbers") == "Hide the numbers"
    assert page.get_attribute("#findings-numbers", "aria-pressed") == "true"
    shown = page.locator("#findings .finding__numbers").all_text_contents()
    assert len(shown) == 5, "one for each finding about terms: none for the change worth finding or for the fit"
    row = coefficients(page)["Temperature"]
    assert shown[0] == (
        f"Temperature: coefficient {row['Coefficient']} (95% interval {row['95% low']} to {row['95% high']}), "
        f"p {row['p-value']}"
    ), "the table's own"
    assert shown[1].startswith("Pressure: coefficient -")
    assert "; Flow: coefficient " in shown[1], "each of the other real factors"

    # The choice is the reader's for as long as the page is open: the next analysis shows its numbers too.
    page.select_option("#model", "main_effects")
    settled(page)
    assert page.text_content("#findings-numbers") == "Hide the numbers"
    assert page.locator("#findings .finding__numbers").count() > 0
    page.click("#findings-numbers")
    assert page.locator("#findings .finding__numbers").count() == 0
    assert page.get_attribute("#findings-numbers", "aria-pressed") == "false"


def test_leaving_out_the_terms_with_no_sign_fits_the_smaller_model_and_says_how_it_was_picked(page):
    """The search for a simpler model stands with the terms that show no sign of an effect, and does what it did."""
    unclear = page.locator(FINDINGS, has_text="No sign of an effect in these runs")
    assert unclear.locator("#simplify-ask").text_content() == "Simplify the model"
    assert page.locator("#model-choice #simplify, .model #simplify").count() == 0, (
        "it has moved from the model's controls"
    )
    unclear.locator("#simplify-ask").click()
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    assert (
        unclear.locator("#simplify-words")
        .text_content()
        .startswith(
            "The results support Temperature, Pressure, Flow, Temperature·Pressure and Flow², "
            "without Temperature·Flow, Pressure·Flow, Temperature² and Pressure²."
        )
    ), "the terms the search leaves out are the ones the finding lists"
    before = page.evaluate("window.__pi.count('api_analyze_runs')")
    unclear.locator("#simplify-apply").click()
    page.wait_for_function(f"window.__pi.count('api_analyze_runs') > {before}")
    settled(page)

    assert sorted(coefficients(page)) == sorted(
        ["Intercept", "Temperature", "Pressure", "Flow", "Temperature·Pressure", "Flow²"]
    )
    said = page.locator(FINDINGS).all_text_contents()
    assert not [words for words in said if words.startswith("No sign of")], "every term left shows an effect"
    assert said[-1].startswith(PICKED), "as Learn has said of such a model before"
    # With no term to leave out, the search is offered under the list: it can still bring a term back.
    assert page.locator(".findings__more #simplify-ask").is_visible()
    assert page.text_content("#simplify-words") == "", "the proposal was of the analysis before"


def test_a_mixture_s_findings_offer_no_search_for_a_simpler_model(blank_page):
    """A mixture's model is chosen by name."""
    page = blank_page
    set_factors(page, [{"name": name, "low": "", "high": "", "units": ""} for name in ("Oil", "Water", "Wax")])
    choose(page, design_type="simplex_lattice")
    generate(page)
    practise(page)
    page.click("#go-learn")
    page.wait_for_selector("#analysis .equation")
    assert page.locator("#simplify").count() == 0
    assert page.locator("#findings svg.spark").count() == 0


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_findings_and_their_numbers_fit_the_screen(page):
    page.click("#findings-numbers")
    page.click("#simplify-ask")
    page.wait_for_function("document.querySelector('#simplify-words').textContent.startsWith('The results support')")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    widest = page.locator("#findings svg.spark").evaluate_all(
        "(all) => Math.max(...all.map((one) => one.getBoundingClientRect().right))"
    )
    assert widest <= 390


def test_the_report_holds_the_pictures_and_none_of_the_controls(page):
    page.click("#findings-numbers")
    with page.expect_popup() as popup:
        page.click("#learn-report-view")
    report = popup.value
    report.wait_for_url("blob:**")
    assert report.locator("ul.findings svg.spark").count() == 6
    drawn = report.locator("ul.findings svg.spark polyline.spark__line").first
    assert drawn.get_attribute("points"), "a picture is drawn in the report as on the page"
    assert drawn.evaluate("(line) => getComputedStyle(line).fill") == "none"
    text = report.text_content("body")
    assert "Temperature: coefficient " in text, "the numbers, when the reader has them shown"
    assert "Simplify the model" not in text
    assert "Hide the numbers" not in text
