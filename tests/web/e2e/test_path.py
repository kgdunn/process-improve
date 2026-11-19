"""The runs along the path of steepest ascent: listed in Next, added to the run sheet, and followed to a new study.

The worked example's results call for a move past its ranges. The path's runs are made one at a time, in the path's
order, and are no part of the fit; the best of them is the centre of the next study.
"""

from __future__ import annotations

import pytest

from tests.web.e2e._page import factor_table, reveal, settled

PATH = "#next-out .next__path"


def path_table(page) -> list[list[str]]:
    return page.locator(f"{PATH} tbody tr").evaluate_all(
        "(rows) => rows.map((row) => [...row.cells].map((cell) => cell.textContent.trim()))"
    )


def add_the_path(page) -> None:
    reveal(page, "#next")
    page.click("#path-add")
    settled(page)


def enter(page, results: list[str]) -> None:
    """Type the results of the path's runs, the last runs of the run sheet, in the path's order."""
    page.click("#go-run")
    boxes = page.locator("#design-table input[data-response]")
    first = boxes.count() - 4
    for i, value in enumerate(results):
        boxes.nth(first + i).fill(value)
        boxes.nth(first + i).press("Tab")
    settled(page)
    page.click("#go-next")


def test_the_move_lists_the_path_s_runs_with_their_price(page):
    reveal(page, "#next")
    assert page.text_content(f"{PATH} h4") == "The runs along the path"
    heads = page.locator(f"{PATH} thead th").all_text_contents()
    assert [head.strip() for head in heads] == ["Step", "Temperature", "Pressure", "Flow", "The model predicts"]
    rows = path_table(page)
    assert [row[0] for row in rows] == ["1", "2", "3", "4"]
    assert all(row[1].endswith("degC") and row[4].endswith("%") for row in rows)
    assert float(rows[0][1].split()[0]) > 200, "past the range the study ran"
    assert "Stop at the first run that does worse" in page.text_content(PATH)
    assert page.text_content("#path-add") == "Add these 4 runs to the run sheet"
    assert "4 more runs" in page.text_content(f"{PATH} .price")


def test_a_setting_below_zero_on_the_path_is_marked_and_the_page_says_what_that_means(page):
    """The path takes Pressure below zero in the worked example: the page knows a pressure's range began above it."""
    reveal(page, "#next")
    rows = path_table(page)
    marked = [row[2] for row in rows if row[2].endswith(", below zero")]
    assert marked, "the worked example's path reaches a pressure below zero"
    assert all(setting.startswith("-") for setting in marked)
    assert not any(cell.endswith("below zero") for row in rows for cell in (row[1], row[3])), "no other factor does"
    assert 'A setting marked "below zero" may not be possible: stop the path before it.' in page.text_content(PATH)


def test_the_path_s_runs_join_the_run_sheet_in_order_and_stay_out_of_the_fit(page):
    reveal(page, "#next")
    before = page.text_content("#badge-learn")
    steps = path_table(page)
    add_the_path(page)
    assert page.is_hidden("#path-add")
    assert page.text_content("#badge-run") == "15 of 15", "the design's runs are counted, not the path's"
    assert page.text_content("#badge-learn") == before, "the fit is as it was"
    rows = path_table(page)
    assert [row[0] for row in rows] == ["16", "17", "18", "19"], "by their run numbers, after the design's"
    assert [row[1] for row in rows] == [step[1] for step in steps], "in the path's order"
    # The worked example's results are practice ones, so the path's are simulated with them.
    assert "is the best of them so far" in page.text_content(PATH)
    assert all(row[-1].endswith("%") for row in rows), "each with its result"
    page.click("#go-run")
    assert page.locator("#design-table tbody tr").count() == 19


def test_the_path_is_followed_to_the_first_run_that_does_worse_and_a_study_is_centred_on_the_best(page):
    add_the_path(page)
    enter(page, ["", "", "", ""])
    assert "None of these runs has a result yet" in page.text_content(PATH)
    assert page.is_hidden("#path-study")
    enter(page, ["70", "75"])
    assert "Run 17 is the best of them so far (75%). Make run 18 next." in page.text_content(PATH)
    enter(page, ["70", "75", "60"])
    assert "Run 18 did worse than run 17: stop here." in page.text_content(PATH)
    centre = float(path_table(page)[1][1].split()[0])
    assert page.text_content("#path-study") == "Start a new study centred on run 17"
    page.click("#path-study")
    settled(page)
    assert page.get_attribute("#workspace", "data-phase") == "plan"
    temperature = factor_table(page)[0]
    assert (float(temperature["low"]), float(temperature["high"])) == pytest.approx((centre - 25, centre + 25))
    assert page.input_value("#response") == "Yield", "the brief is this study's"


def test_the_path_s_runs_come_off_the_run_sheet_in_one_press(page):
    add_the_path(page)
    page.click("#path-remove")
    settled(page)
    assert page.is_visible("#path-add"), "offered again"
    page.click("#go-run")
    assert page.locator("#design-table tbody tr").count() == 15


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_path_s_runs_fit_the_screen(page):
    add_the_path(page)
    assert page.is_visible("#path-study")
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
