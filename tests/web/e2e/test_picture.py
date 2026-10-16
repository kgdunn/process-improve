"""The picture of the settings that can be run, under the limits: a panel per pair of numeric factors.

Each panel is hatched where no setting can be run and shaded where some setting of the other factors
meets every limit, with each straight limit's boundary numbered as the list numbers it, the center of
the ranges and the run sheet's runs. A panel opens to look inside the region, with the other factors
held where the reader sets them.
"""

from __future__ import annotations

from typing import Any

from tests.web._support import FACTORS, factors_named
from tests.web.e2e._page import candidates_done, generate, limits_done, reveal, set_factors, settled, type_limit

HEAT = "T + 25 * P <= 250"


def panels(page: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Read each panel as a reader can: its pair, its description, and what it draws."""
    return page.evaluate(
        """() => [...document.querySelectorAll('#region-panels .region__cell')].map((cell) => {
             const drawing = cell.querySelector('svg');
             const described = cell.querySelector('[aria-describedby]')?.getAttribute('aria-describedby');
             const description = described
               ? document.getElementById(described).textContent
               : drawing.getAttribute('aria-label');
             return {
               pair: [drawing.dataset.x, drawing.dataset.y],
               description,
               limits: [...drawing.querySelectorAll('.region__tag text')].map((t) => t.textContent),
               runs: drawing.querySelectorAll('.region__run').length,
               traced: Boolean(drawing.querySelector('.region__edge')),
             };
           })"""
    )


def test_the_picture_shows_each_pair_with_each_limit_s_boundary_numbered(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    assert page.locator("#region").is_visible()
    shown = {tuple(p["pair"]): p for p in panels(page)}
    assert list(shown) == [("T", "P"), ("T", "F"), ("P", "F")]
    assert shown[("T", "P")]["limits"] == ["1"], "the limit names T and P"
    assert shown[("T", "F")]["limits"] == []
    assert shown[("T", "P")]["description"] == (
        "T against P, for some setting of the other factors: the settings that can be run have the corners "
        "T 150 degC, P 1 bar; T 200 degC, P 1 bar; T 200 degC, P 2 bar; T 175 degC, P 3 bar; T 150 degC, P 3 bar; "
        "limit 1 crosses from T 175 degC, P 3 bar to T 200 degC, P 2 bar."
    )


def test_the_picture_draws_the_run_sheet_s_runs(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    candidates_done(page)
    generate(page)
    reveal(page, "#limits")
    runs = page.evaluate("document.querySelectorAll('#design-table tbody tr').length")
    assert [p["runs"] for p in panels(page)] == [runs] * 3
    assert page.text_content("#region-legend").endswith(f"a run of the run sheet ({runs}).")


def test_looking_inside_a_pair_holds_the_other_factors_where_the_reader_sets_them(blank_page):
    page = blank_page
    type_limit(page, "T + 25 * P + 5 * F <= 350")
    page.get_by_role("button", name="Look inside T and P").click()
    assert page.locator("#region-panels").is_hidden()
    assert page.evaluate("document.activeElement.name") == "held-F", "the keyboard goes to the first factor held"
    assert page.input_value("[name=held-box-F]") == "15", "held at the center of its range"
    page.fill("[name=held-box-F]", "18")
    page.wait_for_timeout(400)
    settled(page)
    assert page.input_value("[name=held-F]") == "18", "the slider follows the box"
    drawing = page.get_attribute("#slice-panel svg", "aria-label")
    assert drawing == (
        "T against P, with F at 18 L/min: the settings that can be run have the corners T 150 degC, P 1 bar; "
        "T 200 degC, P 1 bar; T 200 degC, P 2.4 bar; T 185 degC, P 3 bar; T 150 degC, P 3 bar; "
        "limit 1 crosses from T 185 degC, P 3 bar to T 200 degC, P 2.4 bar."
    )
    page.click("#slice-close")
    assert page.locator("#region-panels").is_visible()
    assert page.evaluate("document.activeElement.dataset.x + document.activeElement.dataset.y") == "TP"


def test_a_curved_limit_is_traced_through_where_it_crosses_the_grid(fresh_page):
    page = fresh_page
    set_factors(page, FACTORS[:2])
    candidates_done(page)
    type_limit(page, "T * P <= 450")
    [panel] = panels(page)
    assert panel["traced"], "a smooth boundary, not the steps of a mask"
    assert panel["limits"] == ["1"], "the curve is numbered as the list numbers it"
    assert "the settings that can be run are shaded, inside a curved limit; limit 1 curves from" in panel["description"]
    for end in ("T 150 degC, P 3 bar", "T 200 degC, P 2.25 bar"):
        assert end in panel["description"], end


def test_a_curved_limit_that_names_two_of_three_factors_is_traced_on_their_panel(blank_page):
    page = blank_page
    type_limit(page, HEAT)
    type_limit(page, "T * F <= 3600")
    shown = {tuple(p["pair"]): p for p in panels(page)}
    assert shown[("T", "F")]["traced"]
    assert shown[("T", "F")]["limits"] == ["2"]
    assert not shown[("T", "P")]["traced"], "the curve names F, which the panel of T and P leaves out"


def test_from_five_factors_the_picture_shows_every_pair_on_request(fresh_page):
    page = fresh_page
    set_factors(page, factors_named(*"ABCDE"))
    candidates_done(page)
    type_limit(page, "A + D <= 1")
    assert [p["pair"] for p in panels(page)] == [["A", "D"]], "the pair the limit names"
    page.click("#every-pair")
    limits_done(page)
    assert len(panels(page)) == 10
    assert page.get_attribute("#every-pair", "aria-pressed") == "true"
