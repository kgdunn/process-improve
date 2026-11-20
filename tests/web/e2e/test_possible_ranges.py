"""Each numeric factor's possible range, in the browser (#1322).

Under the factors, "What each factor can be set to" holds a lowest and a highest possible setting for each numeric
factor. The design is made with them, a study keeps them, and a design's runs past them are named in its notes.
"""

from __future__ import annotations

from typing import Any

from tests.web.e2e._page import calls, generate, reveal, saved_studies, set_factors, settled

FACTORS = [
    {"name": "T", "low": 150, "high": 200, "units": "degC"},
    {"name": "P", "low": 1, "high": 3, "units": "bar"},
]


def bound(page: Any, words: str, name: str) -> Any:  # noqa: ANN401 - a Playwright Page and Locator
    """Return the box of a factor's lowest or highest possible setting."""
    return page.get_by_label(f"{words} setting of {name}")


def opened(page: Any, factors: list[dict]) -> None:  # noqa: ANN401 - a Playwright Page
    """Type the factors and open the possible ranges under them."""
    set_factors(page, factors)
    settled(page)
    reveal(page, "#possible")
    page.click("#possible summary")


def test_a_categorical_factor_takes_no_possible_range(blank_page):
    page = blank_page
    opened(page, [*FACTORS, {"name": "Supplier", "low": "A", "high": "B"}])
    assert page.locator("#possible-rows th").all_text_contents() == ["T", "P"]


def test_a_possible_range_is_kept_by_the_design_and_the_study_and_runs_past_it_are_named(blank_page):
    page = blank_page
    opened(page, FACTORS)
    bound(page, "Lowest possible", "P").fill("0.7")  # two factors: the low axial run of P is at 0.586 bar
    bound(page, "Lowest possible", "P").press("Tab")
    assert page.text_content("#possible-count") == "1"
    generate(page, design_type="ccd", alpha="rotatable")
    made = calls(page, "api_make_design")[-1]["payload"]
    assert made["possible"] == {"P": [0.7, None]}
    assert "sets P past what it can be, at least 0.7 bar" in page.text_content("#design-summary")
    kept = saved_studies(page, lambda studies: any(s["design"]["spec"].get("possible") for s in studies))
    assert kept[0]["design"]["spec"]["possible"] == {"P": [0.7, None]}
    assert kept[0]["v"] == 15
