"""Step 4: every design the factor count allows, generated as step 1 would make it and scored side by side.

A column per design, a row per measure in its group, the FDS curves and a correlation map
per design; and any design can be taken from the comparison into step 1.
"""

from __future__ import annotations

from tests.web.e2e._page import calls, settled, status


def design_columns(page) -> list[str]:
    heads = page.locator("#comparison thead th").all_text_contents()
    assert heads[:2] == ["Measure", "Better"]
    return [head.removesuffix("Use this design") for head in heads[2:]]


def test_the_comparison_has_a_column_per_design_and_a_row_per_measure(page, first_paint):
    compared = first_paint["comparison"]["result"]
    assert design_columns(page) == compared["designs"]
    shown = [
        m for row, m in enumerate(compared["measures"]) if any(compared["display"][d][row] for d in compared["designs"])
    ]
    groups = page.locator("#comparison tbody tr.group").all_text_contents()
    assert groups == list(dict.fromkeys(m["group"] for m in shown))
    assert page.locator("#comparison tbody tr:not(.group)").count() == len(shown)


def test_comparing_generates_each_design_then_scores_them_all(page, first_paint):
    page.click("#compare")
    settled(page)
    made = first_paint["comparison"]["made"]
    fns = [c["fn"] for c in calls(page)][1:]  # after the catalogue
    assert fns == ["api_compare_plan", *["api_make_design"] * len(made), "api_compare"]
    assert status(page) == (f"{len(made)} designs compared.", "ok")
    assert page.locator("#compare-charts .js-plotly-plot").count() == 1 + len(made)
    assert "compare_designs(designs" in page.text_content("#py-compare code")


def test_a_design_taken_from_the_comparison_reaches_step_1_and_is_simulated(page, first_paint):
    chosen = next(m for m in first_paint["comparison"]["made"] if m["design_type"] == "d_optimal")
    column = page.locator("#comparison thead th", has_text=chosen["label"])
    column.get_by_role("button", name="Use this design").click()
    settled(page)
    assert page.input_value("#design-type") == "d_optimal"
    assert page.locator("#design-table tbody tr").count() == chosen["design"]["n_runs"]
    assert calls(page)[-3]["fn"] == "api_simulate", "the same process, noise and seed, on the new runs"
    assert page.locator("#design-table input.sim").count() == chosen["design"]["n_runs"]
