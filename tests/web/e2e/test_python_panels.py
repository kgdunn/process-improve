"""The "Show the Python" panel of each step: the process-improve code that does what the step did, to copy."""

from __future__ import annotations

import base64

from tests.web._support import FACTORS, fill_workbook, quadratic
from tests.web.e2e._page import settled, type_results


def code(page, step: str) -> str:
    return page.text_content(f"#py-{step} code")


def test_each_step_shows_the_python_that_does_the_same(page):
    assert 'generate_design(factors, design_type="box_behnken"' in code(page, "design")
    assert "TrueProcess(" in code(page, "run")
    assert "analyze_experiment(" in code(page, "analyse")
    assert 'compare_designs(designs, model="main_quadratic")' in code(page, "compare")
    for step in ("design", "run", "analyse", "compare"):
        assert page.is_visible(f"#py-{step}"), step


def test_copy_puts_the_code_on_the_clipboard(context, page, site):
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=site.rstrip("/"))
    page.click("#py-design summary")
    page.click("#py-design .copy")
    assert page.evaluate("navigator.clipboard.readText()") == code(page, "design")
    assert page.text_content("#py-design .copy") == "Copied"
    page.wait_for_function("document.querySelector('#py-design .copy').textContent === 'Copy the code'")


def test_after_an_upload_the_panels_show_the_workbook_s_design_and_results(page, api, tmp_path):
    design = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS})
    path = tmp_path / "filled.xlsx"
    path.write_bytes(base64.b64decode(fill_workbook(design["xlsx_base64"], quadratic, blank_rows=(0,))))
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    settled(page)
    assert 'design_type="ccd"' in code(page, "design")
    assert "TrueProcess" not in code(page, "run"), "the results were read, not simulated"
    assert "y = pd.Series([None, " in code(page, "run"), "the first run is still to do"
    assert "analyze_experiment(" in code(page, "analyse")


def test_the_analysis_panel_goes_with_the_analysis(page):
    type_results(page, dict.fromkeys(range(15), ""))
    assert page.is_hidden("#analysis")
    assert page.is_hidden("#py-analyse")
