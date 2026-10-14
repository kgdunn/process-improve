"""The "Show the Python code" panel of each step: the process-improve code that does what the step did, to copy."""

from __future__ import annotations

import base64
import re

from tests.web._support import FACTORS, fill_workbook, quadratic
from tests.web.e2e._page import reveal, settled, type_results


def code(page, step: str) -> str:
    return page.text_content(f"#py-{step} code")


def test_each_step_shows_the_python_that_does_the_same(page):
    assert 'generate_design(factors, design_type="box_behnken"' in code(page, "design")
    assert "TrueProcess(" in code(page, "run")
    assert "analyze_experiment(" in code(page, "analyse")
    for step in ("design", "run", "analyse"):
        reveal(page, f"#py-{step}")
        assert page.is_visible(f"#py-{step}"), step
        assert page.text_content(f"#py-{step} summary") == "Show the Python code"
    reveal(page, "#compare")
    page.click("#compare")
    assert 'compare_designs(designs, model="main_quadratic")' in code(page, "compare")
    assert page.is_visible("#py-compare"), "in the detailed comparison"


def test_each_panel_holds_the_steps_before_it_so_its_code_runs_on_its_own(page):
    design, run, analyse = code(page, "design"), code(page, "run"), code(page, "analyse")
    assert "generate_design(" in run
    assert "generate_design(" in analyse
    assert "TrueProcess(" in analyse
    first = analyse.splitlines()[:3]
    assert first == [
        "import pandas as pd",
        "from process_improve.experiments import Factor, TrueProcess, analyze_experiment, generate_design",
        "",
    ], "the imports of every step, merged, at the top"
    assert design.split("\n\n", 1)[1] in run, "the design as its own panel shows it"


def test_copy_puts_the_code_on_the_clipboard(context, page, site):
    context.grant_permissions(["clipboard-read", "clipboard-write"], origin=site.rstrip("/"))
    reveal(page, "#py-design")
    page.click("#py-design summary")
    page.click("#py-design .copy")
    assert page.evaluate("navigator.clipboard.readText()") == code(page, "design")
    assert page.text_content("#py-design .copy") == "Copied"
    assert page.is_visible("#toast")
    assert page.text_content("#toast") == "Code copied to the clipboard: paste it into a script or a notebook."
    page.wait_for_function("document.querySelector('#py-design .copy').textContent === 'Copy the code'")


def test_after_an_upload_the_panels_show_the_workbook_s_design_and_results(page, api, tmp_path):
    design = api.ok("api_make_design", {"design_type": "ccd", "factors": FACTORS})
    path = tmp_path / "filled.xlsx"
    path.write_bytes(base64.b64decode(fill_workbook(design["xlsx_base64"], quadratic, blank_rows=(0,))))
    reveal(page, "#tab-real")
    page.click("#tab-real")
    page.set_input_files("#upload", path)
    settled(page)
    assert 'design_type="ccd"' in code(page, "design")
    assert "TrueProcess" not in code(page, "run"), "the results were read, not simulated"
    assert re.search(r'"y": \[\s*None,', code(page, "run")), "the first run is still to do"
    assert 'y = runs["y"]' in code(page, "run")
    assert "analyze_experiment(" in code(page, "analyse")


def test_the_analysis_panel_goes_with_the_analysis(page):
    type_results(page, dict.fromkeys(range(15), ""))
    assert page.is_hidden("#analysis")
    assert page.is_hidden("#py-analyse")


def test_a_refused_clipboard_selects_the_code_to_copy_by_hand(page):
    page.evaluate(
        "() => { navigator.clipboard.writeText = () =>"
        " Promise.reject(new DOMException('Write permission denied.', 'NotAllowedError')); }"
    )
    reveal(page, "#py-design")
    page.click("#py-design summary")
    page.click("#py-design .copy")
    page.wait_for_function("document.querySelector('#py-design .copy').textContent !== 'Copy the code'", timeout=5_000)
    assert page.text_content("#py-design .copy") == "Selected: copy it with Ctrl+C or ⌘C"
    assert page.evaluate("getSelection().toString()") == code(page, "design")
