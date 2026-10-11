"""The "Show the Python" panel of each step: the process-improve code that does what the step did, to copy."""

from __future__ import annotations


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
