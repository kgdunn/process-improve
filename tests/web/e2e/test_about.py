"""How Factorial works, in the browser: reached from the app's footer, listing the versions this copy loads."""

from __future__ import annotations

import json

import pytest

from tests.web._support import WEB


def test_the_footer_opens_it_with_the_versions_this_copy_loads(page, plotly_js):
    runtime = json.loads((WEB / "runtime.json").read_text(encoding="utf-8"))
    page.click("footer >> text=Read more here")
    page.wait_for_url("**/about.html")
    page.wait_for_function("document.querySelector('#stack-pyodide').textContent.startsWith('Pyodide ')")
    assert page.text_content("#stack-pyodide") == f"Pyodide {runtime['pyodide']}"
    assert page.text_content("#stack-packages") == ", ".join(runtime["packages"])
    assert page.text_content("#stack-plotly-js") == f"Plotly.js {plotly_js[0]}"
    assert page.text_content("#stack-pypi").startswith(", ".join(runtime["micropip"]))
    page.click("text=Back to Factorial")
    page.wait_for_selector("#go-plan")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_it_reads_on_a_phone_without_scrolling_sideways(context, site):
    page = context.new_page()
    page.goto(f"{site}about.html")
    wider = page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('pre'))
             .map((e) => e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''))"""
    )
    assert wider == []
