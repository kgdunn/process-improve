"""How Factorial works, in the browser: reached from the app's footer, listing the versions this copy loads.

Each part answers a question in plain words; what only some readers ask is under the answer, in a fold.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.web._support import WEB


def test_the_footer_opens_it_with_the_versions_this_copy_loads(page, plotly_js):
    runtime = json.loads((WEB / "runtime.json").read_text(encoding="utf-8"))
    page.click("footer >> text=How it works")
    page.wait_for_url("**/technical-details-how-factorial-works.html")
    page.wait_for_function("document.querySelector('#stack-pyodide').textContent.startsWith('Pyodide ')")
    assert page.text_content("#stack-pyodide") == f"Pyodide {runtime['pyodide']}"
    assert page.text_content("#stack-packages") == ", ".join([*runtime["packages"], *runtime["fitting"]])
    assert page.text_content("#stack-plotly-js") == f"Plotly.js {plotly_js[0]}"
    assert page.text_content("#stack-pypi").startswith(", ".join(runtime["micropip"]))
    assert page.is_hidden("#stack"), "the versions are in a fold, for a reader who asks"
    page.click("text=The versions this copy loads")
    assert page.is_visible("#stack-pyodide")
    page.click("text=Back to Factorial")
    page.wait_for_selector("#go-plan")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_it_reads_on_a_phone_without_scrolling_sideways(context, site):
    page = context.new_page()
    page.goto(f"{site}technical-details-how-factorial-works.html")
    wider = page.evaluate(
        """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('pre'))
             .map((e) => e.tagName.toLowerCase() + (e.className ? '.' + e.className : ''))"""
    )
    assert wider == []
    page.evaluate("() => document.querySelectorAll('details.fold').forEach((fold) => { fold.open = true; })")
    assert (
        page.evaluate(
            """() => [...document.querySelectorAll('body *')]
             .filter((e) => e.getBoundingClientRect().right > document.documentElement.clientWidth + 1)
             .filter((e) => !e.closest('pre')).length"""
        )
        == 0
    ), "with every fold open too"


def test_each_part_shows_its_answer_and_opens_its_fold_on_a_press_or_from_the_keyboard(context, site):
    page = context.new_page()
    page.goto(f"{site}technical-details-how-factorial-works.html")
    folds = page.locator("details.fold")
    assert folds.count() >= 8
    assert folds.evaluate_all("(folds) => folds.every((fold) => !fold.open)"), "the page opens on its answers alone"
    assert page.locator(".feat-cards li").count() == 6, "and on what it comes to, in short"
    part = page.locator("section[aria-labelledby='h-servers']")
    assert part.locator("h2").text_content() == "What does a server see?"
    assert "nothing of your work" in part.locator("p").first.text_content()
    assert part.locator("text=the first 24 bits").is_hidden()
    part.locator("summary").click()
    assert part.locator("text=the first 24 bits").is_visible()
    part.locator("summary").press("Enter")
    assert part.locator("text=the first 24 bits").is_hidden()
    # The link to the study file leads to its part, from the part on where the studies are kept.
    page.click("section[aria-labelledby='h-kept'] >> text=the study file")
    assert page.url.endswith("#h-file")


def test_the_picture_of_a_locker_is_done_in_front_of_the_reader(context, site):
    """The proof under the picture encrypts a line with a new key, by web/locker.js, and opens it again; each press is a
    new key, so new bytes. On a wide screen the picture reads left to right.
    """
    page = context.new_page()
    page.goto(f"{site}technical-details-how-factorial-works.html#h-locker")
    page.wait_for_selector("#locker-proof", state="visible")
    plain = page.text_content("#proof-plain")
    held = page.text_content("#proof-held")
    assert plain == page.text_content("#proof-opened") == "Temperature 175 °C, pressure 2.5 bar: yield 62.3 %"
    assert re.fullmatch(r"(?:[0-9a-f]{2} ){23}[0-9a-f]{2}", held), held
    assert page.text_content("#proof-size") == "the first 24 of 4,124 bytes", (
        "padded to the smallest size a locker takes"
    )
    page.click("#proof-again")
    page.wait_for_function(f"document.querySelector('#proof-held').textContent !== {held!r}")
    assert page.is_visible(".locker-picture__wide")
    assert page.is_hidden(".locker-picture__tall")


@pytest.mark.parametrize("context_options", [{"viewport": {"width": 390, "height": 844}, "is_mobile": True}])
def test_on_a_phone_the_picture_of_a_locker_reads_top_to_bottom(context, site):
    page = context.new_page()
    page.goto(f"{site}technical-details-how-factorial-works.html#h-locker")
    assert page.is_hidden(".locker-picture__wide")
    assert page.is_visible(".locker-picture__tall")
    width = page.evaluate("document.querySelector('.locker-picture__tall').getBoundingClientRect().width")
    assert width >= 320, "its words drawn at about their own size"
