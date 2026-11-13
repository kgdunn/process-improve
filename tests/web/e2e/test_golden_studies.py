"""Studies saved by an earlier release open unchanged: the same run sheet, fitted model and best settings.

A reader keeps a study as a link in an email or a note, or as the file "Download all" writes, and expects
it to open a year later as it did the day it was made. ``golden/studies.json`` holds ten links made with
factori.al on 13 November 2026, one for each common kind of study, with what each showed: the run sheet,
the model and transform chosen, and, for each response, every coefficient (with its standard error and
p-value) and every best setting. ``golden/studies.factorial.json`` is the file "Download all" wrote there
once every link had been opened.

The run sheet is the reader's own data, so it must open exactly as it was kept. A number the page
works out may differ by one unit in the last digit shown: a value that lies on a rounding tie
(1.1875 shown to four figures) rounds one way natively and the other way under Pyodide.

A change that is meant to alter what a kept study shows updates the golden file in the same pull
request, so that the change is seen in review.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.web.e2e._page import headers, run_sheet, settled
from tests.web.e2e.test_responses import pick, tabs

GOLDEN = Path(__file__).parent / "golden"
CASES = json.loads((GOLDEN / "studies.json").read_text(encoding="utf-8"))["cases"]
STUDY_FILE = GOLDEN / "studies.factorial.json"
BANNER = "#campaign-banner"
#: A number as the page shows it, and what follows it, such as its units: "18.04 L/min", "-0.1438", "< 0.0001".
NUMBER = re.compile(r"^(-?\d+(?:\.(\d+))?)(.*)$")


def _table(page: Any, selector: str) -> tuple[list[str], list[list[str]]]:  # noqa: ANN401 - a Playwright Page
    """Read a table's column headings and its rows of cells."""
    table = page.locator(selector).first
    rows = [row.locator("td, th").all_text_contents() for row in table.locator("tbody tr").all()]
    return table.locator("thead th").all_text_contents(), rows


def _response(page: Any) -> dict[str, Any]:  # noqa: ANN401
    """Read what Learn shows for the response chosen.

    That is each term's coefficient, standard error and p-value, and each factor's (or component's) best
    setting when the page proposes one.
    """
    heads, rows = _table(page, "#analysis table")
    columns = [heads.index(h) for h in ("Coefficient", "Std. error", "p-value")]
    shown: dict[str, Any] = {"coefficients": {row[0]: [row[k] for k in columns] for row in rows}}
    best = "#best-out table.data:not(.known-table)"
    if page.locator(best).count():
        shown["best"] = {row[0]: row[1] for row in _table(page, best)[1]}
    return shown


def _readings(page: Any) -> dict[str, Any]:  # noqa: ANN401
    """Read a study as it opened: its run sheet, then in Learn the model, the transform and every response."""
    settled(page)
    seen = {"headers": headers(page), "run_sheet": run_sheet(page)}
    page.click("#go-learn")
    settled(page)
    seen |= {"model": page.input_value("#model"), "transform": page.input_value("#transform"), "responses": {}}
    names = [name for name, _ in tabs(page)]
    for name in names or [seen["headers"][-1]]:
        if names and dict(tabs(page))[name] != "true":
            pick(page, name)
        seen["responses"][name] = _response(page)
    return seen


def _close(shown: str, kept: str) -> bool:
    """Whether two numbers as shown agree to within one unit in the last digit kept, and the rest matches exactly."""
    a, b = NUMBER.match(shown), NUMBER.match(kept)
    if not (a and b) or a[3] != b[3]:
        return shown == kept
    return abs(float(a[1]) - float(b[1])) <= 10.0 ** -len(b[2] or "") * 1.000001


def _differences(seen: dict[str, Any], kept: dict[str, Any]) -> list[str]:
    """List each coefficient, standard error, p-value or best setting that no longer agrees with what was kept."""
    out = [] if seen.keys() == kept.keys() else [f"responses {sorted(seen)}, kept {sorted(kept)}"]
    for response, then in kept.items():
        now = seen.get(response, {})
        for part in ("coefficients", "best"):
            old, new = then.get(part) or {}, now.get(part) or {}
            out += [f"{response} {part}: {sorted(new)}, kept {sorted(old)}"] if old.keys() != new.keys() else []
            for key in old.keys() & new.keys():
                pairs = zip(
                    new[key] if part == "coefficients" else [new[key]],
                    old[key] if part == "coefficients" else [old[key]],
                    strict=True,
                )
                if not all(_close(s, k) for s, k in pairs):
                    out.append(f"{response} {part} {key}: {new[key]}, kept {old[key]}")
    return out


def _opened(open_context: Any, site: str, link: str) -> Any:  # noqa: ANN401 - a Playwright Page
    """Open a link in a browser of its own, as the reader it was sent to would, once Python has answered."""
    page = open_context().new_page()
    page.goto(site + link)
    page.wait_for_selector(BANNER, state="visible")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    return page


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_a_link_made_by_an_earlier_release_opens_the_same_study(case, open_context, site):
    page = _opened(open_context, site, case["link"])
    assert page.text_content(BANNER).startswith(f"Opened from a link: {case['title']}.")
    seen = _readings(page)
    assert seen["headers"] == case["headers"]
    assert seen["run_sheet"] == case["run_sheet"], "the design and the results typed in are the reader's own"
    assert (seen["model"], seen["transform"]) == (case["model"], case["transform"]), "the analysis the reader chose"
    assert not _differences(seen["responses"], case["responses"]), _differences(seen["responses"], case["responses"])


def test_a_file_written_by_an_earlier_release_opens_every_study_in_it(open_context, site):
    page = open_context().new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    page.set_input_files("#study-upload", str(STUDY_FILE))
    page.wait_for_selector(BANNER, state="visible")
    assert page.text_content(BANNER) == f"{len(CASES)} studies added from {STUDY_FILE.name}."
    page.click("#library summary")
    titles = page.locator("#library-list li .library__study button").all_text_contents()
    assert sorted(titles) == sorted(case["title"] for case in CASES)
    for case in CASES:
        if not page.locator("#library").evaluate("(box) => box.open"):
            page.click("#library summary")  # opening a study folds the list away
        page.locator("#library-list li .library__study button", has_text=case["title"]).click()
        settled(page)
        assert page.text_content(BANNER).startswith(f"Opened {case['title']}"), case["name"]
        assert run_sheet(page) == case["run_sheet"], case["name"]
