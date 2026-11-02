"""The study file: every study kept, written to one file, shared one at a time, and opened in another browser.

The page keeps every study, with no limit on how many. "Download all" writes them to one JSON file
(``web/study-file.js``), each study's Share hands its own file to the share sheet where the browser
can share a file and saves it where it cannot, and the dropzone under the list opens a file in any
browser: a file of one study puts it on the page, a file of several adds them all.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tests.web.e2e._page import generate, practise, saved_studies, settled

if TYPE_CHECKING:
    from pathlib import Path

BANNER = "#campaign-banner"
#: How a study made from the worked example's brief is titled: its response, then its design and factors.
YIELD = "Yield (%), maximise: "


def two_studies(page) -> list[str]:
    """Make two studies from the worked example's brief, each with results, and return their titles."""
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.fill("#design-table input[data-index='0']", "12.5")  # a result typed in: this study is kept
    generate(page, design_type="ccd", n_center_points=2)
    practise(page)
    saved_studies(page, lambda saved: len(saved) == 2 and all(any(s["responses"]) for s in saved))
    page.wait_for_function("document.querySelector('#library-count').textContent === '2'")
    return [f"{YIELD}Full factorial (2-level) in T, P and F", f"{YIELD}Central composite (CCD) in T, P and F"]


def downloaded(page, click: str, tmp_path: Path) -> tuple[str, dict]:
    """Press ``click`` and return the file the page saves: its name and its JSON."""
    with page.expect_download() as download:
        page.click(click)
    target = tmp_path / download.value.suggested_filename
    download.value.save_as(target)
    return download.value.suggested_filename, json.loads(target.read_text(encoding="utf-8"))


def opened_page(open_context, site: str):
    """Open the page in another browser, its storage empty, once Python has answered."""
    page = open_context().new_page()
    page.goto(site)
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    return page


def test_your_studies_and_its_dropzone_show_before_any_study_is_saved(fresh_page):
    page = fresh_page
    assert page.is_visible("#library summary")
    assert page.text_content("#library-count") == "0"
    page.click("#library summary")
    assert page.is_visible("#study-drop")
    assert page.is_disabled("#download-all"), "nothing to download yet"


def test_every_study_is_kept_with_no_limit_on_how_many(fresh_page, site):
    page = fresh_page
    page.evaluate(
        """async (url) => {
            const { saveCampaign } = await import(new URL("store.js", url).href);
            for (let i = 0; i < 40; i++) {
              await saveCampaign({ id: `s${i}`, title: `Study ${i}`, updated: i, created: i });
            }
        }""",
        site,
    )
    assert len(saved_studies(page, lambda saved: len(saved) == 40)) == 40, "the 30 most recent were kept before"


def test_download_all_writes_every_study_to_one_file(page, tmp_path):
    titles = two_studies(page)
    page.click("#library summary")
    name, file = downloaded(page, "#download-all", tmp_path)
    assert name.startswith("factorial-studies-")
    assert name.endswith(".json")
    assert file["format"] == "factorial-studies"
    assert file["version"] == 1
    assert file["app"]["name"] == "Factorial"
    assert sorted(s["study"]["title"] for s in file["studies"]) == sorted(titles)
    assert all(len(s["sha256"]) == 64 and s["id"] and s["created"].endswith("Z") for s in file["studies"])
    assert all(s["study"]["design"]["rows"] for s in file["studies"]), "each study whole, its run sheet too"


def test_a_file_of_several_studies_opens_in_another_browser_and_a_second_time_adds_nothing(
    page, open_context, site, tmp_path
):
    titles = two_studies(page)
    page.click("#library summary")
    name, _ = downloaded(page, "#download-all", tmp_path)
    other = opened_page(open_context, site)
    other.set_input_files("#study-upload", str(tmp_path / name))
    other.wait_for_selector(BANNER, state="visible")
    assert other.text_content(BANNER) == f"2 studies added from {name}."
    assert other.text_content("#library-count") == "2"
    assert sorted(other.locator("#library-list li .library__study button").all_text_contents()) == sorted(titles)
    other.set_input_files("#study-upload", str(tmp_path / name))
    other.wait_for_function(f"document.querySelector('{BANNER}').textContent.includes('already here')")
    assert other.text_content(BANNER) == f"0 studies added from {name}, 2 already here."
    assert other.text_content("#library-count") == "2", "the same studies are not saved twice"


def test_a_study_s_own_file_opens_it_on_the_page_in_another_browser(page, open_context, site, tmp_path):
    titles = two_studies(page)
    page.click("#library summary")
    full_factorial = page.locator("#library-list li", has_text=titles[0])
    button = full_factorial.get_by_role("button", name=f"Download {titles[0]}")
    with page.expect_download() as download:
        button.click()
    name = download.value.suggested_filename
    assert name == "yield-maximise-full-factorial-2-level-in-t-p-and-f.factorial.json"
    download.value.save_as(tmp_path / name)
    other = opened_page(open_context, site)
    other.click("#library summary")
    # Dropped on the dropzone, as a file dragged from the desktop is.
    other.evaluate(
        """(text) => {
            const files = new DataTransfer();
            files.items.add(new File([text], "shared.factorial.json", { type: "application/json" }));
            const zone = document.querySelector("#study-drop");
            zone.dispatchEvent(new DragEvent("dragover", { dataTransfer: files, bubbles: true, cancelable: true }));
            zone.dispatchEvent(new DragEvent("drop", { dataTransfer: files, bubbles: true, cancelable: true }));
        }""",
        (tmp_path / name).read_text(encoding="utf-8"),
    )
    other.wait_for_selector(BANNER, state="visible")
    settled(other)
    assert other.text_content(BANNER) == (
        f"Opened {titles[0]} from shared.factorial.json. It is saved in this browser now, under Your studies."
    )
    assert other.locator("#design-table tbody tr").count() == 8
    assert other.text_content("#library-count") == "1"


def test_share_hands_the_study_s_file_to_the_share_sheet(open_context, site):
    """Where the browser can share a file (an iPad, a phone), Share opens the share sheet with the study's file."""
    context = open_context()
    context.add_init_script(
        """navigator.canShare = () => true;
           navigator.share = async ({ files, title }) => {
             window.__shared = { title, name: files[0].name, text: await files[0].text() };
           };"""
    )
    page = context.new_page()
    page.goto(f"{site}?example=worked")
    page.wait_for_function("window.__pi.count('api_catalogue') === 1 && window.__pi.idle()")
    generate(page, design_type="full_factorial", n_center_points=0)
    practise(page)
    page.wait_for_function("document.querySelector('#library-count').textContent === '1'")
    page.click("#library summary")
    title = f"{YIELD}Full factorial (2-level) in T, P and F"
    page.get_by_role("button", name=f"Share {title}").click()
    page.wait_for_function("window.__shared")
    shared = page.evaluate("window.__shared")
    assert shared["title"] == title
    assert shared["name"] == "yield-maximise-full-factorial-2-level-in-t-p-and-f.factorial.json"
    assert [s["study"]["title"] for s in json.loads(shared["text"])["studies"]] == [title]


@pytest.mark.parametrize(
    ("text", "said"),
    [
        (b"{", "This file could not be opened: it is not JSON."),
        (b'{"format": "a spreadsheet"}', "This file could not be opened: it is not a Factorial study file."),
    ],
    ids=["not-json", "not-a-study-file"],
)
def test_a_file_that_is_not_a_study_file_is_refused_in_a_sentence(fresh_page, text, said):
    page = fresh_page
    page.click("#library summary")
    page.set_input_files("#study-upload", {"name": "notes.json", "mimeType": "application/json", "buffer": text})
    page.wait_for_selector(f"{BANNER}.error", state="visible")
    assert page.text_content(BANNER) == said
    assert page.text_content("#library-count") == "0"
