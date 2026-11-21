"""Turn the browser app's copy on GitHub Pages into a signpost to https://factori.al/ (#1237).

The copy at ``/app`` was rebuilt with every deploy, and for ten minutes after one a reader's cached scripts could run
with the new Python, which factori.al's server does not allow. Older links and releases of the package still point
here, so the copy stays, with ``pages_moved.js`` in each page: a browser with no study saved at this address goes on to
the same page at factori.al, and one with studies is told how to take them along. factori.al's build is not touched.

Usage::

    python scripts/move_pages_copy.py docs/_build/html/app
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SCRIPT = (Path(__file__).with_name("pages_moved.js")).read_text(encoding="utf-8")

#: What marks a page done, so a second run leaves it as it is.
MARK = "Factorial has moved to https://factori.al/ (#1237)"


def mark(folder: Path) -> int:
    """Put the signpost first in each page's ``<head>`` under ``folder``; return how many pages were marked."""
    marked = 0
    for page in sorted(folder.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        if MARK in text:
            continue
        head = re.search(r"<head\b[^>]*>", text, re.IGNORECASE)
        if head is None:
            raise ValueError(f"{page} has no <head> to put the signpost in.")
        at = head.end()
        page.write_text(f"{text[:at]}\n<script>\n{SCRIPT}</script>{text[at:]}", encoding="utf-8")
        marked += 1
    return marked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="the copy of the app GitHub Pages serves at /app")
    folder = parser.parse_args().folder
    print(f"Marked {mark(folder)} pages of {folder} as moved to factori.al.")
