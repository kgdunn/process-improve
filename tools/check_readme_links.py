"""Check that every link in the README and the sklearn notes still answers.

``tests/test_readme_guards.py`` checks the links into this repository offline, on every pull
request. Links to other sites (the textbook, openmv.net, PyPI, the badges) can break with no
change here, so this script requests each one; ``.github/workflows/readme-links.yml`` runs it
every Monday and on pull requests that touch these pages.

The README's examples read their data from openmv.net, while ``tests/test_readme.py`` checks the
printed numbers against the copies bundled with the package. A reader fetches the live files, so
each one must also hold the same lines as the bundled copy (line endings aside), or the README's
numbers would not reproduce for that reader while CI stayed green.

    python tools/check_readme_links.py      # exit status 1 if any link fails
"""

from __future__ import annotations

import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ("README.md", "SKLEARN_COMPATIBILITY.md")
DATASETS = ROOT / "src" / "process_improve" / "datasets"
URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")
#: Some hosts refuse urllib's default agent, so the request says what it is.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; process-improve README link check)"}
ATTEMPTS = 3


def page_urls() -> list[str]:
    """Return every distinct URL in the pages, in order of first appearance."""
    urls: list[str] = []
    for page in PAGES:
        for url in URL.findall((ROOT / page).read_text(encoding="utf-8")):
            cleaned = url.rstrip(".,;:")
            if cleaned not in urls:
                urls.append(cleaned)
    return urls


def fetch(url: str) -> bytes:
    """Return the body of ``url``, retrying a server error, a rate limit or a timeout."""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as response:  # noqa: S310 - https URLs from our own pages
                return response.read()
        except urllib.error.HTTPError as error:
            if attempt == ATTEMPTS or (error.code < 500 and error.code != 429):
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == ATTEMPTS:
                raise
        time.sleep(2**attempt)
    msg = f"no attempt was made to fetch {url}"
    raise RuntimeError(msg)


def bundled_copy(url: str) -> Path | None:
    """Return the dataset bundled with the package that a ``.csv`` URL serves, if exactly one has its name."""
    if not url.endswith(".csv"):
        return None
    matches = list(DATASETS.rglob(url.rsplit("/", 1)[-1]))
    return matches[0] if len(matches) == 1 else None


def main() -> int:
    """Request every URL in the pages and report the ones that fail; return the exit status."""
    urls, failures = page_urls(), []
    for url in urls:
        try:
            body = fetch(url)
        except (urllib.error.URLError, TimeoutError) as error:  # HTTPError is a URLError
            failures.append(f"{url}\n    {error}")
            continue
        copy = bundled_copy(url)
        if copy is not None and body.decode("utf-8", "replace").splitlines() != copy.read_text("utf-8").splitlines():
            reason = f"its data differ from {copy.relative_to(ROOT)}, so a reader would not get the README's numbers"
            failures.append(f"{url}\n    {reason}")
    for failure in failures:
        print(failure)
    print(f"{len(urls) - len(failures)} of {len(urls)} links answered.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
