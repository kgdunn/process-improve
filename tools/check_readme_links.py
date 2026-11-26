"""Check that every link in the README, the sklearn notes and factori.al's page for learning still answers.

``tests/test_readme_guards.py`` checks the links into this repository offline, on every pull
request. Links to other sites (the textbook, openmv.net, PyPI, the badges) can break with no
change here, so this script requests each one; ``.github/workflows/readme-links.yml`` runs it
every Monday and on pull requests that touch these pages.

The page for learning (``web/map/index.html``) links to the book's chapter, to the course and to
the course's videos on YouTube. YouTube refuses a script's request for a video's page, and answers
a browser's even when the video is gone, so a video and its playlist are asked for through
YouTube's oEmbed endpoint, which answers a script and says when either is gone.

The README's examples read their data from openmv.net, while ``tests/test_readme.py`` checks the
printed numbers against the copies bundled with the package. A reader fetches the live files, so
each one must also hold the same lines as the bundled copy (line endings aside), or the README's
numbers would not reproduce for that reader while CI stayed green.

    python tools/check_readme_links.py      # exit status 1 if any link fails
"""

from __future__ import annotations

import html
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = ("README.md", "SKLEARN_COMPATIBILITY.md", "web/map/index.html")
DATASETS = ROOT / "src" / "process_improve" / "datasets"
URL = re.compile(r"https?://[^\s<>\"'`)\]}]+")
#: Some hosts refuse urllib's default agent, so the request says what it is.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; process-improve README link check)"}
ATTEMPTS = 3
#: Links to this repository's files are not requested: ``tests/test_readme_guards.py`` finds each one on disk,
#: and github.com answers a script's request for a file page with 503 Service Unavailable. Nor are the badges of
#: this repository's own workflows: the test finds each workflow's file, and a badge of a workflow added in a
#: pull request answers 404 until the workflow is on main.
REPOSITORY_FILE = re.compile(
    r"https://(?:github\.com/kgdunn/process-improve/(?:blob|tree)/main/|raw\.githubusercontent\.com/kgdunn/process-improve/main/"
    r"|github\.com/kgdunn/process-improve/actions/workflows/[^/]+/badge\.svg)"
)


#: A video or a playlist on YouTube, which is asked for through YouTube's oEmbed endpoint: see ``to_request``.
YOUTUBE = re.compile(r"https://www\.youtube\.com/(?:watch|playlist)\?")
OEMBED = "https://www.youtube.com/oembed?format=json&url="


def text_of(page: str) -> str:
    """Return a page's text, an HTML page's with its entities read: ``&amp;`` in a link's address is ``&``."""
    text = (ROOT / page).read_text(encoding="utf-8")
    return html.unescape(text) if page.endswith(".html") else text


def page_urls() -> list[str]:
    """Return every distinct URL in the pages, except links to this repository's files, in order of first appearance."""
    urls: list[str] = []
    for page in PAGES:
        for url in URL.findall(text_of(page)):
            cleaned = url.rstrip(".,;:")
            if cleaned not in urls and not REPOSITORY_FILE.match(cleaned):
                urls.append(cleaned)
    return urls


def to_request(url: str) -> list[str]:
    """Return the addresses whose answers say that ``url`` is alive: itself, unless it is a video on YouTube.

    YouTube answers a script's request for a video's page with 429, and a browser's with 200 even when the video
    is gone. Its oEmbed endpoint answers a script, with 400 or 404 for a video or a playlist that is gone, so a
    link to a video in a playlist is two requests: one for the video, one for the playlist.
    """
    if not YOUTUBE.match(url):
        return [url]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    parts = [f"https://www.youtube.com/watch?v={video}" for video in query.get("v", [])]
    parts += [f"https://www.youtube.com/playlist?list={playlist}" for playlist in query.get("list", [])]
    return [OEMBED + urllib.parse.quote(part, safe="") for part in parts]


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
            body = b"".join(fetch(address) for address in to_request(url))
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
