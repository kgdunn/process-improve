#!/usr/bin/python3 -I
"""Count how factori.al is used, from Caddy's access log: no cookies, no tracking script, no database.

The log has one line per request, written with the reader's address cut to its network
(``factorial.caddy``). From it, for each day (UTC):

    visits     the page loaded in a browser
    catalog    the design catalog (factori.al/catalog/) loaded in a browser: a visit to the app,
               counted apart from the page's
    readers    distinct networks and browsers among the visits and the catalog's: approximate, since
               one network can hold many readers and one reader can move between networks
    python     Python started in the page, which fetches bootstrap.py once Pyodide has loaded
    previews   the link card fetched by a chat or a social site, to show a link to the app

Robots, link-preview fetchers and the deploy's own checks (headless Chromium, Python's urllib)
are not visits. Run it on the server, as a user that can read the log:

    sudo factorial-usage --days 28
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

LOGS = "/var/log/caddy/factorial-access*.log*"
SITE = "factori.al"
PAGE = {"/", "/index.html"}
#: The design catalog's page. factori.al/catalog is redirected to factori.al/catalog/, so it is not counted twice.
CATALOG = {"/catalog/", "/catalog/index.html"}
PYTHON = "/bootstrap.py"
CARD = "/og-card.png"
#: User agents that are not a reader at a browser: robots, link-preview fetchers, monitors, and
#: the deploy's own checks, which run headless Chromium and Python's urllib.
NOT_A_READER = re.compile(
    r"bot|crawl|spider|slurp|preview|externalhit|whatsapp|telegram|discord|slack|skype|embedly|iframely|"
    r"mastodon|headless|lighthouse|python|curl|wget|httpx|go-http|okhttp|java/|libwww|scrapy|uptime|^$",
    re.IGNORECASE,
)
#: The fetchers a chat or a social site sends to show a link's card.
PREVIEWER = re.compile(
    r"slack|externalhit|facebook|twitter|whatsapp|telegram|discord|linkedin|skype|embedly|iframely|"
    r"mastodon|applebot|teams|redditbot|pinterest",
    re.IGNORECASE,
)


@dataclass
class Day:
    """What one day's requests add up to."""

    visits: int = 0
    catalog: int = 0
    readers: set[tuple[str, str]] = field(default_factory=set)
    python: int = 0
    previews: int = 0


def lines(paths: Iterable[Path]) -> Iterator[str]:
    """Yield every line of the log files, the rolled ones gzipped."""
    for path in sorted(paths):
        with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8", errors="replace") as log:
            yield from log


def requests(log: Iterable[str]) -> Iterator[dict]:
    """Yield each logged request, skipping a line a rotation cut short."""
    for line in log:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("request"), dict):
            yield entry


def header(request: dict, name: str) -> str:
    """Return the first value of a request header, or an empty string."""
    return (request.get("headers", {}).get(name) or [""])[0]


def tally(entries: Iterable[dict], since: datetime) -> tuple[dict[str, Day], Counter[str]]:
    """Add up the requests made since ``since``: what each day saw, and which sites visits came from."""
    days: dict[str, Day] = defaultdict(Day)
    referrers: Counter[str] = Counter()
    for entry in entries:
        when = datetime.fromtimestamp(entry.get("ts", 0), timezone.utc)
        request = entry["request"]
        if when < since or request.get("method") != "GET" or entry.get("status") not in {200, 304}:
            continue
        day = days[when.strftime("%Y-%m-%d")]
        path, agent = urlsplit(request.get("uri", "")).path, header(request, "User-Agent")
        if path == CARD and PREVIEWER.search(agent):
            day.previews += 1
        elif NOT_A_READER.search(agent):
            continue
        elif path in PAGE or path in CATALOG:
            if path in CATALOG:
                day.catalog += 1
            else:
                day.visits += 1
            day.readers.add((request.get("client_ip") or request.get("remote_ip", ""), agent))
            came_from = urlsplit(header(request, "Referer")).hostname
            if came_from and came_from != SITE and not came_from.endswith(f".{SITE}"):
                referrers[came_from] += 1
        elif path == PYTHON:
            day.python += 1
    return days, referrers


def report(days: dict[str, Day], referrers: Counter[str], first: date, last: date) -> str:
    """Lay out one row per day from ``first`` to ``last``, the period's totals, and where visits came from."""
    rows = [("Day", "Visits", "Catalog", "Readers", "Python", "Previews")]
    everyone: set[tuple[str, str]] = set()
    totals = [0, 0, 0, 0]
    for n in range((last - first).days + 1):
        name = (first + timedelta(days=n)).isoformat()
        day = days.get(name, Day())
        everyone |= day.readers
        counts = [day.visits, day.catalog, day.python, day.previews]
        totals = [a + b for a, b in zip(totals, counts, strict=True)]
        rows.append((name, *map(str, counts[:2]), str(len(day.readers)), *map(str, counts[2:])))
    rows.append(("All", *map(str, totals[:2]), str(len(everyone)), *map(str, totals[2:])))
    width = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    table = [
        "  ".join([row[0].ljust(width[0]), *(cell.rjust(w) for cell, w in zip(row[1:], width[1:], strict=True))])
        for row in rows
    ]
    out = [
        f"{SITE}, {first} to {last} (UTC)",
        "",
        *table,
        "",
        "Readers are distinct networks and browsers: approximate.",
    ]
    if referrers:
        out += ["", "Where visits came from", *(f"  {host:<30} {n:>6}" for host, n in referrers.most_common(10))]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Print the report for the last ``--days`` days, today included."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=28, help="how many days to report, today included (%(default)s)")
    parser.add_argument("--log", default=LOGS, help="the access log's files, a pattern in the last part (%(default)s)")
    args = parser.parse_args(argv)
    pattern = Path(args.log)
    paths = list(pattern.parent.glob(pattern.name))
    if not paths:
        print(f"no log files match {args.log}", file=sys.stderr)
        return 1
    last = datetime.now(timezone.utc).date()
    first = last - timedelta(days=args.days - 1)
    days, referrers = tally(requests(lines(paths)), datetime.combine(first, time.min, tzinfo=timezone.utc))
    print(report(days, referrers, first, last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
