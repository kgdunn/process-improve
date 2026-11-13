#!/usr/bin/python3 -I
"""Count how factori.al is used, from Caddy's access log: no cookies, no tracking script, no database.

The log has one line per request, written with the reader's address cut to its network
(``factorial.caddy``), and kept for 366 days. From it, for each day (UTC), the report counts:

    pages      each page loaded in a browser: the app (Home), the concept map (Map), the pages
               of the design catalog (Catalog), the multivariate page, About, and any other page
    readers    distinct networks and browsers among them: approximate, since one network can
               hold many readers and one reader can move between networks
    python     Python started in the app: its worker reads runtime.json, past every cache, as it
               starts
    previews   the link card fetched by a chat or a social site, to show a link to the app

Robots, link-preview fetchers and the deploy's own checks (headless Chromium, Python's urllib)
are not readers. Run it on the server, as a user that can read the log:

    sudo factorial-usage                        # the last 28 days, a row a day
    sudo factorial-usage --days 366 --by month  # a year, a row a month

The log forgets a day after 366 days, so ``--keep`` adds each finished day's counts, and nothing
else, to a file of its own (``/var/lib/factorial/usage.csv``). A daily job that ``setup.sh``
installs runs it, and the report reads the file for the days the log no longer holds.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

LOGS = "/var/log/caddy/factorial-access*.log*"
#: Each finished day's counts, kept after the log forgets the day.
KEPT = "/var/lib/factorial/usage.csv"
SITE = "factori.al"
#: The site's pages, in the order the report lists them, each by the paths it is served at. The design catalog is
#: every page under /catalog/; a page no name claims is counted as Other. factori.al/map is redirected to
#: factori.al/map/, so it is not counted twice.
PAGES = {
    "Home": {"/", "/index.html"},
    "Map": {"/map/", "/map/index.html"},
    "Catalog": set(),
    "Multivariate": {"/mva/", "/mva/index.html"},
    "About": {"/about.html"},
    "Other": set(),
}
#: What the report counts each day, in the order of its columns and of the file that keeps them.
COLUMNS = (*PAGES, "Readers", "Python", "Previews")
#: Python is started by the app's worker, which reads runtime.json past every cache as it starts. The app's own
#: Python used to be fetched as bootstrap.py; a build now ships it in an archive kept for a year, which a returning
#: browser does not ask for again. The page reads runtime.json too, for the copy it keeps for use without a network,
#: and so does the multivariate page's worker: the Referer, which names the worker, tells those apart.
PYTHON = "/runtime.json"
WORKER = "/worker.js"
CARD = "/og-card.png"
#: A bar for each row of the report, from none to the most a page had in a row.
BARS = " ▁▂▃▄▅▆▇█"
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

    pages: Counter[str] = field(default_factory=Counter)
    readers: set[tuple[str, str]] = field(default_factory=set)
    python: int = 0
    previews: int = 0

    def counts(self) -> dict[str, int]:
        """Return the day's counts, as the report's columns and the file that keeps them name them."""
        pages = {name: self.pages[name] for name in PAGES}
        return {**pages, "Readers": len(self.readers), "Python": self.python, "Previews": self.previews}


def page(path: str) -> str | None:
    """Name the page a path is, or None for a file that is no page: a script, an image, the data a page reads."""
    for name, paths in PAGES.items():
        if path in paths:
            return name
    if not path.endswith(("/", ".html")):
        return None
    return "Catalog" if path.startswith("/catalog/") else "Other"


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
    """Add up the requests made since ``since``: what each day saw, and which sites readers came from."""
    days: dict[str, Day] = defaultdict(Day)
    referrers: Counter[str] = Counter()
    for entry in entries:
        when = datetime.fromtimestamp(entry.get("ts", 0), UTC)
        request = entry["request"]
        if when < since or request.get("method") != "GET" or entry.get("status") not in {200, 304}:
            continue
        day = days[when.strftime("%Y-%m-%d")]
        path, agent = urlsplit(request.get("uri", "")).path, header(request, "User-Agent")
        if path == CARD and PREVIEWER.search(agent):
            day.previews += 1
        elif NOT_A_READER.search(agent):
            continue
        elif name := page(path):
            day.pages[name] += 1
            day.readers.add((request.get("client_ip") or request.get("remote_ip", ""), agent))
            came_from = urlsplit(header(request, "Referer")).hostname
            if came_from and came_from != SITE and not came_from.endswith(f".{SITE}"):
                referrers[came_from] += 1
        elif path == PYTHON and urlsplit(header(request, "Referer")).path == WORKER:
            day.python += 1
    return days, referrers


def kept(path: Path) -> dict[str, dict[str, int]]:
    """Return the counts the file keeps, by day; none where there is no file. A column it lacks counts nothing."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["Day"]: {name: int(row.get(name) or 0) for name in COLUMNS} for row in csv.DictReader(file)}


def keep(path: Path, days: dict[str, Day], today: date) -> list[str]:
    """Add to the file the counts of each day before ``today`` that it lacks, and return those days.

    The file is written whole, in the order of the days and with the columns of now, so a column added later is
    filled with zeros for the days before it. A day the file keeps is never written again: once the log has
    forgotten part of a day, the file's count is the whole one. The oldest day the log holds when the file is
    first written may be partial.
    """
    rows = kept(path)
    added = sorted(name for name in days if name < today.isoformat() and name not in rows)
    if added:
        rows |= {name: days[name].counts() for name in added}
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        with part.open("w", newline="", encoding="utf-8") as file:
            out = csv.DictWriter(file, fieldnames=["Day", *COLUMNS])
            out.writeheader()
            out.writerows({"Day": name, **rows[name]} for name in sorted(rows))
        part.replace(path)
    return added


def start(when: date, by: str) -> date:
    """Return the first day of the row a day falls in: the day itself, the Monday of its week, or its month's first."""
    if by == "week":
        return when - timedelta(days=when.weekday())
    return when.replace(day=1) if by == "month" else when


def period(when: date, by: str) -> str:
    """Name the row a day falls in: the day itself, the Monday of its week, or its month."""
    return start(when, by).strftime("%Y-%m") if by == "month" else start(when, by).isoformat()


def bars(values: list[float]) -> str:
    """Draw values as a line of bars, the largest the tallest; a row with none is a blank."""
    top = max(values, default=0) or 1
    return "".join(BARS[0] if not v else BARS[max(1, round(v / top * (len(BARS) - 1)))] for v in values)


def report(
    days: dict[str, Day],
    keeps: dict[str, dict[str, int]],
    referrers: Counter[str],
    first: date,
    last: date,
    by: str = "day",
) -> str:
    """Lay out a row per day, week or month from ``first`` to ``last``, the period's totals, and each page's trend.

    A day the file keeps is read from it, and any other from the log. Readers are counted a day: a week's or a
    month's row, and the period's, give the average a day. A period of weeks or months starts with a whole one,
    and the row of the week or month still going is marked as so far.
    """
    first = start(first, by)
    rows: dict[str, list[dict[str, int]]] = defaultdict(list)
    for n in range((last - first).days + 1):
        when = first + timedelta(days=n)
        rows[period(when, by)].append(keeps.get(when.isoformat()) or days.get(when.isoformat(), Day()).counts())

    def summed(counts: list[dict[str, int]]) -> list[str]:
        each = [str(sum(c[name] for c in counts)) for name in COLUMNS]
        readers = sum(c["Readers"] for c in counts) / len(counts)
        each[COLUMNS.index("Readers")] = f"{readers:.0f}" if by == "day" and len(counts) == 1 else f"{readers:.1f}"
        return each

    head = {"day": "Day", "week": "Week of", "month": "Month"}[by]
    going = by != "day" and start(last + timedelta(days=1), by) == start(last, by)  # the last row is not over
    named = [f"{name}*" if going and name == period(last, by) else name for name in rows]
    table = [(head, *COLUMNS), *((name, *summed(counts)) for name, counts in zip(named, rows.values(), strict=True))]
    table.append(("All", *summed([c for counts in rows.values() for c in counts])))
    width = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    lines = [
        "  ".join([row[0].ljust(width[0]), *(c.rjust(w) for c, w in zip(row[1:], width[1:], strict=True))])
        for row in table
    ]
    out = [f"{SITE}, {first} to {last} (UTC), a row a {by}", "", *lines, ""]
    out.append(
        "Readers are distinct networks and browsers in a day: approximate. Over several days, the average a day."
    )
    if going:
        out.append(f"* The {by} so far, to {last}.")
    if len(rows) > 1:
        out += ["", f"Each page over the period, a bar a {by}:"]
        for name in [*PAGES, "Readers"]:
            values = [
                sum(c[name] for c in counts) / len(counts) if name == "Readers" else sum(c[name] for c in counts)
                for counts in rows.values()
            ]
            out.append(f"  {name:<13} {bars(values)}".rstrip())  # a page no one opened shows its name alone
    if referrers:
        out += ["", "Where readers came from", *(f"  {host:<30} {n:>6}" for host, n in referrers.most_common(10))]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """Print the report for the last ``--days`` days, today included; or, with ``--keep``, keep the finished days."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=28, help="how many days to report, today included (%(default)s)")
    parser.add_argument("--by", choices=("day", "week", "month"), default="day", help="a row a day, week or month")
    parser.add_argument(
        "--keep", action="store_true", help="add the finished days the file lacks to it, and print nothing"
    )
    parser.add_argument("--log", default=LOGS, help="the access log's files, a pattern in the last part (%(default)s)")
    parser.add_argument("--kept", default=KEPT, help="the file that keeps each finished day's counts (%(default)s)")
    args = parser.parse_args(argv)
    pattern = Path(args.log)
    paths = list(pattern.parent.glob(pattern.name))
    if not paths:
        print(f"no log files match {args.log}", file=sys.stderr)
        return 1
    last = datetime.now(UTC).date()
    if args.keep:
        days, _ = tally(requests(lines(paths)), datetime.fromtimestamp(0, UTC))
        keep(Path(args.kept), days, last)
        return 0
    first = last - timedelta(days=args.days - 1)
    days, referrers = tally(requests(lines(paths)), datetime.combine(first, time.min, tzinfo=UTC))
    print(report(days, kept(Path(args.kept)), referrers, first, last, args.by))
    return 0


if __name__ == "__main__":
    sys.exit(main())
