"""``deploy/factorial/usage.py``: how factori.al is used, counted from Caddy's access log, and the counts it keeps."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "usage.py"
BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari"
PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
HEADLESS = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.0.0 Safari"
SINCE = datetime(2026, 10, 1, tzinfo=UTC)


def load():
    spec = importlib.util.spec_from_file_location("factorial_usage", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


usage = load()


def logged(uri: str, agent: str = BROWSER, *, when: str = "2026-10-12T10:00", **fields) -> dict:
    """Return one request as Caddy logs it, its address already masked to the network."""
    headers = {"User-Agent": [agent], **({"Referer": [fields["referer"]]} if "referer" in fields else {})}
    ip = fields.get("ip", "203.0.113.0")
    request = {"remote_ip": ip, "client_ip": ip, "proto": "HTTP/2.0", "method": fields.get("method", "GET"),
               "host": "factori.al", "uri": uri, "headers": headers}  # fmt: skip
    ts = datetime.fromisoformat(when).replace(tzinfo=UTC).timestamp()
    return {"level": "info", "ts": ts, "logger": "http.log.access.log0", "msg": "handled request",
            "request": request, "status": fields.get("status", 200), "size": 10,
            "duration": fields.get("duration", 0.001)}  # fmt: skip


def visit(uri: str, agent: str = BROWSER, **fields) -> list[dict]:
    """Return a reader's visit as Caddy logs it: the page, then the stylesheet the reader's browser asks for."""
    sheet = {**fields, "referer": f"https://factori.al{uri.partition('?')[0]}"}
    return [logged(uri, agent, **fields), logged("/style.css", agent, **sheet)]


def day(*entries: dict, on: str = "2026-10-12"):
    days, _ = usage.tally(entries, SINCE)
    return days.get(on, usage.Day())


#: The archive of the app's own Python, in the folder a build names by digest: the last file its worker asks for.
ARCHIVE = "/wheels/0123456789abcdef/bootstrap.zip"


#: What a fetch from the app's worker, from the app itself and from the multivariate page's worker says it came from.
FROM_THE_WORKER, FROM_THE_PAGE, FROM_THE_MVA_WORKER = (
    "https://factori.al/worker.js",
    "https://factori.al/",
    "https://factori.al/mva/worker.js",
)


def test_a_view_of_the_app_is_its_page_loaded_in_a_browser():
    seen = day(
        logged("/"),
        logged("/index.html", status=304),  # coming back: the page is revalidated
        logged("/?utm_source=book"),
        logged("/", method="HEAD"),
        logged("/missing", status=404),
        logged("/", status=500),
        logged("/app.js"),
    )
    assert seen.pages == {"Home": 3}


def test_each_page_of_the_site_is_counted_under_its_own_name():
    seen = day(
        logged("/map/"),
        logged("/map/index.html", status=304),
        logged("/map", status=308),  # redirected to /map/, which is counted
        logged("/map/graph.json"),  # the page's data, not a page
        logged("/catalog/"),
        logged("/catalog/?q=dsd"),
        logged("/catalog/books/montgomery.html"),
        logged("/catalog/index.json"),
        logged("/mva/"),
        logged("/about.html"),  # the page's old address, served before the server redirects it
        logged("/about.html", status=301),  # redirected to the page's new address, which is counted
        logged("/technical-details-how-factorial-works.html"),
        logged("/features.html"),
        logged("/features.html", status=304),
        logged("/FEATURES.md"),  # the document the page reads, not a page
        logged("/a-page-added-later/"),
        logged("/style.css"),
        logged("/no-such-page/", status=404),
    )
    assert seen.pages == {"Map": 2, "Catalog": 3, "Multivariate": 1, "About": 2, "Features": 2, "Other": 1}
    assert seen.counts() == {
        "Home": 0, "Map": 2, "Catalog": 3, "Multivariate": 1, "About": 2, "Features": 2, "Other": 1,
        "Readers": 1, "Python": 0, "Previews": 0,
    }  # fmt: skip
    assert tuple(seen.counts()) == usage.COLUMNS, "in the order of the report's columns"


@pytest.mark.parametrize(
    "agent",
    [
        "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
        "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
        "WhatsApp/2.23.20.0",
        HEADLESS,
        "Python-urllib/3.13",
        "curl/8.5.0",
        "",
    ],
    ids=["search", "slack", "facebook", "whatsapp", "deploy-check", "deploy-wait", "curl", "none"],
)
def test_robots_previews_and_the_deploys_own_checks_are_not_readers(agent):
    seen = day(logged("/", agent), logged("/map/", agent), logged("/runtime.json", agent, referer=FROM_THE_WORKER))
    assert (sum(seen.pages.values()), seen.python, len(seen.readers)) == (0, 0, 0)


def test_readers_are_distinct_networks_and_browsers_whichever_pages_they_open():
    seen = day(
        logged("/"),
        logged("/", when="2026-10-12T15:00"),
        logged("/map/"),  # the same reader, on the map
        logged("/", PHONE),
        logged("/catalog/", ip="198.51.100.0"),
    )
    assert seen.pages == {"Home": 3, "Map": 1, "Catalog": 1}
    assert len(seen.readers) == 3


def test_python_started_is_the_app_s_worker_reading_runtime_json():
    """A build ships the app's Python in an archive kept for a year: a returning browser no longer asks for it.

    The worker reads runtime.json past every cache each time it starts Python, so that is what is counted. The app
    reads the file too, for the copy it keeps, and so does the multivariate page's worker: neither is a start here.
    """
    seen = day(
        logged("/"),
        logged("/runtime.json", referer=FROM_THE_WORKER),
        logged("/runtime.json", PHONE, referer=FROM_THE_WORKER),
        logged("/runtime.json", HEADLESS, referer=FROM_THE_WORKER),  # the deploy's own check
        logged("/runtime.json", referer=FROM_THE_PAGE),
        logged("/runtime.json", referer=FROM_THE_MVA_WORKER),
        logged("/runtime.json"),  # no Referer at all: not the worker
        logged("/bootstrap.py", referer=FROM_THE_WORKER),  # an older build's worker read this too, after runtime.json
        logged(ARCHIVE, referer=FROM_THE_WORKER),
    )
    assert seen.python == 2


def opened_and_started(opened: str, ready: str, agent: str = BROWSER, *, duration: float = 0, **fields) -> list[dict]:
    """Return a start as the log holds it: the app's page, its response ended at ``opened`` after ``duration``
    seconds, and its Python's archive downloaded by ``ready``. Pyodide, from jsDelivr, is not in this log.
    """
    return [
        logged("/", agent, when=opened, duration=duration, **fields),
        logged(ARCHIVE, agent, when=ready, referer=FROM_THE_WORKER, **fields),
    ]


def test_a_start_is_timed_from_when_the_app_began_to_load_to_the_end_of_its_pythons_download():
    # The page's line is written when its response ended, half a second after it began.
    seen = day(*opened_and_started("2026-10-12T10:00:00", "2026-10-12T10:00:07", duration=0.5))
    assert seen.starts == {"First visit": [pytest.approx(7.5)]}


def test_a_readers_first_timed_start_is_a_first_visit_and_any_later_one_returning():
    seen = day(
        *opened_and_started("2026-10-12T10:00:00", "2026-10-12T10:00:09"),
        *opened_and_started("2026-10-12T10:00:30", "2026-10-12T10:00:42", PHONE),  # another reader's first
        *opened_and_started("2026-10-12T15:00:00", "2026-10-12T15:00:03"),  # after a deploy: Pyodide is kept
    )
    assert seen.starts == {"First visit": [pytest.approx(9), pytest.approx(12)], "Returning": [pytest.approx(3)]}


def test_a_start_is_not_timed_without_an_opening_before_it_or_long_after_it():
    seen = day(
        logged(ARCHIVE, when="2026-10-12T09:00", referer=FROM_THE_WORKER),  # no opening of the app before it
        *opened_and_started("2026-10-12T10:00", "2026-10-12T10:11"),  # the page was left open for 11 minutes
        logged(ARCHIVE, when="2026-10-12T10:12", referer=FROM_THE_WORKER),  # that opening was paired already
        *opened_and_started("2026-10-12T11:00", "2026-10-12T11:00:08", HEADLESS),  # the deploy's own check
        logged("/map/", when="2026-10-12T12:00"),  # a page that is not the app starts no Python
        logged(ARCHIVE, when="2026-10-12T12:00:05", referer=FROM_THE_WORKER),
    )
    assert seen.starts == {}


def test_the_report_gives_the_median_start_and_the_time_9_in_10_took_no_longer_than():
    first_visits = [8, 4, 6, 30, 10]  # seconds, a reader each; 9 in 10 of five is the fifth fastest, the slowest
    entries = [
        entry
        for n, seconds in enumerate(first_visits)
        for entry in opened_and_started("2026-10-11T10:00", f"2026-10-11T10:00:{seconds:02}", ip=f"198.51.100.{n}")
    ]
    entries += opened_and_started("2026-10-12T10:00", "2026-10-12T10:00:02", ip="198.51.100.0")  # two returning
    entries += opened_and_started("2026-10-12T11:00", "2026-10-12T11:00:03", ip="198.51.100.1")
    days, referrers = usage.tally(entries, SINCE)
    lines = usage.report(days, {}, referrers, date(2026, 10, 10), date(2026, 10, 12)).splitlines()
    assert lines[-3:] == [
        "From opening the app to Python ready, where the browser downloaded the app's Python:",
        "  First visit  median 8.0 s, 9 in 10 within 30.0 s, of 5",
        "  Returning    median 2.5 s, 9 in 10 within 3.0 s, of 2",
    ]
    assert lines[-4] == "", "a blank line sets it apart"


#: The process-improve wheel, beside the archive: the worker asks for it once Pyodide and its packages are in.
WHEEL = "/wheels/0123456789abcdef/process_improve-2.7.0-py3-none-any.whl"


def started_in_parts(opened: str, worker: str, wheel: str, ready: str, **fields) -> list[dict]:
    """Return a start with the worker's requests that split it: runtime.json as it starts, then the wheel."""
    return [
        logged("/", when=opened, duration=0, **fields),
        logged("/runtime.json", when=worker, referer=FROM_THE_WORKER, duration=0, **fields),
        logged(WHEEL, when=wheel, referer=FROM_THE_WORKER, duration=0, **fields),
        logged(ARCHIVE, when=ready, referer=FROM_THE_WORKER, **fields),
    ]


def test_a_start_splits_into_the_page_pyodide_and_process_improve():
    seen = day(
        *started_in_parts("2026-10-12T10:00:00", "2026-10-12T10:00:01", "2026-10-12T10:00:07", "2026-10-12T10:00:08"),
        logged("/runtime.json", when="2026-10-12T10:00:02", referer=FROM_THE_PAGE),  # the page's own copy: no mark
    )
    assert seen.starts == {"First visit": [pytest.approx(8)]}
    assert seen.parts == {"First visit": [(pytest.approx(1), pytest.approx(6), pytest.approx(1))]}


def test_a_start_whose_worker_requests_are_missing_is_timed_but_not_split():
    seen = day(
        logged("/", when="2026-10-12T10:00:00", duration=0),
        logged(WHEEL, when="2026-10-12T10:00:07", referer=FROM_THE_WORKER),  # no runtime.json from the worker
        logged(ARCHIVE, when="2026-10-12T10:00:08", referer=FROM_THE_WORKER),
        *started_in_parts("2026-10-12T11:00:00", "2026-10-12T11:00:01", "2026-10-12T11:00:07", "2026-10-12T11:00:08"),
    )
    assert seen.starts == {"First visit": [pytest.approx(8)], "Returning": [pytest.approx(8)]}
    assert seen.parts == {"Returning": [(pytest.approx(1), pytest.approx(6), pytest.approx(1))]}, "the second only"


def test_the_report_gives_the_median_of_each_part_of_a_start():
    entries = [
        *started_in_parts("2026-10-12T10:00:00", "2026-10-12T10:00:01", "2026-10-12T10:00:09", "2026-10-12T10:00:10"),
        *started_in_parts(
            "2026-10-12T10:00:00", "2026-10-12T10:00:02", "2026-10-12T10:00:06", "2026-10-12T10:00:08", ip="192.0.2.0"
        ),
    ]
    days, referrers = usage.tally(entries, SINCE)
    lines = usage.report(days, {}, referrers, date(2026, 10, 12), date(2026, 10, 12)).splitlines()
    assert lines[-2:] == [
        "Where the time went, the median of each part (medians, so they need not add up to the one above):",
        "  First visit  page 1.5 s, Pyodide 6.0 s, process-improve 1.5 s, of 2",
    ]


def test_a_link_card_fetched_by_a_chat_or_social_site_is_a_preview():
    seen = day(
        logged("/og-card.png", "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)"),
        logged("/og-card.png", "Twitterbot/1.0"),
        logged("/og-card.png", "LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)"),
        logged("/og-card.png"),  # a reader opening the image is neither a preview nor a page
    )
    assert (seen.previews, sum(seen.pages.values())) == (3, 0)


def test_readers_name_the_sites_they_came_from_but_not_factori_al_itself():
    _, referrers = usage.tally(
        [
            *visit("/", referer="https://learnche.org/pid/design-analysis-experiments/index"),
            *visit("/map/", PHONE, referer="https://learnche.org/pid/contents"),
            *visit("/", referer="https://github.com/kgdunn/process-improve"),
            *visit("/", referer="https://factori.al/"),
            *visit("/catalog/", referer="https://www.factori.al/"),
            *visit("/"),
        ],
        SINCE,
    )
    assert dict(referrers) == {"learnche.org": 2, "github.com": 1}


def test_a_robot_that_names_a_site_to_be_listed_is_not_where_a_reader_came_from():
    """Referrer spam: a robot asks for a page alone, with a made-up Referer, to be listed in reports like this one.

    A reader's browser goes on to ask for the page's stylesheet and scripts, so only such a visit says where it
    came from. The robot's page views are counted as before: it is the list of places that it is kept out of.
    """
    robot = {"ip": "198.51.100.0"}
    spam = [
        logged("/", referer="https://growyoursitetraffic.shop/", **robot),
        logged("/?ref=linkedin", **robot),
        logged("/favicon.svg", **robot),  # an icon is not a stylesheet or a script
    ]
    days, referrers = usage.tally([*spam, *visit("/", PHONE, referer="https://learnche.org/pid/contents")], SINCE)
    assert dict(referrers) == {"learnche.org": 1}
    assert days["2026-10-12"].pages["Home"] == 3


def test_a_reader_is_known_by_a_stylesheet_or_a_script_its_own_browser_asks_for():
    """Whenever in the period it is asked for, a script as well as a stylesheet; another reader's does not count."""
    entries = [
        logged("/map/", when="2026-10-12T10:00:00", referer="https://github.com/kgdunn/process-improve"),
        logged("/map.js", when="2026-10-12T10:00:02", referer="https://factori.al/map/"),
        logged("/", ip="198.51.100.0", referer="https://newsmarketing.cyou/"),
        logged("/style.css", PHONE, ip="198.51.100.0"),  # another browser on the robot's network
        logged("/app.js", "Googlebot/2.1", ip="198.51.100.0"),  # and a robot that says it is one
    ]
    _, referrers = usage.tally(entries, SINCE)
    assert dict(referrers) == {"github.com": 1}


def test_a_tagged_link_is_counted_by_its_tag_on_whichever_page_it_opens():
    """A link says where it was posted (?ref=linkedin), which an app's own browser, sending no Referer, cannot."""
    _, referrers = usage.tally(
        [
            *visit("/?ref=linkedin"),
            *visit("/?example=worked&ref=LinkedIn", PHONE),
            *visit("/features.html?ref=linkedin"),
            *visit("/?ref=newsletter", referer="https://mail.example.com/inbox"),  # by its tag alone, counted once
            *visit("/", referer="https://www.linkedin.com/feed/"),
            *visit("/about.html", referer="https://factori.al/?ref=linkedin"),  # a page opened from the app after
            *visit("/?ref=linkedin", "Googlebot/2.1"),
        ],
        SINCE,
    )
    assert dict(referrers) == {"?ref=linkedin": 3, "?ref=newsletter": 1, "www.linkedin.com": 1}


def test_the_app_drops_from_its_address_the_parameter_the_report_reads():
    app = (Path(__file__).resolve().parents[2] / "web" / "campaign.js").read_text(encoding="utf-8")
    assert f'export const REF_PARAM = "{usage.REF}";' in app


@pytest.mark.parametrize(
    "tag",
    ["", "two%20words", "%1b%5B31mred", "-dash-first", "a" * 25, "caf%C3%A9", "a/b", "<b>"],
    ids=["none", "a space", "an escape", "a dash first", "too long", "an accent", "a path", "markup"],
)
def test_a_tag_that_is_no_short_plain_word_is_not_counted(tag):
    """The log holds whatever a request sent and the report prints it, so a tag is letters, digits, - and _."""
    _, alone = usage.tally(visit(f"/?ref={tag}"), SINCE)
    _, linked = usage.tally(visit(f"/?ref={tag}", referer="https://learnche.org/pid/contents"), SINCE)
    assert (dict(alone), dict(linked)) == ({}, {"learnche.org": 1}), "the view is still a view, from the site it names"


def test_the_longest_tag_fits_the_reports_column():
    _, referrers = usage.tally([*visit(f"/?ref={'a' * 24}"), *visit("/?ref=q4_2026-launch")], SINCE)
    assert dict(referrers) == {f"?ref={'a' * 24}": 1, "?ref=q4_2026-launch": 1}
    days, _ = three_days()
    rows = usage.report(days, {}, referrers, date(2026, 10, 10), date(2026, 10, 12)).splitlines()
    assert rows[-2:] == [f"  ?ref={'a' * 24}       1", "  ?ref=q4_2026-launch                 1"]


def test_requests_before_the_period_are_left_out():
    days, _ = usage.tally([logged("/", when="2026-09-30T23:59"), logged("/", when="2026-10-01T00:00")], SINCE)
    assert {name: d.pages["Home"] for name, d in days.items()} == {"2026-10-01": 1}


def three_days():
    return usage.tally(
        [
            *visit("/", when="2026-10-11T09:00", referer="https://learnche.org/pid/contents"),
            logged("/runtime.json", when="2026-10-11T09:01", referer=FROM_THE_WORKER),
            logged("/", when="2026-10-12T09:00"),  # the same reader the next day
            logged("/", PHONE, when="2026-10-12T10:00"),
            logged("/map/", PHONE, when="2026-10-12T10:05"),  # the same reader, on the map
        ],
        SINCE,
    )


HEAD = "Day         Home  Map  Catalog  Multivariate  About  Features  Other  Readers  Python  Previews"


def test_the_report_has_a_row_for_every_day_the_periods_totals_and_a_bar_a_day_for_each_page():
    days, referrers = three_days()
    assert usage.report(days, {}, referrers, date(2026, 10, 10), date(2026, 10, 12)).splitlines() == [
        "factori.al, 2026-10-10 to 2026-10-12 (UTC), a row a day",
        "",
        HEAD,
        "2026-10-10     0    0        0             0      0         0      0        0       0         0",
        "2026-10-11     1    0        0             0      0         0      0        1       1         0",
        "2026-10-12     2    1        0             0      0         0      0        2       0         0",
        "All            3    1        0             0      0         0      0      1.0       1         0",
        "",
        "Readers are distinct networks and browsers in a day: approximate. Over several days, the average a day.",
        "",
        "Each page over the period, a bar a day:",
        "  Home           ▄█",
        "  Map             █",
        "  Catalog",
        "  Multivariate",
        "  About",
        "  Features",
        "  Other",
        "  Readers        ▄█",
        "",
        "Where readers came from",
        "  learnche.org                        1",
    ]


@pytest.mark.parametrize(
    ("by", "first", "rows", "readers"),
    [
        ("week", "2026-10-05", [["2026-10-05", "1"], ["2026-10-12*", "2"], ["All", "3"]], ["0.1", "2.0", "0.4"]),
        ("month", "2026-10-01", [["2026-10*", "3"], ["All", "3"]], ["0.2", "0.2"]),
    ],
)
def test_a_row_a_week_or_a_month_sums_the_pages_and_gives_the_readers_a_day(by, first, rows, readers):
    """A period of weeks or months starts with a whole one; the one still going is marked, so a short row reads so."""
    days, referrers = three_days()
    lines = usage.report(days, {}, referrers, date(2026, 10, 10), date(2026, 10, 12), by).splitlines()
    assert lines[0] == f"factori.al, {first} to 2026-10-12 (UTC), a row a {by}"
    assert lines[2].startswith({"week": "Week of", "month": "Month"}[by])
    table = lines[3 : 3 + len(rows)]
    assert [line.split()[:2] for line in table] == rows, "the pages summed: here, Home"
    assert [line.split()[-3] for line in table] == readers, "the readers an average a day"
    assert f"* The {by} so far, to 2026-10-12." in lines
    assert (f"Each page over the period, a bar a {by}:" in lines) == (len(rows) > 2), "bars where there are rows"


def test_a_week_or_a_month_that_is_over_is_not_marked():
    days, referrers = three_days()
    lines = usage.report(days, {}, referrers, date(2026, 10, 5), date(2026, 10, 11), "week").splitlines()
    assert [line.split()[0] for line in lines[3:5]] == ["2026-10-05", "All"]
    assert not any(line.startswith("*") for line in lines)
    assert "so far" not in usage.report(days, {}, referrers, date(2026, 10, 12), date(2026, 10, 12)), "a day is a day"


def test_bars_run_from_a_blank_for_none_to_the_tallest_for_the_most():
    assert usage.bars([0, 1, 4, 8]) == " ▁▄█"
    assert usage.bars([0, 0]) == "  "
    assert usage.bars([0.1, 100]) == "▁█", "anything at all shows"


def test_the_finished_days_are_kept_once_and_today_is_not(tmp_path):
    days, _ = three_days()
    kept = tmp_path / "usage.csv"
    assert usage.keep(kept, days, date(2026, 10, 12)) == ["2026-10-11"]
    assert kept.read_text(encoding="utf-8").splitlines() == [
        "Day,Home,Map,Catalog,Multivariate,About,Features,Other,Readers,Python,Previews",
        "2026-10-11,1,0,0,0,0,0,0,1,1,0",
    ]
    assert usage.keep(kept, days, date(2026, 10, 13)) == ["2026-10-12"]
    forgotten = {"2026-10-12": usage.Day()}  # the log has since forgotten most of the day
    assert usage.keep(kept, forgotten, date(2026, 10, 20)) == []
    assert usage.kept(kept)["2026-10-12"]["Home"] == 2, "a day kept is the whole day, never written again"
    assert not (tmp_path / "usage.csv.part").exists()


def test_a_column_the_file_lacks_counts_nothing_and_is_added_when_the_file_is_next_written(tmp_path):
    kept = tmp_path / "usage.csv"
    kept.write_text("Day,Home,Readers\n2026-09-01,7,3\n", encoding="utf-8")
    assert usage.kept(kept)["2026-09-01"] == {name: {"Home": 7, "Readers": 3}.get(name, 0) for name in usage.COLUMNS}
    days, _ = three_days()
    usage.keep(kept, days, date(2026, 10, 12))
    assert kept.read_text(encoding="utf-8").splitlines()[:2] == [
        "Day,Home,Map,Catalog,Multivariate,About,Features,Other,Readers,Python,Previews",
        "2026-09-01,7,0,0,0,0,0,0,3,0,0",
    ]


def test_the_report_reads_the_kept_counts_for_the_days_the_log_no_longer_holds(tmp_path):
    days, referrers = three_days()
    keeps = {"2026-10-10": {**dict.fromkeys(usage.COLUMNS, 0), "Home": 9, "Map": 4, "Readers": 5}}
    keeps["2026-10-11"] = {**dict.fromkeys(usage.COLUMNS, 0), "Home": 1, "Readers": 1, "Python": 1}
    lines = usage.report(days, keeps, referrers, date(2026, 10, 10), date(2026, 10, 12)).splitlines()
    assert lines[3:6] == [
        "2026-10-10     9    4        0             0      0         0      0        5       0         0",
        "2026-10-11     1    0        0             0      0         0      0        1       1         0",
        "2026-10-12     2    1        0             0      0         0      0        2       0         0",
    ]


def test_keep_adds_the_finished_days_and_says_nothing_and_the_report_reads_them(tmp_path, capsys):
    log = tmp_path / "factorial-access.log"
    kept = tmp_path / "kept" / "usage.csv"
    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M")
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    log.write_text(
        "".join(json.dumps(logged(path, when=when)) + "\n" for path, when in (("/", yesterday), ("/map/", now))),
        encoding="utf-8",
    )
    paths = ["--log", str(tmp_path / "factorial-access*.log*"), "--kept", str(kept)]
    assert usage.main([*paths, "--keep"]) == 0
    assert capsys.readouterr().out == ""
    assert list(usage.kept(kept)) == [yesterday[:10]], "the finished day, and not today"
    log.write_text("", encoding="utf-8")  # the log forgets
    assert usage.main([*paths, "--days", "2"]) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[3:5]]
    assert [row[:3] for row in rows] == [[yesterday[:10], "1", "0"], [now[:10], "0", "0"]]


def test_the_report_reads_the_live_log_and_the_rolled_ones_gzipped(tmp_path, capsys):
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")
    (tmp_path / "factorial-access.log").write_text(
        json.dumps(logged("/", when=now)) + "\n" + '{"cut short by a rotation', encoding="utf-8"
    )
    with gzip.open(tmp_path / "factorial-access-2026-10-01T00-00-00.000.log.gz", "wt", encoding="utf-8") as rolled:
        rolled.write(json.dumps(logged("/", PHONE, when=now)) + "\n")
    (tmp_path / "other-site.log").write_text(json.dumps(logged("/", when=now)) + "\n", encoding="utf-8")
    log = ["--log", str(tmp_path / "factorial-access*.log*"), "--kept", str(tmp_path / "none.csv")]
    assert usage.main([*log, "--days", "1"]) == 0
    others = ["0"] * (len(usage.PAGES) - 1)  # every page but Home
    assert capsys.readouterr().out.splitlines()[4].split() == ["All", "2", *others, "2", "0", "0"]


def test_the_report_says_when_no_log_matches(tmp_path, capsys):
    assert usage.main(["--log", str(tmp_path / "factorial-access*.log*")]) == 1
    assert "no log files match" in capsys.readouterr().err


# --------------------------------------------------------------------------- the public file
NOW = datetime(2026, 10, 12, 10, 30, tzinfo=UTC)


def write_log(path: Path, *entries: dict, written: datetime = NOW) -> Path:
    """Write the entries as a log file, last written at ``written``, and return its path."""
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    os.utime(path, (written.timestamp(), written.timestamp()))
    return path


def test_the_public_file_holds_the_readers_of_each_day_kept_and_logged_and_nothing_else(tmp_path):
    """The footer draws it: the kept days, then yesterday from the log, then the last 24 hours, as numbers alone."""
    kept = tmp_path / "usage.csv"
    kept.write_text("Day,Home,Readers\n2026-10-08,9,5\n2026-10-09,12,7\n", encoding="utf-8")
    log = write_log(
        tmp_path / "factorial-access.log",
        logged("/", when="2026-10-11T09:00"),
        logged("/map/", PHONE, when="2026-10-11T21:00"),
        logged("/", when="2026-10-12T08:00"),
        logged("/", HEADLESS, when="2026-10-12T08:05"),  # the deploy's own check is no reader
    )
    public = tmp_path / "public" / "usage.json"
    written = usage.publish(public, [log], kept, NOW)
    assert written == {
        "updated": "2026-10-12T10:30:00Z",
        "days": [["2026-10-08", 5], ["2026-10-09", 7], ["2026-10-10", 0], ["2026-10-11", 2]],  # none before the first
        "last_24_hours": 2,  # since 2026-10-11T10:30: the phone at 21:00, and the browser this morning, not at 09:00
    }
    assert json.loads(public.read_text(encoding="utf-8")) == written
    if os.name == "posix":  # the server's: Windows keeps no such mode
        assert public.stat().st_mode & 0o777 == 0o644, "Caddy reads it as another user"
    assert not public.with_name("usage.json.part").exists()


def test_the_public_file_reads_the_log_from_the_first_day_not_kept_and_no_rolled_file_before_it(tmp_path):
    """An hourly run reads a day or two of the log, not a year of it: a kept day is never counted again."""
    kept = tmp_path / "usage.csv"
    kept.write_text("Day,Readers\n2026-10-09,7\n", encoding="utf-8")
    log = write_log(
        tmp_path / "factorial-access.log",
        logged("/", when="2026-10-09T10:00"),  # a kept day, which the file says
        logged("/", when="2026-10-10T10:00"),  # not kept: the daily job did not run
        logged("/", when="2026-10-12T10:00"),
    )
    # A rolled file last written before the first day read: never opened, whatever it holds.
    rolled = write_log(tmp_path / "factorial-access-old.log", logged("/", PHONE, when="2026-10-12T09:00"),
                       written=datetime(2026, 10, 5, tzinfo=UTC))  # fmt: skip
    assert usage.written_since([log, rolled], datetime(2026, 10, 10, tzinfo=UTC)) == [log]
    written = usage.publish(tmp_path / "usage.json", [log, rolled], kept, NOW)
    assert written["days"] == [["2026-10-09", 7], ["2026-10-10", 1], ["2026-10-11", 0]]
    assert written["last_24_hours"] == 1


def test_the_public_file_holds_ninety_days_at_most(tmp_path):
    kept = tmp_path / "usage.csv"
    first = NOW.date() - timedelta(days=200)
    rows = "".join(f"{(first + timedelta(days=i)).isoformat()},{i}\n" for i in range(199))
    kept.write_text("Day,Readers\n" + rows, encoding="utf-8")
    log = write_log(tmp_path / "factorial-access.log", logged("/", when="2026-10-12T09:00"))
    days = usage.publish(tmp_path / "usage.json", [log], kept, NOW)["days"]
    assert len(days) == usage.PUBLIC_DAYS
    assert days[0][0] == (NOW.date() - timedelta(days=usage.PUBLIC_DAYS)).isoformat()
    assert days[-1] == ["2026-10-11", 0], "yesterday, the last finished day: neither kept nor in the log"


def test_public_writes_the_file_and_prints_nothing(tmp_path, capsys):
    write_log(tmp_path / "factorial-access.log", logged("/", when=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M")),
              written=datetime.now(UTC))  # fmt: skip
    public = tmp_path / "usage.json"
    paths = ["--log", str(tmp_path / "factorial-access*.log*"), "--kept", str(tmp_path / "none.csv")]
    assert usage.main([*paths, "--public", str(public)]) == 0
    assert capsys.readouterr().out == ""
    written = json.loads(public.read_text(encoding="utf-8"))
    assert (written["days"], written["last_24_hours"]) == ([], 1), "no finished day yet, and one reader today"


def test_the_hourly_job_writes_the_file_caddy_serves():
    here = SCRIPT.parent
    job = (here / "usage-hourly.sh").read_text(encoding="utf-8")
    assert "exec /usr/local/bin/factorial-usage --public /var/lib/factorial/public/usage.json" in job
    caddy = (here / "factorial.caddy").read_text(encoding="utf-8")
    assert "handle /usage.json {\n\t\troot * /var/lib/factorial/public\n\t\tfile_server\n\t}" in caddy
    setup = (here / "setup.sh").read_text(encoding="utf-8")
    # run-parts runs only what may be run: setup.sh installs it so, whatever mode a checkout gave it.
    assert 'install -o root -g root -m 755 "$here/usage-hourly.sh" /etc/cron.hourly/factorial-usage-public' in setup
