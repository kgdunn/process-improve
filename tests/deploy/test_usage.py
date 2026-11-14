"""``deploy/factorial/usage.py``: how factori.al is used, counted from Caddy's access log, and the counts it keeps."""

from __future__ import annotations

import gzip
import importlib.util
import json
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
            "request": request, "status": fields.get("status", 200), "size": 10, "duration": 0.001}  # fmt: skip


def day(*entries: dict, on: str = "2026-10-12"):
    days, _ = usage.tally(entries, SINCE)
    return days.get(on, usage.Day())


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
        logged("/about.html"),
        logged("/features.html"),
        logged("/features.html", status=304),
        logged("/FEATURES.md"),  # the document the page reads, not a page
        logged("/a-page-added-later/"),
        logged("/style.css"),
        logged("/no-such-page/", status=404),
    )
    assert seen.pages == {"Map": 2, "Catalog": 3, "Multivariate": 1, "About": 1, "Features": 2, "Other": 1}
    assert seen.counts() == {
        "Home": 0, "Map": 2, "Catalog": 3, "Multivariate": 1, "About": 1, "Features": 2, "Other": 1,
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
        logged("/wheels/0123456789abcdef/bootstrap.zip", referer=FROM_THE_WORKER),
    )
    assert seen.python == 2


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
            logged("/", referer="https://learnche.org/pid/design-analysis-experiments/index"),
            logged("/map/", PHONE, referer="https://learnche.org/pid/contents"),
            logged("/", referer="https://github.com/kgdunn/process-improve"),
            logged("/", referer="https://factori.al/"),
            logged("/catalog/", referer="https://www.factori.al/"),
            logged("/"),
        ],
        SINCE,
    )
    assert dict(referrers) == {"learnche.org": 2, "github.com": 1}


def test_requests_before_the_period_are_left_out():
    days, _ = usage.tally([logged("/", when="2026-09-30T23:59"), logged("/", when="2026-10-01T00:00")], SINCE)
    assert {name: d.pages["Home"] for name, d in days.items()} == {"2026-10-01": 1}


def three_days():
    return usage.tally(
        [
            logged("/", when="2026-10-11T09:00", referer="https://learnche.org/pid/contents"),
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
