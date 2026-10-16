"""``deploy/factorial/usage.py``: how factori.al is used, counted from Caddy's access log alone."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "factorial" / "usage.py"
BROWSER = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari"
PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
HEADLESS = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/141.0.0.0 Safari"
SINCE = datetime(2026, 10, 1, tzinfo=timezone.utc)


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
    ts = datetime.fromisoformat(when).replace(tzinfo=timezone.utc).timestamp()
    return {"level": "info", "ts": ts, "logger": "http.log.access.log0", "msg": "handled request",
            "request": request, "status": fields.get("status", 200), "size": 10, "duration": 0.001}  # fmt: skip


def day(*entries: dict, on: str = "2026-10-12"):
    days, _ = usage.tally(entries, SINCE)
    return days.get(on, usage.Day())


def test_a_visit_is_the_page_loaded_in_a_browser():
    seen = day(
        logged("/"),
        logged("/index.html", status=304),  # coming back: the page is revalidated
        logged("/?utm_source=book"),
        logged("/", method="HEAD"),
        logged("/missing", status=404),
        logged("/", status=500),
        logged("/app.js"),
    )
    assert seen.visits == 3


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
def test_robots_previews_and_the_deploys_own_checks_are_not_visits(agent):
    seen = day(logged("/", agent), logged("/bootstrap.py", agent))
    assert (seen.visits, seen.python, len(seen.readers)) == (0, 0, 0)


def test_the_catalog_s_visits_are_counted_apart_from_the_page_s():
    """factori.al/catalog/ is a visit to the app in a bucket of its own; its readers are the app's readers."""
    seen = day(
        logged("/catalog/"),
        logged("/catalog/index.html", status=304),
        logged("/catalog/?q=dsd"),
        logged("/catalog", status=308),  # redirected to /catalog/, which is counted
        logged("/catalog/index.json"),  # the page's data, not a visit
        logged("/catalog/", HEADLESS),
        logged("/"),
    )
    assert (seen.catalog, seen.visits) == (3, 1)
    assert len(seen.readers) == 1, "one browser, in the app and the catalog"


def test_readers_are_distinct_networks_and_browsers():
    seen = day(logged("/"), logged("/", when="2026-10-12T15:00"), logged("/", PHONE), logged("/", ip="198.51.100.0"))
    assert seen.visits == 4
    assert len(seen.readers) == 3


def test_python_started_is_the_page_fetching_bootstrap_py():
    seen = day(logged("/"), logged("/bootstrap.py"), logged("/bootstrap.py", PHONE), logged("/bootstrap.py", HEADLESS))
    assert seen.python == 2


def test_a_link_card_fetched_by_a_chat_or_social_site_is_a_preview():
    seen = day(
        logged("/og-card.png", "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)"),
        logged("/og-card.png", "Twitterbot/1.0"),
        logged("/og-card.png", "LinkedInBot/1.0 (compatible; Mozilla/5.0; Apache-HttpClient +http://www.linkedin.com)"),
        logged("/og-card.png"),  # a reader opening the image is neither a preview nor a visit
    )
    assert (seen.previews, seen.visits) == (3, 0)


def test_visits_name_the_sites_they_came_from_but_not_factori_al_itself():
    _, referrers = usage.tally(
        [
            logged("/", referer="https://learnche.org/pid/design-analysis-experiments/index"),
            logged("/", PHONE, referer="https://learnche.org/pid/contents"),
            logged("/", referer="https://github.com/kgdunn/process-improve"),
            logged("/", referer="https://factori.al/"),
            logged("/", referer="https://www.factori.al/"),
            logged("/"),
        ],
        SINCE,
    )
    assert dict(referrers) == {"learnche.org": 2, "github.com": 1}


def test_requests_before_the_period_are_left_out():
    days, _ = usage.tally([logged("/", when="2026-09-30T23:59"), logged("/", when="2026-10-01T00:00")], SINCE)
    assert {name: d.visits for name, d in days.items()} == {"2026-10-01": 1}


def test_the_report_has_a_row_for_every_day_and_the_periods_totals():
    days, referrers = usage.tally(
        [
            logged("/", when="2026-10-11T09:00", referer="https://learnche.org/pid/contents"),
            logged("/bootstrap.py", when="2026-10-11T09:01"),
            logged("/", when="2026-10-12T09:00"),  # the same reader the next day
            logged("/", PHONE, when="2026-10-12T10:00"),
            logged("/catalog/", PHONE, when="2026-10-12T10:05"),  # the same reader, in the catalog
        ],
        SINCE,
    )
    assert usage.report(days, referrers, date(2026, 10, 10), date(2026, 10, 12)).splitlines() == (
        [
            "factori.al, 2026-10-10 to 2026-10-12 (UTC)",
            "",
            "Day         Visits  Catalog  Readers  Python  Previews",
            "2026-10-10       0        0        0       0         0",
            "2026-10-11       1        0        1       1         0",
            "2026-10-12       2        1        2       0         0",
            "All              3        1        2       1         0",
            "",
            "Readers are distinct networks and browsers: approximate.",
            "",
            "Where visits came from",
            "  learnche.org                        1",
        ]
    )


def test_the_report_reads_the_live_log_and_the_rolled_ones_gzipped(tmp_path, capsys):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    (tmp_path / "factorial-access.log").write_text(
        json.dumps(logged("/", when=now)) + "\n" + '{"cut short by a rotation', encoding="utf-8"
    )
    with gzip.open(tmp_path / "factorial-access-2026-10-01T00-00-00.000.log.gz", "wt", encoding="utf-8") as rolled:
        rolled.write(json.dumps(logged("/", PHONE, when=now)) + "\n")
    (tmp_path / "other-site.log").write_text(json.dumps(logged("/", when=now)) + "\n", encoding="utf-8")
    assert usage.main(["--log", str(tmp_path / "factorial-access*.log*"), "--days", "1"]) == 0
    assert capsys.readouterr().out.splitlines()[-3].split() == ["All", "2", "0", "2", "0", "0"]


def test_the_report_says_when_no_log_matches(tmp_path, capsys):
    assert usage.main(["--log", str(tmp_path / "factorial-access*.log*")]) == 1
    assert "no log files match" in capsys.readouterr().err
