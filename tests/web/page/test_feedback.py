"""The Feedback dialog and the relay on factori.al: what the page sends is what the relay takes.

``web/feedback.js`` repeats the relay's topics and limits, so it can say what is wrong with a report
before it is sent, and cut the page's context to what the relay takes. Here both are read, the
page's under Node and the relay's from ``deploy/factorial/feedback_relay.py``, and every report
the page builds must pass the relay's own checks.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.web._support import ROOT, javascript, source

RELAY = ROOT / "deploy" / "factorial" / "feedback_relay.py"


@pytest.fixture(scope="module")
def relay():
    spec = importlib.util.spec_from_file_location("feedback_relay", RELAY)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


def page(node: str, *calls: tuple[str, list]) -> list:
    """Call ``web/feedback.js`` under Node: each call a function and its arguments, or a constant to read."""
    return javascript(node, source("feedback.js"), list(calls))


def test_the_page_names_the_topics_and_limits_the_relay_takes(node, relay):
    names = ("TOPICS", "MESSAGE_MIN", "MESSAGE_MAX", "EMAIL_MAX", "MAX_SCREENSHOT", "MAX_STUDY", "CONTEXT")
    topics, least, most, email, screenshot, study, context = page(node, *((name, []) for name in names))
    assert topics == relay.TOPICS
    assert (least, most, email) == (relay.MESSAGE_MIN, relay.MESSAGE_MAX, relay.EMAIL_MAX)
    assert (screenshot, study) == (relay.MAX_SCREENSHOT, relay.MAX_STUDY)
    assert {key: tuple(limit) if isinstance(limit, list) else limit for key, limit in context.items()} == relay.CONTEXT


def test_every_report_the_page_builds_passes_the_relay_s_checks(node, relay):
    """Each at an edge: the longest message, emoji in it, a context past every limit, which the page cuts."""
    jpeg = b"\xff\xd8\xff\xe0" + bytes(64)
    shot = relay.JPEG_URL + base64.b64encode(jpeg).decode()
    over = {
        "commit": "f" * 100,
        "tab": "Learn\nand more",
        "model": "\x07".join(["As designed"] * 40),
        "charts": [f"Chart {n}" for n in range(50)],
        "stamp": ["x" * 500] * 20,
        "errors": ["😀" * 400] * 9,
        "browser": "b" * 1000,
        "window": "1280 x 900",
        "cookie": "dropped",
    }
    built = page(
        node,
        ("buildReport", [{"topic": "algorithm", "message": "😀" * relay.MESSAGE_MAX, "context": over}]),
        ("buildReport", [{"topic": "general", "message": "A line\r\nand a tab\t.", "email": " a.b+c@example.co.uk "}]),
        ("buildReport", [{"topic": "interface", "message": "x\u0000" * 20, "screenshot": shot, "study": "jZJN_-"}]),
    )
    for report in built:
        checked = relay.read_report(json.dumps(report).encode())
        assert checked.topic == report["topic"]
    assert relay.read_report(json.dumps(built[2]).encode()).screenshot == jpeg


def test_the_issue_link_fills_in_the_issue_form_s_own_fields(node):
    """The repository takes no blank issue, so the link names the app's form and fills its fields by their ids."""
    yaml = pytest.importorskip("yaml")
    form = yaml.safe_load((ROOT / ".github" / "ISSUE_TEMPLATE" / "web_app.yml").read_text(encoding="utf-8"))
    fields = {item["id"]: item for item in form["body"] if "id" in item}
    [link, template] = page(
        node,
        ("issueUrl", [{"topic": "general", "message": "A message for the form.", "context": {"tab": "Run"}}]),
        ("ISSUE_TEMPLATE", []),
    )
    assert template == "web_app.yml"
    sent = parse_qs(urlsplit(link).query)
    assert set(sent) - {"template", "title"} == set(fields), (
        "every field the link fills is the form's, and each is filled"
    )
    assert all(fields[key]["type"] in {"input", "textarea"} for key in fields), "a link fills only the text fields"
    required = {key for key, item in fields.items() if item.get("validations", {}).get("required")}
    assert required <= set(sent)
    assert sent["title"] == ["[Factorial] General: A message for the form."]
    assert form["title"] == "[Factorial] "


def test_the_screenshot_s_module_is_the_published_release_with_its_licence():
    """``web/vendor/modern-screenshot.mjs`` is npm's file unchanged after a header naming its release and its digest."""
    vendored = (ROOT / "web" / "vendor" / "modern-screenshot.mjs").read_bytes()
    header, _, published = vendored.partition(b" */\n")
    words = header.decode("ascii")
    release = re.search(r"modern-screenshot (\d+\.\d+\.\d+), https://github\.com/qq15725/modern-screenshot", words)
    assert release
    assert f"/modern-screenshot-{release[1]}.tgz" in words
    assert re.search(r"SHA-256 of dist/index\.mjs: ([0-9a-f]{64})", words)[1] == hashlib.sha256(published).hexdigest()
    assert "The MIT License (MIT)" in words
    assert "Copyright (c) 2021-present wxm" in words
    assert 'import("./vendor/modern-screenshot.mjs")' in source("feedback.js"), "imported when it is asked for"
    assert "modern-screenshot" not in source("app.js"), "never at the page's first load"


@pytest.mark.parametrize(
    "address",
    [
        "reader@example.com",
        "first.last+tag@sub.example.co.uk",
        "o'brien@example.ie",
        "reader",
        "reader@localhost",
        "Reader <reader@example.com>",
        "one@example.com, two@example.com",
        "re..ader@example.com",
        "reader@[127.0.0.1]",
        "réader@example.com",
        "reader@exa_mple.com",
    ],
)
def test_an_address_the_page_takes_is_one_the_relay_takes(node, relay, address):
    """The page may be stricter than the relay, never looser: a report it lets go is never refused for its address."""
    [problem] = page(node, ("emailProblem", [address]))
    if problem == "":
        assert relay.reply_address(address) == address
