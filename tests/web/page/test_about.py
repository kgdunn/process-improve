"""How Factorial works (``web/technical-details-how-factorial-works.html``): what it says must stay true of the code.

The page says what reaches the server, where to read the code and how to build and serve the app.
Each of those is checked here against the file it describes, so a change to the server's
configuration, the build script or the repository's layout cannot leave the page describing an
app that no longer exists. The versions it lists are read at run time from the files the app
loads (``runtime.json`` and ``wheels/manifest.json``), so none is written into the page.

The page is written for a reader of the app: each part answers a question in plain words, and
what only some readers ask is under the answer, in a fold. That is checked here too.
"""

from __future__ import annotations

import ast
import json
import re

from tests.web._support import ROOT, WEB, source

REPOSITORY = "https://github.com/kgdunn/process-improve/"


def about() -> str:
    return " ".join(source("technical-details-how-factorial-works.html").split())


def source_of(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


#: Words for a reader who checks how it is done. Each is said in a fold, and none in an answer.
FOR_THE_FOLDS = (
    "IndexedDB",
    "local storage",
    "service worker",
    "Web Worker",
    "WebAssembly",
    "CPython",
    "Pyodide",
    "Plotly.js",
    "JSON",
    "SHA-256",
    "User-Agent",
    "IPv4",
    "MIT license",
    "wheel",
    ".xlsx",
    ".ipynb",
)


def test_each_part_answers_a_question_in_plain_words_and_folds_the_rest():
    text = about()
    parts = re.findall(r'<section aria-labelledby="(h-[\w-]+)">(.*?)</section>', text)
    assert len(parts) >= 8
    for name, part in parts:
        heading = re.search(rf'<h2 id="{name}">([^<]+)</h2>', part)
        assert heading, name
        assert heading[1].endswith("?"), f"{name}: the question a reader asks"
        answer, fold, folded = part.partition('<details class="fold">')
        assert "<p>" in answer, f"{name}: its answer comes first"
        assert fold, f"{name}: the rest is in a fold"
        assert folded.count("<summary>") == 1, f"{name}: one fold, which says what it holds"
        assert folded.strip().endswith("</details>"), f"{name}: nothing follows the fold"
    assert text.count('<details class="fold">') == len(parts)


def test_what_is_in_bold_is_a_sentence_with_its_period():
    """A point opens with a sentence in bold: none trails a comma. The names in the drawing's boxes are no points."""
    text = re.sub(r'<div class="flow__box[^"]*"><strong>[^<]+</strong>', "", about())
    bold = [re.sub(r"<[^>]+>", "", words) for words in re.findall(r"<strong>(.+?)</strong>", text)]
    assert len(bold) > 25
    assert [words for words in bold if not words.endswith(".")] == []
    assert not re.search(r"</strong>[,:;]", text), "and nothing but a new sentence follows it"


def test_the_answers_have_none_of_the_words_kept_for_the_folds():
    """What a reader sees before opening a fold is in plain words: the words of files and servers are inside."""
    text = about()
    shown = text[text.index("<main") : text.index("</main>")]  # what the page says: its scripts are outside it
    shown = re.sub(r"<svg\b.*?</svg>", "", shown)  # a drawing's markup is no words: its words are in its caption
    answers = re.sub(r'<details class="fold">.*?</details>', "", shown)
    assert len(answers) < len(shown) / 2, "most of the page's words are in its folds"
    for word in FOR_THE_FOLDS:
        assert word.lower() not in answers.lower(), word
        assert word.lower() in shown.lower(), f"{word}: said in a fold, for a reader who asks"


PAGE = "technical-details-how-factorial-works.html"


def test_the_app_s_footer_links_to_it():
    link = f'<li><a href="{PAGE}">How it works</a></li>'
    assert link in " ".join(source("index.html").split())


def test_its_old_address_forwards_to_it_with_the_link_s_query_and_place():
    """Links shared before the page was renamed, on 20 November 2026, still reach it, and search engines move on."""
    old = " ".join(source("about.html").split())
    assert f'<link rel="canonical" href="https://factori.al/{PAGE}">' in old
    assert '<meta name="robots" content="noindex">' in old
    assert f'<meta http-equiv="refresh" content="0; url={PAGE}">' in old
    assert f'location.replace("{PAGE}" + location.search + location.hash)' in old
    caddy = source_of("deploy/factorial/factorial.caddy")
    assert f"redir /about.html /{PAGE}{{?query}} permanent" in caddy, "a 301 on the server, keeping ?ref="


def test_a_shared_link_to_it_names_it_with_its_own_address():
    assert f'<link rel="canonical" href="https://factori.al/{PAGE}">' in about()
    assert f'<meta property="og:url" content="https://factori.al/{PAGE}">' in about()


def test_every_link_into_the_repository_names_a_path_that_exists():
    paths = re.findall(
        rf'href="{re.escape(REPOSITORY)}(?:tree|blob)/main/([^"]+)"',
        source("technical-details-how-factorial-works.html"),
    )
    assert len(paths) >= 5, "the links to the code a reader would audit"
    for path in paths:
        assert (ROOT / path).exists(), path


def test_each_function_it_says_python_answers_with_is_one_bootstrap_calls():
    """The worker's functions, a study of one factor's tests among them, are each one ``web/bootstrap.py`` uses.

    It imports most by name. Those that fit a model it takes from the experiments package as it calls them
    (``doe.analyze_experiment``): imported by name they would load statsmodels as the page starts.
    """
    bullet = re.search(r"<li><strong>A Web Worker runs Python\.</strong>(.+?)</li>", about())[1]
    named = set(re.findall(r"<code>(\w+)</code>", bullet))
    assert {"analyze_experiment", "bartlett_test", "kruskal_wallis_test"} <= named
    nodes = list(ast.walk(ast.parse(source("bootstrap.py"))))
    imported = {alias.name for node in nodes if isinstance(node, ast.ImportFrom) for alias in node.names}
    attributes = [node for node in nodes if isinstance(node, ast.Attribute)]
    of_the_package = {node.attr for node in attributes if ast.unparse(node.value) == "doe"}
    assert {"analyze_experiment", "analyze_functional_experiment"} <= of_the_package
    assert named <= imported | of_the_package, named - imported - of_the_package


def test_what_it_says_the_server_keeps_is_what_the_server_is_configured_to_keep():
    caddy = (ROOT / "deploy" / "factorial" / "factorial.caddy").read_text(encoding="utf-8")
    ipv4, ipv6 = re.search(r"ip_mask \{\s*ipv4 (\d+)\s*ipv6 (\d+)", caddy).groups()
    hours = int(re.search(r"roll_keep_for (\d+)h", caddy).group(1))
    text = about()
    assert f"the first {ipv4} bits of an IPv4 address, the first {ipv6} of an IPv6 one" in text
    assert f"kept for {hours // 24} days" in text


def test_what_it_says_of_feedback_is_what_the_dialog_and_the_relay_do():
    """Sent only on Send, the screenshot unless unticked, the study only if ticked, and the relay's code a link away."""
    text = about()
    bullet = re.search(r"<li><strong>Feedback reaches factori\.al only when you press Send\.</strong>(.+?)</li>", text)[
        1
    ]
    form = " ".join(source("feedback.js").split())
    assert '<input type="checkbox" id="feedback-screenshot" name="screenshot" checked>' in form
    assert "a screenshot of the tab unless you untick it" in bullet
    assert '<input type="checkbox" id="feedback-study" name="study">' in form
    assert "your study only if you tick it" in bullet
    assert "the email provider that sends the site's mail" in bullet
    assert f'<a href="{REPOSITORY}blob/main/deploy/factorial/feedback_relay.py"' in bullet
    assert "Read the code" in bullet
    relay = source_of("deploy/factorial/feedback_relay.py")
    assert 'def log_message(self, *_: object) -> None:\n        """Say nothing' in relay, "no request line in the log"
    footer = " ".join(source("index.html").split())
    assert "nothing on our server" in footer
    assert '<button type="button" class="link" data-feedback="general">Send feedback</button>' in footer, (
        "the footer names the one report sent to the server"
    )


def test_what_it_says_of_rooms_is_what_the_locker_and_the_page_do():
    """The sizes, the days, the counts and the code it names are those of web/room.js and the locker."""
    text = about()
    locker = source_of("deploy/factorial/encrypted_study_locker.py")
    room = source("room.js")
    padded = re.search(r"export const PADDED = \[(\d+), (\d+), (\d+)\];", room).groups()
    assert re.search(r"^PADDED = \(" + ", ".join(padded) + r"\)$", locker, flags=re.MULTILINE), "the page's sizes"
    kib = [str(int(size) // 1024) for size in padded]
    assert f"padded to {kib[0]}, {kib[1]} or {kib[2]} KiB" in text

    expiry, rooms, budget = (
        re.search(rf"^{name} = (\d+)", locker, flags=re.MULTILINE)[1] for name in ("EXPIRY_DAYS", "NEW_ROOMS", "BUDGET")
    )
    assert re.search(r"^BUDGET = \d+ \* 2\*\*20$", locker, flags=re.MULTILINE), "the budget in MiB"
    assert f"nobody changes for {expiry} days" in text
    assert f"at most {rooms} new rooms a day" in text
    assert f"when it holds {budget} MiB" in text
    assert 'const LABELS = { key: "factorial room: key", write: "factorial room: write" };' in room
    assert 'WRITE_LABEL = b"factorial room: write"' in locker, "the page and the locker derive the same token"
    assert f'<a href="{REPOSITORY}blob/main/deploy/factorial/encrypted_study_locker.py"' in text
    assert 'def log_message(self, *_: object) -> None:\n        """Say nothing' in locker, "no request line in the log"
    caddy = source_of("deploy/factorial/factorial.caddy")
    assert "request>headers>Authorization delete" in caddy, "the header a room's name and token travel in"


def test_the_picture_of_a_room_says_one_thing_and_points_to_the_code_that_does_it():
    """Both drawings, wide and tall, carry the same words for a reader who cannot see them; the key never reaches the
    server in either; and the code it invites the reader to audit is the code that makes and uses the key.
    """
    text = about()
    part = re.search(r'<section aria-labelledby="h-room">(.*?)<details class="fold">', text)[1]
    labels = re.findall(
        r'<svg class="room-picture__(?:wide|tall)" viewBox="[^"]+" role="img" aria-label="([^"]+)">', part
    )
    assert len(labels) == 2
    assert labels[0] == labels[1], "the two drawings say the same"
    assert "has no key, so it cannot open it" in labels[0]
    assert "The key travels in the link, which you send yourself" in labels[0]
    assert part.count('class="key-words">has no key: cannot open it</text>') == 2
    assert "<figcaption>The study goes through factori.al locked." in part
    audit = re.search(r'<p class="room-audit">(.+?)</p>', part)[1]
    for path in ("web/room.js", "deploy/factorial/encrypted_study_locker.py", "tests/web/e2e/test_rooms.py"):
        assert f'href="{REPOSITORY}blob/main/{path}"' in audit, path
    room = source("room.js")
    for name in ("roomOf", "seal"):  # the functions the paragraph names, which make the key and encrypt with it
        assert f"<code>{name}</code>" in audit
        assert re.search(rf"^export (?:async )?function {name}\(", room, flags=re.MULTILINE), name
    imported = re.search(r'import \{ ([^}]+) \} from "\./room\.js";', text)[1].split(", ")
    for name in imported:  # the proof runs the app's own code, and only what room.js exports
        assert re.search(rf"^export (?:async )?(?:function|const) {name}\b", room, flags=re.MULTILINE), name
    locker = source_of("deploy/factorial/encrypted_study_locker.py")
    assert "factorial room: key" not in locker, "the locker derives no key: only the write token's label is there"


def test_the_build_it_prints_is_the_build_script_s():
    text = source("technical-details-how-factorial-works.html")
    command = re.search(r"uv run python (scripts/build_web_app\.py) (.+)\n", text)
    assert command
    assert (ROOT / command.group(1)).is_file()
    script = source_of(command.group(1))
    for option in re.findall(r"(--[\w-]+)", command.group(2)):
        assert f'"{option}"' in script, option
    port = re.search(r'"--port", type=int, default=(\d+)', script).group(1)
    assert f"http://localhost:{port}" in text


def test_the_addresses_it_says_to_change_for_a_closed_network_are_in_the_files_it_names():
    text = about()
    assert "<code>web/python-boot.js</code> (Pyodide)" in text
    assert "<code>web/charts.js</code> (Plotly.js)" in text
    assert "<code>wheels/manifest.json</code> (the PyPI packages, under <code>on_demand</code>)" in text
    assert "cdn.jsdelivr.net/pyodide" in source("python-boot.js")
    assert "cdn.jsdelivr.net/npm/plotly.js" in source("charts.js")
    # The worker installs what the manifest lists under on_demand, from the addresses it gives.
    assert "exact: manifest.on_demand?.openpyxl" in source("python-boot.js")
    assert "exact: manifest.on_demand?.plotly" in source("python-boot.js")
    assert '"on_demand": exact_files(' in source_of("scripts/build_web_app.py")
    for file in ("offline.js", "sw.js"):  # the copy for use without a network reads the first two addresses there
        assert f"<code>web/{file}</code>" in text
        assert "cdn.jsdelivr.net/pyodide" in source(file)
        assert "cdn.jsdelivr.net/npm/plotly.js" in source(file)


def test_it_writes_no_version_into_the_page():
    """The versions come from the files the app loads, so the page cannot fall behind them."""
    runtime = json.loads((WEB / "runtime.json").read_text(encoding="utf-8"))
    text = source("technical-details-how-factorial-works.html")
    assert runtime["pyodide"] not in text
    for pinned in runtime["micropip"]:
        assert pinned.split("==")[1] not in text, pinned


def test_what_it_says_of_the_study_file_is_what_study_file_js_writes():
    """The format and version in its example are study-file.js's, and it links nowhere the page does not."""
    code = source("study-file.js")
    form = re.search(r'export const FILE_FORMAT = "([^"]+)";', code)
    version = re.search(r"export const FILE_VERSION = (\d+);", code)
    assert form
    assert version
    assert f'"format": "{form.group(1)}"' in about()
    assert f'"version": {version.group(1)},' in about()
    assert '<a href="#h-file">the study file</a>' in about(), "Your studies points to it"
