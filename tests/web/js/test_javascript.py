"""The page's JavaScript, one function at a time: ``node --test`` runs each suite named here.

The suites import the page's modules from ``web/`` as the page does. Node.js reads a ``.js``
module without a package.json as an ES module from version 22.7; on an older Node.js these
tests skip, and CI's web-app job runs them on Node.js 22.
"""

from __future__ import annotations

import json
import keyword
import re
import subprocess
from pathlib import Path

import pytest

from process_improve.experiments._lm import suggest_identifier
from tests.web._support import ROOT

HERE = Path(__file__).parent

#: Each suite, by name, so ``--affected-since`` sends a change to one of them here.
SUITES = (
    "logic.test.mjs",
    "charts.test.mjs",
    "python.test.mjs",
    "best.test.mjs",
    "report.test.mjs",
    "campaign.test.mjs",
    "study-file.test.mjs",
    "picture.test.mjs",
    "catalog.test.mjs",
    "field-chart.test.mjs",
    "reproduced.test.mjs",
    "feedback.test.mjs",
    "comparison.test.mjs",
    "spread.test.mjs",
    "mva-selection.test.mjs",
    "mva-table.test.mjs",
    "mva-views.test.mjs",
)

#: The first Node.js that reads ``web/*.js`` as ES modules without being told.
LEAST = (22, 7)


def node_version(node: str) -> tuple[int, ...]:
    """Return the version of Node.js, as numbers."""
    printed = subprocess.run([node, "--version"], capture_output=True, text=True, check=True).stdout  # noqa: S603
    return tuple(int(part) for part in re.findall(r"\d+", printed)[:3])


def test_every_suite_is_named_here():
    assert sorted(p.name for p in HERE.glob("*.test.mjs")) == sorted(SUITES)


@pytest.mark.parametrize("suite", SUITES)
def test_the_page_s_javascript(node, suite):
    if node_version(node) < LEAST:
        pytest.skip(f"Node.js {'.'.join(map(str, LEAST))} or later reads web/*.js as ES modules")
    done = subprocess.run(  # noqa: S603 - Node.js running this directory's suites
        [node, "--test", str(HERE / suite)], capture_output=True, text=True, encoding="utf-8", check=False, cwd=ROOT
    )
    assert done.returncode == 0, done.stdout[-6000:] + done.stderr[-2000:]


#: Names a reader could type that a model cannot hold, and Python's keywords, which it cannot either.
TYPED_NAMES = [
    *("Water %", "Yield (%)", "Temp (°C)", "Run #", "Conc. [g/L]", "2nd stage", "  Yield  ", "a__b", "Été", "x²"),
    *("a·b", "Size (µm)", "__import__('os')", "", "%", "match", *keyword.kwlist),
]


def test_the_page_suggests_the_names_the_library_does(node):
    """``suggestName`` in ``web/logic.js`` gives each typed name the one ``suggest_identifier`` gives it."""
    if node_version(node) < LEAST:
        pytest.skip(f"Node.js {'.'.join(map(str, LEAST))} or later reads web/*.js as ES modules")
    script = (
        "import { suggestName } from './web/logic.js';"
        f"console.log(JSON.stringify({json.dumps(TYPED_NAMES)}.map(suggestName)));"
    )
    done = subprocess.run(  # noqa: S603 - Node.js running the page's own module
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=ROOT,
    )
    assert json.loads(done.stdout) == [suggest_identifier(name) for name in TYPED_NAMES]


#: Reasons a number is not in the library yet: a long clause naming what it lacks, a short one, one ending in a
#: stop, and none.
MISSING_REASONS = [
    "a step of Bartlett's test, which the library does not have",
    "the library has no all-bias composite design: no construction for it",
    "an estimate of the reweighted fit, which the library cannot make for a quadratic in two factors; it is checked",
    "Lenth's multiplier, which the library makes no use of in this table.",
    "a count from an enumeration of designs",
]


def test_the_book_pages_group_what_is_missing_as_the_reproduction_page_does(node, build_script):
    """``missing_key`` in the build, for a book's page, gives each reason the key ``missingKey`` gives it."""
    if node_version(node) < LEAST:
        pytest.skip(f"Node.js {'.'.join(map(str, LEAST))} or later reads web/*.js as ES modules")
    script = (
        "import { missingKey } from './web/reproduced-logic.js';"
        f"console.log(JSON.stringify({json.dumps(MISSING_REASONS)}.map(missingKey)));"
    )
    done = subprocess.run(  # noqa: S603 - Node.js running the page's own module
        [node, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=ROOT,
    )
    keys = json.loads(done.stdout)
    assert keys == [build_script.missing_key(reason) for reason in MISSING_REASONS]
    assert len(set(keys)) == len(keys), "each reason here is a group of its own"
    assert keys[1] == "The library has no all-bias composite design", "a long clause is the key"
