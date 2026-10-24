"""The page's JavaScript, one function at a time: ``node --test`` runs each suite named here.

The suites import the page's modules from ``web/`` as the page does. Node.js reads a ``.js``
module without a package.json as an ES module from version 22.7; on an older Node.js these
tests skip, and CI's web-app job runs them on Node.js 22.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

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
