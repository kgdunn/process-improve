"""Changelog fragments: each pull request adds a file of its own, filed into ``CHANGELOG.md`` at release.

When every pull request inserted at the same anchor of ``CHANGELOG.md``, every open pull request
conflicted with every other one. A pull request now adds ``changelog.d/<slug>.<type>.md`` instead,
holding its entry as one or more Markdown bullets, so two pull requests never touch the same
lines. ``<type>`` is the heading the entry belongs under, and it picks the release level as the
heading does (see "Choosing the level at release" in ``CONTRIBUTING.md``).

    python tools/changelog.py check   # exit status 1 if a fragment is not well formed
    python tools/changelog.py build   # at release: file every fragment under ## [Unreleased]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS = ROOT / "changelog.d"
#: Each fragment type and its heading, in the order the headings appear under a release.
HEADINGS = {
    "added": "Added",
    "changed": "Changed",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "fixed": "Fixed",
    "security": "Security",
    "documentation": "Documentation",
}
NAME = re.compile(rf"^[a-z0-9][a-z0-9-]*\.({'|'.join(HEADINGS)})\.md$")
EM_DASH = "\u2014"


def fragments(folder: Path) -> list[Path]:
    """Return the fragments in ``folder`` sorted by name; the folder's README is not one."""
    return sorted(path for path in folder.glob("*.md") if path.name != "README.md")


def problems(folder: Path) -> list[str]:
    """Return what is wrong with each fragment in ``folder``: its name or its text."""
    found = []
    for path in fragments(folder):
        text = path.read_text(encoding="utf-8")
        if not NAME.match(path.name):
            found.append(f"{path.name}: name it <slug>.<type>.md, with <type> one of {', '.join(HEADINGS)}")
        if not text.startswith("- "):
            found.append(f"{path.name}: the entry is a Markdown bullet, so it starts with '- '")
        if any(line.startswith("#") for line in text.splitlines()):
            found.append(f"{path.name}: the type in the name picks the heading, so the file holds none")
        if EM_DASH in text:
            found.append(f"{path.name}: no em-dashes; use a colon, a comma or two sentences")
    return found


def _unreleased(lines: list[str]) -> tuple[int, int]:
    """Return where ``## [Unreleased]`` starts and where the release after it starts."""
    start = lines.index("## [Unreleased]")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")), len(lines))
    return start, end


def _filed(lines: list[str], heading: str, entry: list[str]) -> list[str]:
    """Return the changelog's lines with ``entry`` added at the end of ``heading`` under ``[Unreleased]``.

    A heading the section does not have yet is opened in its place in the order of ``HEADINGS``.
    """
    start, end = _unreleased(lines)
    title = f"### {heading}"
    if title not in lines[start:end]:
        order = list(HEADINGS.values())
        later = {f"### {name}" for name in order[order.index(heading) + 1 :]}
        at = next((i for i in range(start, end) if lines[i] in later), end)
        return [*lines[:at], title, "", *entry, "", *lines[at:]]
    begin = lines.index(title, start)
    stop = next((i for i in range(begin + 1, end) if lines[i].startswith("### ")), end)
    while lines[stop - 1] == "":  # after the heading's last bullet, before the blank line that ends it
        stop -= 1
    return [*lines[:stop], *entry, *lines[stop:]]


def build(changelog: Path, folder: Path) -> int:
    """File every fragment under its heading in ``## [Unreleased]``, delete it, and return how many."""
    lines = changelog.read_text(encoding="utf-8").splitlines()
    filed = sorted(fragments(folder), key=lambda path: (list(HEADINGS).index(path.name.split(".")[-2]), path.name))
    for path in filed:
        entry = path.read_text(encoding="utf-8").rstrip("\n").splitlines()
        lines = _filed(lines, HEADINGS[path.name.split(".")[-2]], entry)
    changelog.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in filed:
        path.unlink()
    return len(filed)


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "check"
    if command not in ("check", "build"):
        print(__doc__)
        return 2
    found = problems(FRAGMENTS)
    for problem in found:
        print(problem)
    if found:
        return 1
    if command == "build":
        print(f"{build(CHANGELOG, FRAGMENTS)} fragment(s) filed under ## [Unreleased] in {CHANGELOG.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
