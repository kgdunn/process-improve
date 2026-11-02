"""Every ``DeprecationWarning`` in ``src/`` follows ``docs/development/deprecation_policy.rst``.

The policy fixes the message (the deprecated name, the version that announced it, the version that
removes it, and the replacement) and the schedule (announced in X.Y.0, removed in (X+1).0). The
scan reads each ``warnings.warn(..., DeprecationWarning)`` call's message from the source, so the
test also fails once the package version reaches a removal version: that is the reminder to delete
the shim.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

#: The policy's message: ``process_improve.X.old is deprecated since 1.23.0 and will be removed in 2.0; use ...``.
MESSAGE = re.compile(
    r"^process_improve\.\S+ is deprecated since (?P<since>\d+)\.\d+\.\d+ "
    r"and will be removed in (?P<removal>\d+)\.0; (?:use \S.* instead|no replacement)"
)


def _text(node: ast.expr) -> str | None:
    """Return the text of a string literal, with ``{}`` for each f-string field; ``None`` if it is not a literal."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in node.values)
    return None


def deprecation_messages(source: str) -> list[tuple[int, str | None]]:
    """Return the line and message of each ``warn`` call in ``source`` whose category is ``DeprecationWarning``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        positional = node.args[1] if len(node.args) > 1 else None
        category = next((k.value for k in node.keywords if k.arg == "category"), positional)
        if name == "warn" and isinstance(category, ast.Name) and category.id == "DeprecationWarning":
            found.append((node.lineno, _text(node.args[0])))
    return found


def problems(message: str | None, version: str) -> list[str]:
    """Return how ``message`` breaks the policy for a package at ``version``; empty when it complies."""
    if message is None:
        return ["the message is not a string literal, so it cannot be checked"]
    match = MESSAGE.match(message)
    if match is None:
        return [f"the message does not follow the policy's format: {message!r}"]
    since, removal = int(match["since"]), int(match["removal"])
    found = []
    if removal != since + 1:
        found.append(f"announced in {since}.x, so it is removed in {since + 1}.0, not {removal}.0")
    if int(version.partition(".")[0]) >= removal:
        found.append(f"the package is at {version}, past the removal in {removal}.0: delete the deprecated path")
    return found


def test_every_deprecation_warning_follows_the_policy() -> None:
    """Each deprecation names what goes, when it was announced, when it goes and what replaces it."""
    report = [
        f"{path.relative_to(ROOT)}:{line}: {problem}"
        for path in sorted((ROOT / "src").rglob("*.py"))
        for line, message in deprecation_messages(path.read_text(encoding="utf-8"))
        for problem in problems(message, VERSION)
    ]
    assert not report, "\n".join(report)


def test_the_scan_finds_and_judges_each_kind_of_deprecation() -> None:
    """The scan is not vacuous: it finds both spellings of the call, and it rejects each kind of breach."""
    source = """
import warnings
from warnings import warn
warnings.warn("process_improve.a is deprecated since 2.1.0 and will be removed in 3.0; use b instead.",
              category=DeprecationWarning, stacklevel=2)
warn(f"process_improve.{x} is deprecated since 1.5.0 and will be removed in 2.0; no replacement.", DeprecationWarning)
warnings.warn("old is going away", category=DeprecationWarning)
warnings.warn(message, category=DeprecationWarning)
warnings.warn("process_improve.c is deprecated since 2.1.0 and will be removed in 4.0; use d instead.",
              category=DeprecationWarning)
warnings.warn("not a deprecation", category=UserWarning)
"""
    found = deprecation_messages(source)
    assert [line for line, _ in found] == [4, 6, 7, 8, 9]
    verdicts = [problems(message, "2.1.0") for _, message in found]
    assert verdicts[0] == []
    assert verdicts[1] == ["the package is at 2.1.0, past the removal in 2.0: delete the deprecated path"]
    assert "format" in verdicts[2][0]
    assert "not a string literal" in verdicts[3][0]
    assert verdicts[4] == ["announced in 2.x, so it is removed in 3.0, not 4.0"]
