"""Every ``DeprecationWarning`` and ``FutureWarning`` in ``src/`` follows ``docs/development/deprecation_policy.rst``.

The policy fixes the message (the deprecated name, the version that announced it, the version that
removes it, and the replacement) and the schedule (announced in X.Y.0, removed in (X+1).0). The
scan reads each ``warnings.warn(..., DeprecationWarning)`` call's message from the source, so the
test also fails once the package version reaches a removal version: that is the reminder to delete
the shim.

A ``FutureWarning`` announces a changed default on the same schedule. Its message names the release
that announced it and the major release that makes the change, and the test fails once the package
reaches that release: the reminder to flip the default and delete the warning and its note (#1132).
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


#: A changed default's message: the release that announced it, and the major release that makes the change.
ANNOUNCED = re.compile(r"\b(?:since|announced in) (?P<since>\d+)\.\d+\.\d+")
CHANGED = re.compile(r"\b(?:in|from) (?P<change>\d+)\.0\b")


def _resolve(node: ast.expr, scope: ast.AST, functions: dict[str, ast.FunctionDef]) -> str | None:
    """Return the text of a message, following a name or a key to its assignment and a call to its return."""
    text = _text(node)
    if text is not None:
        return text
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
        returned = (n.value for n in ast.walk(functions[node.func.id]) if isinstance(n, ast.Return) and n.value)
        return next((t for t in map(_text, returned) if t is not None), None)
    if isinstance(node, ast.Name | ast.Subscript):
        for assign in (n for n in ast.walk(scope) if isinstance(n, ast.Assign)):
            if any(ast.unparse(target) == ast.unparse(node) for target in assign.targets):
                return _resolve(assign.value, scope, functions)
    return None


def warn_messages(source: str, category: str = "DeprecationWarning") -> list[tuple[int, str | None]]:
    """Return the line and message of each ``warn`` call in ``source`` whose category is ``category``.

    A message held in a name or a key (``note``, ``out["intervals_note"]``) is read from its assignment in the
    function that warns, and one a function of the module returns (``judged_rows_note(...)``) from its return.
    """
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    scope_of: dict[ast.AST, ast.AST] = {}
    # ast.walk visits an outer function before the ones inside it, so the innermost scope is the last one set.
    for scope in [tree, *(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))]:
        scope_of |= dict.fromkeys(ast.walk(scope), scope)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        positional = node.args[1] if len(node.args) > 1 else None
        given = next((k.value for k in node.keywords if k.arg == "category"), positional)
        if name == "warn" and isinstance(given, ast.Name) and given.id == category:
            found.append((node.lineno, _resolve(node.args[0], scope_of[node], functions)))
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


def future_problems(message: str | None, version: str) -> list[str]:
    """Return how a changed default's ``message`` breaks the policy for a package at ``version``."""
    if message is None:
        return ["the message cannot be read from the source, so it cannot be checked"]
    announced, changed = ANNOUNCED.search(message), CHANGED.search(message)
    if announced is None or changed is None:
        return [f"the message does not name the release that announced it and the one that changes it: {message!r}"]
    since, change = int(announced["since"]), int(changed["change"])
    found = []
    if change != since + 1:
        found.append(f"announced in {since}.x, so it changes in {since + 1}.0, not {change}.0")
    if int(version.partition(".")[0]) >= change:
        found.append(
            f"the package is at {version}, past the change in {change}.0: make the change, and delete the warning "
            "and any note that repeats it"
        )
    return found


def test_every_deprecation_warning_follows_the_policy() -> None:
    """Each deprecation names what goes, when it was announced, when it goes and what replaces it."""
    report = [
        f"{path.relative_to(ROOT)}:{line}: {problem}"
        for path in sorted((ROOT / "src").rglob("*.py"))
        for line, message in warn_messages(path.read_text(encoding="utf-8"))
        for problem in problems(message, VERSION)
    ]
    assert not report, "\n".join(report)


def test_every_future_warning_names_the_release_that_changes_the_default() -> None:
    """Each changed default names when it was announced and when it changes, and goes when the package gets there."""
    found = [
        (path, line, message)
        for path in sorted((ROOT / "src").rglob("*.py"))
        for line, message in warn_messages(path.read_text(encoding="utf-8"), "FutureWarning")
    ]
    report = [
        f"{path.relative_to(ROOT)}:{line}: {problem}"
        for path, line, message in found
        for problem in future_problems(message, VERSION)
    ]
    assert found, "the scan found no FutureWarning in src/; check that it still reads the calls"
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
    found = warn_messages(source)
    assert [line for line, _ in found] == [4, 6, 7, 8, 9]
    verdicts = [problems(message, "2.1.0") for _, message in found]
    assert verdicts[0] == []
    assert verdicts[1] == ["the package is at 2.1.0, past the removal in 2.0: delete the deprecated path"]
    assert "format" in verdicts[2][0]
    assert "not a string literal" in verdicts[3][0]
    assert verdicts[4] == ["announced in 2.x, so it is removed in 3.0, not 4.0"]


def test_the_scan_reads_each_kind_of_future_warning_message() -> None:
    """A changed default's message is read through a name, a key and a function's return, and each breach is caught."""
    source = """
import warnings
def note(level):
    if level is None:
        return None
    return f"process_improve.f: level=None gives {level} until 3.0 and from 3.0 another, as announced in 2.4.0."
def f(level, out):
    text = note(level)
    warnings.warn(text, FutureWarning, stacklevel=2)
    out["note"] = "process_improve.g's default is deprecated since 2.3.1 and will change in 3.0."
    warnings.warn(out["note"], category=FutureWarning)
    warnings.warn("process_improve.h is deprecated since 2.4.0 and an error from 4.0.", FutureWarning)
    warnings.warn("process_improve.k changes one day.", FutureWarning)
    warnings.warn(make_message(), FutureWarning)
    warnings.warn("process_improve.m is deprecated since 2.4.0 and an error from 3.0.", DeprecationWarning)
"""
    found = warn_messages(source, "FutureWarning")
    assert [line for line, _ in found] == [9, 11, 12, 13, 14]
    verdicts = [future_problems(message, "2.4.0") for _, message in found]
    assert verdicts[0] == []
    assert verdicts[1] == []
    assert verdicts[2] == ["announced in 2.x, so it changes in 3.0, not 4.0"]
    assert "does not name the release" in verdicts[3][0]
    assert "cannot be read" in verdicts[4][0]
    (past,) = future_problems(found[1][1], "3.0.0")
    assert past.startswith("the package is at 3.0.0, past the change in 3.0: make the change")
