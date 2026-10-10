"""Guard the parts of the README that ``test_readme.py`` does not run.

``test_readme.py`` executes every code block and checks the numbers they print. These tests
cover the rest of the page, offline: the calls quoted in its text and tables, the links into
this repository and its documentation, and the installation instructions, which must match
``pyproject.toml``. Links to other sites are checked weekly by ``tools/check_readme_links.py``.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pkgutil
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest

import process_improve

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
#: Extras a reader is not told to install: the contributors' tools and a deprecated empty one.
UNDOCUMENTED_EXTRAS = {"dev", "ilp"}
_PATH = r"([^\s)\"'#?<>]*)"  # a URL's path, up to a query, an anchor or the end of the link
REPOSITORY_LINK = re.compile(
    r"https://(?:github\.com/kgdunn/process-improve/(?:blob|tree)/main|"
    r"raw\.githubusercontent\.com/kgdunn/process-improve/main)/" + _PATH
)
DOCS_LINK = re.compile(r"https://kgdunn\.github\.io/process-improve/" + _PATH)


def text_outside_code_blocks(markdown: str) -> str:
    """Return the Markdown with its fenced code blocks removed; ``test_readme.py`` runs those."""
    return re.sub(r"^```.*?^```", "", markdown, flags=re.MULTILINE | re.DOTALL)


def package_name(requirement: str) -> str:
    """Return the normalised distribution name of a requirement such as ``'scikit-learn>=1.7'``."""
    match = re.match(r"[A-Za-z0-9._-]+", requirement.strip())
    assert match, requirement
    return re.sub(r"[-_.]+", "-", match.group()).lower()


@functools.cache
def package_api() -> dict[str, list[Callable[..., object]]]:
    """Map every public function, class and method name in ``process_improve`` to the objects that carry it."""
    index: dict[str, list[Callable[..., object]]] = defaultdict(list)
    modules = pkgutil.walk_packages(process_improve.__path__, "process_improve.")
    for name in (m.name for m in modules if not any(part.startswith("_") for part in m.name.split(".")[1:])):
        for attribute, obj in vars(importlib.import_module(name)).items():
            if attribute.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if not obj.__module__.startswith("process_improve"):
                continue  # imported from another library
            index[attribute].append(obj)
            if inspect.isclass(obj):
                for member, value in inspect.getmembers(obj, callable):
                    if not member.startswith("_"):
                        index[member].append(value)
    return index


def accepts(function: Callable[..., object], keywords: set[str]) -> bool:
    """Whether ``function`` (or a class's constructor) takes all of ``keywords``."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return True  # no signature to check against
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    return takes_any or keywords <= parameters.keys()


@pytest.mark.slow
def test_every_call_in_the_readme_text_exists_in_the_package() -> None:
    """A call quoted in the text or a table names a real function, class or method, with keywords it takes.

    Only the code blocks run; a call in the question table such as
    ``model.invert(y_desired=20.9)`` would otherwise go stale silently after a rename.
    """
    api, problems = package_api(), []
    for span in re.findall(r"`([^`\n]+)`", text_outside_code_blocks(README)):
        try:
            tree = ast.parse(span, mode="eval")
        except SyntaxError:
            continue  # not Python: a shell command, a file name, an expression fragment
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, (ast.Name, ast.Attribute)):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            if name not in api:
                problems.append(f"`{span}`: process_improve has no function, class or method {name!r}")
            elif not any(accepts(candidate, keywords) for candidate in api[name]):
                problems.append(f"`{span}`: no {name!r} in process_improve takes {sorted(keywords)}")
    assert not problems, "\n".join(problems)


def docs_sources(page: str) -> list[Path]:
    """Return the files that could build ``page``, a path on the documentation site."""
    if page == "app/":
        return [ROOT / "web" / "index.html"]  # scripts/build_web_app.py publishes web/ at /app
    stem = page.removesuffix(".html") if page and not page.endswith("/") else f"{page}index"
    return [ROOT / "docs" / f"{stem}{suffix}" for suffix in (".rst", ".md", ".ipynb")]


def anchor(heading: str) -> str:
    """Return the anchor GitHub gives a Markdown heading: lower case, punctuation dropped, spaces hyphenated."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_links_into_this_repository_and_its_documentation_resolve() -> None:
    """Every link into the repository names a file on disk, every docs link a page source, every anchor a heading."""
    problems = []
    for page in ("README.md", "SKLEARN_COMPATIBILITY.md"):
        text = (ROOT / page).read_text(encoding="utf-8")
        headings = {anchor(h) for h in re.findall(r"^#+ (.+)$", text_outside_code_blocks(text), flags=re.MULTILINE)}
        problems += [
            f"{page}: {path} is not in the repository"
            for path in REPOSITORY_LINK.findall(text)
            if not (ROOT / path).exists()
        ]
        problems += [
            f"{page}: the docs page {docs_page!r} has no source in docs/"
            for docs_page in DOCS_LINK.findall(text)
            if not any(source.exists() for source in docs_sources(docs_page))
        ]
        problems += [
            f"{page}: no heading produces #{target}"
            for target in re.findall(r"\]\(#([^)]+)\)", text)
            if target not in headings
        ]
    assert not problems, "\n".join(problems)


def test_installation_instructions_match_pyproject() -> None:
    """The Python floor, the core dependencies and each extra's packages are what ``pyproject.toml`` declares."""
    floor = re.search(r"Requires Python ([\d.]+) or newer", README)
    assert floor, "the README no longer states the oldest supported Python"
    assert PYPROJECT["requires-python"] == f">={floor.group(1)}"

    core = re.search(r"The core install pulls in (.+?)\.\n", README, flags=re.DOTALL)
    assert core, "the README no longer lists the core dependencies"
    listed_core = {package_name(name) for name in re.findall(r"`([^`]+)`", core.group(1))}
    assert listed_core == {package_name(requirement) for requirement in PYPROJECT["dependencies"]}

    extras = {
        name: {package_name(requirement) for requirement in requirements}
        for name, requirements in PYPROJECT["optional-dependencies"].items()
    }
    install_lines = dict(re.findall(r"pip install 'process-improve\[(\w+)\]'\s+# (.+)", README))
    assert set(install_lines) == {name for name, packages in extras.items() if packages} - UNDOCUMENTED_EXTRAS

    listed = {}
    for extra, comment in install_lines.items():
        if extra == "all":
            assert comment.startswith("everything above"), comment
            continue
        assert comment.startswith("adds "), f"[{extra}]: say which packages it adds"
        listed[extra] = {package_name(name) for name in re.split(r", | and ", comment[5:].split(" (")[0])}
        assert listed[extra] == extras[extra], f"[{extra}]: the README lists {sorted(listed[extra])}"
    missing = set().union(*listed.values()) - extras["all"]
    assert not missing, f"the README says [all] installs everything above, but it lacks {sorted(missing)}"
