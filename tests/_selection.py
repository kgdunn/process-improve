"""Choose which tests to run: by what a change touches, and by shard.

Two independent tools, wired into pytest by ``tests/conftest.py``.

``affected_test_files``: test impact analysis from a static import graph
    Every ``.py`` file under ``src/`` and ``tests/`` is parsed, never imported.
    Each one gets an edge to every first-party module it imports, found in three
    places: ``import`` statements; string literals naming a ``process_improve``
    module (the lazy registries in ``tool_spec.py`` and ``recipes.py`` import
    their members that way); and Python source held in a string literal (the
    snippets tests run with ``subprocess.run([sys.executable, "-c", code])``).
    A test file is affected when it can reach a changed file along those edges.

    ``from package import name`` follows ``name`` to the module that defines it,
    through any ``__init__.py`` that only re-exports it. Without that step,
    ``from process_improve import PCA`` would reach the top-level ``__init__``,
    which imports ``tool_spec``, whose registry names every ``tools.py``: every
    test would depend on everything. The price is that a facade's import-time
    side effects are not seen; the ``coverage`` CI job runs the whole suite.

    A changed file that is not part of the graph (a CSV dataset, ``README.md``, a
    docs page) affects whichever Python files name it in a string literal: by
    file name, by stem, or failing those by its nearest named parent directory.
    Three things widen the selection rather than narrow it:

    - a test that executes code the graph cannot follow (``exec``, a module
      loaded from a file path) is "opaque" and runs on every ``src/`` change;
    - an affected ``conftest.py`` affects every test file below it;
    - a deleted file, or a build file in ``FULL_SUITE``, selects the whole suite.

``assign_shard``: split the suite across ``N`` CI runners
    Test files are placed by Longest-Processing-Time-first: sort them by recorded
    duration, then hand each to the currently lightest shard. The greedy rule is
    within 4/3 of the best possible split, it is deterministic (every runner and
    every xdist worker computes the same answer without talking to the others),
    and whole files stay together so module-scoped fixtures are built once.
"""

from __future__ import annotations

import ast
import heapq
import json
import re
import statistics
import textwrap
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# A change to any of these can alter every test, so it selects the whole suite.
FULL_SUITE = frozenset({"pyproject.toml", "pytest.ini", ".coveragerc", "uv.lock"})

_MODULE_NAME = re.compile(r"^process_improve(\.\w+)+$")

# Calls that run code from a file the graph cannot see: README examples, docs
# pages, a script loaded by path.
_OPAQUE_CALLS = frozenset({"exec", "spec_from_file_location", "run_path", "run_module"})


@dataclass
class ImportGraph:
    """Python files under ``src/`` and ``tests/``, and how they reach each other."""

    edges: dict[Path, set[Path]] = field(default_factory=dict)
    opaque: set[Path] = field(default_factory=set)
    # Each path component that appears in a string literal, and the files it appears in.
    mentions: defaultdict[str, set[Path]] = field(default_factory=lambda: defaultdict(set))

    def importers_of(self, starts: set[Path]) -> set[Path]:
        """Return ``starts`` plus every file that imports one of them, directly or not."""
        reverse: defaultdict[Path, set[Path]] = defaultdict(set)
        for source, targets in self.edges.items():
            for target in targets:
                reverse[target].add(source)
        seen, queue = set(starts), deque(starts)
        while queue:
            for importer in reverse[queue.popleft()] - seen:
                seen.add(importer)
                queue.append(importer)
        return seen

    def files_naming(self, path: Path, root: Path) -> set[Path]:
        """Python files that name ``path``, or failing that its nearest named parent directory."""
        candidates = [path.name, path.stem, *(p.name for p in path.relative_to(root).parents if p.name)]
        for name in candidates:
            if name in self.mentions:
                return self.mentions[name]
        return set()


def _module_name(path: Path, root: Path) -> str:
    """Dotted module name: ``src/process_improve/x.py`` -> ``process_improve.x``, ``tests/t.py`` -> ``tests.t``."""
    base = root / "src" if path.is_relative_to(root / "src") else root
    parts = list(path.relative_to(base).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


# ``import a.b`` is ("a.b", ""); ``from a import b`` is ("a", "b"): ``b`` may be a
# submodule of ``a`` or a name bound in it, which only the graph can tell apart.
Import = tuple[str, str]


def _absolute(node: ast.ImportFrom, package: str) -> str:
    """Return the module a ``from ... import`` names; ``from ..x import y`` inside ``a.b.c`` is ``a.b.x``."""
    if not node.level:
        return node.module or ""
    anchor = package.rsplit(".", node.level - 1)[0]
    return f"{anchor}.{node.module}" if node.module else anchor


def _imports(tree: ast.AST, package: str) -> set[Import]:
    """Every ``import`` and ``from ... import`` in ``tree``, wherever it sits."""
    found: set[Import] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update((alias.name, "") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.update((_absolute(node, package), alias.name) for alias in node.names)
    return found


def _module_level(nodes: Iterable[ast.AST]) -> Iterator[ast.AST]:
    """Nodes that run at import time: everything outside function and class bodies."""
    for node in nodes:
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield from _module_level(ast.iter_child_nodes(node))


def _reexports(tree: ast.Module, package: str) -> dict[str, Import]:
    """Names a module binds with ``from x import y`` at import time: bound name -> (x, y)."""
    return {
        alias.asname or alias.name: (_absolute(node, package), alias.name)
        for node in _module_level(tree.body)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name != "*"
    }


def _embedded_imports(literal: str) -> set[Import]:
    """Find the imports made by Python source held in a string, such as a ``python -c`` snippet."""
    if "import " not in literal:
        return set()
    try:
        return _imports(ast.parse(textwrap.dedent(literal)), package="")
    except SyntaxError:
        return set()


@dataclass
class _Parsed:
    imports: set[Import]
    reexports: dict[str, Import]
    components: set[str]  # path components found in string literals
    opaque: bool


def _parse(path: Path, module: str) -> _Parsed:
    """Read one file's imports, re-exports, string-literal path components and opacity."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    parsed = _Parsed(_imports(tree, package), _reexports(tree, package), set(), opaque=False)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literal = node.value
            if _MODULE_NAME.match(literal):
                parsed.imports.add((literal, ""))
            parsed.imports |= _embedded_imports(literal)
            parsed.components.update(re.split(r"[/\\]", literal))
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            parsed.opaque |= called in _OPAQUE_CALLS
    return parsed


def build_import_graph(root: Path) -> ImportGraph:
    """Parse every Python file under ``src/`` and ``tests/`` into an :class:`ImportGraph`."""
    files = sorted(p for folder in ("src", "tests") for p in (root / folder).rglob("*.py"))
    by_module = {_module_name(p, root): p for p in files}
    parsed = {path: _parse(path, _module_name(path, root)) for path in files}
    reexports = {_module_name(path, root): info.reexports for path, info in parsed.items()}

    def resolve(module: str, name: str) -> Path | None:
        """File that defines ``module.name``: a submodule, or where a re-export leads."""
        seen: set[Import] = set()
        while name and (module, name) not in seen:
            seen.add((module, name))
            if f"{module}.{name}" in by_module:
                return by_module[f"{module}.{name}"]
            if name not in reexports.get(module, {}):
                break
            module, name = reexports[module][name]
        return by_module.get(module)

    graph = ImportGraph()
    for path, info in parsed.items():
        targets = {resolve(module, name) for module, name in info.imports}
        graph.edges[path] = {t for t in targets if t is not None} - {path}
        for component in info.components:
            graph.mentions[component].add(path)
        if info.opaque and path.is_relative_to(root / "tests"):
            graph.opaque.add(path)
    return graph


def affected_test_files(changed: list[str], root: Path) -> set[Path] | None:
    """Test files that ``changed`` (paths relative to ``root``) can affect; ``None`` means all of them."""
    graph = build_import_graph(root)
    seeds: set[Path] = set()
    for name in changed:
        path = (root / name).resolve()
        if name in FULL_SUITE or not path.exists():
            return None
        seeds |= {path} if path in graph.edges else graph.files_naming(path, root)

    affected = graph.importers_of(seeds)
    if any(seed.is_relative_to(root / "src") for seed in seeds):
        affected |= graph.importers_of(graph.opaque)

    tests = {p for p in graph.edges if p.name.startswith("test_") and p.is_relative_to(root / "tests")}
    for conftest in {p for p in affected if p.name == "conftest.py"}:
        affected |= {t for t in tests if t.is_relative_to(conftest.parent)}
    return affected & tests


def assign_shard(tests_per_file: dict[str, int], durations: dict[str, float], shard: int, n_shards: int) -> set[str]:
    """Files belonging to ``shard`` (1-based) of ``n_shards``, balanced by duration.

    A file with a recorded duration costs that many seconds. A file without one
    (a new test file) is estimated from its test count, at the median
    seconds-per-test of the files that have a recorded duration.
    """
    rates = [durations[f] / n for f, n in tests_per_file.items() if f in durations and n]
    per_test = statistics.median(rates) if rates else 1.0
    cost = {f: durations.get(f, n * per_test) for f, n in tests_per_file.items()}

    shards = [(0.0, k) for k in range(1, n_shards + 1)]  # (load in seconds, shard number), a min-heap
    owner: dict[str, int] = {}
    for name in sorted(cost, key=lambda f: (-cost[f], f)):
        load, k = heapq.heappop(shards)
        owner[name] = k
        heapq.heappush(shards, (load + cost[name], k))
    return {f for f, k in owner.items() if k == shard}


def load_durations(path: Path) -> dict[str, float]:
    """Read the recorded seconds per test file; nothing when the file is missing."""
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
