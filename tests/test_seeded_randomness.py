"""Every random draw in the tests is seeded, so a test gives the same verdict on every run (#1073).

A test that draws unseeded data or folds can pass on one run and fail on the next, and the
nightly mutation tests then kill a mutant on some runs only. This scan reads the source of every
Python file under ``tests/`` and fails on a call that draws from an unseeded generator: a
generator made without a seed, a draw from NumPy's or Python's global generator in a function
that does not seed it, a shuffling scikit-learn splitter without a ``random_state``, and an
integer ``cv`` without a ``random_state``, which the package's functions shuffle into folds.

A call that writes its seed as ``None``, such as ``random_state=None``, is left alone: that is a
choice, and a test that compares a function's default settings has to write it. The scan sees only
the calls written in the tests: a package function whose default ``cv`` is an integer, called
without a ``random_state``, also draws unseeded folds, so seed such calls by hand.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent

#: Calls that make a generator, seeded by their first argument.
GENERATORS = {"default_rng", "RandomState", "Random"}
#: Attributes of ``np.random`` and ``random`` that do not draw from the global generator.
NOT_GLOBAL_DRAWS = {
    *GENERATORS,
    "BitGenerator",
    "Generator",
    "MT19937",
    "PCG64",
    "PCG64DXSM",
    "Philox",
    "SFC64",
    "SeedSequence",
    "SystemRandom",
    "get_state",
    "getstate",
    "seed",
    "set_state",
    "setstate",
}
#: scikit-learn splitters that shuffle only when asked to, and those that always shuffle.
SHUFFLE_IF_ASKED = {"KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold"}
ALWAYS_SHUFFLE = {
    "GroupShuffleSplit",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    "train_test_split",
}
#: Keywords that, given an integer, make the package shuffle rows into that many folds.
FOLD_COUNTS = ("cv", "outer_cv", "inner_cv")


def _dotted(node: ast.expr) -> str:
    """Return ``"np.random.normal"`` for the callee ``np.random.normal``, and ``""`` for a computed one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _literal(node: ast.expr | None) -> object:
    """Return the value of a literal argument, and ``None`` for an absent or computed one."""
    return node.value if isinstance(node, ast.Constant) else None


def _global_draw(dotted: str) -> bool:
    """Return whether a callee such as ``np.random.normal`` or ``random.choice`` draws from a global generator."""
    module, _, name = dotted.rpartition(".")
    return module in ("np.random", "numpy.random", "random") and name not in NOT_GLOBAL_DRAWS


def unseeded_draw(call: ast.Call, *, seeded_globally: bool) -> str | None:
    """Return why ``call`` draws from an unseeded generator, or ``None`` when it does not.

    ``seeded_globally`` is true inside a function that seeds the global generators itself.
    """
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if None in keywords:  # **kwargs: the seed may be in there
        return None
    dotted = _dotted(call.func)
    name = dotted.rpartition(".")[2]
    unseeded = "random_state" not in keywords
    shuffle = _literal(keywords.get("shuffle"))
    if name in GENERATORS and not call.args and not keywords:
        return f"{dotted}() makes a generator without a seed"
    if _global_draw(dotted) and not seeded_globally:
        return f"{dotted}() draws from the global generator, which this function does not seed"
    if unseeded and ((name in ALWAYS_SHUFFLE and shuffle is not False) or (name in SHUFFLE_IF_ASKED and shuffle)):
        return f"{dotted}() shuffles without a random_state"
    for keyword in FOLD_COUNTS:
        count = _literal(keywords.get(keyword))
        if unseeded and isinstance(count, int) and not isinstance(count, bool):
            return f"{dotted}({keyword}={count}) has no random_state, and the package shuffles an integer {keyword}"
    return None


class _Scan(ast.NodeVisitor):
    """Collect each unseeded draw in a module, with its line number."""

    def __init__(self) -> None:
        self.found: list[tuple[int, str]] = []
        self.seeded_globally = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        outer = self.seeded_globally
        self.seeded_globally = outer or any(
            isinstance(call, ast.Call) and _dotted(call.func) in ("np.random.seed", "numpy.random.seed", "random.seed")
            for call in ast.walk(node)
        )
        self.generic_visit(node)
        self.seeded_globally = outer

    def visit_Call(self, node: ast.Call) -> None:
        reason = unseeded_draw(node, seeded_globally=self.seeded_globally)
        if reason:
            self.found.append((node.lineno, reason))
        self.generic_visit(node)


def unseeded_draws(source: str) -> list[tuple[int, str]]:
    """Return the line and the reason of every unseeded draw in a module's source."""
    scan = _Scan()
    scan.visit(ast.parse(source))
    return scan.found


def test_no_test_draws_from_an_unseeded_generator() -> None:
    found = [
        f"{path.relative_to(TESTS.parent)}:{line}: {reason}"
        for path in sorted(TESTS.rglob("*.py"))
        for line, reason in unseeded_draws(path.read_text(encoding="utf-8"))
    ]
    assert not found, (
        "These calls draw from an unseeded generator, so the tests can give another verdict on the next run. "
        "Pass a seed or a random_state, or a splitter such as KFold(k):\n" + "\n".join(found)
    )


@pytest.mark.parametrize(
    ("unseeded", "seeded"),
    [
        ("np.random.default_rng()", "np.random.default_rng(seed=0)"),
        ("np.random.RandomState()", "np.random.RandomState(0)"),
        ("random.Random()", "random.Random(0)"),
        ("np.random.normal(size=3)", "rng.normal(size=3)"),
        ("random.choice([1, 2])", "rng.choice([1, 2])"),
        ("KFold(4, shuffle=True)", "KFold(4, shuffle=True, random_state=0)"),
        ("RepeatedKFold(n_splits=4)", "KFold(4)"),
        ("train_test_split(X, y)", "train_test_split(X, y, shuffle=False)"),
        ("StratifiedShuffleSplit(5)", "StratifiedShuffleSplit(5, random_state=0)"),
        ("PLS.select_n_components(X, Y, cv=4)", "PLS.select_n_components(X, Y, cv=4, random_state=0)"),
        ("PLS.nested_cv(X, Y, outer_cv=3)", "PLS.nested_cv(X, Y, outer_cv=3, random_state=0)"),
    ],
)
def test_the_scan_finds_each_unseeded_draw_and_passes_its_seeded_form(unseeded: str, seeded: str) -> None:
    assert len(unseeded_draws(unseeded)) == 1
    assert unseeded_draws(seeded) == []


def test_a_function_that_seeds_the_global_generator_may_draw_from_it() -> None:
    """A test that checks a function leaves the caller's global state alone has to draw from it."""
    source = "def test():\n    np.random.seed(5)\n    np.random.random()\n\ndef other():\n    np.random.random()\n"
    assert [line for line, _ in unseeded_draws(source)] == [6]


def test_a_seed_written_as_none_is_a_choice() -> None:
    """A test that compares a function's default settings writes ``random_state=None`` on purpose."""
    calls = [
        "np.random.default_rng(None)",
        "KFold(4, shuffle=True, random_state=None)",
        "PLS.nested_cv(outer_cv=3, random_state=None)",
    ]
    assert unseeded_draws("\n".join(calls)) == []


def test_a_call_with_unpacked_keywords_is_not_judged() -> None:
    assert unseeded_draws("KFold(4, shuffle=True, **settings)") == []
