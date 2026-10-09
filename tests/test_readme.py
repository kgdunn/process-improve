"""Run every Python example in the README and the sklearn notes, and check the output each promises.

A page's blocks share one namespace, as a reader pasting them one after another would.
They read their data from openmv.net; here those URLs are served from the copies bundled with
the package, so the test needs no network. A ``print(...)`` followed by a comment on the same
line promises that output, and the comment is compared with what the call printed. Numbers may
differ in their last printed digit, so a platform's floating point does not fail the test.
"""

from __future__ import annotations

import ast
import math
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import pytest

mpl.use("Agg")
pytestmark = [pytest.mark.dataset, pytest.mark.slow]

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "process_improve" / "datasets" / "multivariate"
#: The openmv.net files the README reads, and the copies bundled with the package.
BUNDLED = {
    "https://openmv.net/file/LDPE.csv": DATA / "LDPE" / "LDPE.csv",
    "https://openmv.net/file/cheddar-cheese.csv": DATA / "cheddar-cheese.csv",
}
FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
ECHO = re.compile(r"^print\((?P<args>.*)\)  # (?P<promised>.+)$")


def python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, code)`` for every fenced Python block of a Markdown file."""
    text = path.read_text(encoding="utf-8")
    return [(text.count("\n", 0, match.start(1)) + 1, match.group(1)) for match in FENCE.finditer(text)]


def close(printed: object, promised: object) -> bool:
    """Compare two parsed outputs: floats to the last printed digit, everything else exactly."""
    if isinstance(printed, dict) and isinstance(promised, dict):
        return list(printed) == list(promised) and all(close(printed[k], promised[k]) for k in printed)
    if isinstance(printed, (list, tuple)) and isinstance(promised, (list, tuple)):
        return len(printed) == len(promised) and all(close(a, b) for a, b in zip(printed, promised, strict=True))
    if isinstance(printed, float) or isinstance(promised, float):
        numbers = (int, float)
        return (
            isinstance(printed, numbers)
            and isinstance(promised, numbers)
            and not isinstance(printed, bool)
            and math.isclose(printed, promised, rel_tol=1e-3, abs_tol=0.011)
        )
    return type(printed) is type(promised) and printed == promised


def same(printed: str, promised: str) -> bool:
    """Whether a printed line matches the output the README promises for it."""
    try:
        return close(ast.literal_eval(printed), ast.literal_eval(promised))
    except (ValueError, SyntaxError):
        return printed.split() == promised.split()


@pytest.mark.parametrize("page", ["README.md", "SKLEARN_COMPATIBILITY.md"])
def test_examples_run_and_print_what_they_promise(page: str, monkeypatch: pytest.MonkeyPatch) -> None:
    read_csv = pd.read_csv

    def offline_read_csv(source: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            if source not in BUNDLED:
                pytest.fail(f"{page} reads {source}: add its bundled copy to BUNDLED in {__name__}")
            source = BUNDLED[source]
        return read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", offline_read_csv)
    echoes: list[tuple[str, str, str]] = []  # where, what was printed, what the page promises

    def echo(where: str, promised: str, *values: object) -> None:
        echoes.append((where, " ".join(map(str, values)), promised))

    namespace: dict[str, object] = {"__echo__": echo}
    blocks = python_blocks(ROOT / page)
    assert blocks, f"no Python blocks found in {page}"
    for start, code in blocks:
        lines = code.splitlines()
        for i, line in enumerate(lines):
            if match := ECHO.match(line):
                lines[i] = f"__echo__({f'{page}:{start + i}'!r}, {match['promised']!r}, {match['args']})"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile("\n".join(lines), f"{page}:{start}", "exec"), namespace)  # noqa: S102
        ours = [w for w in caught if issubclass(w.category, DeprecationWarning) and "process_improve" in str(w.message)]
        assert not ours, f"{page}:{start} uses deprecated API: {ours[0].message}"

    assert echoes, "no promised outputs found: the check would pass vacuously"
    wrong = [f"{where}: printed {printed!r}, the page promises {promised!r}" for where, printed, promised in echoes]
    wrong = [line for line, (_, printed, promised) in zip(wrong, echoes, strict=True) if not same(printed, promised)]
    assert not wrong, "\n".join(wrong)
