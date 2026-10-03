"""Run the Python examples of the design-of-experiments user-guide pages, in page order.

Each page is one script: its ``.. code-block:: python`` blocks run in order in a shared
namespace, as a reader would paste them. A block that raises, or that triggers one of the
library's own deprecation warnings, fails the page, so the documented examples cannot rot.
"""

from __future__ import annotations

import re
import textwrap
import warnings
from pathlib import Path

import matplotlib as mpl
import pytest

mpl.use("Agg")

_PAGES = Path(__file__).resolve().parent.parent / "docs" / "user_guide"
_DIRECTIVE = re.compile(r"^(?P<indent>\s*)\.\. code-block:: python\s*$")


def python_blocks(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, code)`` for every Python code block in an RST file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    found, i = [], 0
    while i < len(lines):
        match = _DIRECTIVE.match(lines[i])
        i += 1
        if not match:
            continue
        indent = len(match["indent"])
        while i < len(lines) and lines[i].lstrip().startswith(":"):  # directive options
            i += 1
        start, body = i, []
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent):
            body.append(lines[i])
            i += 1
        found.append((start, textwrap.dedent("\n".join(body))))
    return found


@pytest.mark.parametrize(
    "page",
    [
        "constrained_designs.rst",
        "design_evaluation.rst",
        "screening_and_space_filling.rst",
        "doe_strategy.rst",
        pytest.param("omars_designs.rst", marks=pytest.mark.slow),
    ],
)
def test_page_examples_run(page: str) -> None:
    path = _PAGES / page
    blocks = python_blocks(path)
    assert blocks, f"no Python blocks found in {page}"
    namespace: dict[str, object] = {}
    for line, code in blocks:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            exec(compile(code, f"{path}:{line}", "exec"), namespace)  # noqa: S102
        ours = [w for w in caught if issubclass(w.category, DeprecationWarning) and "process_improve" in str(w.message)]
        assert not ours, f"{page}:{line} uses deprecated API: {ours[0].message}"
