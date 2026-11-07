"""Every module defines each of its functions and classes once (#1144).

A second ``def`` of a name replaces the first without a word: Python looks the name up when a
function is called, so every caller then runs the second. ``tests/test_sensory.py`` defined
``_predictive_case`` twice, and the tests written for the first ran on the data of the second.
ruff reports a redefinition only while the first is unused, and here the first was used.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_each_module_level_function_and_class_is_defined_once() -> None:
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    repeated = {}
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]):
        counts = Counter(
            node.name for node in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(node, definitions)
        )
        names = sorted(name for name, count in counts.items() if count > 1)
        if names:
            repeated[str(path.relative_to(ROOT))] = names
    assert repeated == {}
