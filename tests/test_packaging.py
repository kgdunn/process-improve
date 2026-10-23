"""Tests of what the package publishes: its extras, and the files its wheel carries."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRAS: dict[str, list[str]] = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "optional-dependencies"
]


def test_the_all_extra_is_exactly_the_union_of_the_others() -> None:
    """``[all]`` is hand-listed, so a package or a floor changed in one extra must change there too (#513)."""
    others = {requirement.replace(" ", "") for name, group in EXTRAS.items() if name != "all" for requirement in group}
    listed = {requirement.replace(" ", "") for requirement in EXTRAS["all"]}
    assert listed == others, (
        f"[all] lacks {sorted(others - listed)} and has {sorted(listed - others)} that no other extra declares"
    )
