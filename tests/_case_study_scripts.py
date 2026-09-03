"""Helpers for the tests that pin the numbers printed by the batch case-study scripts.

The case studies live as plain scripts next to their documentation pages in
``docs/user_guide/case_studies/batch/``. They are not importable as a package,
so :func:`load_script` imports one by path; the tests then call its functions
and assert on the models and frames they return, which keeps the documented
numbers and the tested numbers the same code path.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from typing import TYPE_CHECKING, TypeVar

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

T = TypeVar("T")

CASE_STUDY_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "case_studies" / "batch"

# The SBR workbook is served from openmv.net; until it is uploaded, or when
# working offline, point the SBR tests at a local copy with a file:// URL.
SBR_URL_OVERRIDE = os.environ.get("PROCESS_IMPROVE_SBR_URL")


def load_script(name: str) -> ModuleType:
    """Import ``docs/user_guide/case_studies/batch/<name>.py`` as a module."""
    path = CASE_STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"case_study_{name}", path)
    if spec is None or spec.loader is None:
        msg = f"Cannot import the case-study script {path}."
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_or_skip(loader: Callable[[], T]) -> T:
    """Call a remote data loader, skipping the calling test when the download fails."""
    try:
        return loader()
    except RuntimeError as exc:
        raise pytest.skip.Exception(f"Cannot download the dataset: {exc}") from exc
