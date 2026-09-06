"""Tests for the shared remote sample-dataset reader, :mod:`process_improve._remote_data`.

No test here performs network access: ``urlopen`` is monkeypatched, or a
``file://`` URL points at a temporary file.
"""

from __future__ import annotations

import io
import re
import sys
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from process_improve import _remote_data as remote_data
from process_improve.config import settings

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable
    from typing import Self


class _FakeResponse:
    """Minimal stand-in for the ``urlopen`` context-manager response."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


def _workbook_bytes() -> bytes:
    pytest.importorskip("openpyxl")
    trajectories = pd.DataFrame({"batch_id": [1, 1, 2, 2], "time": [1, 2, 1, 2], "Temp": [0.1, 0.2, 0.3, 0.4]})
    quality = pd.DataFrame({"batch_id": [1, 2], "Quality": [10.0, 11.0]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        trajectories.to_excel(writer, sheet_name="X_batch", index=False)
        quality.to_excel(writer, sheet_name="Y_quality", index=False)
    return buffer.getvalue()


def _serve(payload: bytes) -> Callable[..., _FakeResponse]:
    """Return a ``urlopen`` stand-in that always serves ``payload``."""

    def _fake_urlopen(_url: str, timeout: float | None = None) -> _FakeResponse:
        return _FakeResponse(payload)

    return _fake_urlopen


def test_read_remote_excel_parses_every_sheet(monkeypatch: pytest.MonkeyPatch) -> None:
    """All sheets come back keyed by name, and the default timeout is the settings value."""
    payload = _workbook_bytes()
    seen: dict[str, float | None] = {}

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
        seen["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _fake_urlopen)
    sheets = remote_data.read_remote_excel("https://openmv.net/file/example.xlsx")

    assert set(sheets) == {"X_batch", "Y_quality"}
    assert sheets["X_batch"].shape == (4, 3)
    assert sheets["Y_quality"].shape == (2, 2)
    assert sheets["X_batch"]["Temp"].dtype.kind == "f"
    assert seen["timeout"] == settings.dataset_fetch_timeout


def test_read_remote_excel_single_sheet_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single requested sheet is still returned inside a one-entry dict."""
    payload = _workbook_bytes()
    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _serve(payload))
    sheets = remote_data.read_remote_excel("https://openmv.net/file/example.xlsx", sheet_name="Y_quality")
    assert list(sheets) == ["Y_quality"]
    assert sheets["Y_quality"].shape == (2, 2)


def test_read_remote_excel_wraps_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed download surfaces as the documented ``RuntimeError`` naming the URL."""

    def _boom(_url: str, timeout: float | None = None) -> _FakeResponse:
        raise OSError("name resolution failed")

    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _boom)
    expected = re.escape("Could not download the sample dataset from 'https://openmv.net/file/x.xlsx'")
    with pytest.raises(RuntimeError, match=expected):
        remote_data.read_remote_excel("https://openmv.net/file/x.xlsx")


def test_read_remote_excel_wraps_bad_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A payload that is not a workbook is reported as a download problem, not a parser traceback."""
    pytest.importorskip("openpyxl")
    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _serve(b"not a workbook"))
    with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
        remote_data.read_remote_excel("https://openmv.net/file/x.xlsx")


def test_read_remote_csv_wraps_bad_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unparseable CSV bytes map onto the same ``RuntimeError``."""
    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _serve(b""))
    with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
        remote_data.read_remote_csv("https://openmv.net/file/x.csv")


def test_fetch_remote_bytes_explicit_timeout_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``timeout`` is passed through to ``urlopen`` unchanged."""
    seen: dict[str, float | None] = {}

    def _fake_urlopen(url: str, timeout: float | None = None) -> _FakeResponse:
        seen["timeout"] = timeout
        return _FakeResponse(b"a,b\n1,2\n")

    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _fake_urlopen)
    assert remote_data.fetch_remote_bytes("https://openmv.net/file/x.csv", timeout=2.5) == b"a,b\n1,2\n"
    assert seen["timeout"] == 2.5


def test_read_remote_csv_reads_a_local_file_url(tmp_path: pathlib.Path) -> None:
    """A ``file://`` URL reads a local copy, which is how a downloaded workbook is used offline."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("batch_id,time,Temp\n1,1,0.5\n1,2,0.6\n")
    frame = remote_data.read_remote_csv(csv_path.as_uri())
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"batch_id": [1, 1], "time": [1, 2], "Temp": [0.5, 0.6]}))


def test_read_remote_excel_names_the_extra_when_openpyxl_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without openpyxl the error tells the caller which extra to install."""
    monkeypatch.setattr(remote_data.urllib.request, "urlopen", _serve(b"irrelevant"))
    monkeypatch.setitem(sys.modules, "openpyxl", None)  # makes ``import openpyxl`` raise ImportError
    with pytest.raises(ImportError, match="batch"):
        remote_data.read_remote_excel("https://openmv.net/file/x.xlsx")
