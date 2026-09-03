# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Bounded download of the sample datasets that are not bundled with the package.

The remote sample datasets (the `openmv.net <https://openmv.net>`_ files behind
:mod:`process_improve.experiments.datasets` and
:mod:`process_improve.batch.datasets`) are fetched on demand. Every fetch goes
through :func:`fetch_remote_bytes`, so one place enforces the contract:

- the URL is a fixed ``https`` location chosen by the library, or a ``file://``
  path a caller passes explicitly to read a local copy; it is never free text
  from an untrusted source;
- the download is bounded by ``settings.dataset_fetch_timeout`` (30 s by
  default; ``PROCESS_IMPROVE_DATASET_FETCH_TIMEOUT`` overrides it), so a
  black-holing host raises instead of hanging the caller indefinitely (#508);
- network failures, timeouts and parse failures surface as one clear
  ``RuntimeError`` naming the URL, rather than a lower-level error.

The content is trusted only as far as the remote host is.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd

from process_improve._extras import require_extra
from process_improve.config import settings


def _download_error(url: str, exc: Exception) -> RuntimeError:
    """Return the one ``RuntimeError`` every failure mode maps onto."""
    return RuntimeError(
        f"Could not download the sample dataset from {url!r}: {exc}. "
        "Check your network connection; this dataset is fetched from a remote host."
    )


def fetch_remote_bytes(url: str, timeout: float | None = None) -> bytes:
    """Download ``url`` and return the raw payload.

    Parameters
    ----------
    url : str
        The fixed ``https`` URL of the file to fetch, or a ``file://`` URL
        for a local copy.
    timeout : float, optional
        Seconds to wait before giving up. Defaults to
        ``settings.dataset_fetch_timeout``.

    Returns
    -------
    bytes
        The downloaded content.

    Raises
    ------
    RuntimeError
        On any network failure or timeout (``TimeoutError`` and
        ``urllib.error.URLError`` are both ``OSError`` subclasses), with the
        URL in the message.
    """
    if timeout is None:
        timeout = settings.dataset_fetch_timeout
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https or file URLs
            return response.read()
    except OSError as exc:
        raise _download_error(url, exc) from exc


def read_remote_csv(url: str, timeout: float | None = None) -> pd.DataFrame:
    """Fetch a CSV file with :func:`fetch_remote_bytes` and parse it.

    Parameters
    ----------
    url : str
        The fixed ``https`` (or ``file://``) URL of the CSV file.
    timeout : float, optional
        Seconds to wait before giving up. Defaults to
        ``settings.dataset_fetch_timeout``.

    Returns
    -------
    pd.DataFrame
        The parsed table.

    Raises
    ------
    RuntimeError
        On a download failure or when the payload is not a readable CSV.
    """
    payload = fetch_remote_bytes(url, timeout=timeout)
    try:
        return pd.read_csv(io.BytesIO(payload))
    except ValueError as exc:
        raise _download_error(url, exc) from exc


def read_remote_excel(
    url: str,
    timeout: float | None = None,
    *,
    sheet_name: str | list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch an Excel workbook with :func:`fetch_remote_bytes` and parse its sheets.

    Reading ``.xlsx`` files needs ``openpyxl``, which ships with the ``batch``
    extra; a missing install raises an ``ImportError`` that names the extra.

    Parameters
    ----------
    url : str
        The fixed ``https`` (or ``file://``) URL of the workbook.
    timeout : float, optional
        Seconds to wait before giving up. Defaults to
        ``settings.dataset_fetch_timeout``.
    sheet_name : str or list of str, optional
        Sheets to read. ``None`` (default) reads every sheet.

    Returns
    -------
    dict[str, pd.DataFrame]
        One table per sheet, keyed by sheet name, also when a single sheet
        name was requested.

    Raises
    ------
    RuntimeError
        On a download failure or when the payload is not a readable workbook.
    ImportError
        When ``openpyxl`` is not installed.
    """
    payload = fetch_remote_bytes(url, timeout=timeout)
    try:
        import openpyxl  # noqa: F401, PLC0415 - probed here so the error names the extra to install
    except ImportError as exc:
        raise require_extra("openpyxl", "batch") from exc
    try:
        sheets = pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, engine="openpyxl")
    except (ValueError, zipfile.BadZipFile) as exc:
        raise _download_error(url, exc) from exc
    if isinstance(sheets, pd.DataFrame):
        return {str(sheet_name): sheets}
    return {str(name): frame for name, frame in sheets.items()}
