"""Tests for the experiments dataset loaders.

``boilingpot()`` loads from a CSV bundled with the package; ``oildoe()``
and ``distillateflow()`` fetch from openmv.net. The remaining functions
in ``process_improve.experiments.datasets`` are still docstring-only
stubs (they return ``None``); we exercise them so they keep counting
toward coverage.
"""

from __future__ import annotations

import re
import urllib.error
from collections.abc import Callable

import pandas as pd
import pytest
from _pytest.outcomes import Skipped

from process_improve.experiments import datasets


def _load_or_skip(loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Call the network-backed loader, ``pytest.skip`` on any network error.

    ``RuntimeError`` is the one that matters in practice, and it has to be
    listed explicitly.  ``_read_remote_csv`` deliberately converts the
    lower-level ``OSError`` / ``ValueError`` into a ``RuntimeError`` carrying an
    actionable message, and ``RuntimeError`` is not an ``OSError``.  Catching
    only the urllib types therefore skipped nothing: a remote outage (openmv.net
    returning 502, say) failed the build rather than skipping, contrary to what
    this helper and the tests below claim.  The other types are kept in case a
    loader ever raises one directly.
    """
    try:
        return loader()
    except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        pytest.skip(f"could not fetch from openmv.net: {exc}")


def test_load_or_skip_skips_when_the_remote_host_is_unavailable() -> None:
    """A remote outage must skip, not fail the build.

    ``_read_remote_csv`` wraps network failures in a ``RuntimeError``, so that
    is what the helper actually sees.  Pinning it here because the mismatch is
    invisible until openmv.net has an outage, at which point every build fails.
    """

    def unavailable() -> pd.DataFrame:
        raise RuntimeError(
            "Could not download the sample dataset from 'https://openmv.net/file/oil-company-doe.csv': "
            "HTTP Error 502: Bad Gateway."
        )

    with pytest.raises(Skipped, match=re.escape("could not fetch from openmv.net")):
        _load_or_skip(unavailable)


def test_load_or_skip_returns_the_frame_when_the_loader_succeeds() -> None:
    """The guard must not swallow a working fetch."""
    frame = pd.DataFrame({"A": [1], "y": [2]})
    assert _load_or_skip(lambda: frame) is frame


def test_boilingpot_loads() -> None:
    """``boilingpot()`` returns the documented 11x4 factorial frame."""
    df = datasets.boilingpot()
    assert df.shape == (11, 4)
    assert set(df.columns) == {"A", "B", "C", "y"}


@pytest.mark.dataset
def test_oildoe_loads() -> None:
    """``oildoe()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.oildoe)
    assert df.shape == (19, 5)
    assert set(df.columns) == {"A", "B", "C", "D", "y"}


@pytest.mark.dataset
def test_distillateflow_loads() -> None:
    """``distillateflow()`` fetches the openmv.net file (skipped if offline)."""
    df = _load_or_skip(datasets.distillateflow)
    assert df.shape == (44640, 1)
    assert "Flow" in df.columns


def test_pollutant_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.pollutant() is None


def test_golf_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.golf() is None


def test_solar_returns_none() -> None:
    """The stub should be callable and return None."""
    assert datasets.solar() is None


def test_data_dispatch_signature_typed() -> None:
    """``datasets.data`` is the planned dispatcher; verify its signature
    is preserved so we notice if it is later refactored away.
    """
    # The annotation is stored as a string under PEP 563
    # (``from __future__ import annotations``).
    assert datasets.data.__annotations__["return"] == "pd.DataFrame"
    assert datasets.data.__annotations__["dataset"] == "str"


class TestRemoteFetchRetries:
    """A transient blip at the remote host must not fail a whole run.

    Sample data is fetched live, so a momentary 502 from openmv.net has failed
    an entire CI job.  Retrying rides that out; the tests below pin both that it
    retries when it should and that it does not when retrying cannot help.
    """

    def test_transient_error_is_retried_and_succeeds(self, monkeypatch) -> None:
        frame = pd.DataFrame({"A": [1]})
        calls = {"n": 0}

        def flaky(_url):
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.HTTPError(_url, 502, "Bad Gateway", {}, None)
            return frame

        monkeypatch.setattr(datasets.pd, "read_csv", flaky)
        monkeypatch.setattr(datasets.time, "sleep", lambda _s: None)
        assert datasets._read_remote_csv("https://openmv.net/file/x.csv") is frame
        assert calls["n"] == 3

    def test_gives_up_after_the_attempt_budget(self, monkeypatch) -> None:
        calls = {"n": 0}

        def always_down(_url):
            calls["n"] += 1
            raise urllib.error.HTTPError(_url, 503, "Service Unavailable", {}, None)

        monkeypatch.setattr(datasets.pd, "read_csv", always_down)
        monkeypatch.setattr(datasets.time, "sleep", lambda _s: None)
        with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
            datasets._read_remote_csv("https://openmv.net/file/x.csv")
        assert calls["n"] == datasets._FETCH_ATTEMPTS

    @pytest.mark.parametrize("status", [404, 403, 410])
    def test_settled_http_status_is_not_retried(self, monkeypatch, status: int) -> None:
        """A 404 says the same thing next time; do not pay the backoff for it."""
        calls = {"n": 0}

        def missing(_url):
            calls["n"] += 1
            raise urllib.error.HTTPError(_url, status, "nope", {}, None)

        monkeypatch.setattr(datasets.pd, "read_csv", missing)
        monkeypatch.setattr(datasets.time, "sleep", lambda _s: pytest.fail("should not back off"))
        with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
            datasets._read_remote_csv("https://openmv.net/file/x.csv")
        assert calls["n"] == 1

    def test_parse_error_is_not_retried(self, monkeypatch) -> None:
        """A ValueError means the fetch worked and the content was wrong."""
        calls = {"n": 0}

        def bad_content(_url):
            calls["n"] += 1
            raise ValueError("could not parse")

        monkeypatch.setattr(datasets.pd, "read_csv", bad_content)
        monkeypatch.setattr(datasets.time, "sleep", lambda _s: pytest.fail("should not back off"))
        with pytest.raises(RuntimeError, match="Could not download the sample dataset"):
            datasets._read_remote_csv("https://openmv.net/file/x.csv")
        assert calls["n"] == 1

    def test_connection_error_is_retried(self, monkeypatch) -> None:
        """A bare OSError (DNS, reset connection) is worth another try."""
        calls = {"n": 0}

        def dns_fail(_url):
            calls["n"] += 1
            raise OSError("name resolution failed")

        monkeypatch.setattr(datasets.pd, "read_csv", dns_fail)
        monkeypatch.setattr(datasets.time, "sleep", lambda _s: None)
        with pytest.raises(RuntimeError):
            datasets._read_remote_csv("https://openmv.net/file/x.csv")
        assert calls["n"] == datasets._FETCH_ATTEMPTS

    def test_attempts_can_be_disabled(self, monkeypatch) -> None:
        calls = {"n": 0}

        def down(_url):
            calls["n"] += 1
            raise urllib.error.HTTPError(_url, 502, "Bad Gateway", {}, None)

        monkeypatch.setattr(datasets.pd, "read_csv", down)
        with pytest.raises(RuntimeError):
            datasets._read_remote_csv("https://openmv.net/file/x.csv", attempts=1)
        assert calls["n"] == 1
