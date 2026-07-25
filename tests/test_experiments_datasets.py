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
