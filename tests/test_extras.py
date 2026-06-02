"""Direct tests for ``process_improve._extras`` (ENG-13 / #295).

The optional-extras fallback branches (``except ImportError:`` at each
gated import site) cannot fire when CI installs ``--all-extras``;
they ship with ``# pragma: no cover``. The ``_MissingExtra`` proxy
and ``require_extra`` helper *are* exercised here, by constructing
the proxy directly and asserting its documented exception semantics.
"""

from __future__ import annotations

import pytest

from process_improve._extras import _extra_message, _MissingExtra, require_extra


class TestRequireExtra:
    def test_returns_import_error(self) -> None:
        exc = require_extra("plotly", "plotting")
        assert isinstance(exc, ImportError)

    def test_message_mentions_extra_and_install_hint(self) -> None:
        msg = str(require_extra("pyDOE3", "expt"))
        assert "pyDOE3" in msg
        assert "expt" in msg
        assert "pip install 'process-improve[expt]'" in msg
        assert "pip install 'process-improve[all]'" in msg


class TestExtraMessage:
    def test_format(self) -> None:
        text = _extra_message("numba", "fast")
        assert "numba" in text
        assert "fast" in text
        assert "process-improve[fast]" in text


class TestMissingExtraProxy:
    """Stand-in for a not-installed optional module."""

    def test_attribute_access_raises_attribute_error(self) -> None:
        # __getattr__ must raise AttributeError so ``hasattr`` works
        # (CodeQL py/non-standard-exception-raised-in-special-method).
        stub = _MissingExtra("plotly", "plotting")
        with pytest.raises(AttributeError, match=r"plotly.*plotting"):
            _ = stub.Figure

    def test_attribute_access_message_carries_install_hint(self) -> None:
        stub = _MissingExtra("plotly", "plotting")
        with pytest.raises(AttributeError) as info:
            _ = stub.Figure
        assert "pip install 'process-improve[plotting]'" in str(info.value)
        assert "Figure" in str(info.value)

    def test_hasattr_returns_false(self) -> None:
        # Direct corollary of __getattr__ raising AttributeError.
        stub = _MissingExtra("plotly", "plotting")
        assert hasattr(stub, "Figure") is False
        assert hasattr(stub, "anything_else_at_all") is False

    def test_call_raises_import_error(self) -> None:
        # __call__ has no equivalent special-method convention; raise
        # ImportError (the natural class for "you need to install this").
        stub = _MissingExtra("pyDOE3", "expt")
        with pytest.raises(ImportError, match=r"pyDOE3.*expt"):
            stub()

    def test_call_with_args_raises_import_error(self) -> None:
        stub = _MissingExtra("pyDOE3", "expt")
        with pytest.raises(ImportError):
            stub(1, 2, foo="bar")

    def test_message_includes_both_install_commands(self) -> None:
        """Each exception body lists the specific extra *and* the [all] meta extra."""
        stub = _MissingExtra("seaborn", "plotting")
        with pytest.raises(ImportError) as info:
            stub()
        text = str(info.value)
        assert "'process-improve[plotting]'" in text
        assert "'process-improve[all]'" in text
