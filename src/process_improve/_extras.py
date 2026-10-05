"""ENG-13 (#295): clean errors when an optional extra is not installed.

``process_improve`` ships a small core (numpy, pandas, scipy, scikit-learn,
statsmodels, patsy, pydantic, pyyaml, tqdm) and gates heavier
optional dependencies behind extras::

    pip install 'process-improve[plotting]'    # matplotlib + plotly + seaborn + ridgeplot
    pip install 'process-improve[expt]'        # pyDOE3 (DOE / experiments helpers)
    pip install 'process-improve[batch]'       # scikit-image + openpyxl + ruptures
    pip install 'process-improve[mcp]'         # mcp
    pip install 'process-improve[fast]'        # numba (JIT)
    pip install 'process-improve[control]'     # osqp (mid-course correction QP)
    pip install 'process-improve[all]'         # everything above

Modules that need an optional dependency import it inside a
``try / except ImportError`` and re-raise via :func:`require_extra`
so the caller sees a concrete remediation::

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise require_extra("plotly", "plotting") from exc
"""

from __future__ import annotations


def _extra_message(missing: str, extra: str) -> str:
    """Format the canonical 'install the extra' remediation message."""
    return (
        f"process_improve needs {missing!r}, which is part of the optional "
        f"{extra!r} extra. Install it with:\n"
        f"    pip install 'process-improve[{extra}]'\n"
        f"or install every optional dependency at once with:\n"
        f"    pip install 'process-improve[all]'"
    )


def require_extra(missing: str, extra: str) -> ImportError:
    """Return an ``ImportError`` whose message tells the caller which extra to install.

    Use as ``raise require_extra(...) from exc`` inside an
    ``except ImportError`` block. We return the exception (rather than
    raising it directly) so the caller can use ``raise ... from`` and
    preserve the original traceback.

    Parameters
    ----------
    missing
        Name of the missing module the caller tried to import (e.g. ``"plotly"``).
    extra
        Name of the process-improve extra that provides it
        (e.g. ``"plotting"``).
    """
    return ImportError(_extra_message(missing, extra))


class _MissingExtra:
    """Stand-in for an optional module that was not installed.

    Using the stub in any way that needs the real package raises
    :class:`ImportError` with the canonical "install the extra" message, so a
    missing ``plotting`` extra fails the same way as a missing ``expt`` extra.
    Attribute access returns a child stub that remembers the dotted path, so
    ``go.Figure(...)`` and ``go.layout.Template(...)`` raise ``ImportError`` at
    the call, and indexing (``pio.templates[name]``) raises it too.

    Dunder lookups (``__array__``, ``__deepcopy__``, ...) raise
    :class:`AttributeError` instead, so protocol probes such as ``copy``,
    ``pickle`` and numpy see a missing attribute, per the Python data model.

    Lets a module top-level write::

        try:
            import plotly.graph_objects as go
        except ImportError:
            go = _MissingExtra("plotly", "plotting")

    without breaking ``from <that module> import <core-name>`` for users
    who never touch the optional surface.
    """

    __slots__ = ("_extra", "_missing", "_path")

    def __init__(self, missing: str, extra: str, path: str = "") -> None:
        self._missing = missing
        self._extra = extra
        self._path = path

    def _error(self) -> ImportError:
        hint = f"\n(Attempted to use {self._path!r}.)" if self._path else ""
        return ImportError(f"{_extra_message(self._missing, self._extra)}{hint}")

    def __getattr__(self, name: str) -> _MissingExtra:
        if name.startswith("__") and name.endswith("__"):
            # __getattr__ must raise AttributeError for protocol lookups so
            # ``hasattr`` and friends behave per the data model (CodeQL
            # py/non-standard-exception-raised-in-special-method).
            raise AttributeError(
                f"{_extra_message(self._missing, self._extra)}\n(Attempted attribute access: {name!r}.)"
            )
        return _MissingExtra(self._missing, self._extra, f"{self._path}.{name}" if self._path else name)

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        raise self._error()

    def __getitem__(self, _key: object) -> object:
        raise self._error()

    def __setitem__(self, _key: object, _value: object) -> None:
        raise self._error()
