"""ENG-13 (#295): clean errors when an optional extra is not installed.

``process_improve`` ships a small core (numpy, pandas, scikit-learn,
statsmodels, patsy, pydantic, pyyaml, tqdm) and gates heavier
optional dependencies behind extras::

    pip install 'process-improve[plotting]'    # matplotlib + plotly + seaborn + ridgeplot
    pip install 'process-improve[expt]'        # pyDOE3 (DOE / experiments helpers)
    pip install 'process-improve[batch]'       # scikit-image + openpyxl
    pip install 'process-improve[mcp]'         # mcp
    pip install 'process-improve[fast]'        # numba (JIT)
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

    Attribute access raises :class:`AttributeError` (Python's special-method
    convention -- ``hasattr(stub, 'Figure')`` returns ``False``) with the
    canonical "install the extra" remediation message in the body. Direct
    *calls* on the stub raise :class:`ImportError` because that is the
    natural error class for "you need to install this dependency".

    Lets a module top-level write::

        try:
            import plotly.graph_objects as go
        except ImportError:
            go = _MissingExtra("plotly", "plotting")

    without breaking ``from <that module> import <core-name>`` for users
    who never touch the optional surface.
    """

    __slots__ = ("_extra", "_missing")

    def __init__(self, missing: str, extra: str) -> None:
        self._missing = missing
        self._extra = extra

    def __getattr__(self, name: str) -> object:
        # __getattr__ must raise AttributeError so ``hasattr`` and friends
        # behave per the Python data model (CodeQL py/non-standard-exception
        # -raised-in-special-method). The install-hint message is preserved
        # in the exception text.
        raise AttributeError(
            f"{_extra_message(self._missing, self._extra)}\n"
            f"(Attempted attribute access: {name!r}.)"
        )

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        # __call__ has no equivalent convention; raise ImportError here
        # because callers that try to use the stub directly are asking
        # for the missing package.
        raise require_extra(self._missing, self._extra)
