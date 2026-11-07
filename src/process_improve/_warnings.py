"""(c) Kevin Dunn, 2010-2026. MIT License.

The ``stacklevel`` that points a warning at the caller's line, for every warning the package raises.

It sits at the top of the package, and imports nothing from it, so that any module can import it
at module level: ``multivariate`` imports ``experiments``, so a helper kept in ``multivariate``
could be reached from ``experiments`` only by an import inside each function.
"""

from __future__ import annotations

import inspect

#: The packages a warning must not point into: this one, scikit-learn, whose ``_fit_context``
#: wraps ``fit``, and mutmut, whose trampoline calls each function that the nightly mutation run
#: mutates (#1092). A frame is read by its module's name, which stays the same wherever the
#: file is, so the copy of this package that mutmut mutates is still this package.
_LIBRARY_PACKAGES = frozenset({"process_improve", "sklearn", "mutmut"})


def user_stacklevel() -> int:
    """Return the ``stacklevel`` that points a warning at the first frame outside the library: the caller's line.

    Pass it as ``warnings.warn(..., stacklevel=user_stacklevel())``. A fixed count is wrong as
    soon as a frame comes between the caller and the warning: a private helper, sklearn's
    ``_fit_context`` wrapper, a subclass's ``fit``, an inner fit, or mutmut's trampoline. The
    warning then names a line in the library, which the caller cannot act on (#1079).
    """
    frame = inspect.currentframe()
    frame = frame.f_back if frame is not None else None  # the function that warns: stacklevel 1
    level = 1
    while frame is not None and str(frame.f_globals.get("__name__", "")).partition(".")[0] in _LIBRARY_PACKAGES:
        frame = frame.f_back
        level += 1
    return level
