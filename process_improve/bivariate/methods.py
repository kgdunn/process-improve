"""Backwards-compatible re-exporter for ``process_improve.bivariate``.

The implementation now lives in
:mod:`process_improve.bivariate._elbow_peak` (ENG-23 / #305): the renamed
file makes filename-ranked tooling (Jump-to-File, fuzzy search, codecov
reports) less ambiguous about which ``methods.py`` is being shown.

Every public name remains importable as before::

    from process_improve.bivariate.methods import find_elbow_point, find_line_intersection
"""

from process_improve.bivariate._elbow_peak import (
    find_elbow_point,
    find_line_intersection,
    fit_robust_lm,
)

__all__ = [
    "find_elbow_point",
    "find_line_intersection",
    "fit_robust_lm",
]
