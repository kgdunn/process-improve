"""Backwards-compatible re-exporter for ``process_improve.regression``.

The implementation now lives in
:mod:`process_improve.regression._robust_regression` (ENG-23 / #305): the
renamed file makes filename-ranked tooling (Jump-to-File, fuzzy search,
codecov reports) less ambiguous about which ``methods.py`` is being shown.

Every public name remains importable as before::

    from process_improve.regression.methods import OLS, robust_regression, repeated_median_slope
"""

from process_improve.regression._robust_regression import *  # noqa: F403
