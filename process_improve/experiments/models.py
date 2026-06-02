"""Backwards-compatible re-exporter for ``process_improve.experiments``.

The implementation now lives in :mod:`process_improve.experiments._lm`
(ENG-23 / #305): the renamed file makes filename-ranked tooling
(Jump-to-File, fuzzy search, codecov reports) less ambiguous about which
``models.py`` is being shown.

Every public name remains importable as before::

    from process_improve.experiments.models import Model, lm, predict, summary, validate_formula_is_safe
"""

from process_improve.experiments._lm import *  # noqa: F403
