"""Backwards-compatible re-exporter for ``process_improve.multivariate``.

The implementation now lives in :mod:`process_improve.multivariate._pca_pls`
(ENG-23 / #305): the renamed file makes filename-ranked tooling (Jump-to-File,
fuzzy search, codecov reports) less ambiguous about which ``methods.py`` is
being shown.

Every public name remains importable as before::

    from process_improve.multivariate.methods import PCA, PLS, MCUVScaler
"""

from process_improve.multivariate._pca_pls import *  # noqa: F403
