# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Backward-compatibility shim for ``process_improve.multivariate`` (ENG-01).

The implementation that used to live here has been split into focused
submodules (:mod:`~process_improve.multivariate._pca`, ``_pls``, ``_tpls``,
``_mbpls``, ``_mbpca``, ``_preprocessing``, ``_nipals``, ``_limits``,
``_diagnostics``, ``_resampling`` and ``plots``) and is aggregated by
:mod:`process_improve.multivariate.methods`. This shim re-exports that public
surface (plus the private ``_nz`` helper) so that existing imports such as
``from process_improve.multivariate._pca_pls import PCA`` keep working.

The star import below is controlled by ``methods.__all__``, so it does not
pollute the namespace (see ENG-23 / #305).
"""

from __future__ import annotations

from . import methods as _methods

# Re-exported (listed in __all__ below) for back-compat: some tests import _nz
# from this module.
from ._common import _nz
from .methods import *  # noqa: F403  re-export the full public surface

__all__ = [*_methods.__all__, "_nz"]  # noqa: PLE0604 -- _methods.__all__ is a list[str]
