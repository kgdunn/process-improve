# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Shared base class and mixins for the latent-variable estimators (ENG-17).

PCA and PLS duplicated a large block of identical scaffolding: the ENG-05
convenience methods (``score_plot``, ``vip``, ``spe_limit``, ...), the
``hotellings_t2_limit`` / ``ellipse_coordinates`` wrappers, and the attribute
rename ``__getattr__``. This module factors that out:

* :class:`_RenameGetattrMixin` - the migration ``__getattr__`` driven by a
  per-class ``_ATTRIBUTE_RENAMES`` dict and ``_RENAME_CONTEXT`` label.
* :class:`_HotellingsT2LimitMixin` - the ``hotellings_t2_limit`` method shared,
  byte-identically, by PCA / PLS / MBPLS / MBPCA (it reads ``self.n_components``
  and ``self.n_samples_``).
* :class:`_LatentVariableModel` - the PCA/PLS convenience-method surface plus
  ``ellipse_coordinates``.

A fix to any of this shared scaffolding now propagates to all derived classes at
once. TPLS keeps its own copies (its ``hotellings_t2_limit`` / ``ellipse_coordinates``
read differently-named fitted attributes), so it does not inherit these.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ._common import _model_method
from ._diagnostics import (
    eigenvalue_summary as _eigenvalue_summary,
)
from ._diagnostics import (
    observation_contributions as _observation_contributions,
)
from ._diagnostics import (
    project_variables as _project_variables,
)
from ._diagnostics import (
    squared_cosine as _squared_cosine,
)
from ._diagnostics import (
    vip as _vip,
)
from ._limits import (
    ellipse_coordinates as _ellipse_coordinates,
)
from ._limits import (
    hotellings_t2_limit as _hotellings_t2_limit,
)
from ._limits import (
    score_limit as _score_limit,
)
from ._limits import (
    spe_limit as _spe_limit,
)
from .plots import (
    correlation_loadings_plot as _correlation_loadings_plot,
)
from .plots import (
    explained_variance_plot as _explained_variance_plot,
)
from .plots import (
    loading_plot as _loading_plot,
)
from .plots import (
    score_plot as _score_plot,
)
from .plots import (
    spe_plot as _spe_plot,
)
from .plots import (
    t2_plot as _t2_plot,
)


class _LazyFrame:
    """Expose a private ndarray as a lazily-built, cached :class:`pandas.DataFrame` (ENG-18).

    Declared as a class attribute, e.g.::

        scores_   = _LazyFrame("_scores",   index="_sample_index",  columns="_component_names")
        loadings_ = _LazyFrame("_loadings", index="_feature_names", columns="_component_names")

    The private ndarray (``self._scores``) is the source of truth; the public
    ``DataFrame`` is built on first access from the ndarray plus the index/column
    metadata attributes, cached in ``self.__dict__["_frame_cache"]`` (so repeated
    access returns the same object and is cheap), and excluded from pickling by
    :meth:`_LatentVariableModel.__getstate__`. Internal math reads the ndarray
    directly and avoids the per-call ``.values`` conversion.

    On an unfitted model the backing ndarray is absent, so ``getattr`` raises
    ``AttributeError`` - the same "not fitted" signal as before this change, so
    ``hasattr`` / ``check_is_fitted`` behave identically.
    """

    def __init__(self, source: str, *, index: str, columns: str) -> None:
        self._source = source
        self._index = index
        self._columns = columns

    def __set_name__(self, owner: type, name: str) -> None:
        self._public_name = name

    def __get__(self, obj: object, objtype: type | None = None) -> object:
        if obj is None:
            return self
        cache = obj.__dict__.get("_frame_cache")
        if cache is None:
            cache = obj.__dict__["_frame_cache"] = {}
        cached = cache.get(self._public_name)
        if cached is not None:
            return cached
        frame = pd.DataFrame(
            getattr(obj, self._source),
            index=getattr(obj, self._index),
            columns=getattr(obj, self._columns),
        )
        cache[self._public_name] = frame
        return frame


class _RenameGetattrMixin:
    """``__getattr__`` that raises a helpful message for renamed attributes.

    Subclasses set ``_ATTRIBUTE_RENAMES`` (old name -> new name) and
    ``_RENAME_CONTEXT`` (the label used in the message, e.g. ``"PCA"``).
    """

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {}
    _RENAME_CONTEXT: typing.ClassVar[str] = ""

    def __getattr__(self, name: str):
        """Provide helpful error messages for old attribute names."""
        renames = type(self)._ATTRIBUTE_RENAMES
        if name in renames:
            raise AttributeError(
                f"'{name}' was renamed to '{renames[name]}' in the {type(self)._RENAME_CONTEXT} refactoring. "
                f"Please update your code to use '{renames[name]}'."
            )
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class _HotellingsT2LimitMixin:
    """The ``hotellings_t2_limit`` method shared by PCA / PLS / MBPLS / MBPCA.

    Reads ``self.n_components`` and ``self.n_samples_``; identical across those
    four estimators. (TPLS reads a differently-named row count and keeps its own.)
    """

    if typing.TYPE_CHECKING:  # fitted attributes provided by concrete subclasses
        n_components: int
        n_samples_: int

    def hotellings_t2_limit(self, conf_level: float = 0.95) -> float:
        """Hotelling's T2 limit at the given confidence level (see :func:`hotellings_t2_limit`)."""
        return _hotellings_t2_limit(
            conf_level=conf_level,
            n_components=self.n_components,
            n_rows=self.n_samples_,
        )


class _LatentVariableModel(_RenameGetattrMixin, _HotellingsT2LimitMixin, BaseEstimator):
    """Convenience-method scaffolding shared by :class:`PCA` and :class:`PLS`.

    Hosts the ENG-05 convenience methods that forward to the standalone functions
    (so ``help`` / ``inspect.signature`` stay accurate and the methods are
    overridable), plus ``ellipse_coordinates``. Subclasses add their own sklearn
    mixins (``TransformerMixin`` / ``RegressorMixin``) and ``BaseEstimator``, and
    provide the algorithm-specific ``fit`` / ``predict`` / ``transform``.
    """

    if typing.TYPE_CHECKING:  # fitted attribute provided by concrete subclasses
        scaling_factor_for_scores_: pd.Series

    def __getstate__(self) -> dict:
        """Exclude the lazily-built DataFrame cache from pickling (ENG-18).

        Only the private ndarrays and their index/column metadata are pickled;
        the public DataFrame views are rebuilt on demand after unpickling.
        """
        state = super().__getstate__()
        if isinstance(state, dict):
            state.pop("_frame_cache", None)
        return state

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)

    score_plot = _model_method(_score_plot)
    spe_plot = _model_method(_spe_plot)
    t2_plot = _model_method(_t2_plot)
    loading_plot = _model_method(_loading_plot)
    explained_variance_plot = _model_method(_explained_variance_plot)
    correlation_loadings_plot = _model_method(_correlation_loadings_plot)
    spe_limit = _model_method(_spe_limit)
    score_limit = _model_method(_score_limit)
    vip = _model_method(_vip)
    squared_cosine = _model_method(_squared_cosine)
    observation_contributions = _model_method(_observation_contributions)
    eigenvalue_summary = _model_method(_eigenvalue_summary)
    project_variables = _model_method(_project_variables)

    def ellipse_coordinates(
        self,
        score_horiz: int,
        score_vert: int,
        conf_level: float = 0.95,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the T2 confidence ellipse (see :func:`ellipse_coordinates`)."""
        return _ellipse_coordinates(
            score_horiz=score_horiz,
            score_vert=score_vert,
            conf_level=conf_level,
            n_points=n_points,
            n_components=self.n_components,
            scaling_factor_for_scores=self.scaling_factor_for_scores_,
            n_rows=self.n_samples_,
        )
