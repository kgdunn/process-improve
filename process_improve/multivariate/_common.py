# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Shared primitives for the multivariate package (ENG-01).

This leaf module holds the small, dependency-free building blocks used across
the PCA / PLS / TPLS / multiblock implementations: the :data:`DataMatrix` type
alias, the :data:`epsqrt` tolerance, the NIPALS denominator floor helper
:func:`_nz`, and :class:`SpecificationWarning`. Nothing here imports any of the
sibling submodules, so it sits at the base of the dependency graph.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

# Names re-exported to the rest of the package. Declared explicitly so CodeQL
# does not flag the ``DataMatrix`` type alias (only ever referenced in lazy
# annotation strings under ``from __future__ import annotations``) as unused.
__all__ = ["DataMatrix", "SpecificationWarning", "epsqrt"]

DataMatrix: TypeAlias = np.ndarray | pd.DataFrame

epsqrt = np.sqrt(np.finfo(float).eps)

#: Smallest positive float; used to floor NIPALS denominators away from zero.
_DENOM_FLOOR = float(np.finfo(float).tiny)


def _nz(denominator: float) -> float:
    """Floor a non-negative NIPALS denominator away from zero.

    Sum-of-squares and vector-norm denominators (``v @ v``, ``norm(v)``) are
    non-negative. When a score or loading vector collapses to (near) zero during
    NIPALS - a fully-deflated component, or a degenerate / perfectly collinear
    block - the denominator is ~0 and the division would yield ``inf``/``nan``
    that silently poisons the fitted model. Flooring to the smallest positive
    float leaves every well-conditioned value untouched (real denominators are
    far larger) while turning the degenerate ``0/0`` into a finite (~0)
    projection, since the numerator collapses with the same vector.
    """
    return max(_DENOM_FLOOR, denominator)


class SpecificationWarning(UserWarning):
    """Parent warning class."""


def _model_method(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a module-level ``fn(model, ...)`` as an introspectable instance method.

    ENG-05: estimators (PCA / PLS / ...) expose convenience methods such as
    ``score_plot``, ``spe_limit`` and ``vip`` that forward to the standalone
    functions with ``self`` supplied as the ``model`` argument. Defining them
    via this factory at class-body time - rather than binding
    ``functools.partial`` instances in ``fit`` - means ``help`` and
    ``inspect.signature`` report the underlying function (minus ``self``), the
    fitted model stays picklable, and subclasses can override cleanly.
    """

    @functools.wraps(fn)
    def method(self: object, *args, **kwargs) -> object:
        return fn(self, *args, **kwargs)

    return method
