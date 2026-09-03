# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Helpers shared by the batch estimator classes."""

from __future__ import annotations

import functools
import typing

from sklearn.utils.validation import check_is_fitted

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def inner_method(fn: Callable[..., typing.Any], *, inner: str, fitted: str) -> Callable[..., typing.Any]:
    """Wrap a module-level ``fn(model, ...)`` as a method that forwards an inner estimator.

    The standalone plots, limits and contribution functions in
    :mod:`process_improve.multivariate` read only fitted attributes that the
    inner PCA or PLS of a batch class carries (``scores_``, ``spe_``,
    ``hotellings_t2_``, the loadings or weights, ...), so the batch classes
    forward to them with that estimator as the ``model`` argument.

    Parameters
    ----------
    fn : callable
        The standalone function, taking the model as its first argument.
    inner : str
        Name of the attribute holding the inner estimator (``"_pca"``, ``"_pls"``).
    fitted : str
        Fitted attribute that ``check_is_fitted`` probes before forwarding.

    Returns
    -------
    callable
        The method. ``functools.wraps`` keeps ``help`` and ``inspect.signature``
        reporting the underlying function, as ``_model_method`` does for the
        multivariate estimators (ENG-05).
    """

    @functools.wraps(fn)
    def method(self: object, *args: object, **kwargs: object) -> object:
        check_is_fitted(self, fitted)
        return fn(getattr(self, inner), *args, **kwargs)

    return method
