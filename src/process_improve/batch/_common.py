# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Helpers shared by the batch estimator classes."""

from __future__ import annotations

import functools
import inspect
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


_BATCH_ROWS = """\
On a batch model, ``X`` is the batches, as :meth:`transform` takes them, with their
Z block as ``initial_conditions`` when the model was fitted with one: they are
unfolded and scaled as ``transform`` reads them. A DataFrame is taken to be the
:meth:`unfold_and_scale` output, and is used unchanged."""


def contribution_method(fn: Callable[..., typing.Any], *, inner: str, fitted: str) -> Callable[..., typing.Any]:
    """Wrap a contribution function ``fn(model, X, ...)`` as a method that takes the batches ``transform`` takes.

    The methods took only the ``unfold_and_scale`` output, and a batch dictionary, the
    ``X`` of ``transform``, raised a pandas error (#1144). They now take either. ``help``
    and ``inspect.signature`` show the ``initial_conditions`` keyword, and the docstring
    says what ``X`` is, before the function's own.

    Parameters
    ----------
    fn : callable
        The standalone contribution function, taking the model and ``X`` first.
    inner : str
        Name of the attribute holding the inner estimator (``"_pca"``, ``"_pls"``).
    fitted : str
        Fitted attribute that ``check_is_fitted`` probes before forwarding.

    Returns
    -------
    callable
        The method.
    """

    @functools.wraps(fn)
    def method(
        self: typing.Any,  # noqa: ANN401 - the batch estimator, which has _scaled_wide
        X: object,
        *args: object,
        initial_conditions: object = None,
        **kwargs: object,
    ) -> object:
        check_is_fitted(self, fitted)
        if isinstance(X, dict):
            X = self._scaled_wide(X, initial_conditions)
        elif initial_conditions is not None:
            msg = (
                "initial_conditions goes with a batch dictionary X; the unfold_and_scale output holds "
                "the Z block already, so pass that alone."
            )
            raise ValueError(msg)
        return fn(getattr(self, inner), X, *args, **kwargs)

    parameters = list(inspect.signature(fn).parameters.values())
    rest = [p for p in parameters[2:] if p.kind is not inspect.Parameter.VAR_KEYWORD]
    keyword = inspect.Parameter(
        "initial_conditions", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="pd.DataFrame | None"
    )
    method.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            parameters[1].replace(annotation="dict[Hashable, pd.DataFrame] | pd.DataFrame"),
            *rest,
            keyword,
            *[p for p in parameters[2:] if p.kind is inspect.Parameter.VAR_KEYWORD],
        ]
    )
    summary, _, details = inspect.cleandoc(fn.__doc__ or "").partition("\n\n")
    method.__doc__ = f"{summary}\n\n{_BATCH_ROWS}\n\n{details}"
    return method
