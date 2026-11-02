# (c) Kevin Dunn, 2010-2026. MIT License.
"""Response transforms for :func:`analyze_experiment`: validation, application, and re-use.

The model is fitted on the transformed response, so everything computed from it
(predictions, intervals, a confirmation run's comparison) is on the transformed
scale. :func:`to_model_scale` puts a new raw observation on that same scale, using the
exact transform the fit used, Box-Cox ``lambda`` included.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .box_cox import box_cox_profile, box_cox_transform

#: The transforms ``analyze_experiment`` accepts.
TRANSFORMS = ("log", "log10", "sqrt", "inverse", "box_cox")


def validate_transform(transform: str | None) -> None:
    """Raise ``ValueError`` for a transform name that is not one of :data:`TRANSFORMS`."""
    if transform is not None and transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}; use one of {', '.join(TRANSFORMS)}, or None.")


def _check_domain(values: np.ndarray, transform: str, what: str) -> None:
    """Raise when ``values`` fall outside the domain of ``transform``."""
    if transform in ("log", "log10", "box_cox") and np.any(values <= 0):
        raise ValueError(
            f"transform={transform!r} needs every {what} to be positive; it has values <= 0. "
            "Shift the response, or pick a different transform."
        )
    if transform == "sqrt" and np.any(values < 0):
        raise ValueError(
            f"transform='sqrt' needs every {what} to be non-negative; it has negative values. "
            "Shift the response, or pick a different transform."
        )
    if transform == "inverse" and np.any(values == 0):
        # SEC-26 (#275): a zero would give inf, then a LinAlgError downstream.
        raise ValueError(
            f"transform='inverse' is undefined when the {what} contains zero. Remove the zero "
            "observations or pick a different transform."
        )


def transform_response(
    y: np.ndarray,
    transform: str | None,
    exog: Callable[[], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform the response, and describe the transform so it can be re-applied.

    Parameters
    ----------
    y : numpy.ndarray
        The raw response.
    transform : str or None
        One of :data:`TRANSFORMS`, or ``None``.
    exog : Callable[[], numpy.ndarray]
        Returns the model matrix; called only for Box-Cox, whose ``lambda`` is chosen
        by the profile likelihood of that model.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, Any]]
        The transformed response, and ``{"transform": name}`` plus ``"box_cox_lambda"``
        for Box-Cox.
    """
    validate_transform(transform)
    y = np.asarray(y, dtype=float)
    if transform is None:
        return y, {"transform": None}
    _check_domain(y, transform, "response")
    info: dict[str, Any] = {"transform": transform}
    if transform == "box_cox":
        lmbda, _ = box_cox_profile(y, exog())
        info["box_cox_lambda"] = lmbda
    return to_model_scale(y, info), info


def to_model_scale(values: np.ndarray, info: dict[str, Any], what: str = "response") -> np.ndarray:
    """Apply the transform described by ``info`` (from :func:`transform_response`) to raw ``values``."""
    values = np.asarray(values, dtype=float)
    transform = info.get("transform")
    if transform is None:
        return values
    _check_domain(values, transform, what)
    if transform == "log":
        return np.log(values)
    if transform == "log10":  # the common log, whose sums of squares are the natural log's over ln(10)^2
        return np.log10(values)
    if transform == "sqrt":
        return np.sqrt(values)
    if transform == "inverse":
        return 1.0 / values
    return box_cox_transform(values, float(info["box_cox_lambda"]))


def to_original_scale(values: ArrayLike, info: dict[str, Any]) -> np.ndarray:
    """Undo the transform described by ``info``: the inverse of :func:`to_model_scale`.

    A prediction put back this way is the response's median, not its mean, when the errors are
    symmetric on the model's scale: a monotone transform keeps the median and not the mean. A value
    below the transform's range, such as an interval's end below 0 for ``"sqrt"``, is the edge of the
    response's range (0, or ``inf`` for a negative power); ``"inverse"`` gives ``inf`` at 0.
    """
    values = np.asarray(values, dtype=float)
    transform = info.get("transform")
    if transform is None:
        return values
    with np.errstate(divide="ignore"):
        if transform != "box_cox":
            return _INVERSES[transform](values)
        lmbda = float(info["box_cox_lambda"])
        return np.exp(values) if lmbda == 0.0 else np.power(np.maximum(lmbda * values + 1.0, 0.0), 1.0 / lmbda)


#: The inverse of each fixed transform, for :func:`to_original_scale`.
_INVERSES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log": np.exp,
    "log10": lambda v: np.power(10.0, v),
    "sqrt": lambda v: np.square(np.maximum(v, 0.0)),
    "inverse": lambda v: 1.0 / v,
}
