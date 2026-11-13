"""Process Improvement using Data.

A Python package for multivariate data analysis, designed experiments, and
process monitoring. Companion to the online textbook
`Process Improvement using Data <https://learnche.org/pid>`_.

Subpackages
-----------
multivariate
    PCA, PLS, TPLS, MCUVScaler, and related plotting functions.
univariate
    Statistical metrics (t-value, confidence intervals, outlier detection).
experiments
    Factorial and response surface experiment designs.
monitoring
    Control charts (Shewhart, EWMA, CUSUM, and a robust Holt-Winters chart that
    blends Shewhart and CUSUM-like behaviour) and process capability metrics.
batch
    Batch process data analysis.
regression
    Robust regression methods.
bivariate
    Elbow detection, peak finding, area under curve.
visualization
    General-purpose plotting utilities.

(c) Kevin Dunn, 2010-2026. MIT License.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from process_improve.tool_spec import execute_tool_call, get_tool_specs

if TYPE_CHECKING:
    from process_improve.monitoring.control_charts import ControlChart
    from process_improve.monitoring.metrics import calculate_cpk
    from process_improve.multivariate.methods import PCA, PLS, MCUVScaler

__all__ = [
    "PCA",
    "PLS",
    "ControlChart",
    "MCUVScaler",
    "calculate_cpk",
    "execute_tool_call",
    "get_tool_specs",
]

#: The names this package offers from its subpackages, and where each lives. They are imported when first asked
#: for (PEP 562), not when the package is: ``import process_improve`` then loads neither scikit-learn nor
#: statsmodels, which only some of the subpackages need (#1235).
_LAZY_NAMES = {
    "ControlChart": "process_improve.monitoring.control_charts",
    "calculate_cpk": "process_improve.monitoring.metrics",
    "PCA": "process_improve.multivariate.methods",
    "PLS": "process_improve.multivariate.methods",
    "MCUVScaler": "process_improve.multivariate.methods",
}

#: The subpackages, each an attribute of the package once it is asked for, as when the package imported them itself.
_SUBPACKAGES = frozenset(
    {
        "batch",
        "bivariate",
        "chemistry",
        "experiments",
        "monitoring",
        "multivariate",
        "regression",
        "sensory",
        "simulation",
        "univariate",
        "visualization",
        "workbench",
    }
)


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a class, a function or a subpackage
    """Import one of the package's lazy names, or a subpackage, the first time it is asked for."""
    if name in _LAZY_NAMES:
        value = getattr(importlib.import_module(_LAZY_NAMES[name]), name)
        globals()[name] = value  # found in the package itself from here on
        return value
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_NAMES, *_SUBPACKAGES})
