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
    Control charts (Shewhart, CUSUM, EWMA).
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

from process_improve.monitoring.control_charts import ControlChart
from process_improve.monitoring.metrics import calculate_cpk
from process_improve.multivariate.methods import PCA, PLS, MCUVScaler
from process_improve.tool_spec import execute_tool_call, get_tool_specs

__all__ = [
    "PCA",
    "PLS",
    "ControlChart",
    "MCUVScaler",
    "calculate_cpk",
    "execute_tool_call",
    "get_tool_specs",
]
