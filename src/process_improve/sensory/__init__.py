"""(c) Kevin Dunn, 2010-2026. MIT License.

Descriptive panel-data analysis.

This subpackage provides a small, generic pipeline for descriptive panel
data: validate the data, identify and (optionally) correct panel anomalies,
and relate the panel attributes back to the product. The product can be
described either by **designed** factors (controlled experimental runs, so
the relationship is analysed as effects) or by **observational** descriptors
(measured covariates of products whose formulation is unknown, so the
relationship is analysed by PLS as association rather than causation).

The public entry points are :func:`validate_descriptive` and
:func:`analyze_descriptive`. Agent-callable wrappers live in
``process_improve.sensory.tools``.
"""

from process_improve.sensory.analysis import (
    AnalysisResult,
    aggregate_to_product,
    analyze_descriptive,
    product_means,
    relate_designed,
    relate_observational,
)
from process_improve.sensory.panel import PanelScorecard, apply_correction, panel_scorecard
from process_improve.sensory.validation import (
    DESCRIPTIVE_LONG_COLUMNS,
    ValidationResult,
    validate_descriptive,
)

__all__ = [
    "DESCRIPTIVE_LONG_COLUMNS",
    "AnalysisResult",
    "PanelScorecard",
    "ValidationResult",
    "aggregate_to_product",
    "analyze_descriptive",
    "apply_correction",
    "panel_scorecard",
    "product_means",
    "relate_designed",
    "relate_observational",
    "validate_descriptive",
]
