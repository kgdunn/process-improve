"""(c) Kevin Dunn, 2010-2026. MIT License.

Descriptive panel-data analysis.

This subpackage provides a small, generic pipeline for descriptive panel
data: validate the data, identify and (optionally) correct panel anomalies,
and relate the panel attributes back to the product. For now the product is
described by **observational** descriptors (measured covariates of products
whose formulation is unknown), and the relationship is analysed by PLS as
association rather than causation. A **designed** mode (controlled experimental
runs, analysed as factor effects) is stubbed and planned for a later release.

The public entry points are :func:`validate_descriptive` and
:func:`analyze_descriptive`. Agent-callable wrappers live in
``process_improve.sensory.tools``.
"""

from typing import NoReturn

from process_improve.sensory.analysis import (
    AnalysisResult,
    aggregate_to_product,
    analyze_descriptive,
    find_predictive_descriptors,
    permutation_column_null,
    product_means,
    relate_designed,
    relate_observational,
)
from process_improve.sensory.designed import (
    ComparisonResult,
    compare_products,
    dunnett_vs_control,
    factorial_anova,
    tukey_hsd,
)
from process_improve.sensory.diagnostics import (
    assessor_variance_equality,
    boundary_occupancy,
    detection_rate,
)
from process_improve.sensory.ingest import reshape_to_long
from process_improve.sensory.mam import MAMResult, align_scores, mixed_assessor_model
from process_improve.sensory.panel import PanelScorecard, apply_correction, panel_scorecard
from process_improve.sensory.recipes import SENSORY_RECIPES
from process_improve.sensory.validation import (
    DESCRIPTIVE_LONG_COLUMNS,
    ValidationResult,
    validate_descriptive,
)

__all__ = [
    "DESCRIPTIVE_LONG_COLUMNS",
    "SENSORY_RECIPES",
    "AnalysisResult",
    "ComparisonResult",
    "MAMResult",
    "PanelScorecard",
    "ValidationResult",
    "aggregate_to_product",
    "align_scores",
    "analyze_descriptive",
    "apply_correction",
    "assessor_variance_equality",
    "boundary_occupancy",
    "compare_products",
    "detection_rate",
    "dunnett_vs_control",
    "factorial_anova",
    "find_predictive_descriptors",
    "mixed_assessor_model",
    "panel_scorecard",
    "permutation_column_null",
    "product_means",
    "relate_designed",
    "relate_observational",
    "reshape_to_long",
    "tukey_hsd",
    "validate_descriptive",
]


# ---------------------------------------------------------------------------
# Migration helpers - old names raise helpful errors
# ---------------------------------------------------------------------------

_RENAMED = {
    "discriminate_observational": "find_predictive_descriptors",
}


def __getattr__(name: str) -> NoReturn:
    """Raise a helpful error when a renamed module attribute is accessed."""
    if name in _RENAMED:
        new = _RENAMED[name]
        raise AttributeError(f"{name!r} has been renamed to {new!r}. Use: from {__name__} import {new}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
