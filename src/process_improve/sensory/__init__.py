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

from process_improve.sensory.analysis import (
    AnalysisResult,
    aggregate_to_product,
    analyze_descriptive,
    discriminate_observational,
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
    "compare_products",
    "discriminate_observational",
    "dunnett_vs_control",
    "factorial_anova",
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
