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

__all__: list[str] = []
