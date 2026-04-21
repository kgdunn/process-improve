"""Enums and type constants for chart specs.

These enums define the vocabulary for backend-agnostic chart specs: mark
types, scale types, annotation types, and the DOE-specific plot types.
The :class:`DOEPlotType` enum lists plot-type keys used by
:func:`~process_improve.experiments.visualization.visualize_doe`; other
enums are shared by generic chart classes as well.
"""

from __future__ import annotations

from enum import Enum


class MarkType(str, Enum):
    """Visual mark used in a layer."""

    bar = "bar"
    line = "line"
    scatter = "scatter"
    contour = "contour"
    heatmap = "heatmap"
    surface = "surface"
    area = "area"
    text = "text"
    wireframe = "wireframe"
    boxplot = "boxplot"


class ScaleType(str, Enum):
    """Axis scale type."""

    linear = "linear"
    log = "log"
    category = "category"


class AnnotationType(str, Enum):
    """Annotation overlay type."""

    reference_line = "reference_line"
    significance_threshold = "significance_threshold"
    constraint_region = "constraint_region"
    reference_band = "reference_band"
    label = "label"


class DOEPlotType(str, Enum):
    """All 20 supported DOE plot types plus the diagnostic panel."""

    # Effect / significance plots
    pareto = "pareto"
    half_normal = "half_normal"
    daniel = "daniel"

    # Factor-effect plots
    main_effects = "main_effects"
    interaction = "interaction"
    perturbation = "perturbation"

    # Diagnostic plots
    residuals_vs_fitted = "residuals_vs_fitted"
    normal_probability = "normal_probability"
    residuals_vs_order = "residuals_vs_order"
    box_cox = "box_cox"

    # Response-surface plots
    contour = "contour"
    surface_3d = "surface_3d"
    prediction_variance = "prediction_variance"

    # Cube / special
    cube_plot = "cube_plot"

    # Optimisation plots
    desirability_contour = "desirability_contour"
    overlay = "overlay"
    ridge_trace = "ridge_trace"
    steepest_ascent_path = "steepest_ascent_path"

    # Design-quality plots
    fds_plot = "fds_plot"
    power_curve = "power_curve"
