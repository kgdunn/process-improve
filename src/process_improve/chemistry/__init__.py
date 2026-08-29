"""(c) Kevin Dunn, 2010-2026. MIT License.

Preprocessing a product-by-compound block for a sensory-to-chemistry analysis.

One row per product, one column per compound, values being concentrations or
integrated peak areas. The pipeline is four steps in a fixed order, trim then
transform then centre then scale, and each fitting step has an
``apply_fitted_*`` partner so held-out rows can be preprocessed with training
constants alone:

.. code-block:: python

    from process_improve.chemistry import (
        apply_transform,
        center_and_scale,
        classify_zero_states,
        normalisation_check,
        trim_by_prevalence,
    )

    states = classify_zero_states(chem, lod={"linalool": 0.02})
    totals, outside = normalisation_check(chem)
    kept, dropped, presence = trim_by_prevalence(chem, min_nonzero=3)
    transformed, applied = apply_transform(kept, lod={"linalool": 0.02})
    scaled, constants = center_and_scale(transformed, presence[kept.columns])

See :mod:`process_improve.chemistry.preprocessing` for the reasoning behind
each step.
"""

from process_improve.chemistry.preprocessing import (
    SCALING_METHODS,
    TRANSFORM_RULES,
    ZERO_STATES,
    apply_fitted_center_scale,
    apply_fitted_transform,
    apply_transform,
    center_and_scale,
    choose_transform,
    classify_zero_states,
    normalisation_check,
    trim_by_prevalence,
)

__all__ = [
    "SCALING_METHODS",
    "TRANSFORM_RULES",
    "ZERO_STATES",
    "apply_fitted_center_scale",
    "apply_fitted_transform",
    "apply_transform",
    "center_and_scale",
    "choose_transform",
    "classify_zero_states",
    "normalisation_check",
    "trim_by_prevalence",
]
