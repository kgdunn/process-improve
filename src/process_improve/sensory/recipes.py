"""(c) Kevin Dunn, 2010-2026. MIT License.

Analysis recipes for the descriptive sensory pipeline.

Each recipe is a guided, multi-step workflow the agent follows, chaining the
sensory tools (``sensory_reshape_to_long``, ``sensory_validate_descriptive``,
``sensory_panel_check``, ``sensory_analyze_descriptive``) in the right order.
The recipes are registered into the package-wide catalog on import; see
:mod:`process_improve.recipes` for the framework and the general
``select_analysis_recipe`` tool.
"""

from __future__ import annotations

from process_improve.recipes import AnalysisRecipe, RecipeStep, register_recipe

_DOMAIN = "sensory"


_INTAKE = AnalysisRecipe(
    key="sensory_intake",
    title="Sensory data intake: organise and check a panel file",
    summary=(
        "Take a freshly received descriptive-panel spreadsheet from raw rows to a validated, canonical "
        "long table ready for analysis. Resolve the layout and column roles, reshape to the "
        "descriptive_long schema with round-trip checks, then validate the schema, score range and panel "
        "balance. Use this whenever new panel data arrives and needs organising and checking."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "panel file",
        "panel data",
        "load panel",
        "import panel",
        "ingest",
        "raw panel",
        "new sensory data",
        "received data",
        "got a spreadsheet",
        "wide panel",
        "reshape",
        "organise the data",
        "organize the data",
        "tidy the data",
        "check the data",
        "clean the file",
        "descriptive panel",
    ],
    inputs_needed=[
        "the parsed spreadsheet as rows (the front end or a code sandbox reads the file first; these "
        "recipes do not read files themselves)",
        "which columns hold the panelist, product, attribute, replicate, session and score",
        "the layout: one column per attribute, one column per product, or already one row per score",
        "the valid score range if known (for example 0 to 10)",
    ],
    stages=[
        RecipeStep(
            order=1,
            directive=(
                "Inspect the parsed rows (header names and a few sample rows) to decide the layout and the "
                "column roles. Choose 'wide_by_attribute' (one column per attribute), 'wide_by_product' "
                "(one column per product plus an attribute-label column), or 'long' (already one row per "
                "score). Note any nuisance columns (a site or batch code) to ignore."
            ),
        ),
        RecipeStep(
            order=2,
            directive=(
                "Reshape to the canonical descriptive_long schema with sensory_reshape_to_long, passing the "
                "explicit column mapping and any nuisance columns to ignore. Confirm checks.ok is true (the "
                "grand, per-attribute and per-panelist means and the cell count are preserved). If it is "
                "false the mapping is wrong: fix the column roles and rerun rather than proceeding."
            ),
            tools=["sensory_reshape_to_long"],
            arg_hints={
                "layout": "<one of long / wide_by_attribute / wide_by_product>",
                "panelist_id": "<panelist column name>",
                "product": "<product column name, for long / wide_by_attribute>",
                "ignore": "<list of nuisance columns to drop, if any>",
            },
        ),
        RecipeStep(
            order=3,
            directive=(
                "Validate the reshaped table (and the product-covariate table if one was supplied) with "
                "sensory_validate_descriptive. Read ok, warnings, errors, stats and content_hash. Stop and "
                "report the errors if ok is false; warnings (such as panel-imbalance notes) can be carried "
                "forward."
            ),
            tools=["sensory_validate_descriptive"],
            arg_hints={
                "mode": "observational",
                "score_min": "<lower score bound if known>",
                "score_max": "<upper score bound if known>",
            },
        ),
        RecipeStep(
            order=4,
            directive=(
                "Summarise for the user: counts of panelists, products, attributes and replicates, the "
                "score range, any balance warnings, and the content hash that identifies this dataset. The "
                "validated canonical descriptive_long table is the artifact the panel-processing recipe "
                "consumes next."
            ),
        ),
    ],
)


_PANEL_PROCESSING = AnalysisRecipe(
    key="sensory_panel_processing",
    title="Panel consistency and scale-use correction",
    summary=(
        "Check whether the panel is consistent and how each assessor uses the scale, explain the findings "
        "in plain language, and produce a corrected canonical table ready for relating to covariates. "
        "Covers the per-panelist scorecard, the Mixed Assessor Model scaling coefficient (beta), and "
        "rescaling the panel onto a common scale. Use this after intake and before relating."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "panel consistency",
        "panel performance",
        "panel check",
        "panelist",
        "assessor",
        "consistent",
        "agreement",
        "scale usage",
        "scale use",
        "using the scale",
        "beta",
        "scaling coefficient",
        "harmonise the panel",
        "harmonize the panel",
        "align the panel",
        "correct the panel",
        "process the panel",
        "ready for analysis",
    ],
    inputs_needed=[
        "a validated canonical descriptive_long table (from the intake recipe)",
        "a decision on whether to rescale every panelist, drop anomalous panelists, or both",
    ],
    stages=[
        RecipeStep(
            order=1,
            directive=(
                "Run sensory_panel_check on the validated panel (no covariates are needed). Read the "
                "scorecard (discrimination, agreement, scale_shift, scale_spread, drift), the flagged "
                "panelists and their reasons, and the Mixed Assessor Model results: each panelist's scaling "
                "coefficient beta per attribute, and the MAM versus classical product-effect F-tests."
            ),
            tools=["sensory_panel_check"],
        ),
        RecipeStep(
            order=2,
            directive=(
                "Interpret the numbers for a non-statistician. beta near 1 means the panelist uses the scale "
                "like the rest of the panel; beta below 1 means they compress it (scores bunched together); "
                "beta above 1 means they stretch it. A large offset means they sit consistently high or low. "
                "Low agreement means their ranking of the products does not track the panel, which is "
                "genuine disagreement rather than a scale habit. A panelist is flagged only when it is both "
                "an outlier and genuinely poor on agreement or discrimination. When the MAM product F-test "
                "is larger than the classical one, removing scale-use differences makes the products "
                "separate more clearly."
            ),
        ),
        RecipeStep(
            order=3,
            directive=(
                "Decide the correction and produce the corrected canonical table. Call sensory_panel_check "
                "with align=true to rescale every panelist onto a common scale (a location lever removes "
                "their offset, a scale lever divides by beta); the returned aligned_panel is the new "
                "canonical descriptive_long table, with scale-use artefacts removed but genuine "
                "disagreement preserved. A panelist who truly disagrees (low agreement, not just scaling) is "
                "better dropped than rescaled; aligning and dropping can be combined."
            ),
            tools=["sensory_panel_check"],
            arg_hints={"align": "true", "align_method": "both"},
        ),
        RecipeStep(
            order=4,
            directive=(
                "Hand off: the corrected canonical descriptive_long table (aligned, with any genuine "
                "disagreers earmarked for dropping) plus the recorded correction decision are the inputs to "
                "the relate-to-covariates recipe. Report what was changed and why, in plain language."
            ),
        ),
    ],
)


_RELATE_COVARIATES = AnalysisRecipe(
    key="sensory_relate_covariates",
    title="Relate panel attributes to product covariates",
    summary=(
        "Relate each sensory attribute to measured product covariates and separate genuine drivers from "
        "proxies and chance correlations. Runs PLS plus per-pair correlations and the cross-validated "
        "discriminator (out-of-sample Q-squared gate, selectivity ratio, collinear clustering), then "
        "interprets which descriptors really carry signal. Use this after the panel has been processed."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "relate to",
        "relate the",
        "relate attributes",
        "covariate",
        "what drives",
        "drivers of",
        "which measurements",
        "instrumental",
        "descriptors",
        "chemistry",
        "correlate attributes",
        "predict liking",
        "explain sweetness",
        "link sensory",
        "selectivity",
        "confound",
        "spurious",
        "causal",
        "root cause",
    ],
    inputs_needed=[
        "the corrected canonical panel from the panel-processing recipe",
        "a product-covariate table: one row per product with the measured numeric descriptors",
        "mode is observational (designed / DoE relate is not implemented yet)",
    ],
    stages=[
        RecipeStep(
            order=1,
            directive=(
                "Assemble the product-covariate table: one row per product, numeric descriptors only (drop "
                "id and non-numeric columns). Confirm every product in the panel has covariates."
            ),
        ),
        RecipeStep(
            order=2,
            directive=(
                "Run sensory_analyze_descriptive on the corrected panel and the covariate table. Set "
                "correction to what was already applied in the processing recipe (use 'none' if the panel "
                "is already aligned) and drop the genuine disagreers identified there. This one call "
                "validates, relates the attributes to the descriptors with PLS and per-pair correlations, "
                "and runs the cross-validated discriminator."
            ),
            tools=["sensory_analyze_descriptive"],
            arg_hints={
                "mode": "observational",
                "correction": "<none if already aligned, else align>",
                "drop_flagged": "<true to drop the flagged disagreers>",
            },
        ),
        RecipeStep(
            order=3,
            directive=(
                "Read the marginal associations in relate.associations: each (attribute, descriptor) "
                "correlation with a Benjamini-Hochberg q_value and a significant flag. These flag every "
                "descriptor that correlates in this sample, genuine drivers, proxies and coincidences "
                "alike."
            ),
        ),
        RecipeStep(
            order=4,
            directive=(
                "Read the discriminator in relate.discriminator: per_attribute gives the cross-validated "
                "q2_cv and a predictable flag; descriptors gives, per (attribute, descriptor), the "
                "selectivity_ratio, a permutation q_value, a discriminator_significant flag, and a "
                "cluster_id."
            ),
        ),
        RecipeStep(
            order=5,
            directive=(
                "Interpret for a non-statistician. An association that survives the discriminator (the "
                "attribute is predictable out of sample and the descriptor is selectivity-significant) "
                "carries real, transferable predictive signal. An association that the marginal test flags "
                "but the discriminator demotes is most likely a coincidence of this sample. Descriptors "
                "that share a cluster_id carry the same information and cannot be told apart, so a "
                "significant one may be a proxy riding on the true driver."
            ),
        ),
        RecipeStep(
            order=6,
            directive=(
                "State the limit plainly. From one observational panel you cannot prove causation or rank "
                "descriptors within a collinear cluster; separating a genuine driver from a collinear proxy "
                "needs an external dataset that breaks the collinearity, a designed experiment, or "
                "mechanistic knowledge. Report the drivers grouped by cluster, with a genuine / proxy / "
                "coincidence verdict per descriptor."
            ),
        ),
    ],
)


_VISUALISATION = AnalysisRecipe(
    key="sensory_visualisation",
    title="Sensory visualisation (planned)",
    summary=(
        "A planned workflow to turn relate and discriminator output into sensory maps, driver biplots and "
        "small-multiple panels. Not yet available; this entry advertises the workflow so the agent can tell "
        "the user it is coming rather than improvising plots."
    ),
    domain=_DOMAIN,
    cue_phrases=[
        "visualise sensory",
        "visualize sensory",
        "sensory map",
        "perceptual map",
        "biplot",
        "loadings plot",
        "spider plot",
        "plot the drivers",
        "chart the panel",
    ],
    inputs_needed=[
        "not yet available - this visualisation workflow is planned for a later release",
    ],
    stages=[],
    status="planned",
)


SENSORY_RECIPES: list[AnalysisRecipe] = [
    _INTAKE,
    _PANEL_PROCESSING,
    _RELATE_COVARIATES,
    _VISUALISATION,
]

for _recipe in SENSORY_RECIPES:
    register_recipe(_recipe)


__all__ = ["SENSORY_RECIPES"]
