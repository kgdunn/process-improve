Descriptive panel data: validate, check the panel, relate to the product
========================================================================

A descriptive panel study asks a group of assessors (panelists) to score a set
of products on a set of attributes, usually with replicates and sometimes
across several sessions. The recurring question is: *which product
characteristics drive the attribute scores, and can we trust the panel that
produced them?*

The :mod:`process_improve.sensory` subpackage answers that with a small,
generic pipeline. The data is described only as panelist, product, attribute,
replicate, and score; nothing about the pipeline is specific to any particular
kind of product. The flow has three steps:

1. **validate** the panel data and a product-covariate table;
2. **check the panel** and optionally drop anomalous panelists;
3. **relate** each attribute back to the product.

The data contract
-----------------

Panel data is supplied in the ``descriptive_long`` schema, one row per score:

================  ========================================================
Column            Meaning
================  ========================================================
``panelist_id``   Who gave the score.
``session``       Which sitting the score came from.
``product``       Which product was scored.
``attribute``     Which attribute was scored.
``replicate``     Which repeat of that product/attribute for that panelist.
``score``         The numeric rating.
================  ========================================================

Alongside it you supply a **product-covariate table**: one row per product,
describing what each product *is*. This table comes in two flavours, and the
distinction decides how the relate step works.

Designed versus observational
-----------------------------

The covariate table is one of two kinds, and you must say which with the
``mode`` argument:

- **Observational** (``mode="observational"``) - *implemented*. You did not set
  the formulation (for example the products are existing market products), but
  you measured each one, for example by chemical or instrumental analysis. These
  measured *descriptors* are correlated covariates, not a designed matrix, so
  the relate step reports *association*, not causation: the attribute block is
  related to the descriptors with PLS, and per-descriptor correlations show
  which descriptors track which attributes.

- **Designed** (``mode="designed"``) - *planned, not implemented yet*. You
  controlled the formulation, so each product is an experimental run and the
  covariates are the *design factors*. Because the factors were deliberately
  varied, the relate step will speak of *effects*: each attribute regressed on
  the factors, so a coefficient estimates how changing a factor changes the
  attribute. For now ``mode="designed"`` raises ``NotImplementedError``.

Keep the interpretation in mind: an observational analysis supports only "this
descriptor is associated with this attribute", whereas the future designed
analysis will support "increasing this factor raises this attribute".

Step 1: validate
----------------

:func:`~process_improve.sensory.validate_descriptive` coerces the inputs to the
schema and checks them: required columns and dtypes, the score range, panel
balance (the fraction of the full panelist x product x attribute x replicate
grid that is missing), label encoding, and mode-specific covariate checks. It
returns a result whose ``ok`` flag gates the rest of the pipeline.

.. code-block:: python

   import pandas as pd
   from process_improve.sensory import validate_descriptive, analyze_descriptive

   # panel: a DataFrame in the descriptive_long schema (loaded from your data).
   # descriptors: one row per product, with the measured (e.g. instrumental)
   # covariates for each product.
   descriptors = pd.DataFrame(
       {"product": ["A", "B", "C", "D"], "sodium": [0.2, 0.5, 0.8, 1.0], "fat": [3.1, 3.0, 2.9, 3.2]}
   )

   validated = validate_descriptive(panel, descriptors, mode="observational", score_min=0, score_max=10)
   print(validated.ok, validated.warnings)

Step 2 and 3: check the panel and relate
----------------------------------------

:func:`~process_improve.sensory.analyze_descriptive` runs the rest. It first
builds a per-panelist scorecard
(:func:`~process_improve.sensory.panel_scorecard`) rating each panelist on
discrimination (do they separate the products), agreement (do they rank
products like the rest of the panel), scale use, and drift across sessions. A
panelist is flagged only when it is both an outlier and genuinely poor on
agreement or discrimination. Passing ``drop_panelists="auto"`` removes the
flagged panelists before relating, so a noisy panelist does not contaminate the
product conclusions.

.. code-block:: python

   result = analyze_descriptive(validated, drop_panelists="auto")

   print(result.panel.flagged)          # panelists flagged as anomalous
   print(result.dropped)                # panelists actually removed

   # Observational relate: which descriptors are associated with which attributes.
   drivers = pd.DataFrame(result.relate["vip"])             # PLS descriptor importance
   assoc = pd.DataFrame(result.relate["associations"])      # attribute-descriptor links
   print(assoc[assoc["significant"]])

The observational relate output is:

- ``result.relate["vip"]`` ranks descriptors by their PLS variable-importance.
- ``result.relate["associations"]`` gives the per-(attribute, descriptor)
  correlation with a raw p-value, a Benjamini-Hochberg ``q_value`` across the
  whole family of tests, and a ``significant`` flag.

The result also carries supporting context: ``result.product_means`` (each
product-by-attribute mean with a confidence interval) and ``result.pca`` (a PCA
sensory map of the products over the attributes).

The designed-mode relate (factor *effects* rather than associations) is planned;
``mode="designed"`` raises ``NotImplementedError`` for now.

Using the tools from an agent
-----------------------------

The same two steps are exposed as agent-callable tools,
``sensory_validate_descriptive`` and ``sensory_analyze_descriptive`` (see
:func:`process_improve.sensory.tools.get_sensory_tool_specs`). They take the
panel and covariate tables as lists of row-records and return JSON. The analyze
tool validates first and refuses to run if validation fails, so an agent cannot
skip the gate.
