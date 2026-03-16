T-shaped Partial Least Squares (TPLS)
======================================

TPLS is a multi-block method designed for **T-shaped data** structures that
arise naturally in batch processes, formulation studies, and similar settings
where information is organized in interconnected blocks.

When to Use TPLS
----------------

Use TPLS when your data has a natural multi-block structure that standard PLS
cannot represent. Typical applications include:

- **Pharmaceutical manufacturing** — formulation recipes, raw material
  properties, process conditions, and tablet quality form distinct blocks.
- **Chemical reaction optimization** — catalyst properties, feed
  compositions, operating conditions, and product quality.
- **Food processing** — ingredient properties, recipes, process settings,
  and sensory or nutritional outcomes.
- **Biotechnology** — media composition, strain properties, fermentation
  trajectories, and yield/quality metrics.

If your data fits naturally into a single X matrix and a single Y matrix,
standard :doc:`PLS <pls>` is simpler and should be preferred.

Data Structure
--------------

TPLS operates on four interconnected data blocks:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Block
     - Name
     - Description
   * - **D**
     - Properties
     - Intrinsic properties of raw materials or design factors.
       Rows = materials, columns = properties.
   * - **F**
     - Formulations
     - How materials are combined in each batch/experiment.
       Rows = batches, columns = materials. Column names must match D's row
       index.
   * - **Z**
     - Conditions
     - Process conditions or final-state measurements for each batch.
       Rows = batches, columns = condition variables.
   * - **Y**
     - Quality
     - Response or quality variables for each batch.
       Rows = batches, columns = quality variables.

The "T-shape" comes from the way D and F link: D describes the materials
(rows) while F describes how those same materials (columns) are used in each
batch (rows). F, Z, and Y must all have the same number of rows (batches).

Basic Usage
-----------

Data blocks are organized using ``DataFrameDict`` — a dictionary of
DataFrames, optionally grouped:

.. code-block:: python

   from process_improve.multivariate.methods import TPLS, DataFrameDict

   data = DataFrameDict(
       {
           "D": {"Group_A": properties_a, "Group_B": properties_b},
           "F": {"Group_A": formulas_a, "Group_B": formulas_b},
           "Z": {"Conditions": process_conditions},
           "Y": {"Quality": quality_responses},
       }
   )

   model = TPLS(n_components=3, d_matrix=data["D"])
   model.fit(data)

**Key requirements:**

- Column names in each F group must match the row index of the corresponding
  D group — this is how TPLS knows which material properties correspond to
  which formulation amounts.
- All F, Z, and Y DataFrames must have the same number of rows.
- The ``d_matrix`` parameter passed to the constructor should be the D block
  (material properties).
