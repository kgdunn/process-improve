Multi-block PCA and PLS (MBPCA / MBPLS)
=======================================

Generic multi-block latent-variable models for the case where your X
data is naturally organized into several semantically distinct blocks —
for example, one block per processing zone, plant unit, or sensor group.

Multi-block models give you per-block diagnostics (which block is
driving a fault? which block is most predictive of Y?) that get lost
when every variable is dumped into a single big-X model.

When to Use
-----------

- You have **multiple X-blocks** (typically 2-10) sharing the same row
  observations but different column variables.
- Each block has a clear semantic meaning (e.g. ``zone1``, ``zone2``,
  ``feed``, ``utilities``, ``quality_lab``).
- You want **per-block bookkeeping**: per-block R²X, per-block VIPs,
  per-block SPE / Hotelling's T², per-block contribution plots.
- For MBPLS, you also have a single Y-block and want to know how each
  X-block contributes to predicting Y.

If you only have one big X (no semantic block structure), use
:doc:`PCA <pca>` or :doc:`PLS <pls>` instead. If your data follows the
fixed D / F / Z / Y structure (database of properties, formulations,
process conditions, quality), use :doc:`TPLS <tpls>`.

How It Works
------------

The multi-block / hierarchical / consensus formulation of Westerhuis,
Kourti & MacGregor (1998):

1. Each X-block is preprocessed independently (mean-centred and
   unit-variance scaled with :class:`~process_improve.multivariate.methods.MCUVScaler`).
2. Each block is divided by ``sqrt(K_b)`` (where ``K_b`` is the number
   of variables in block ``b``) so blocks of unequal width contribute
   fairly to the consensus super-score.
3. Hierarchical NIPALS alternates between (i) computing per-block scores
   and weights / loadings against the current super-score, and (ii)
   collecting block scores into a super-block and refining the
   super-score / super-loading.
4. After convergence, each block is deflated using the super-score and
   the corresponding block loading.

The block-weighting in step 2 is fundamental: without it, blocks with
many variables would dominate the super-score simply by virtue of their
size.

API at a glance
---------------

Both classes use the same ``dict[str, pd.DataFrame]`` API for X-blocks::

   x_blocks = {
       "zone1": df_with_zone1_columns,    # all blocks share the row index
       "zone2": df_with_zone2_columns,
   }

   from process_improve.multivariate.methods import MBPCA, MBPLS

   pca = MBPCA(n_components=3).fit(x_blocks)
   pls = MBPLS(n_components=3).fit(x_blocks, y_df)

After fitting, every model exposes:

- ``super_scores_``, ``super_loadings_`` (or ``super_weights_`` for MBPLS)
- ``block_scores_``, ``block_loadings_`` — both ``dict[str, DataFrame]``
- ``r2_x_per_block_cumulative_``, ``r2_x_per_block_per_component_``
- ``block_vip_`` — per-block VIPs as ``dict[str, Series]``
- ``block_spe_``, ``block_hotellings_t2_``, ``super_hotellings_t2_``
- ``predict(X_new)`` — returns super-scores, block-scores, per-block SPE,
  super Hotelling's T² (and Y predictions for MBPLS)
- ``spe_contributions(X)`` — per-variable squared residuals for fault
  diagnosis
- ``block_spe_limit(name, conf_level)``, ``super_spe_limit(conf_level)``,
  ``hotellings_t2_limit(conf_level)``

Worked example: LDPE tubular reactor (MBPLS)
--------------------------------------------

The LDPE dataset shipped with this package is a tubular polymer reactor
with two zones; the Y-block is five quality variables. Splitting the
X-block by reactor zone is a natural multi-block setup.

.. code-block:: python

   import pathlib
   import pandas as pd
   from process_improve.multivariate.methods import MBPLS, randomization_test_mbpls

   folder = pathlib.Path("process_improve/datasets/multivariate/LDPE")
   values = pd.read_csv(folder / "LDPE.csv", index_col=0)

   # Reactor-zone split (1-based MATLAB indexes -> 0-based Python)
   zone_1_idx = [0, 1, 2, 5, 7, 9, 11, 13]
   zone_2_idx = [3, 4, 6, 8, 10, 12]
   x_blocks = {
       "zone1": values.iloc[:, zone_1_idx],
       "zone2": values.iloc[:, zone_2_idx],
   }
   y_df = values.iloc[:, 14:]

   model = MBPLS(n_components=3).fit(x_blocks, y_df)
   print(model.display_results())

   # How predictive is each component? Lower risk_pct = more significant.
   sig = randomization_test_mbpls(model, x_blocks, y_df, n_permutations=200, seed=0)
   print(sig)

   # Top 5 variables contributing to a high-SPE observation
   contribs = model.spe_contributions(x_blocks)
   for name, df in contribs.items():
       worst_row = df.sum(axis=1).idxmax()
       print(name, df.loc[worst_row].nlargest(5))

   # Quick visual: super-score plot, RMSEE plot
   model.super_score_plot(pc_horiz=1, pc_vert=2).show()
   model.predictions_vs_observed_plot(y_df, variable=str(y_df.columns[0])).show()

References
----------

- Westerhuis, J. A., Kourti, T. & MacGregor, J. F. *Analysis of
  multiblock and hierarchical PCA and PLS models.* Journal of
  Chemometrics, 12 (1998), 301-321.
- Westerhuis, J. A. & Smilde, A. K. *Deflation in multiblock PLS.*
  Journal of Chemometrics, 15 (2001), 485-493.
- Wiklund, S., Nilsson, D., Eriksson, L., Sjöström, M., Wold, S. &
  Faber, K. *A randomization test for PLS component selection.*
  Journal of Chemometrics, 21 (2007), 427-439.
