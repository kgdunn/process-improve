Multi-block PCA and PLS (MBPCA / MBPLS)
=======================================

Generic multi-block latent-variable models for the case where your X
data is naturally organized into several semantically distinct blocks -
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
- ``block_scores_``, ``block_loadings_`` - both ``dict[str, DataFrame]``
- ``r2_x_per_block_cumulative_``, ``r2_x_per_block_per_component_``
- ``block_vip_`` - per-block VIPs as ``dict[str, Series]``
- ``block_spe_``, ``block_hotellings_t2_``, ``super_hotellings_t2_``
- ``predict(X_new)`` - returns super-scores, block-scores, per-block SPE,
  super Hotelling's T² (and Y predictions for MBPLS)
- ``spe_contributions(X)`` - per-variable squared residuals for fault
  diagnosis
- ``block_spe_limit(name, conf_level)``, ``super_spe_limit(conf_level)``,
  ``hotellings_t2_limit(conf_level)``

Worked example: LDPE tubular reactor (MBPLS)
--------------------------------------------

The LDPE dataset shipped with this package is a tubular polymer reactor
with two reactor zones plus a separately measured pressure variable;
the Y-block is five quality variables (cumulative conversion ``Conv``,
number- and weight-average molecular weights ``Mn`` / ``Mw``, long-
and short-chain branching ``LCB`` / ``SCB``).

The natural multi-block split is:

- **block 1 (zone1)**: ``Tin, Tmax1, Tout1, Tcin1, z1, Fi1, Fs1`` -
  reactor-zone-1 process variables; the common feed inlet temperature
  ``Tin`` is grouped here by convention.
- **block 2 (zone2)**: ``Tmax2, Tout2, Tcin2, z2, Fi2, Fs2`` -
  reactor-zone-2 process variables.
- **block 3 (pressure)**: ``Press`` - a single common operating
  variable.
- **Y**: ``Conv, Mn, Mw, LCB, SCB`` - the quality block.

.. code-block:: python

   import pathlib
   import pandas as pd
   from process_improve.multivariate.methods import MBPLS, randomization_test_mbpls

   folder = pathlib.Path("process_improve/datasets/multivariate/LDPE")
   values = pd.read_csv(folder / "LDPE.csv", index_col=0)

   # Natural 4-block split
   x_blocks = {
       "zone1":    values.iloc[:, [0, 1, 2, 5, 7, 9, 11]],
       "zone2":    values.iloc[:, [3, 4, 6, 8, 10, 12]],
       "pressure": values.iloc[:, [13]],
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

   # Quick visual: super-score plot, predicted vs observed
   model.super_score_plot(pc_horiz=1, pc_vert=2).show()
   model.predictions_vs_observed_plot(y_df, variable=str(y_df.columns[0])).show()

A fully runnable walkthrough of this example, including missing-data
handling, lives at
:doc:`case_studies/latent-variable-modelling/mbpls-ldpe-missing-data`.

Handling missing data
---------------------

Both :class:`~process_improve.multivariate.methods.MBPCA` and
:class:`~process_improve.multivariate.methods.MBPLS` fit directly on
incomplete X-blocks (and incomplete Y for MBPLS) using an iterative
mask-aware NIPALS solver. No imputation, no listwise deletion: the
algorithm computes each per-block score using only the observed
entries of every row, and the super-score is built from those
per-block scores in the usual way.

Two arguments control this:

- ``algorithm`` (``"auto"`` by default): selects between ``"dense"``
  (closed-form, NumPy SVD-style; complete data only) and ``"nipals"``
  (iterative, mask-aware). ``"auto"`` switches to ``"nipals"`` as soon
  as any X-block or Y has a NaN entry, and stays on the faster
  ``"dense"`` path otherwise.
- ``missing_data_settings`` (``dict``, optional): convergence settings
  for the NIPALS inner loop. The defaults are sane and rarely need
  tuning; the available keys are ``md_tol`` (per-component convergence
  tolerance; default ``sqrt(eps)`` ~ 1.5e-8) and ``md_max_iter``
  (cap on inner-loop iterations per component; default 1000).

After fitting, two attributes record the choices the model made:

- ``algorithm_`` - the algorithm actually used, after ``"auto"`` is
  resolved.
- ``has_missing_data_`` - ``True`` iff any input block contained at
  least one NaN.

Minimal example::

   from process_improve.multivariate.methods import MBPLS

   model = MBPLS(
       n_components=3,
       algorithm="auto",
       missing_data_settings={"md_tol": 1e-9, "md_max_iter": 2000},
   ).fit(x_blocks_with_nans, y_df_with_nans)

   assert model.algorithm_ == "nipals"
   assert model.has_missing_data_ is True

   # Per-component NIPALS iteration counts (one entry per component)
   model.fitting_info_["iterations"]

Degeneracy guards
~~~~~~~~~~~~~~~~~

Two patterns can make a NIPALS deflation step ill-posed; the solver
detects them and raises a clear error rather than returning silent
garbage:

- A row in which every entry of an X-block is NaN ("row-all-NaN" in
  that block) - the per-block score for that row is undefined.
- A row in which every entry of Y is NaN (MBPLS only) - the
  regression weight column for that row is undefined.

The fix is one of: keep at least one observed value per row in every
block, drop the offending rows, or impute the offending row to a
domain-sensible value before fitting.

When *not* to use the missing-data path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mask-aware NIPALS solver is the right tool for moderate MCAR
(missing-completely-at-random) patterns: dropped sensor readings,
unscheduled samples, communication-loss gaps. For *structured*
missingness (a whole shift's worth of one tag is gone; one block is
entirely absent for one sample) Trimmed Score Regression (TSR) is
usually a better fit; ``PCA`` exposes it via ``algorithm="tsr"`` for
the single-block case.

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
- Nelson, P. R. C., Taylor, P. A. & MacGregor, J. F. *Missing data
  methods in PCA and PLS: score calculations with incomplete
  observations.* Chemometrics and Intelligent Laboratory Systems,
  35 (1996), 45-65.
- Walczak, B. & Massart, D. L. *Dealing with missing data: Part I.*
  Chemometrics and Intelligent Laboratory Systems, 58 (2001), 15-27.
