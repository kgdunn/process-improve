- **A training row with nothing observed has an SPE of NaN in `PCA`, `PLS` and `TPLS`**, as in `MBPCA`
  and `MBPLS`, and as `diagnose` gives such a row. It was 0. The fit still places the row at the centre,
  so its scores and T² stay 0. `spe_limit` (and TPLS's `spe_limit`) is over the rows that have an SPE,
  so a model fitted with such a row has a slightly higher limit than before. (#1114)
- **One warning for a column with no variance across the family.** `PCA`, `PLS` (and `PLSDA`, `PRM`),
  `OPLS`, `MBPCA` and `MBPLS` fitted a column whose observed values are all equal without a word; they
  now give the `UserWarning` `TPLS` gave, for X and for Y: "Block X has columns with no variance, which
  carry no information: 'c'. Drop them, or check the data for a stuck sensor." Its text no longer says
  "and are excluded from the model", which is not so for an un-centred `PCA`. Only a fit the caller asks
  for warns: a fit inside a cross-validation, or on a matrix another estimator derived (`ASCA`, `FPCA`),
  does not. A block in which nothing varies gets the estimator's `NotEnoughVarianceError` alone. (#1114)
- **The R² and RMSE of a weighted fit are measured in its weights.** `PLS` fitted with `sample_weight`,
  and `PRM`, whose fit is a weighted `PLS`, reported `r2_cumulative_`, `r2_per_component_`,
  `r2_per_variable_`, `r2y_per_variable_` and `rmse_` over every row with equal weight, outliers and rows
  of weight 0 included. They are now over the rows in the weights the fit used, so a fit with zero
  weights reports what the fit on the other rows reports. `PRM`'s R² and RMSE describe the rows it
  trusts: on 30 rows with 9 shifted by 25 in y, its two-component R² rises from 0.014 to 0.866 and its
  RMSE falls from 9.5 to 0.64, so its `prediction_interval`, which uses `rmse_`, narrows from about 39
  to 2.7 wide. (#1114)
- **A count that is not a whole number is refused**: `max_components`, `n_repeats` and `n_permutations`
  of `PLS.select_n_components`, `PCA.select_n_components`, `MBPLS.select_n_components`, `PLS.nested_cv`
  and `compare_cv_criteria` (and its `n_cv_permutations`), the `n_permutations` of
  `randomization_test_mbpls`, `PLSDA.permutation_test`, `ASCA.permutation_test`, `ASCA.vasca` and
  `GPA.consensus_test`, the `n_components` of `pseudo_validation_set`, and `impute_low_rank`'s
  `n_components`. 2.5 was read as 2 without a word, or failed in a slice with a `TypeError`; it now
  raises "max_components must be a whole number; got 2.5.". A float that holds a whole number is refused
  too, as scikit-learn's `Integral` constraint refuses it in an estimator's settings. (#1114)
