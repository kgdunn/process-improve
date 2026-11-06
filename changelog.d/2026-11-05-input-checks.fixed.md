- **`PLS.prediction_interval` refuses a `cross_validate` result from a bootstrap**, which holds no rows
  out and so has no `rmse_cv`, and says to pass a jackknife or K-fold result. It failed with an
  `IndexError` from numpy. (#1077)
- **`PLS.invert` refuses a `y_desired` DataFrame with other than one row**, and says to call `invert`
  once per row. Two rows failed in numpy's broadcasting. Its error for a missing target names only the
  targets that are missing, not every target. (#1077, #1144)
- **`PLSDA.confusion`, `PLSDA.score` and `PLSDA.roc_auc` refuse labels whose count differs from the rows
  of X**, naming both counts, as `fit` does. They failed in `zip`, in numpy or in scikit-learn. (#1077)
- **`MBPCA`, `MBPLS` and `TPLS` refuse a block with no columns by name**: "Block X["b2"] has no
  columns; a block needs at least one." numpy's "at least one array or dtype is required" named
  neither. (#1077)
- **One wording for a count below 1.** `PRM`'s `max_weight_iter`, the `max_components` of
  `MBPLS.select_n_components` and the `n_permutations` of `GPA.consensus_test` say "must be >= 1; got
  0.", as every other count does, where they said "must be at least 1". (#1144)
- **The errors of `OPLS` name the value they refuse**: "n_orthogonal_components must be >= 0; got -1.",
  "y_desired has 2 values; OPLS.invert expects a single desired response value." and "y_desired must be
  a finite value; got inf." (#1077)
