- **`PCA` and `PLS` `transform`, `diagnose` and `predict` give a new row with a missing cell estimated scores,
  T², SPE and predictions, where they gave NaN.** The row is scored from the cells it has by trimmed score
  regression, the estimator `project()` uses by default, through the same code, and its SPE is over those cells,
  as `MBPCA`, `MBPLS` and `TPLS` already scored such a row. **Code that used NaN from these methods to find rows
  with missing cells must now check for missing cells itself**, for example with `X.isna().any(axis=1)`. A complete
  row keeps its numbers exactly. A row with nothing observed is NaN, as before, and `PLS.diagnose` no longer gives
  it a T² of 0; it gave that too to a row with a missing cell. A row with too few observed cells for the estimator,
  typically fewer than `n_components`, is NaN with a `SpecificationWarning`; `project()` raises for it, and
  estimates it with a `ridge` above 0. What is built on these methods follows: `PLSDA` and `PRM`, which inherit
  them (`PLSDA` predicted the first class for such a row), `PLS.score`, which raised `ValueError`, and
  `PLS.cross_validate` on X with gaps, whose PRESS, RMSE and Q² left out every held-out row with a missing cell.
  (#1144)
