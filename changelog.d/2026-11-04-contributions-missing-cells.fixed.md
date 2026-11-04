- **The contributions of a row with a missing cell add up to the score, T² and SPE that the model gives that
  row.** `t2_contributions`, `spe_contributions`, `score_contributions` and `group_contributions` of `PCA` and
  `PLS`, and of the estimators built on them, now take by default (`method=None`) the estimator their `transform`
  and `diagnose` use: trimmed score regression. They took single-component projection, so for such a row they
  split another score than the one `diagnose` reports. `method="scp"` gives the old numbers, with which a
  training row reproduces the stored `scores_`. A row too sparse for the estimator is NaN with a warning, where
  `method="tsr"` or `"pmp"` raised. `AdaptivePCA` and `AdaptivePLS` keep single-component projection, which
  their `transform` uses. `MBPCA` and `MBPLS` `score_contributions` divided every row by the loadings' or
  weights' sum of squares over all the cells, and a row with a missing cell now divides by it over its own cells,
  as its super score does: on one test row the contributions summed to 0.296 against a super score of 0.458. Their
  `group_contributions` counts a missing cell as 0, where the column mean skipped it and averaged over the other
  rows alone. On complete rows the values do not change beyond rounding. (#1144)
