- **`selectivity_ratio` and `target_projection` skip a missing cell of `X`, as the NIPALS fit does.** One
  missing value made every target-projected score and loading NaN, and every selectivity ratio 0, with no
  warning, so `find_predictive_descriptors` reported nothing for a descriptor block with gaps. A row's score is
  now regressed on the observed part of the target-projected weight, a feature's loading on the scores of the
  rows where it is observed, and both sums of squares of the ratio run over the feature's observed cells. On
  complete data the values do not change beyond rounding. (#1114)
- **A column the fit left out, having no observed value (#1098), contributes 0 to `t2_contributions`,
  `spe_contributions`, `score_contributions` and `group_contributions`, as `transform` reads it at 0.** For
  `PCA` and `PLS`, a value a new row had in it was a whole SPE contribution, and a NaN there sent every row down
  the missing-data path and gave the column NaN; `MBPLS` gave the column NaN. `PLS.invert` now gives such a
  column NaN in `x_new`, where it gave 0 on X's original scale, a value the model knows nothing about. A
  response the fit left out needs no value in `y_desired`, and its `y_hat` is NaN, as `predict` gives it;
  `invert` refused a target without it, and predicted it at 0. (#1114)
