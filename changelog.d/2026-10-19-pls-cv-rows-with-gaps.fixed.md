- **`PLS.select_n_components` and `PLS.nested_cv` predict a held-out row with a missing X cell
  (#676).** Such a row was scored by a plain product with the direct weights, so its prediction was
  NaN, and its residual dropped out of PRESS while the row still counted in the divisor. With 15%
  of X missing, 21 of 40 rows had no prediction and RMSECV read 3.5 to 7.6 against about 1 for the
  complete data. A held-out row with a gap is now projected over its observed columns by trimmed
  score regression, as `PLS.project` projects a new row, with the model of each component count.
  Separately, with gaps in the training data the first columns of a fold model's direct weights
  are not the smaller model's, so every held-out row was scored with the wrong weights; each
  component count now uses its own. A count that leaves a held-out row with fewer observed
  columns than components unpredicted has no RMSECV, Q2 or randomization p-value, and is not
  recommended. Results on complete data do not change.
