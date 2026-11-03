- **`MBPCA` and `MBPLS` fit a row with every value missing, as `PCA`, `PLS` and `TPLS` do.** The row scores 0
  and has no weight in the fit; `MBPLS` predicts it at Y's mean. Both models refused it, and `MBPCA` also refused
  a row with nothing in only one block: that row now keeps a super score from the blocks it has, with NaN for
  its block score and block SPE in the empty block, as in `MBPLS`. `MBPLS` also fits a row with no observed
  response, which gets a Y-score of 0. Scored as a new row by `diagnose`, `transform` or `predict`, a row with
  nothing observed has NaN scores, SPE, T² and predictions, as in `PCA` and `PLS`. `MBPCA.diagnose` now gives NaN
  for the SPE of every new row with a missing cell, whose scores were NaN already; that SPE was 0. (#1098)
- **A column with every value missing is named in a `UserWarning` and left out of the model by `PCA`, `PLS`,
  `MBPCA` and `MBPLS`, as `TPLS` does since #982.** `MBPCA` and `MBPLS` refused the fit; `PCA` and `PLS` fitted it
  without a word, and a NaN that a new row had in the column made every score and prediction of that row NaN.
  The column now gets zero loadings and weights, the rest of the fit is the fit of the data without the column,
  and a value that a new row has in it changes nothing. The column still counts in `n_features_in_` and
  `feature_names_in_`, so new data keeps the columns of the fit. A response with every value missing (`PLS`,
  `MBPLS`) has NaN predictions; `PLS` predicted it as 0. The warning names the block: `X`, `Y`, or `X["name"]`
  for a block of a multiblock model, for example `Block X has columns with no observed values, which carry no
  information and are excluded from the model: 'c'. Drop them, or check the data for a sensor that was never
  logged.` (#1098)
