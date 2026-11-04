- **The predicted-vs-observed plots give a number when a Y cell is missing.** The RMSE annotation of
  `PLS.predictions_vs_observed_plot` (and of `predictions_vs_observed_plot`, which `BatchPLS` also draws) and
  the RMSEE annotation of `MBPLS.predictions_vs_observed_plot` were `nan`, and the PLS plot's `y = x` line had
  no extent. Both now count only the observed cells, as `PLS.rmse_` does. On complete data the plots do not
  change. (#1081)
- **`interaction_terms` accepts an `x_log` with `pd.NA` in a nullable column, and reads it as a missing
  value, as the estimators do.** Its check that `x_log` is centred and unit-variance scaled raised
  `TypeError`. (#1081)
- **`AdaptivePLS` scores a row with missing values the same way in its prediction channels as in `predict`.**
  The `static`, `preprocessing` and `kernel` channels of `decompose_prediction` and `prediction_channels_`
  put a missing value at the centre, while `predict`, `update` and `predictions_` project the row over its
  observed values one component at a time. The `adaptive` channel therefore differed from the prediction for
  such a row. All three channels now project the row, the `static` and `preprocessing` ones with the training
  model, so the `adaptive` channel is the prediction. The predictions do not change, and neither do the
  channels of a complete row beyond rounding. (#1081)
