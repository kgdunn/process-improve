- **`PLS.rmse_` counts the observed cells of each response, as the R² does.** It averaged the
  squared residuals of every row, so one missing Y cell made that response's `rmse_` NaN at every
  component, and `PLS.prediction_interval`, which takes its error from `rmse_`, gave NaN bounds
  for that response. (#1027)
