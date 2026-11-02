- **`PRM` gives the same model in any units.** With `scale=True`, a column of X or Y whose robust
  spread (1.4826 times its median absolute deviation) was below 1.5e-8 was left in its own units.
  A column of X in units 1e9 times smaller then dropped out of the model and moved the predictions
  by up to twice y's standard deviation, and a Y in such units stopped the fit with no variance
  left in Y. A column is now left unscaled only when its median absolute deviation is exactly 0.
  The residuals are also standardised in the units the fit runs in, so the floor on their median
  no longer held the residual weights of a Y in small units near 1, which had left vertical
  outliers in the fit. (#1070)
