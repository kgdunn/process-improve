- **`feasible_region` allows for responses whose errors move together.** Its probability that one new
  run meets every window multiplied the responses' chances, as if their errors were independent. A
  new argument, `residual_correlation`, takes the correlation of the responses' errors on one run,
  which the correlation of the models' residuals estimates. Each response keeps its own Student's t
  distribution, and the responses are joined by a Gaussian copula, integrated with Genz's method.
  Without the argument the result is unchanged.
- **`analyze_experiment` reports the inverse of `X'X`.** The `"coefficients"` analysis gains
  `"unscaled_covariance"`, in the order of its coefficients and on their scale. With the model
  summary's `mse_residual` and `df_residual` it is what `feasible_region` works the scatter of a new
  run from, so that function now takes the analysis result as it is, as `optimize_responses` does.
  The `analyze_experiment` agent tool leaves the matrix out of its reply.
