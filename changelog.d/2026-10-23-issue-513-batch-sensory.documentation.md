- **`robust_regression` says which of its outputs are robust (#513).** Only the slope and the
  intercept are. The standard errors, `t_value`, `R2`, the confidence intervals and `pi_range` use the
  least-squares formulas on the residuals around the robust line, so one outlier widens the intervals
  without moving the slope. `R2` is `RegSS / (RegSS + RSS)`, which is not the `1 - RSS / TSS` of
  `multiple_linear_regression`. The `robust_regression` tool's description says the same.
