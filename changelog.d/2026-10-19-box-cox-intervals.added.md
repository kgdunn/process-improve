- **The Box-Cox interval as Box, Hunter and Hunter and Montgomery draw it.** `box_cox_profile` takes
  `interval="t"`, the allowance S_min(1 + t²/ν) on the model's residual degrees of freedom, beside the
  default likelihood-ratio interval, and the `box_cox` analysis reports both under `intervals`, each
  with its `critical_sse`, with the least SSE(λ) as `sse` and its `residual_df`. It reproduces BHH's
  critical value 0.353 for the toxic agents, and the books' limits, which they read off a plot, to
  within 0.03. Each end of an interval is now solved for, rather than taken at a point of the 0.001
  grid, so `lambda_ci` can move by less than 0.001 (#801 F1).
