- **The `robust_regression` tool fits at most 1,500 points.** With its default resamples a call
  computes 40 repeated medians (the fit and 39 resamples), each costing time in proportion to the
  square of the number of points, so a hosted call above about 1,500 points could run past the
  10 s timeout on a slow machine; at 5,000 points it took 31 s even on a fast one. A call with more points is now refused by the tool's
  input contract, before any worker starts, with a `ToolInputInvalidError` that says why and
  suggests fitting a random sample or calling `process_improve.regression.robust_regression` in
  Python, which keeps its own limit of 5,000 points. The tool's schema gives the limit as
  `maxItems` on `x` and `y`. (#1130)
