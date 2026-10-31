- **`robust_regression`'s intervals now resist outliers, as its slope does (#953).** `conf_intervals`,
  `conf_interval_intercept` and `pi_range` keep their keys, shapes and meaning, but are now percentile-bootstrap
  intervals over 999 resamples of the `(x, y)` pairs, each refitted with the repeated median. They used the
  least-squares formulas on the residuals around the robust line, so one gross outlier widened every interval
  even when it left the slope where it was. With 20% of 60 points moved far off a line, the bootstrap slope
  interval covers the true slope in 97% of simulated data sets and is about ten times narrower than the
  least-squares one; without outliers it is about a quarter wider, because the repeated-median slope varies more
  than the least-squares slope. The prediction band adds to each resampled line a normal error scaled by 1.4826
  times that resample's median absolute residual. New keyword arguments: `n_boot` (default 999), `random_state`
  (default 0, so a repeated call returns the same numbers) and `intervals="least_squares"`, which returns the
  numbers of earlier releases. `SE`, `standard_errors`, `t_value` and `R2` are unchanged. The bootstrap takes
  about 0.1 s for 100 points and over 10 s for 1,000; lower `n_boot` for large data sets. The
  `robust_regression` tool takes `n_boot` and `random_state` too, and says which intervals it returns.
- **`repeated_median_slope` forms the pairwise slopes as one N x N matrix instead of a Python loop**, and returns
  the same slope. It is several times faster; at the 5,000-point cap it needs about 425 MB of memory.
