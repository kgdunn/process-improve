- **A confidence level outside (0, 1) is refused, by every function that takes one.** A level
  between -1 and 0 gave a reversed interval without a word: `robust_regression(x, y,
  conflevel=-0.5)` returned a slope interval whose lower limit lay above its upper one. A level of 0
  gave an interval of zero width, and one of 1 or more an infinite or NaN one. `robust_regression`, `OLS`,
  `multiple_linear_regression`, `OLS.prediction_interval`, `ttest_one_sample`,
  `ttest_independent`, `ttest_paired` (and their `_from_df` forms), `confidence_interval`,
  `least_squares_means`, `intraclass_correlation_interval`, `BlockedFit.predict`,
  `selectivity_ratio`, `product_means`, `compare_products` and `analyze_descriptive` now raise
  `ValueError: conflevel must lie in (0, 1); got -0.5.` (or `conf_level`, as the function names it)
  before any work is done, as the multivariate limits already did. `robust_regression` gave NaN
  intervals for a level of NaN, infinity or above 1 since 2.3.1; it raises the same error now.
  `variance_interval` and `nonlinear_regression` already refused such a level; their message now
  reads the same as the others. (#1132)
- **A perfect fit no longer warns that `robust_regression`'s default intervals change in 3.0.** A
  line through every point has intervals of zero width whether they are found by least squares or
  by the bootstrap, so the change leaves them alone, and the `FutureWarning` and `intervals_note`
  are gone there. Residuals of rounding size, such as those of `y = 0.07 x - 0.3` at x = 0.1, 0.2,
  and so on, count as a perfect fit. (#1132)
