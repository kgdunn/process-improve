- **`analyze_experiment` no longer calls a model estimable because its factors, coded to -1/+1, would estimate it.**
  With `A * B` the same in every run, `y ~ A:B` has rank 1 of 2, but it was reported as estimable with a warning
  that blamed the units, and the browser app showed the intercept and `A:B` each as half the mean, with a small
  p-value. Now `rank_deficient` is true and the app refuses the model, as it does any model the runs cannot
  estimate. A hierarchical model in very unequal units, or with a tiny span far from zero, is still not blamed on
  the design. (#997)
