- **F tests between nested models.** `compare_models(reduced, full, error=None)` is the
  extra-sum-of-squares F test of two models from `lm` (or the statsmodels formula API) fitted to the
  same runs, as R's `anova(reduced, full)` gives it, optionally against the residual of a third model
  that contains the full one. `group_test(model, terms)` tests a group of a model's terms together,
  with the reduced model built by leaving them out and fitted to the same runs. Both check that the
  models are nested, by the columns of their model matrices, and refuse models fitted to different
  runs. They reproduce Montgomery's lack-of-fit tests (Tables 6.7 and 6.22), the sequential model sums
  of squares of Table 11.7, the extra sum of squares of Example 10.6 and the mixture test of Tables
  11.21 and 11.22 (#789 B2).
