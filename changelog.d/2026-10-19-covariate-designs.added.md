- **Designs on units with known covariates.** `designs_covariates.covariate_design(units, factors,
  n_runs, CovariateOptions(...))` chooses which of a list of experimental units to use, each at
  most once, and the factors' settings for each, D-, A- or I-optimally for a model with the
  units' covariates beside the factors. With every unit used it places the settings in time
  for a trend. On Goos and Jones's polypropylene samples it chooses the printed 18 of 40, and
  with more starts a design 0.05% better; with the time in the model, the best order of the
  2^3 is trend-robust.
