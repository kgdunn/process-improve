- **`TPLS` warns when a component does not converge.** A component whose loop stopped at
  `max_iter` was the state of the last pass, with no warning, and that state can differ from one
  platform to another: the TPLS loop can alternate between two states for ever, as the third
  component of the `tpls_new_materials` example did. The fit now gives a `SpecificationWarning` in
  the words of `MBPLS` and `MBPCA`, and `fitting_statistics["converged"]` records, per component,
  whether it converged. (#1163)
- **A fitted model keeps its own copy of the caller's row and column labels.** `PCA`, `PLS`,
  `MBPLS`, `MBPCA` and `TPLS` kept the caller's index objects, so a name given to the caller's
  axis after the fit, such as `X.index.name = "batch"`, renamed the model's scores, loadings and
  other frames too. The numbers were not affected.
