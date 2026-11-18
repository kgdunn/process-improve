- **`feasible_region` reads a prior `sigma`.** A model from `analyze_experiment(..., sigma=...)` holds
  `prior_sigma`; the chance of meeting a specification now scales a new run by it and reads the normal
  distribution, where before it used the residual mean square and its degrees of freedom. A saturated fit with a
  prior `sigma` now has a chance, where before it had none.
