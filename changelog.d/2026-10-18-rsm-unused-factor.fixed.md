- **A factor name that no model term uses no longer turns a maximum into a ridge.**
  `optimize_responses` with `"stationary_point"` or `"canonical_analysis"` built the quadratic's
  matrix over every name in `factor_names`, so a block column or a copy of a factor in other units
  added an eigenvalue of zero, and a maximum was reported as a `stationary_ridge` with no warning.
  Such a name is now refused, with a `ValueError` that names it.
