- **`optimize_responses(..., flat_tolerance=)`** reads an eigenvalue of the pure-quadratic matrix
  at most this share of the largest in size as zero, for `"stationary_point"` and
  `"canonical_analysis"`, so a surface the sources read as a ridge can be classified as one. The
  slope along such a direction is judged on the same scale, so a ridge rises only when its slope is
  above that share: Box, Hunter and Hunter's stationary ridge (eigenvalues -9.02 and 0.0017, slope
  0.005) is a stationary ridge at `flat_tolerance=0.05`, and their rising ridge (slope 2.97) still
  rises. The default, 1e-8, reads only rounding as zero, as before. A
  `classification_note` now says when an eigenvalue is below 5% of the largest though not flat. The
  `optimize_responses` tool takes the same argument.
