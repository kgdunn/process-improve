- **`optimize_responses(method="stationary_point")` lists the eigenvalues largest in size first**, the
  order `canonical_analysis` already used. It gave them in ascending order, so a caller who matched
  the two results by position paired the wrong values: Davies's three-factor composite design
  (11.73) read (-3.190, -0.069, 0.780) in one and (-3.190, 0.780, -0.069) in the other.
