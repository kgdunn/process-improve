- **Constraints on categorical factors: disallowed combinations and limits on some levels only.** A constraint
  may compare a categorical factor with its levels (`==`, `!=`, `in`, `not in`) and join comparisons with `and`,
  `or` and `not`: `"not (catalyst == 'B' and solvent == 'Y')"` disallows a combination of levels, and
  `"not (catalyst == 'B' and T > 180)"` limits `T` when the catalyst is B. The optimal designs place runs only
  where the constraints allow. A disallowed combination is a missing cell that leaves part of an interaction
  inestimable over the whole region: the design is built for the columns that can be estimated, a warning names
  the term and the missing cell, and `metadata["inestimable_columns"]` records it. `DesignRegion` takes the
  same constraints: `feasible()` and `broken_limits()` read each point's levels, `sample_frame()` and
  `support_frame()` draw the levels with the settings, `allowed_combinations()` lists the allowed combinations
  and the limits on each, and `given()` returns the region of one combination. `evaluate_design` averages and
  maximises over the allowed combinations only, each weighted equally, and drops the same inestimable columns. A
  level name the factor does not have is an error, and two inequalities joined by `or` are refused. Design types
  that cannot follow a constraint naming a categorical factor refuse it instead of warning, and such constraints
  are not supported beside mixture components yet.
