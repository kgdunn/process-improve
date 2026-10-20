- **Tukey's test for nonadditivity, and estimates of missing runs.** `tukey_nonadditivity(design_matrix,
  formula)` fits the additive model again with the square of its fitted values: the sum of squares on
  one degree of freedom, its F test against the remainder (or, with `error="pure"`, against the pure
  error of replicated cells), the regressor's coefficient (Box, Hunter and Hunter's theta), and the
  power of the response that would make the effects add. `estimate_missing(design_matrix, formula)`
  gives the least-squares estimates of the runs whose response is missing (Yates's formula for one), and
  the approximate analysis of the table they complete, with the residual degrees of freedom reduced
  for each estimate. They reproduce Montgomery's impurity example, Box, Hunter and Hunter's penicillin
  and toxic-agents tests, and Davies's estimates and approximate analysis (#800 E1, E2).
