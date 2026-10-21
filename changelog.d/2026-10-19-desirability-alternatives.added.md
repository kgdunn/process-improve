- **Alternative near-best settings.** `optimize_responses(..., method="desirability",
  n_alternatives=k)` reports, beside the best setting, up to k other local optima of its
  multistart search under `alternatives`: each with a composite desirability of at least 95% of
  the best's, and at least 0.5 apart in coded units from the best and from each other, so each is
  a different choice (one cheaper, safer or easier to run, say). The default of 0 leaves the
  search and its result as they were.
