- **A cumulative sum about a target, the variance of a sampling scheme, and defects per million.**
  `monitoring.cusum_about_target(values, target)` gives the plain cumulative sum of the deviations
  from a target, which Box, Hunter and Hunter plot (Table 17.1), beside the tabular CUSUM.
  `sampling_scheme_variance(components, schemes, costs=None, target=None)` turns variance components
  into the variance of the mean of each sampling and testing scheme, Σ σ²_k / n_k, its cost, and the
  cheapest scheme that reaches a target variance (BHH Table 17.4). `calculate_cpk` also returns
  `defects_per_million` and `defects_per_million_nearer`, the parts per million outside the
  specifications of a normal process with that capability (#802 G2, G4, G5).
