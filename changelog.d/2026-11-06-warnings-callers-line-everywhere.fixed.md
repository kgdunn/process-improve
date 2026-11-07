- **Every warning the package raises names the line that made the call, not only those in
  `multivariate` and `batch`.** Some named a line inside `process_improve` instead:
  `generate_design`'s warnings that splitting a CCD's cube into blocks confounds an interaction,
  that no Plackett-Burman design of the smallest size is built, that a supersaturated budget
  leaves factors fully aliased, and that `hard_to_change` is ignored; `analyze_experiment`'s that
  whole plots are analysed by ordinary least squares, and its two that the fit changes, or loses
  rank, once the factors are coded; and the Mixed Assessor Model's warning that a panelist did
  not score every product, when `analyze_descriptive` raised it. The others named the caller only
  when it called the function that warns directly. (#1169)
