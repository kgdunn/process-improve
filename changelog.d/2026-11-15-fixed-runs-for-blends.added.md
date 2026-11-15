- An optimal mixture design keeps the blends already made: `generate_design(components, design_type="d_optimal",
  fixed_runs=blends_made, budget=...)` puts them first, as given, and chooses the other runs as the blends that
  add most to them for the model, under the same bounds, constraints, `min_when_used` and `count_nonzero`
  limits. `fixed_runs` holds the proportions, a column per component; the budget counts them. A blend made
  outside the region is kept with a warning, and `metadata["n_fixed_runs"]` and
  `metadata["n_fixed_runs_outside_region"]` record them. It was refused for mixture designs. A split-plot
  mixture design (`hard_to_change`) is still refused.
