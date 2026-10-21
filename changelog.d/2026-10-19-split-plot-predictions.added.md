- **Predictions and optimisation from a split-plot fit.** `analyze_experiment(...,
  analysis_type="split_plot", new_points=...)` predicts from the generalized least-squares fit:
  each point's mean with its standard error, degrees of freedom and interval (Kenward and Roger's
  when asked for), and the interval for one new run there, whose variance adds every variance
  component. The split-plot result carries `response_name`, `factor_names` and `coding`, so
  `optimize_responses` takes it as a fitted model and works from its coefficients rather than the
  ordinary least-squares ones. On Goos and Jones's battery cell it finds the best corner with
  the printed 7.42, where the ordinary least-squares fit predicts 7.75.
