- **`feasible_region`: the settings that meet a specification.** A new function in
  `process_improve.experiments` takes each response's model as `optimize_responses` does, a window
  for each response (`(low, high)`, an end of `None` open) and the settings to judge, and returns
  for each setting every response's prediction, whether every prediction is inside its window
  (`feasible`), and the `probability` that one new run there meets every window. The probability is
  the share of a new run's distribution inside the window, Student's t about the prediction by the
  standard error of a new observation, multiplied over the responses; a model holds what it is
  worked from as `mse_residual`, `df_residual` and `unscaled_covariance`, and without them the
  probability is NaN. The settings that meet the windows with a high probability, 0.9 say, are the
  design space of quality by design.
