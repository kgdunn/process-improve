- **A mixture analysed with an explicit formula keeps its coefficients in proportions.** A
  formula over blends alone that holds the constant, such as `"y ~ x1 + x2 + x3 + x1:x2 - 1"`, is
  analysed as the Scheffé model it is, and with `coding="auto"` its coefficients are now in
  proportions, as for the `scheffe_*` keywords. Before, each component was coded from -1 to +1, as
  for a process factor, and the coefficients meant nothing. `coding="coded"` now raises for it, and
  neither `"coefficients"` nor `"split_plot"` reports a `factor_coding` for its components.
  `"model_selection"` on a Scheffé model refits the selected model with the constant that blends
  rounded for printing make up, as the main fit does, so its `r_squared` and `r_squared_adj` are
  about the mean, not about zero. (#1094)
