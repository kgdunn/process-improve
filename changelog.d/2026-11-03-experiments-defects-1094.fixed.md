- **A mixture analysed with an explicit formula keeps its coefficients in proportions.** A
  formula over blends alone that holds the constant, such as `"y ~ x1 + x2 + x3 + x1:x2 - 1"`, is
  analysed as the Scheffé model it is, and with `coding="auto"` its coefficients are now in
  proportions, as for the `scheffe_*` keywords. Before, each component was coded from -1 to +1, as
  for a process factor, and the coefficients meant nothing. `coding="coded"` now raises for it, and
  neither `"coefficients"` nor `"split_plot"` reports a `factor_coding` for its components.
  `"model_selection"` on a Scheffé model refits the selected model with the constant that blends
  rounded for printing make up, as the main fit does, so its `r_squared` and `r_squared_adj` are
  about the mean, not about zero. (#1094)
- **A definitive screening design built from a larger conference matrix says so.** No conference
  matrix of order 22 or 34 can be built, so 21 or 22 factors take order 24 (49 runs, not 45) and
  33 or 34 factors take order 38 (77 runs, not 69). Only the docstring said so; `generate_design`
  now gives a `UserWarning` and the same text as `metadata["note"]`. (#1094)
- **The `residuals_vs_order` plot draws the residuals in run order.** It numbered them in the rows'
  order, so a design stored in standard order was drawn in standard order, which hides the drift
  over time the plot exists to show. When `design_data` has a `RunOrder` column that gives each
  residual a distinct number, the residuals are now drawn against it, as `residual_diagnostics`
  already orders them for its `run_order` check. (#1094)
