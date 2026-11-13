- **A design-quality plot is drawn for a factor named `I` or `C`.** `prediction_variance`,
  `fds_plot` and `power_curve`, given a fitted model that squares a factor, stopped with
  `UnsafeFormulaError`: the fitted formula calls `I()`, and a column of that name hides it. The
  plots now build their model matrix with such a factor under a stand-in name, as
  `evaluate_design` does, and show the factor under its own name. factori.al draws the
  prediction-variance chart of such a study with it. (#1229)
