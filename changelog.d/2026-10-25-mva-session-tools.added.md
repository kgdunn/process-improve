- **Multivariate session tools for agents.** `mva_open`, `mva_fit`, `mva_explain`, `mva_predict` and
  `mva_script` let an agent work a multivariate analysis step by step, as the Multi- and Megavariate book does: fit,
  find the rows beyond the T² or SPE limits and the variables that set them apart, leave rows out or transform a
  column, and fit again. The tools keep no state: each takes the analysis so far as a JSON *session* (the source,
  its recipe, and every model with the one it grew from and why) and returns it with the step added, so the session
  works in safe mode and is the whole record. `mva_script` writes the Python that rebuilds each model.
