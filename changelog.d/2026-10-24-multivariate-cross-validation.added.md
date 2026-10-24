- **Choose the number of components on the multivariate page.** A "Cross-validate" button,
  never run on its own, scores every number of components on values the model was not fitted
  to and draws R² beside Q² with one standard error, the recommendation marked; clicking a bar
  uses that number. In Python, `process_improve.workbench.Variant.cross_validate` does the same
  over a recipe's table, leaving the excluded rows out, and returns a `CrossValidation` that
  writes the code repeating it. For PLS, the targets weigh alike, as the fitted model weighs
  them.
