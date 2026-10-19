- `augment_design(..., "add_runs_optimal", target_model=<formula>)` builds the model rows of a
  formula written as main effects, `A:B` products and `I(A ** 2)` squares with numpy instead of
  patsy, so the exchange's polish no longer rebuilds a patsy matrix at every move: 0.5 s instead
  of 7 s for eight factors. Other formulas still go through patsy.
