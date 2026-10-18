- **`optimize_responses` and `evaluate_model` evaluate any polynomial in the factors.** A term may
  be a whole power, `I(A ** 3)`, or a product of powers, `A:I(B ** 2)`, so a cubic model is
  optimised with `method="desirability"` or `"pareto_front"`. The second-order analyses
  (`stationary_point`, `canonical_analysis`, `ridge_analysis`) refuse such a term by name, as they
  refuse `A:B:C`.
