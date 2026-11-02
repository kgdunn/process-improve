- **`ASCA` on an unbalanced design.** A design missing a whole combination of levels was called
  balanced (`is_balanced_`), and its sums of squares, which do not partition the total, came without
  the warning that says so. The warning now follows the sums of squares themselves, so a fraction
  fitted for its main effects, whose parts do add to the total, still does not warn. And the
  intercept of the sum-coded fit, which on an unbalanced design is not the column mean, was left in
  `residuals_`, adding to the residual sum of squares and offsetting every term's PCA with
  `add_residuals=True`. It is now part of `grand_mean_`, so `residuals_` is what no term explains;
  on a balanced design both are as before. (#1029)
