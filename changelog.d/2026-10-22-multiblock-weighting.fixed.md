- **MBPLS and MBPCA: constant columns no longer lower a block's weight (#913).** Each block is divided by the
  square root of its number of columns so that blocks of unequal width carry equal weight, and a constant
  column (a column of zeros once centred and scaled) used to count. Twelve constant columns beside four
  varying ones moved a fitted MBPLS model's R²Y from 0.919 to 0.708. The weighting now counts the columns that
  vary, recorded in `block_informative_widths_`, in the fit, `transform`, `predict`, the score contributions and
  the block VIP; `block_widths_` keeps the full width.
- **PLS and MBPLS: a component with nothing left in Y to explain (#913).** When what is left of Y is
  uncorrelated with what is left of X (as after one component of an orthogonal factorial design), NIPALS used
  to change sign at every iteration until `max_iter`, and the warning blamed convergence. The fit now warns
  that the data support fewer components and fits each later one along the largest direction left in X: its
  Y-loading is about zero, so the predictions, R²Y and the cross-validated PRESS stay as they were, and every
  fitted frame keeps the number of components asked for.
