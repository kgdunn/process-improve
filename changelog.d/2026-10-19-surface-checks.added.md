- **Checks on a response surface before it is read (#797 R2 to R4).**
  - `third_order_contrasts(design, response_col, factor_cols, sigma=None)` gives, for each factor of a
    composite design, the slope its axial runs give less the slope its cube gives, with its standard
    error: the lack of fit along each axis, which the lack-of-fit F test pools. Box, Hunter and Hunter's
    1.28 and -1.93, each with a standard error of 1.06, reproduce.
  - `pooled_pure_error(designs, response_col, factor_cols)` pools the replicated runs of the designs
    of one investigation, each design's replicates compared with their own mean (Box, Hunter and
    Hunter's 4.07, then 3.18).
  - `evaluate_design(..., metric="alias_matrix", alias_terms="quadratic")` also aliases the model
    against the squares of the numeric factors, so the matrix shows how curvature biases a
    first-order design's intercept (Davies, 11.33: b0 -> b0 + b11 + b22 + ...).
