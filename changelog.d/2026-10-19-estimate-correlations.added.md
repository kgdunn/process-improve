- **`analyze_experiment`'s coefficients carry the correlations of the estimates and the contour of
  their joint confidence region (#789 B10).** `estimate_correlations` gives the correlation of each
  pair of coefficient estimates, from (X'X)^-1 alone, so it exists for a saturated design too; BHH
  print -0.8250 for the impurity regression's b1 and b2 (eq. 14.41). `joint_confidence_region`
  gives the residual sum of squares that bounds all the coefficients together, S_R (1 + p F /
  (n - p)) at `1 - significance_level` (BHH eq. 14.45: 4.2053 at 90%, with F rounded to 4.32;
  4.2082 from the exact F), or S_R plus sigma squared times the chi-square point with a `sigma`.
