- **`analyze_experiment` predicts a new point in the block it names (#800).** In a blocked model, a `Block`
  column in `new_points` gives each point its block's contrasts, so a prediction in block j no longer needs a
  refit with `C(Block, Sum)`. A point with no `Block`, or a missing one, is still predicted for the average block,
  and a block the runs do not have is refused. Confirmation tests and the split-plot and GLM predictions take the
  same points.
- **`visualize_doe(plot_type="reference_distribution")` draws Box, Hunter and Hunter's reference distribution
  (#800).** It takes the result of `least_squares_means`: a dot plot of the levels' averages under the t
  distribution on the residual degrees of freedom, scaled by `reference_scale`, centred at the mean of the
  averages.
- **The reference catalogue checks the numbers of #800's new functions.** Box, Hunter and Hunter's reference
  scales (0.97 for the blood-coagulation diets, 0.037 for the toxic agents, 20.50 and 17.06 for the wear Youden
  square) are `least_squares_means` checks, and the toxic agents' tests for nonadditivity (SNA 0.1537, remainder
  0.0964 and F 6.9; on the rates SNA 0.454 and the remainder's mean square 0.223) are `tukey_nonadditivity`
  checks. Davies's approximate analysis of a table with five missing values (103.07 and 51.17) is an
  `estimate_missing` check. The rates' reference scale (0.123, where the runs give 0.1225) and Davies's
  organic-chemical blends (10.94 and 2.28, where the table completed with the two estimates gives 10.92 and 2.33)
  are recorded disagreements.
