- **`analyze_experiment`'s `model_summary` gives the sums of squares in the conventions sources print
  (#789 B1).** `ss_total`, `ss_mean` (the correction for the mean, n times the mean squared) and
  `ss_total_uncorrected` (the crude total, the sum of the squared responses); `ss_model` on the basis
  of `r_squared`, `ss_model_uncorrected` (the sum of the squared fitted values) and `ss_residual`; and
  `coefficient_of_variation`, Design-Expert's "C.V. %". A model with blocks adds `ss_blocks` and
  `r_squared_without_blocks`, `r_squared_adj_without_blocks` and `r_squared_pred_without_blocks`, over
  the total less the blocks' sum of squares, as Design-Expert reports them.
- **The ANOVA of a model with no intercept term says that its rows do not add up** to a model sum of
  squares, and where `model_summary` gives the model's sums. The rows of NIST's yarn-elongation
  mixture model add to 1952.33, which is neither its uncorrected model sum, 2878.27, nor its sum about
  the mean, 128.30.
