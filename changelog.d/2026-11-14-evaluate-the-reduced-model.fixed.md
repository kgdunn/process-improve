- `evaluate_design` judges a design on the columns its region can estimate when the region's limits cost the
  model a column: the interactions of factors that a `count_nonzero(...)` rule keeps from being used together,
  and one column of a group of mixture components that the limits hold to a fixed total. It reported such a
  design, as `generate_design` builds it, rank-deficient with every metric `None`. The columns left out are
  named in `result["notes"]["inestimable_columns"]`. A design that cannot estimate the columns the region
  leaves is still reported rank-deficient.
- The warning of `analyze_experiment`, when its default interactions model leaves out the interaction of two
  factors that no run uses together, says that no run has the two both above their lowest values. It said
  that one factor takes one value on a level of the other.
