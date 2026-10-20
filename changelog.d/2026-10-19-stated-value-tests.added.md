- **Results against a value stated before the data** (`process_improve.univariate`, #789 B5 and B6).
  `variance_about(x, mean)` gives the variance about a known mean on all n degrees of freedom;
  `ttest_new_observation(y, mean, x, mean_known=True)` tests one new observation against a stated
  mean with the standard deviation of earlier data; and `grand_mean_test(y, groups, value)` splits
  a one-way layout's sum of squares about a norm and tests the grand mean against it on the
  residual. They reproduce Box, Hunter and Hunter's five cars (2.96, and t = 1.86 on 5 degrees of
  freedom), the production record's 0.36 and 0.60, and the blood coagulation times against the
  norm of 63 (364 = 24 + 228 + 112, t = 2.07 on 20 degrees of freedom).
