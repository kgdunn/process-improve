- **The scale of Box, Hunter and Hunter's reference distribution.** `least_squares_means` also returns
  `reference_scale`, the scale of the t distribution the book slides under a set of means, on the
  residual degrees of freedom. By default it is the square root of half the average variance of a
  difference of two means: s/√n with n runs at each level, and √(k s²/(λt)) in a balanced incomplete
  block design. `Comparisons(reference_rule="average_replication")` gives s/√n̄ instead, with n̄ the
  average number of runs at a level, as the book scales its unequally replicated blood-coagulation
  diets. They reproduce the book's 0.97 (blood coagulation), 0.037 (toxic agents) and 20.50 and 17.06
  (the wear-testing Youden square) (#800 E3).
