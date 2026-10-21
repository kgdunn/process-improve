- **Factorial takes a study of one factor: two treatments compared** (#905), such as two suppliers, an old
  and a new catalyst, or a temperature at its low and its high. Design sizes the comparison by power: a table
  gives, for each number of runs at each level, the chance that the t-test finds the smallest change worth
  finding, and the design is made with the fewest runs that reach 80% (up to 50 at each level). Learn leads
  with the difference between the two levels and its 95% interval, each level's average with its own interval,
  and the difference tested four ways: the t-test with one variance for both levels (the model's own test),
  Welch's t-test, the randomization test of `process_improve.univariate`, and Wilcoxon's rank-sum test. A chart
  shows every run at its level beside the level's average, in place of the response surface. The Python panel
  and the notebook give the same four tests. The catalog's one-factor designs at two levels now open in
  Factorial, among them Box, Hunter and Hunter's comparison of two methods.
