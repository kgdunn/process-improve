- **The ANOVA of a model with a nested factor no longer prints meaningless rows.** When a factor's
  levels each fall in one level of another, as the times of spinning within each square of Davies's
  yarn experiment do, the model matrix is short of full rank, and `analyze_experiment`'s Type II
  table counted every column as a degree of freedom: Davies's squares, times and days came out on
  3, 11 and 11 df, with sums of squares of 0.11, 0.41 and 0.41, and a warning about rank only.
  Each term is now tested on the degrees of freedom it adds (times 6.794 and days 2.317 on 8 df,
  as Davies prints them). A term that adds none, such as the squares, gets no test, and
  `anova_note` names the `multistratum_anova` strata that separate it.
