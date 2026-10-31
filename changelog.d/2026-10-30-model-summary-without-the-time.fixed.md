- **A model summary no longer prints the date and time of the call.** `Model.summary()` and
  `summary()` kept statsmodels' `Date:` and `Time:` cells, so two summaries of one fit differed,
  and so did the `summary_text` of two `fit_linear_model` calls a second apart. The cells are left
  out, and the left-hand cells below them move up; every number in the summary is as before. (#952)
