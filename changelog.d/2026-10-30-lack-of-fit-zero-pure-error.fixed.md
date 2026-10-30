- **The lack-of-fit analysis now reports the pure-error and lack-of-fit rows when the repeated
  runs agree exactly, or when the model leaves no degrees of freedom for lack of fit.** It returned
  `{"error": "Insufficient degrees of freedom for lack-of-fit test."}` in both cases, though the
  degrees of freedom are there. A pure error of 0 on its degrees of freedom (or a lack of fit of 0
  on 0 degrees of freedom) is now reported with `f_statistic`, `p_value` and `significant` set to
  `None` and a `note` that says why no F ratio is formed. Pure error at the level of round-off
  counts as zero. "No replicated points" is unchanged. (#969)
