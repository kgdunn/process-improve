- **`optimize_responses` says how to avoid "no start converged inside the region" under a `count_nonzero` limit**
  (#1259). When the region counts a factor that has no `min_when_used`, any amount of it above 0 counts as used,
  and the best setting can hold a trace of it that the search never reaches. The error now names those factors
  and says to give each a `min_when_used`. It still starts with the same words.
