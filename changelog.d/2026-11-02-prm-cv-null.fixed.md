- **`PRM.compare_cv_criteria` tests PRM's held-out score correlation against a null of PRM's
  own.** Its permutation null refitted the folds by kernel PLS, and took the observed correlation
  from the same PLS pipeline, so `r_cv_p` and `r_cv_threshold` were plain PLS's, not those of the
  `r_cv` beside them. Each permutation now repeats PRM's own cross-validation, refitting every
  fold with PRM. The numbers change, and each permutation is about a hundred times slower, so
  PRM's default is now 49 permutations instead of 199 (`n_cv_permutations=None`): the smallest
  attainable p-value is 0.02, still enough to decide at `alpha=0.05`. Pass `n_cv_permutations`
  to draw more. `n_cv_permutations=0` still gives the quick normal approximation. `PLS`'s null
  and its default of 199 are unchanged. (#1087)
