- **`PRM.compare_cv_criteria` tests PRM's held-out score correlation against a null of PRM's
  own.** Its permutation null refitted the folds by kernel PLS, and took the observed correlation
  from the same PLS pipeline, so `r_cv_p` and `r_cv_threshold` were plain PLS's, not those of the
  `r_cv` beside them. Each permutation now repeats PRM's own cross-validation, refitting every
  fold with PRM. The numbers change, and the test is about a hundred times slower: with the
  default 199 permutations, a minute or two rather than a second for 40 rows, 7 folds and
  3 components. `n_cv_permutations=0` still gives the quick normal approximation. `PLS`'s null is
  unchanged. (#1087)
