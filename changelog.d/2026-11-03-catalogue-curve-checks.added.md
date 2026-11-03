- **The reference catalogue checks the functional PCA numbers of a curve response.** Two check
  kinds read a curve through `multivariate.FPCA`: `fpc_variance_explained`, the percent of
  variance its first components explain, and `fpc_score`, a run's score. `argument_scale` reads
  a time printed in other units, such as hours / 24, in the units the source integrates over.
  McMullen and Jurica's (2024) 16 FPCA numbers are now recomputed rather than listed as a gap in
  the library. They do not agree, and each records why: the variance share differs by smoothing
  method (92.9% against 90.717%), and the scores are 0.984 times the library's plus an offset of
  32.7 that is not explained yet.
