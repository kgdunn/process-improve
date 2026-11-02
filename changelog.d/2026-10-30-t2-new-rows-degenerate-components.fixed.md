- **The Hotelling's T² of a new row leaves out the components `fit` leaves out.** `fit` drops a
  component from `hotellings_t2_` when its score standard deviation is at most 1.5e-8 times the
  largest, as the last component of a rank-deficient fit is (an sd near 1e-16). `PCA.diagnose`,
  `PCA.project`, `PLS.diagnose`, `PLS.project` and `PLS.invert` divided by that sd all the same, so
  a row off the model plane got a T² near 1e32, and a training row a T² other than its fitted one.
  `PCA(n_components=None)` on fewer rows than columns fits such a component, so every new row of a
  wide block was flagged. The same held for the `pca_predict` and `pls_predict` tools, which now
  also refuse a NaN, negative or empty `scaling_factor_for_scores` in `model_params`. `MBPCA` and
  `MBPLS` counted such a super score in `super_hotellings_t2_` and in `diagnose`, and now leave it
  out by the same rule. Models without such a component give the same numbers as before. (#1026)
