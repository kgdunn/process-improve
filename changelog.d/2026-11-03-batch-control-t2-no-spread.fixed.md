- **`midcourse_correction` leaves a component with no spread out of its T², as `fit` does.** With
  no `score_covariance`, it weighted each component by `1 / explained_variance_`, so the last
  component of a rank-deficient `BatchPLS` fit, whose variance is near 0, got a weight near 1/0 and
  ruled the T² penalty, the T² cap and the reported `t2`. It now uses the rule `hotellings_t2_`
  uses: a component whose score standard deviation is at most 1.5e-8 times the largest one gets
  weight 0. Models with no such component give the same numbers as before. (#1086)
