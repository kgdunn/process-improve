- **The contribution functions take the same `X` as `transform`, `predict` and `diagnose`, and apply the model's
  own centring and scaling to it.** `t2_contributions`, `spe_contributions`, `score_contributions`,
  `group_contributions`, `meda` and `omeda`, as functions and as methods, used `X` as given. A `PLS(scale=True)`
  model, the default, and `PLSDA` and `PRM`, which inherit it, centre and scale `X` inside `fit`, so its own
  training data gave contributions that bore no relation to the stored values, with no warning: 12,528,124 for a
  row whose stored T² is 0.0528. The docstrings said that passing the training data reproduces the stored
  `hotellings_t2_`, and now it does for these models too. **Code that scaled `X` by hand for a
  `scale=True` model now scales it twice, and gets different numbers: pass the unscaled data instead.** `PCA` and
  `PLS(scale=False)` do no scaling of their own, so nothing changes for them; `MBPLS` and `MBPCA` already applied
  theirs. `AdaptivePCA` and `AdaptivePLS` now centre and scale with their current centre and spread, as their
  `transform` does. `target_projection` and `selectivity_ratio` already applied the model's scaling; their
  docstrings now say so. (#1125)
- **A row with no observed value is placed at the model centre by the contribution functions and by
  `PCA.project` and `PLS.project`, as `fit` places it, where they refused it.** Its scores, T² and SPE are 0, its
  contributions are NaN at every cell and sum to 0, a PLS model predicts the training mean of Y for it, and its
  `condition_number` is inf. `projection_matrix` with no observed feature returns an operator with no column. (#1125)
- **`group_contributions` leaves out a missing cell, where it made the whole column NaN.** Each row takes part
  with the contributions `score_contributions` gives it, from its observed cells alone, so the result still sums
  to the weighted combination of the scores. On complete data the values do not change beyond rounding. (#1125)
