- **`PLS.select_n_components` and `PLS.nested_cv` score each held-out row as its fold model
  predicts it.** They scored the held-out rows with the fold's own autoscaling alone, but the
  fold model may centre and scale again: the default `PLS(scale=True)` does, and so does `PRM`,
  which always centres at its robust centre. With `scale_inside_folds=False` the fold models'
  predictions were therefore read in the wrong units, so RMSECV, the validated R² and Q², the
  out-of-fold predictions and the recommended number of components were not those of the models
  fitted (a held-out prediction of 1.14 where its fold model predicts 1.49). With
  `scale_inside_folds=True` a PLS fold model's own scaling is the identity to rounding, and the
  numbers are unchanged; a `PRM` fold model's robust centre is now taken into account too. (#1027)
