- **`compare_cv_criteria` and `pseudo_validation_set` refuse every false `scale`, not only
  `False`.** Each training fold is autoscaled, so the model fitted to all rows must be too, and
  `scale=False` was refused for that reason. The check was `scale is False`, so `scale=0`,
  `scale=None` and `scale=np.False_` passed it and fitted the full-data model unscaled, whose weight
  angles, D ratios and alarm rates were then compared with fold models in another space. All four
  are now refused, with the value named. (#1029)
