- **`AdaptivePCA` and `AdaptivePLS` refuse a NaN EWMA factor.** `lambda_center`, `alpha_scale`,
  `lambda_center_y` and `alpha_scale_y` were checked to lie in [0, 1] with two comparisons a NaN
  passes, so a NaN factor was accepted and made every centre or scale it moved NaN. It is now
  refused with the same `ValueError` as a factor outside [0, 1], as `forgetting_factor` and
  `gamma` already were. (#1029)
