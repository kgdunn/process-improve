- **`AdaptivePCA` and `AdaptivePLS` leave a component with no spread out of Hotelling's T2,** as
  `PCA`, `PLS`, `MBPCA` and `MBPLS` do. Its score variance used to be raised to 2.2e-16 instead, so
  with a constant tag and as many components as tags, a row that moved the constant tag by 0.01
  had a T2 of 4.5e11 and was out of control. `scaling_factor_for_scores_` is now the square root
  of `explained_variance_`, as for the batch models. (#1029)
