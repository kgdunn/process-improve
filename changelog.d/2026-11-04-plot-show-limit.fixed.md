- **`spe_plot` and `t2_plot` leave out the limit line when `settings={"show_limit": False}`.** Both
  plots accepted and validated the setting, then drew the confidence-limit line anyway, on every estimator
  that offers them (`PCA`, `PLS`, `PLSDA`, `PRM`, the adaptive models, `BatchPCA`, `BatchPLS`). The line is now
  drawn only when the setting asks for it, and the limit is computed only then, so `t2_plot` without its line
  gives no `FutureWarning` about a limit it does not show. The `y = 0` reference line stays. (#1150)
