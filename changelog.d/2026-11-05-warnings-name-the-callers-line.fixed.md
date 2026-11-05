- **Every warning in `multivariate` and `batch` names the line that made the call.** Some named a
  line inside `process_improve` or in scikit-learn's `base.py` instead: `PCA.fit`'s warning that
  more components were asked for than the data hold, `TPLS.fit`'s that a column has no variance or
  no observed value, `PCA.select_n_components`'s that the folds of `cv_scheme="ek"` cannot reach
  the components, `PLS.compare_cv_criteria`'s that X is already autoscaled, `MCA.fit`'s that it was
  asked for more axes than the table has, `BatchScaler.fit`'s that a tag has a zero range, and the
  warnings of the PLS that `PLSDA`, `PRM` and `AdaptivePLS` fit inside their own `fit`. The others
  named the caller only when it called the method that warns directly. (#1079)
- **PCA's, PLS's and TPLS's debug log says whether each component converged.** It said a component
  "converged" also when it reached `max_iter` without converging. (#1079)
