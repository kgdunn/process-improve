- **Non-convergence warnings name the caller's line.** The warning that a PCA, PLS, MBPLS or MBPCA
  component did not converge named a line inside `process_improve` or in scikit-learn's `base.py`.
  It now names the line that called `fit`, also through a subclass such as `PLSDA` or an estimator
  that fits PLS inside its own `fit`. So does PLS's warning that its imputed cells were still
  moving, which named the caller only when `PLS.fit` was called directly. (#1079)
