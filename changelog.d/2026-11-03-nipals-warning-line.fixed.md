- **Non-convergence warnings name the caller's line.** The warnings that a PCA, PLS, MBPLS or MBPCA
  component did not converge, and PLS's warning that its imputed cells were still moving, named a
  line inside `process_improve` or in scikit-learn's `base.py`. They now name the line that called
  `fit`, also through a subclass such as `PLSDA` or an estimator that fits PLS inside its own `fit`.
  (#1079)
