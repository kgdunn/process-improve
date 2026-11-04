- **`MBPLS.fit` refuses X-blocks with no variation, as `PLS.fit` does.** When every column of every
  block is constant or has no observed value, it raises `NotEnoughVarianceError`: `There is no
  variance in X: every column of every block is constant or has no observed value, so there is
  nothing to explain Y by.` `MBPLS` used to fit them with every weight 0, and `predict` then gave
  NaN for every new row. (#1128)
- **`MBPLS.fit` refuses a component once Y is fitted exactly, as `PLS.fit` does.** It raises
  `NotEnoughVarianceError`: `There is no variance left in the data array for Y: cannot compute any
  more components beyond component 1.`, naming the components fitted. It used to fit that component,
  and every later one, along X, with a `SpecificationWarning`. A Y with variation left that X cannot
  reach is still fitted along X, as before. (#1128)
- **`OPLS.fit` refuses a constant response with `NotEnoughVarianceError`:** `There is no variance in
  Y: the response is constant, so there is nothing for X to explain.` It raised a `LinAlgError` with
  an orthogonal component, and with none it predicted NaN for every row. (#1128)
- **`random_state` takes a `numpy.random.Generator` in `MBPLS.select_n_components`, and in
  `PLS.select_n_components`, `PLS.nested_cv` and `PLS.cross_validate`, as every function of the
  package that draws random numbers does.** The folds are drawn from the Generator's next draw of an
  int; an int draws the same folds as before. A Generator raised a `ValueError` from scikit-learn.
  (#1128)
- **`pseudo_validation_set` refuses more components than the data support.** A component fitted
  after there is nothing left in Y that X can explain has a Y-loading of 0, and so no D ratio, and
  every cell of `X_pv` came back NaN; in 2.3.0 that Y-loading was round-off, and `X_pv` could be
  about 1e15 times too large. It now raises `NotEnoughVarianceError`, naming the count the data
  support. (#1128)
