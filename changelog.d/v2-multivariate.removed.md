- **`PCA.predict`, `MBPCA.predict`, `MBPLS.predict` and `TPLS.predict`.** Each one forwarded to
  `diagnose`, which returns the same `Bunch`, so call `diagnose` instead. `PLS.predict` and
  `OPLS.predict` are the scikit-learn regressor method and stay.
- **`TPLS.tolerance_`.** Read `TPLS.tol` instead.
- **`permutation_q2` and `pipeline_null`.** Use `check_predictive_signal(x, y, fit_predict)` and
  `count_discoveries_under_null(select, x, y)`; `check_predictive_signal` takes the blocks first.
- **The `empirical_fdr` key of the `count_discoveries_under_null` result.** Read
  `null_to_observed_ratio`, which is the same ratio without the clip to `[0, 1]`.
- **`OPLS(max_iter=..., tol=...)`.** The single-response algorithm is closed form and never read
  them, so leave them out; passing either now raises `TypeError`.
- **The `md_tol` and `md_max_iter` keys of `missing_data_settings` on `PCA`, `PLS`, `MBPCA` and
  `MBPLS`.** Pass `tol` and `max_iter` to the constructor instead. A dict that still holds either
  key raises `ValueError` at `fit`, since ignoring a convergence setting would change the fit.
- **`PCA.select_n_components(n_iter=..., tol=..., threshold=...)`.** Use `ekf_max_iter` and
  `ekf_tol`. `threshold` was already ignored; `selection_rule="q2_increment"` with
  `min_q2_increase` gives a similar preference for fewer components.
- **`PCA.select_n_components(cv_scheme="row_wise")` and its `**pca_kwargs`.** Use `"ekf"`,
  `"ckf"`, `"ek"`, `"sacv"` or `"gcv"`, which the `ValueError` for `"row_wise"` now lists.
  `**pca_kwargs` reached `PCA` only under that scheme, so an unknown keyword now raises `TypeError`
  instead of being ignored.
