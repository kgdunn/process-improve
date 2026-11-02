- **`MBPCA`, `MBPLS` and `TPLS` work from the components they fitted.** After
  `set_params(n_components=k)` without a refit, `transform`, `diagnose`, `predict`, the score
  contributions, the plots, `display_results` and `randomization_test_mbpls` read `k` instead of
  the fitted count: a larger `k` raised `IndexError`, and a smaller one raised `ValueError` or, in
  `MBPCA`, scored new rows on fewer components than the model has. `TPLS.hotellings_t2_limit` and
  `TPLS.ellipse_coordinates` moved to a `k`-component model's without a word. Each now reads the
  fitted `n_components_`, as `PCA` and `PLS` do; `TPLS` gains that attribute. (#1028)
