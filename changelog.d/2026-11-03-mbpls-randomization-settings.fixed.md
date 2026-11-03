- **`randomization_test_mbpls` refits each permutation with the fitted model's settings.** It copied the
  model's number of components only, and refitted with the defaults for `tol`, `max_iter`, `algorithm` and
  `missing_data_settings`, so a model fitted with other settings was compared with the null of another
  estimator. Each permuted model is now a clone of the model under test, with only the rows of Y changed. A
  model fitted with the defaults gets the same `risk_pct` as before. (#1072)
