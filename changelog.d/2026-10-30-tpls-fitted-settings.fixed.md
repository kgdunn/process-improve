- **`TPLS` keeps the settings it was fitted with.** After `set_params` without a refit,
  `display_results` read the new `max_iter`, so a component that had run out of iterations lost
  its "max iter reached" flag (or one that had converged gained it) and the printed setting
  changed, and `diagnose` read the new `skip_f_matrix_preprocessing`, so it stopped centring and
  scaling new F blocks that the model was fitted on scaled, or started doing so. Both now read what
  the fit used, stored in the new `max_iter_` and `skip_f_matrix_preprocessing_`. `MBPCA` and
  `MBPLS` already decide convergence at fit time. (#1028)
