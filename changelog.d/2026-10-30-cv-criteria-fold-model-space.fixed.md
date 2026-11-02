- **`compare_cv_criteria` and `pseudo_validation_set` read each held-out row in its fold model's
  own space.** Both scored the held-out rows with the fold's autoscaling alone, but a fold model
  may centre and scale its rows again: `PRM`, which inherits both class methods from `PLS`, always
  centres at its robust centre and scales by its robust spread. `PRM.compare_cv_criteria` therefore
  reported a PRESS of 72.67 where its fold models' own predictions give 73.11, and every criterion
  built on the held-out scores moved with it. `PRM.pseudo_validation_set` also mapped `X_pv` back to
  the data's units with an autoscaler fitted to X rather than the PRM's own scaler, and fitted its
  fold models in that autoscaled space, so the full-data model gave `X_pv` scores off by up to 5.6
  where they should equal the D-scaled fold scores. The rows now pass through both scalings, and
  the covariance permutation test reads the blocks in the full-data model's space. `PLS` gives the
  same numbers as before: its own scaling of rows already autoscaled is the identity, to rounding.
  (#1029)
