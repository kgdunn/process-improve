- **`PLSDA`'s `max_iter` defaults to 500**, as on `PLS` and every other iterative estimator since
  #588, which left `PLSDA` at the 1000 that `PLS` had before. A fit that needs more than 500 NIPALS
  iterations now warns, as the same fit by `PLS` does. (#1029)
