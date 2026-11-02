- **`TPLS` fits with a loose `tol`.** The loop of each component tested its tolerance before its
  first pass, against a previous score primed as the seed plus 1, whose relative change is about
  0.7 for a centred and scaled Y column. With a `tol` above that the loop never began and `fit`
  raised `UnboundLocalError`. The first pass now always runs, as in `PCA`, `PLS`, `MBPCA` and
  `MBPLS`. (#1028)
