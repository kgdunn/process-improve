- **NIPALS no longer reports a component that converged on its last allowed pass as not
  converged.** PCA, PLS, MBPLS and MBPCA warned "without converging" whenever a component used all
  `max_iter` passes, even when that last pass met `tol`. The warning, and the `converged` entry of
  `MBPLS` and `MBPCA`'s `fitting_info_`, now follow the convergence test itself. (#1080)
