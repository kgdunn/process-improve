- **`fit_gaussian_process` takes a noise term for a physical experiment.** With `noise=True` the model adds a
  nugget, the noise variance over the process variance (`noise_ratio`), estimated by maximum likelihood with the
  correlation parameters. Repeated runs at one setting are then allowed, the prediction smooths the runs rather than
  passing through each, and `predict` adds `std_error_new_run`, the standard error of one new run there. Without
  `noise` the fit is the interpolating model, as before (#1374).
