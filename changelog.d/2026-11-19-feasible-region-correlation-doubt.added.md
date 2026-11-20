- **`feasible_region` can count the doubt in an estimated error correlation.** The new keyword arguments
  `correlation_degrees_of_freedom` and `random_state` go with `residual_correlation`: each of the integration's
  draws then takes a correlation of its own from the inverse Wishart posterior of the error covariance, so the chance
  that one new run meets every window is averaged over what the residuals leave possible. Taking an estimate as known
  understates that chance, by up to a few points with three responses and five degrees of freedom. Without the new
  argument nothing changes. The browser app (factori.al) passes it for a specification of several responses, and
  the Python it shows does too.
