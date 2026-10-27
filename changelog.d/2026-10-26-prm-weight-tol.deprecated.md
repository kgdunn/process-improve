- **`PRM(weight_tol=...)` is deprecated and has no effect.** `PRM` stops, as Serneels et al. (2005) do, on the
  relative change in the norm of its regression on the scores, set by the new `coefficient_tol`. Passing
  `weight_tol` warns with a `DeprecationWarning`; it will be removed in 3.0.
