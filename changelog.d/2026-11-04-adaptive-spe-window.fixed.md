- **`AdaptivePCA` and `AdaptivePLS` refuse a `spe_limit_window` too short for the rolling SPE limit.**
  The rolling limit is refitted only once the window holds `max(20, n_components + 2)` SPE values, as
  fewer give an unreliable limit. A shorter window never holds that many, so with
  `adaptive_spe_limit=True` the limit stayed at its training value for good, with no sign of it. `fit`
  now raises `ValueError: spe_limit_window must be >= 20 (...); got 10.` for such a window, whether or not
  `adaptive_spe_limit` is set. A window that was long enough gives the same limits as before. (#1082)
- **`AdaptivePCA.partial_fit` and `AdaptivePLS.partial_fit` number the rows of an array on from the last
  label.** Each call numbered its rows from 0 again, so after two calls `scores_`, `spe_`,
  `hotellings_t2_`, `distance_` and the other history frames had duplicate index labels. A row given no
  label, from an array or from `update` without `label`, is now numbered one past the last label when that
  is an integer, and by the number of rows recorded so far when it is not. A DataFrame's row labels are
  recorded as before. (#1082)
