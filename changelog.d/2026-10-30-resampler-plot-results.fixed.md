- **`Resampler.plot_results` now draws every resampled value.** It estimated each density on a
  fixed grid from 0 to 2, so a parameter below 0 or above 2 (a negative coefficient, a VIP of 2.5)
  was left off the plot, in part or in full. The grid now runs from the smallest to the largest
  resampled value. (#1025)
- **`Resampler.plot_results` accepts an accessor that returns an array or a list.** Such
  parameters have integer labels, which the ridgeline plot could not use, so the call failed with a
  `TypeError`. They are now labelled `"0"`, `"1"` and so on. (#1025)
- **`Resampler` states its error for three resampling settings as one sentence pair.** Setting
  `use_jackknife`, `bootstrap_rounds` and `fraction_excluded` together raised a `ValueError` whose
  message printed as a Python tuple of two strings. (#1025)
