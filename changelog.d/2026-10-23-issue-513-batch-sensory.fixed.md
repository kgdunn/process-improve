- **The batch feature functions no longer add columns to the caller's DataFrame (#513).** Without a
  `batch_col` or `phase_col`, every `f_*` feature inserted `__batch_grouper__` and `__phase_grouper__`
  into the frame it was given; they now go on a copy.
- **`f_iqr` omits missing values, as `f_std` and `f_robust_mad` do (#513).** One missing value made
  the whole batch's IQR NaN.
- **`f_area` works on a datetime or timedelta time axis, and measures it in seconds (#513).** The
  widths were timedeltas, so the area could not be stored as a number. A numeric axis is unchanged.
- **`f_crossing` gives NaN for a batch that never crosses the threshold, as documented (#513).** It
  gave an empty array.
- **`batch_dtw` keeps each batch's own key (#513).** With keys of mixed types, such as `1` and
  `"two"`, every key was cast to the type of the last one, so batch `1` came back as `"1"`.
- **`batch_dtw` always runs its first iteration (#513).** A `tolerance` at or above the square root of
  the number of aligned columns skipped the alignment and raised an `IndexError`.
- **`resample_to_reference` refuses a batch of fewer than 2 rows, naming it (#513).** It divided by
  zero and then failed inside scipy with "`x` array is all-nan".
- **`OLS` gives a near-perfect fit its `pi_range_` grid (#513).** Below a residual standard error of
  about 1e-7 the grid was NaN, and `multiple_linear_regression` left out its `pi_range` key; the band
  now has the near-zero width that `prediction_interval()` already gave.
