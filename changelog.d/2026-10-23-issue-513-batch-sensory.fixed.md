- **The batch feature functions no longer add columns to the caller's DataFrame (#513).** Without a
  `batch_col` or `phase_col`, every `f_*` feature inserted `__batch_grouper__` and `__phase_grouper__`
  into the frame it was given; they now go on a copy.
- **`f_iqr` omits missing values, as `f_std` and `f_robust_mad` do (#513).** One missing value made
  the whole batch's IQR NaN.
- **`f_area` works on a datetime or timedelta time axis, and measures it in seconds (#513).** The
  widths were timedeltas, so the area could not be stored as a number. A numeric axis is unchanged.
- **`f_crossing` gives NaN for a batch that never crosses the threshold, as documented (#513).** It
  gave an empty array.
