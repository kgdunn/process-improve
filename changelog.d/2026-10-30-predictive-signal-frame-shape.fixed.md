- **`check_predictive_signal` checks the shape of a `fit_predict` that returns a DataFrame.** Only
  an array was checked: a DataFrame with too few columns failed with an `IndexError`, and one with
  too many rows or columns with a broadcasting error, rather than the documented `ValueError` that
  names both shapes. A DataFrame is now compared with the response row by row and column by column
  as an array is, so its labels are not read. (#1025)
