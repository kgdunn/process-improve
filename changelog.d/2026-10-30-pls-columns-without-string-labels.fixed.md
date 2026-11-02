- **`PLS` reads new columns that carry no string labels by position, whether or not it scales.**
  sklearn records no feature names for integer (or other non-string) column labels, so
  `validate_data`, the internal scaler and `PLS.project` read such columns by position. With
  `scale=False`, `PLS.transform`, `PLS.diagnose` and `PLS.predict` multiplied them by the direct
  weights by label instead: the same integer labels in another order gave the scores of the
  columns by label but an SPE of them by position, which matched neither, and labels the fit had
  not seen raised "matrices are not aligned". All four methods now read new rows the same way.
  Columns with string labels are matched by name, as before. (#1027)
