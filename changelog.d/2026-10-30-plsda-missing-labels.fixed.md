- **`PLSDA` refuses a row with no label**, as `None` or NaN, with a `ValueError` that says how
  many rows have none. A NaN failed with a `KeyError` at the end of the fit, and `None` among text
  labels with a `TypeError` from sorting them. (#1029)
