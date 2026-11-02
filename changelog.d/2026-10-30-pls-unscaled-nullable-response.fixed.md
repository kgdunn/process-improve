- **`PLS(scale=False).fit` reads a pandas missing value (`pd.NA`) in the response as a missing
  cell.** With `scale=True` the internal scaler converted the responses to floats; without it
  nothing did, and a nullable `Float64` or `Int64` response column holding `pd.NA` raised
  `TypeError` where NIPALS would have skipped the cell. The responses are now read as floats,
  as X already was. (#1027)
