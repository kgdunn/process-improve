- **`AdaptivePCA` and `AdaptivePLS` read a frame of mixed column types with a missing value.**
  `partial_fit`, `transform`, `predict` and `decompose_prediction` read such a frame row by row,
  and a row holding a nullable column's missing value (`pd.NA`) beside a flag of booleans raised a
  `TypeError`. Each frame is now read whole as floats, a missing value as NaN. (#1029)
