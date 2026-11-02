- **`CA.transform` and `CA.transform_columns` place a supplementary point with an empty cell.** The
  check for a point with no profile summed along the wrong axis: one new row with a zero count in
  any column was refused as "sums to zero", and so was every `MCA.transform` of observations that
  leave a level out, while a row of zeros among other rows came back as NaN coordinates. Each row
  (or column) is now checked on its own total. (#1029)
