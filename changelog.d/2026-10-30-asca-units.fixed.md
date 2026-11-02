- **`ASCA` gives the same answer in any units.** With `scale=True` a column whose standard deviation
  was below 1.5e-8, in small units, was taken for a constant one and left unscaled, which moved
  every term's share; columns are now scaled as `MCUVScaler` scales them, constant only when all
  their values are equal. And `vasca` set every z-score to 0 where the null's spread was below
  1.5e-8 in absolute terms, so in small units it selected the first subset to pass instead of the
  one furthest above its null; the spread is now judged against the null's own size. (#1029)
