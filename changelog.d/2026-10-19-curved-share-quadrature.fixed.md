- **A curved limit's share is integrated, not sampled.** `DesignRegion.volume_fraction` now
  integrates the share when one factor runs through every limit of a group that a curved limit
  ties together: a curved limit on one or two factors, or one beside a straight limit that shares a
  factor with it. At each value of that factor, each other factor's allowed length is found along a
  line across the box (exactly where a straight limit bounds it), and their product is integrated by
  adaptive Gauss-Legendre quadrature, between the points where a limit meets the end of such a line.
  The share is good to about 1e-10, where the fixed sample of 16384 points was within about 0.005:
  the guide's `T*F <= 3600` removes 0.041404, which the sample read as 0.041626, and a band
  `|A + B| <= 0.0001` beside `A*C <= 0` keeps 4.999875e-05, which the sample read as 0. A curved
  limit on three factors is still sampled (#810 item 1).
