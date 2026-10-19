- **`error_transmission`: the variance a quantity carries from the errors of the measurements it is
  worked out from** (`process_improve.univariate`, #801 F4). It linearizes the function at the
  inputs' means, as Box, Hunter and Hunter's section 17.2 does, and returns the variance and its
  square root, each input's slope and the variance it transmits (with its share, which says which
  measurement to improve first), and the linearity check: the function at each input's mean plus
  and minus three standard deviations, beside the straight line's change. With a fitted model and a
  noise factor it gives Montgomery's propagation of error for the filtration example (Example 12.1):
  4.75 at the recommended setting.
