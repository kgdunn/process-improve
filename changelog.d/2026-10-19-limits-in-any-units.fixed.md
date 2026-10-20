- **A limit reads the same whatever the units of its factors (#807).** Several tests held a tolerance of
  `1e-9` in a limit's own units, which is as wide as a factor's whole range when that range is about
  `1e-9` of its units (a trace mole fraction, a nanomolar concentration in mol/L). The region dropped
  `X <= 0.5e-9`, `DesignRegion.feasible()` passed every setting, D-optimal, I-optimal and space-filling
  designs put runs beyond the limit, and a curved limit on such a factor was read as a straight one.
  Now a straight limit is judged by its distance in coded units, `1e-9`, however it is scaled, and a
  curved one by `1e-9` of the largest value it takes over the box. The test that tells a straight
  limit from a curved one compares its fit with the limit's own values and with the rounding of its
  terms. A setting between `1e-9` and `1e-9` divided by a gentle limit's coded gradient from its
  boundary used to pass and now fails. A comparison whose factors cancel exactly, such as
  `0.1*A + 0.2*A <= 0.3*A`, is read in exact arithmetic, as one that compares numbers.
