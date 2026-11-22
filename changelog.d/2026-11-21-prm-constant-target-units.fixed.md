- **A constant target no longer changes how an unscaled PRM fits the others** (#1092). `PRM(scale=False)` gave a
  target with no spread a unit of 1 in Y's units, and the floor of its convergence test followed the largest unit.
  Beside a target in units 1e9 times smaller, the floor was a billion times too high: on 30 rows the reweighting
  stopped after 3 iterations instead of 9, with other weights and coefficients. The floor is now taken over the
  targets that vary, so the fit of the others is the same with or without the constant one, in any units. A fit
  with no constant target is unchanged.
