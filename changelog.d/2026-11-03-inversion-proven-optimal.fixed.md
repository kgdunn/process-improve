- **`optimal_inversion` reports `proven_optimal=True` for an optimum whose multipliers do not settle.** With many
  inputs at a bound, the binding rows are dependent and the least-squares multipliers can miss at a true optimum, so
  `kkt_residual` stayed above 1e-3 and the answer was reported as not proven. The answer is now also proven when one
  linear program, over the rules, the bounds and each limit's tangent plane at the answer, does no better. The
  restarts that look for a better point stop on the same proof. On the bundled TPLS example, bounded by its
  training range, the cheapest blend under two quality windows (cost 6.955066, `kkt_residual` 0.035) is now
  proven. (#1126)
- **`InversionRules` accepts numpy numbers as bounds and minimum amounts.** A bound taken from an integer column's
  `min()` or `max()` (`numpy.int64`) was refused as not a number. (#1126)
