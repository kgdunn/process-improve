- **`Factor(min_when_used=...)`: an ingredient that is left out, or used at no less than a minimum.**
  A continuous factor with `low=0` and `min_when_used=5` is 0 or between 5 and `high`, which keeps
  traces out of a formulation. `generate_design` then chooses a D-optimal design when none is asked
  for, and the candidate exchange places no run between 0 and the minimum: a grid level that falls
  there moves up to the minimum, so a quadratic model still gets three levels, and candidates,
  vertices and polishing moves that fall there are dropped. A classical design that would set the
  factor there, in a run or a centre point, is refused with the factor's name; `fixed_runs` are kept
  as given. `DesignRegion.feasible` is False for such a setting, `broken_limits` writes it as
  `"A = 0 or A >= 5"`, and the region a design records carries the minimum.
