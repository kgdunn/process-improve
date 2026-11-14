- `optimize_responses` searches a constrained region several times faster when its constraints are straight
  lines, as limits on group totals and a blend's sum to 1 are. SLSQP estimated the gradient of every
  constraint by finite differences, a call of the constraint per factor at every iteration; it is now given
  the exact gradient, and a search is given only the constraints that a factor free to move enters. A
  formulation of some tens of ingredients, most of them left out of any one blend, was searched in minutes
  and is searched in seconds, with the same optimum.
- `optimize_responses` no longer reads a `factor_ranges` entry for a mixture component of `region` as a coded
  range. It reported an `optimal_actual` for the blend that did not sum to 1; a proportion is the same on
  either scale, so `optimal_actual` now repeats the blend.
