- `optimize_responses` no longer estimates the gradient of a straight constraint by finite differences, a call of
  the constraint per factor at every iteration, which took nearly all of a search's time over a formulation
  with a limit for each group total. A straight constraint now carries its exact gradient, and a search is
  given only the constraints that a factor free to move enters.
- `optimize_responses` no longer reads a `factor_ranges` entry for a mixture component of `region` as a coded
  range. It reported an `optimal_actual` for the blend that did not sum to 1; a proportion is the same on
  either scale, so `optimal_actual` now repeats the blend.
