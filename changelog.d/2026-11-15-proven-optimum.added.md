- `optimize_responses` proves its optimum when every model is a straight line in the factors and every
  constraint is one, which is the usual case for a formulation of many ingredients. The best setting is then
  found by mixed-integer linear programming (HiGHS), not by a search from starting points: it bounds the
  composite desirability of every set of factors in use and returns the setting with the proof that no setting
  of the region scores higher, to one part in a million. Several responses, targets, `in_range` limits,
  weights and importances are covered, with `Factor(min_when_used=...)`, `count_nonzero` limits, limits on
  totals, and blends beside process factors. The desirability result says so in a new key, `proven_optimal`.
  A formulation of 26 or of 45 ingredients, most of them left out of any one blend, is solved in under a
  second; the search it replaces took from 40 seconds to 13 minutes and gave no proof.
- With a model that has an interaction or a squared term, or a curved constraint, the optimum is still searched
  for, and `proven_optimal` is False unless the composite desirability reaches 1. That search is stronger where
  factors can be left out: it starts from the most promising settings of 30 different sets of factors, tries
  the sets next to the best three results, and uses the exact search on the models' tangents at the best
  setting to pick a set that no neighbouring change reaches. Against the best of every allowed set of ten
  components, with blending terms in three responses, it found the best setting in 44 of 44 cases, where the
  search it replaces found it in 8 of the first 12.
