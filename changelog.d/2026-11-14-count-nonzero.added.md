- **`count_nonzero(...)` in a constraint limits how many factors a run uses.** A factor at 0 is
  left out, and `"count_nonzero(A, B, C, D) <= 2"`, `"count_nonzero(A, B, C) == 1"` or
  `"8 <= count_nonzero(A, B, C, ...) <= 14"` says how many of the named factors a run may use. It
  counts continuous factors and mixture components whose `low` is 0 or more; one with a
  `min_when_used` is left out or at no less than that minimum. The settings such a rule allows are
  pieces, one for each set of the factors in use, and the optimal designs take their candidates
  piece by piece: every allowed set when the factors that can be left out are 10 or fewer, a sample
  of the sets otherwise. A formulation of 26 ingredients that holds 8 to 14 of them, with limits on
  group totals, gets a 40-run design in 5 seconds; without the rule its runs use 5 to 7. Columns no
  allowed run can estimate, such as the interaction of two factors never used together, are left
  out and listed in `metadata["inestimable_columns"]`. `DesignRegion.feasible`, `broken_limits` and
  `sample` know the rule, `optimize_responses` returns an optimum it allows, `augment_design` adds
  runs inside it, and a design type that cannot follow it is refused.
