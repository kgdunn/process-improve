- **Hard limits on a factor's range.** `Factor(..., hard_limits=True)` marks a factor's `low` and `high` as limits of the
  process, not only the range studied, and every design tool keeps runs inside them.
  `generate_design` makes a CCD face-centred by default and refuses a design that would
  set the factor past its limits. `augment_design(..., hard_limits=[...])` keeps axial runs
  on the faces of the cube and refuses an `alpha` above 1. `recommend_strategy` names the
  limits in its assumptions. `recommend_next_step` never moves or widens such a range: the
  path of steepest ascent moves only the other factors, and the reasons say which factors
  would improve past their limits. The MCP tools take `hard_limits` in each factor's
  specification, and `augment_design` takes the list of names.
