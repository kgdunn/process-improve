- **`steepest_path` sets out a path of steepest ascent as sources do, and `next_design` centres the
  next design on one of its steps (#798 S1, S2).** `steepest_path(fit, step={"x1": 10}, n_steps=7)`
  steps a round amount of one leading factor, in its own units, and moves the others in proportion to
  their slopes, as Davies (Table 11-22), Box, Hunter and Hunter (Table 15.2) and Schmidt and Launsby
  set their paths out; a number for `step` keeps the distance in coded units that
  `optimize_responses` steps. The path can start away from the centre (`base=`), where it follows the
  slopes there, and goes down with `goal="minimize"`. `next_design(path, at_step, scale={...},
  design_type=...)` makes a design centred on a step, each factor's range kept or rescaled. A model
  flat at the base raises `NoSteepestDirectionError`, a `ValueError`.
