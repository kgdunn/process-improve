- **A `budget` too small for the design raises a `ValueError` (#957).** Optimal, mixture and
  mixture-process designs used to raise such a budget and log a warning, so a caller who set the run
  count as a limit got a larger design than asked for, with no Python warning to catch. They now refuse
  it, and say how many runs are needed, when the budget is below the model's number of coefficients,
  when it leaves too few runs for the block effects of `n_blocks`, or when it leaves too few new runs
  for the coefficients that `fixed_runs` do not estimate. A budget left at its default is still sized
  to the design, and now counts the block effects as well. `metadata["budget_requested"]`, which
  recorded the raised budget, is no longer set by these designs. The browser app refuses such a run
  count in the same way, where it used to build the larger design and add a note saying so.
  `augment_design` still adds the runs an augmentation needs, and says so in its explanation.
