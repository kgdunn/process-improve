- **A `budget` below the model's number of coefficients raises a `ValueError` (#957).** Optimal,
  mixture and mixture-process designs used to raise such a budget to the coefficient count and log a
  warning, so a caller who set the run count as a limit got a larger design than asked for, with no
  Python warning to catch. They now refuse it and name the coefficient count. A budget left at its
  default is still raised to the coefficient count, since nobody chose that number. With `fixed_runs`,
  a budget that leaves too few new runs, and blocks that add block effects, are still raised with a
  warning, as before.
  The browser app refuses such a run count in the same way, where it used to build the larger design
  and add a note saying so.
