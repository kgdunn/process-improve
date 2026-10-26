- **`PRM` is now the Partial Robust M-regression of Serneels et al. (2005), step for step.** It cited the paper
  but differed from it in five steps, and its fitted values on the PAC data of Varmuza and Filzmoser were 11 to 15
  (root mean square, y's standard deviation 81) from those of the R package chemometrics, written by one of the
  paper's authors; at its defaults its weights did not settle. Now X is centred once at its L1 median and y at its
  median; the leverage weight measures each row's scores with every component scaled to unit length, so the first
  components no longer decide it alone; the residual weight divides by the median absolute residual itself; the
  loop stops when the norm of the regression on those scores changes by less than `coefficient_tol` (default 1%);
  and the intercept is the median residual. With `scale=False` it reproduces chemometrics's `prm` to 1e-7, once
  that code's one slip is undone: it subtracts the scores' L1 median as `T - mt`, which R recycles down the columns,
  so with two or more components the median lands in the wrong cells. `scale=True` (the default) now divides each
  column by 1.4826 times its median absolute deviation, once, instead of standardising with the weights at every
  iteration. Every number a `PRM` returns changes.
