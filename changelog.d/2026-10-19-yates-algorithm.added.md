- **Yates's algorithm, and the algorithm run in reverse.** `yates_table(design, response)` puts the
  runs of a full two-level factorial in standard order and returns the algorithm's columns, each
  effect's estimate and sum of squares, and the columns' sums of squares, each twice the one before,
  as the classical sources print their working. `reverse_yates(table, keep)` runs it backwards on the
  contrasts of the effects a model keeps, and returns the fitted values and residuals with the same
  checks. They reproduce Box, Hunter and Hunter's pilot-plant Tables 10.5, 10A.1 and 10A.2 (#803 H1).
