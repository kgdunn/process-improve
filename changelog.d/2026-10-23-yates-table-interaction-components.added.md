- **The reference catalogue checks Yates's working with `yates_table` and `reverse_yates` (#803).** A check of
  kind `yates` reads a column of the algorithm, one of its entries, or one of the sums of squares that check the
  columns. Box, Hunter and Hunter's pilot plant checks 17 numbers that were manual notes: columns (1) to (3) of
  Table 10.5 and its contrast totals 514, 92 and 40, the estimates of Table 10A.1, the checks 34,342 to 274,736,
  and, run in reverse on T, C and TK, the crude sum of squares of the fitted values, 34,332.5, and the checks
  274,660 to 2,197,280. `component_ss` reads its sum of squares from `interaction_components`, and adds up
  `parts`, each a component and its own runs: Davies's unconfounded three-factor interactions, X + Z + W + Y =
  10,796 from the replicates in which each is not confounded, reproduce.
