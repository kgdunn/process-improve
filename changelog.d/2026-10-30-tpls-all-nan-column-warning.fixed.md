- **`TPLS` warns when it leaves out a column with no observed value.** A constant column in any
  block (D, F, Z or Y) already gave a `UserWarning` naming the block and the column, but a column
  with every value missing, or with a single observed value, was excluded from the fit without a
  word. Both now give the same warning, worded for their own cause. (#982)
