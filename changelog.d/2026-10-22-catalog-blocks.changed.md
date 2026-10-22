- **A catalog record keeps each run's block apart from the design's factors.** `catalog_records` gives a
  published design's blocks, run by run as its source prints them, as `blocks`, and leaves the block column out
  of `factors` and `coded_runs`: a 3^2 in three blocks is described as the 3^2 it is, with 2 factors. A
  reference entry names its block column with `design["blocks"]` when its source calls the blocks something
  else (a batch, an operator, a week); 35 entries now do. Text that names blocks without the design being run
  in them, such as "no blocks", no longer marks a design blocked.
