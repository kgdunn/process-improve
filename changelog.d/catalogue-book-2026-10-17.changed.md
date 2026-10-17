- **The book's reference catalogue reads Process Improvement using Data 2026.10.17.** Its quotations and
  line references follow the book to commit 28e5b88. The worst-case prediction variances of the design
  chapter are now the maxima over the whole cube, searched for with `evaluate_design(...,
  max_search="optimize")` as the book now computes them (the thirteen-run OMARS design's G is 13.00, not
  the sample's 12.50), and the worked OMARS study's numbers follow designs returned in one orientation. A
  `design_metric` check takes `"times_runs": true`, to compare the scaled prediction variance G, the run
  count times the largest prediction variance.
