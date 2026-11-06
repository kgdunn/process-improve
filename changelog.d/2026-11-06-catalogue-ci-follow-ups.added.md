- **The reference catalogue checks Montgomery's Bartlett and Kruskal-Wallis tests with the library's own.**
  The six numbers of the etching example that it listed as missing from the library (q, c, χ²0 and P of
  Bartlett's test; H and P of the Kruskal-Wallis test) are now recomputed with `bartlett_test` and
  `kruskal_wallis_test`. A `classical_test` check takes `by`, a column whose values each pick one group of
  runs. Both P-values differ in the last digit, as the book works them from its rounded statistic. (#1100)
