- `compare_designs` reads a column of labels as a categorical factor of two levels, coded -1 and +1 in sorted order.
  It has no square, and I, G and the FDS curves take it at its two levels only, as `evaluate_design` does. A
  `DesignResult` with a two-level categorical factor, such as an optimal design's, can now be compared; before, its
  column of labels was refused as "not numeric". A factor of three levels or more is refused with its count.
