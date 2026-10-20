- **Two designs set against each other.** `compare_designs(...).relative(design, reference)` gives
  the design's D-, A-, I- and G-efficiencies relative to the reference (each above 1 when the
  design is the better), each coefficient's variance over the reference's, and where the two
  designs' FDS curves cross, with the share of the design space where the design's curve is the
  lower. Every number comes from the information matrices the scores use, X'V^-1 X for a
  split-plot design, and the FDS curves from a seeded uniform sample (`n_points`,
  `random_state`). For Goos and Jones's wind tunnel, the I-optimal split-plot design and the
  second option cross at 0.19 of the design space (the book reads 0.20 off its figure), and the
  second option's relative D-efficiency is 91.4%.
