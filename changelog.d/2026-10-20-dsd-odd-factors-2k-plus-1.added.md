- **`generate_design(..., design_type="dsd", budget=2k + 1)` builds Jones and Nachtsheim's (2011) 2k + 1-run
  design for an odd number of factors (5, 7, 9 or 11, all continuous).** The default definitive screening design
  for an odd k comes from a conference matrix of order k + 1 and has 2k + 3 runs. A budget of 2k + 1 or 2k + 2 runs
  now gives the design of the paper's Figure 1 instead, two runs fewer. Its main effects are still orthogonal to
  every second-order effect but correlated with each other (|r| up to 0.25, 0.167, 0.125 and 0.3), and it has no
  fake factors, so the full second-order model leaves no error degrees of freedom; the design's
  `metadata["note"]` says both. A budget below 2k + 1 still raises, and its message now names the smaller size.
