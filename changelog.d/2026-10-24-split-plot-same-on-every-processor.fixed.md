- **A split-plot design no longer depends on the processor it is built on.** The coordinate exchange
  behind `generate_design(..., hard_to_change=...)` (and `split_plot_design`, `strip_plot_design`)
  climbed from random starts, and a start that could not estimate the model was scored on M plus a
  small ridge, with a condition number of about 1e9. The criteria of its first exchanges then differed
  in the eighth digit from one BLAS kernel to another, so the same `random_state` gave different
  designs on AVX2 and AVX-512 processors: a 48-run I-optimal design with a six-level categorical
  factor had an average prediction variance of 0.640 on one and 0.621 on the other. A start that
  cannot estimate the model is now drawn again, up to 50 times. Designs whose starts all estimated the
  model are unchanged; the others change once, and are then the same on every processor.
