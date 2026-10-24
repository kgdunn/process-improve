- **`signal_to_noise(readings, kind)` works Taguchi's signal-to-noise ratios.** Larger is better, smaller is
  better and nominal is best, in decibels, of one setting's readings or of a table of them with a row for each
  setting; a blank reading is left out. It is exported from `process_improve.experiments`, beside
  `cross_arrays`, and the reference catalogue's `sn_*` statistics use it.
