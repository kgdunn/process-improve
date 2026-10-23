- **The batch feature functions no longer add columns to the caller's DataFrame (#513).** Without a
  `batch_col` or `phase_col`, every `f_*` feature inserted `__batch_grouper__` and `__phase_grouper__`
  into the frame it was given; they now go on a copy.
