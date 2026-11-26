- **`backend="pyoptex"` is deprecated** in `generate_design` and the optimal-design dispatchers: it raises a
  `DeprecationWarning` since 2.7.0 and will be removed in 3.0. The default, `backend="auto"`, builds every
  optimal design with the built-in engines, split-plot designs included, so a call that leaves `backend` alone
  is unaffected.
