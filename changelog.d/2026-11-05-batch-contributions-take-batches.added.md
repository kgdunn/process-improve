- **`BatchPCA` and `BatchPLS` contributions take the batches, as `transform` does.**
  `t2_contributions`, `spe_contributions` and `score_contributions` take a batch dictionary, with
  the new `initial_conditions` keyword for its Z block, and unfold and scale it as `transform` reads
  it. A batch dictionary raised a pandas error ("If using all scalar values, you must pass an
  index"). A DataFrame is still taken to be the `unfold_and_scale` output, so existing calls give
  the same numbers (#1144).
