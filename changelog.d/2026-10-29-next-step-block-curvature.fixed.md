- **`recommend_next_step` no longer reads a shift between blocks as curvature when the centre runs are in a block
  of their own.** With the corners in one round and the centre runs in the next (`block="Round"`), the centre runs'
  mean was compared with the corners' and the blocks ignored, so a process that ran 8 units higher in the second
  round showed significant curvature (p = 0.0003) and the step was `augment_to_rsm`. The runs cannot tell that
  shift from curvature, so curvature is now not judged (`"source": None`), a reason in the step says why, and the
  step is the same with the shift as without it (#998).
