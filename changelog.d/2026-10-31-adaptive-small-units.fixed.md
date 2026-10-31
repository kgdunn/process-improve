- **`AdaptivePCA` and `AdaptivePLS` keep a variable measured in small units.** Each update set a
  moving scale below 1.5e-8 to 1, taking it for a scale that had collapsed to zero, so a tag (or,
  for `AdaptivePLS`, a response) in units 1e9 times smaller dropped out of the model after the
  first update: on a test stream its Hotelling's T2 moved by up to 4.3, and the predictions of a
  response in such units were off by some 2e9 of its standard deviations. A scale is now set to 1
  only when it is exactly 0, as `MCUVScaler` scales a column of equal values, in any units. (#1029)
