- **The prediction variance over a candidate set.** `evaluate_design(..., region=points)`, with
  `points` a DataFrame of one row per point and a column per factor, takes the region-based
  metrics (`average_prediction_variance`, `g_efficiency`, `fds`) over exactly those points, each
  counted once, with no sampling. This is how NIST reports the I-efficiency of a design chosen
  from candidates: its 12-run D-optimal design gives 12 x 0.388542 = 4.6625 over the 20 points of
  Table 5.1, where the cube gives 0.2555. `evaluate_all` takes the same region.
