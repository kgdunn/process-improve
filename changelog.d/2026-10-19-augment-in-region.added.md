- **`augment_design(..., region=...)` keeps every new run inside the limits on the settings** (#809). The
  region is a `DesignRegion` of the design's factors, or its dict, such as `metadata["region"]` of a design
  `generate_design` made. `"add_runs_optimal"` chooses its D-optimal runs among the settings the limits allow,
  with the existing runs held fixed, and reaches the D-criterion that `generate_design` reaches with those runs
  as `fixed_runs`. `"add_center_points"` puts the centre runs at the region's central setting when the centre of
  the box breaks a limit. The region's ranges are limits, so axial runs sit on the faces of the cube, as with
  `hard_limits`. Any other new run that breaks a limit (a folded, axial or centre run, or a copy) is refused,
  and the message names the run and the limit. Existing runs outside the region are kept, with a warning, and
  listed under `existing_runs_outside_region`. The `augment_design` tool takes the region too.
- **`DesignRegion.broken_limits(x)`** lists, for each point, the factor ranges it leaves and the limits it
  breaks.
