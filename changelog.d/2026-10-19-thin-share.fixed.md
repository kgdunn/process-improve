- **`DesignRegion.volume_fraction` no longer reads 0 for a thin region beside a curved limit (#810
  item 2).** When the straight limits leave a region too thin for any point of the fixed sample,
  such as `T >= 149.9999` with T from 100 to 150, the curved limit's part is now counted over points
  spread uniformly through that region, so `["T >= 149.9999", "T * D <= 6000"]` keeps 1.0e-6 of the
  box (half of the straight part's 2e-6) instead of 0. The points are fixed, so the share is the
  same on every run. Regions of more than five factors keep the fraction of the box's points.
