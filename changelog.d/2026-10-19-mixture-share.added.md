- **`DesignRegion.volume_fraction` gives a mixture region's share (#810 item 3).** It raised a
  `ValueError` before. The share is of the blends the components' ranges allow (the whole simplex
  when every range is 0 to 1), and it is exact: the volume of the polytope the limits leave over
  that of the polytope the ranges leave, each the hull of its extreme vertices. A limit `A <= 0.5`
  on three components leaves 0.75 of the simplex.
