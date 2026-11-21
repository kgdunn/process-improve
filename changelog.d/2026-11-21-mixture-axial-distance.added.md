- **Axial check blends at a chosen distance** (#973): `augment_design(design, "add_axial_points",
  axial_distance=r)` on a mixture design places each axial check blend at (1 - (q - 1) r, r, ..., r) on the
  design's simplex, as Box and Draper (Section 16.5) write it, between the vertex (r = 0) and the centroid
  (r = 1/q). The default, midway, is unchanged; `axial_distance` on any other design or augmentation raises
  `ValueError`. Box and Draper's axial blends of Exercises 16.3 and 16.4 and Figure 16.10 are now checked in the
  reference catalogue; two disagree by the book's working from a rounded r.
