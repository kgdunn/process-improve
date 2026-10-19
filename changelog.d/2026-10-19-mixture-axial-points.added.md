- **Axial check blends for a mixture design.** `augment_design(design, "add_axial_points")` on a
  mixture design (each run's proportions sum to 1) adds its axial check blends, each midway
  between the centroid of the simplex and a vertex: (2/3, 1/6, 1/6) and its permutations for
  three components. The design for the linear Scheffe model with them is Ullmann's axial simplex
  (7 runs), and the simplex centroid design with them is Montgomery's augmented simplex lattice
  (10 runs). `"add_center_points"` adds runs at the centroid. Both work on the whole simplex and
  on an L-pseudocomponent simplex the runs span.
