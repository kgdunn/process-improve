- **A factorial whose factors are all categorical gets no centre runs by default** (`generate_design`,
  #788 A5). A categorical factor has no centre, so the three default "centre runs" of such a design
  only repeated cells it already had: a 3 x 3 factorial had 12 runs, with three of its diagonal cells
  twice, and a 2 x 2 had 7. It now has 9 (and the 2 x 2 has 4). An explicit `n_center_points` still
  adds runs, and a design with a numeric factor keeps its three centre runs.
