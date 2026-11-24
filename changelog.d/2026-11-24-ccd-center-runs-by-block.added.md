- **A blocked central composite design takes its center runs block by block.** `generate_design(...,
  design_type="ccd", n_blocks=3, n_center_points=[2, 2, 2])` puts the given center runs in each block, the cube
  blocks first and the axial block last, and the axial distance that blocks the design orthogonally follows from
  them. It builds the designs of the NIST/SEMATECH e-Handbook 5.3.3.6.4, Tables 3.30 to 3.32, with their printed
  axial distances (1.63299, 2 and 2). A number still puts half the center runs, rounded down, with the cube runs.
  The cube blocks take the same number, as orthogonal blocking needs; other lists raise `ValueError` (A11 of #788).
