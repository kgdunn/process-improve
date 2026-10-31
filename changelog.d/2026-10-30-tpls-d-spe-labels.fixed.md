- **`TPLS.spe["D"]` is labelled by material.** Each group's SPE was indexed 0, 1, 2, ... rather
  than by the materials of `d_matrix`, because the first deflation turned the residual D block
  into a DataFrame with a fresh index. `d_mats` holds arrays again, as documented. (#1028)
