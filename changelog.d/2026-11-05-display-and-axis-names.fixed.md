- **`MBPCA.display_results` and `MBPLS.display_results` line up their columns.** The headers were
  as wide as their text and the values nine characters wide, so with a block name of any other
  length the R² values sat away from their headers. The table is now printed by pandas, with each
  value right-aligned under its header; it has no `|` separators or dashed rule any more.
  `TPLS.display_results` without a Z block printed its `-` placeholder one character narrower than
  a number, which moved the F, Y and timing columns one place to the left; they now line up. (#1083)
- **`TPLS` keeps the names of the axes it labels.** A D block whose index is named, such as `Lot#`
  in the pyphi example data, gave `r_loadings_f`, `p_loadings_f`, `spe["D"]`,
  `feature_importance["F"]` and the F residuals of `diagnose` an unnamed axis. Each now keeps the
  name, and so do the frames labelled by D's columns, Z's columns and Y's columns: `s_loadings_d`,
  `v_loadings_d`, `w_loadings_z`, `p_loadings_z`, `q_loadings_y`, the predictions `hat_` and `hat`,
  and those of `diagnose`. `material_names` and the other `*_names` attributes are still lists.
  (#1083)
