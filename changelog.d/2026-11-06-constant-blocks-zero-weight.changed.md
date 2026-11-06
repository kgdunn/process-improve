- **An X in which nothing varies is refused across the family.** When every column of every block is
  constant or has no observed value, `PCA`, `PLS` (and `PLSDA` and `PRM`), `OPLS`, `MBPCA`, `MBPLS`,
  `BatchPCA` and `BatchPLS` raise `NotEnoughVarianceError` in the words `MBPLS` used: "There is no
  variance in X: every column is constant or has no observed value, so there is nothing to explain Y
  by." (`PCA` and `MBPCA` end "so there is nothing for a component to describe"; with several blocks
  it reads "every column of every block"). `PCA` and `MBPCA` fitted such an X, with scores that told
  no row apart; `PLS` and `OPLS` refused it with "There is no variance left in the data array for X:
  cannot compute any more components beyond component 0.". The error comes alone, before any warning
  names the columns. `PLS` judges X over the rows its `sample_weight` keeps. (#1144)
- **A block with no variance, beside blocks that vary, gets zero weight, and one warning names it as a
  block.** `MBPCA` and `MBPLS` already fitted it with zero block scores, weights, VIP and
  contributions; the warning listed every one of its columns, and now reads: 'Block X["b"] has no
  variance: every column is constant or has no observed value, so it gets zero weight in the model.
  Drop the block, or check the data for a stuck sensor.' A block with only some constant columns keeps
  the warning that names them. (#1144)
- **`TPLS` fits a group of F or D with no variation with zero weight.** A group whose F has no
  variation over the blends, scaled or used as given (`skip_f_matrix_preprocessing`), or whose D has
  none over its materials, beside a group that varies, takes no part in the model: its
  `r_loadings_f`, `s_loadings_d`, `v_loadings_d` and `p_loadings_f` are 0, so the fit, the
  predictions and the scores `diagnose` gives a new row are those of the model without that group,
  whatever the row holds in it, and its VIP is 0. Such a group of F was refused ("TPLS cannot fit
  component 1: its scores are not finite ..."), and so was a group of D with one material; a constant
  group of D was fitted, with its F deflated. The warning above names the group, as `D["B"]` or
  `F["B"]`. When no group has a column of F and a column of D that vary, the fit is refused, now also
  when Z varies: "There is no variance in F and D: no group has a column of F and a column of D that
  vary, so the blends have no F score to explain Y by. A TPLS model needs one, even when Z varies.".
  `TPLS` names columns with no variance after these refusals, as `MBPLS` does, so a Y with no variance
  gets its error alone. The error for scores that are not finite now ends "Check that each group of F
  covaries with the first Y block, or fit fewer components." (#1144)
