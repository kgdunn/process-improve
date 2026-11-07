- **`TPLS.vip` is finite beside a group with no variation.** One group of D with no variance made the
  VIP of every group of D NaN, with numpy "Mean of empty slice" warnings: the block's R² per
  component, which weighs the components, averaged that group's 0 / 0 in. The group is left out of
  that average, so its VIP is 0, as its loadings are, and the others are finite. (#1169)
