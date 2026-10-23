- **Each VIP says which variance weights its components (#513).** `vip()` weights by R2X on a PCA
  and by R2Y on a PLS, with one formula for both; `MBPLS.block_vip_` weights by each block's R2X
  while `MBPLS.super_vip_` weights by R2Y; `MBPCA.block_vip_` and `TPLS.vip()` weight by the block's
  own variance. The docstrings now say so.
- **`PCA.diagnose` and `PLS.diagnose` say what shape their `hotellings_t2` is (#513).** PCA returns
  the cumulative T² as a DataFrame with one column per component; PLS, MBPLS and MBPCA return the
  full model's T² as a Series. Both docstrings now give each field's type and shape, and name the
  difference: the Series is the DataFrame's last column.
