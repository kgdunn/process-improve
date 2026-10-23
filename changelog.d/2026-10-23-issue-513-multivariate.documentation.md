- **Each VIP says which variance weights its components (#513).** `vip()` weights by R2X on a PCA
  and by R2Y on a PLS, with one formula for both; `MBPLS.block_vip_` weights by each block's R2X
  while `MBPLS.super_vip_` weights by R2Y; `MBPCA.block_vip_` and `TPLS.vip()` weight by the block's
  own variance. The docstrings now say so.
