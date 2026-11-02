- **`MBPLS` gives no SPE where a row has nothing observed in a block, and leaves that row out of
  the block's SPE limit.** Such a row has NaN for its block score and for its `block_spe_` there.
  `diagnose` gave it a block SPE of 0, the SPE of a row the model fits exactly, and
  `block_spe_limit` and `super_spe_limit` were NaN for any model fitted with such a row.
  `diagnose` now gives NaN, `block_spe_limit` is set on the rows that have an SPE in the block,
  and `super_spe_limit` counts a block a row has nothing in as leaving no residual. (#1028)
