- **Blocking a fraction warns when the blocks take a two-factor interaction through an alias** (#788):
  `generate_design(..., n_blocks=...)`, `generate_design(..., block_generators=...)` and
  `augment_design(..., "add_blocks")` raise a `UserWarning` that names each word confounded with blocks that the
  fraction aliases with a two-factor interaction, such as ABC with DE for the resolution V half fraction of five
  factors in two blocks, which Box and Draper's Table 5.14 marks "Not available". Before, only
  `metadata["blocking"]["aliases"]` held it. The `"add_blocks"` explanation now names the aliases too. The blocks
  themselves are unchanged.
