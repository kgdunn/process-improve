- **`generate_design(..., block_generators=[...])`: a two-level factorial blocked by the
  interaction words the user gives** (#788 A8). Until now the library chose the words itself, so
  a published blocking could not be rebuilt. `block_generators=["ADE", "BCE"]` makes four blocks
  with ADE, BCE and their product ABCD confounded with them, as in Montgomery's 2^5 example; names
  longer than one letter are joined with `*`. Words that confound a main effect (directly or
  through an alias), depend on each other, or lie in the fraction's defining relation are refused.
  For a fraction, `metadata["blocking"]["aliases"]` now lists each confounded word's aliases,
  which go with it to the blocks (ABCE with ADEF in Montgomery's photoresist half fraction), for
  the library's own choice of words as well.
