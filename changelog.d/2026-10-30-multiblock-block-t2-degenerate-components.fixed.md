- **A multiblock model's block T² leaves out a block component with no spread.** `MBPCA` and
  `MBPLS` divided each block score by its second moment however small, so when a block had no
  variation left for a later component (block scores with a second moment near 1e-32), that
  component added noise over noise, of order 1, to every row of `block_hotellings_t2_`. A block
  component is now left out by the rule every other T² follows: its score standard deviation is at
  most 1.5e-8 times the largest of the block's. Models without such a component give the same
  numbers as before. (#1028)
