- **`MBPCA.transform` and `MBPCA.diagnose` score a new row with a missing cell over the cells it has, as the
  fit scores its own rows.** One missing cell gave the row NaN super and block scores, a NaN T², and an SPE of 0
  in every block, the SPE of a row the model fits exactly. Each score is now the fit's masked regression, block
  by block and then over the blocks the row has a score in, so scoring the training data again gives every row
  its fitted scores and SPE. A block SPE is over the cells the row has in the block, and NaN for a block it has
  nothing observed in; a row with nothing observed at all has NaN scores, SPE and T². A column the fit left out
  (#1098) counts as missing, as in `MBPLS`. On complete rows the values do not change beyond rounding. (#1076)
