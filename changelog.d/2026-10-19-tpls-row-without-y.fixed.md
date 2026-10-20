- **`TPLS.fit` fits a row with every Y value missing, and predicts it (#709).** Such a row (a
  material added before any quality was measured) turned every fitted value NaN and ran each
  component to the 500-iteration cap, about 20 s on eight materials. It now carries no Y
  information, as a row with only some Y values missing already did, so it shapes no weight and is
  predicted from its materials through F. A row that leaves nothing to compute a score from, a
  material with no observed value in D or a row with none in F, now stops the fit at once with a
  `ValueError` that names it, instead of running to the cap and returning NaN everywhere.
