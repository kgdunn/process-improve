- **`TPLS` pairs the materials of F with the rows of D by name.** `fit` paired each column of a
  group of F with the row of D in the same position, so F that listed the materials of a group in
  another order than D fitted each material with another material's properties, with no error and
  no warning, and `diagnose` then scored the training blends up to 3.2 away from the fit. A model
  fitted to such data now gives other scores, loadings, predictions and limits: the ones it gives
  for F in D's order. F whose materials are already in D's order fits as before. A material of F
  that D does not have, a material of D with no column in F, and a material that D names twice
  are now refused with a `ValueError` that names them; the first was fitted without a word, and the
  second failed on mismatched shapes. A material that no blend uses still fits, with a column of
  zeros in F. (#1197)
