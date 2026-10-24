- **`process_improve.workbench`: a recipe that refines a table step by step and writes the Python that repeats
  it.** A `Recipe` is a source file (`SourceFile`, with its SHA-256 digest) and a list of steps: drop, keep or
  rename columns, drop rows with a reason, keep the rows where a condition holds, add a computed column,
  transform a column (log, log10, sqrt, reciprocal, logit, signed log, signed power), read text as numbers, and
  identify the rows by a column. `Recipe.apply` runs it, `Recipe.to_python` writes a pandas script that rebuilds
  the same table bit for bit (and refuses a different file of the same name), and `Recipe.to_dict` /
  `Recipe.from_dict` round-trip it through JSON. Computed columns and conditions use a small expression language
  that is parsed and never evaluated as Python, so a recipe read from someone else's file stays data.
- **`process_improve.workbench.Variant`: a PCA or PLS model over a recipe's table that excludes rows without
  losing them.** An excluded row is left out of the fit and of its centring and scaling, then projected through
  the fitted model, so `Fit.rows()` still reports its scores, Hotelling's T2 and SPE, flagged. `Variant.to_python`
  writes the model code that follows the recipe's script. `Fit.explain_group` answers what a cluster on a score
  plot asks: per component, each variable's share of the difference between the group's average score and the
  rest's (they add up to that difference), the variables to look at first, their means in the table's own units,
  and their raw values in run order to draw beside the bars.
- **`process_improve.workbench.Project`: one zip file with a source file, its recipe, its model variants and the
  reader's saved selections.** It stores the source file's own bytes, a `manifest.json`, the recipe as a script
  to run next to the source, and each variant's fitted scores, loadings, T2 and SPE as `.npy` arrays. Opening
  executes nothing from the file (JSON checked field by field, and `numpy.load(allow_pickle=False)`), and
  `Project.open` refits every variant and compares it with the stored arrays, naming any that no longer
  reproduce.
