- **`process_improve.workbench`: a recipe that refines a table step by step and writes the Python that repeats
  it.** A `Recipe` is a source file (`SourceFile`, with its SHA-256 digest) and a list of steps: drop, keep or
  rename columns, drop rows with a reason, keep the rows where a condition holds, add a computed column,
  transform a column (log, log10, sqrt, reciprocal, logit, signed log, signed power), read text as numbers, and
  identify the rows by a column. `Recipe.apply` runs it, `Recipe.to_python` writes a pandas script that rebuilds
  the same table bit for bit (and refuses a different file of the same name), and `Recipe.to_dict` /
  `Recipe.from_dict` round-trip it through JSON. Computed columns and conditions use a small expression language
  that is parsed and never evaluated as Python, so a recipe read from someone else's file stays data.
