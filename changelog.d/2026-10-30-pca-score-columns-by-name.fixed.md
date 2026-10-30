- **`PCA.score` matches a DataFrame's columns by name, as `transform` does.** It took the scores
  from `transform`, which puts a DataFrame's columns in the fitted order, but subtracted their
  reconstruction from the DataFrame as given, so a block with its columns in another order was
  scored against the wrong variables (-4.84 for a block whose score is -0.82). `PCA.diagnose` and
  `PCA.score` also raised `TypeError` on a pandas nullable column holding `pd.NA`, which
  `transform` and `project` read as a missing value; all four now read new rows the same way.
  (#1026)
