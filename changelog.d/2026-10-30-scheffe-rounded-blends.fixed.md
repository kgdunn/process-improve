- **`analyze_experiment` fits a mixture model on blends rounded for printing as it does on exact
  blends.** A Scheffé model has no intercept, and its components make up the constant, so R², the
  model df and the overall F are about the mean. On blends that sum to 0.999 or 1.001 (1/3 printed
  as 0.333) the fit lost that constant and reported R², df_model and F about zero: one more model
  df, and an R² and F far larger than the exact blends give. Blends that sum to 1 to within 0.01
  now hold the constant, with a warning that says so; they are also mixture data for the default
  model. A Scheffé model of settings that do not sum to 1 warns that its summary is about zero.
  An explicit formula with no intercept over blends alone, such as
  `"y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 - 1"` or `"y ~ I(x1 + x2 + x3) + x4 - 1"`, whose terms
  make up the constant, is now analysed as the Scheffé model: its ANOVA tests the linear blending
  as one block on q - 1 df, as the `scheffe_*` models do, instead of each component against zero.
  (#964)
