- **`model_selection` searches mixture-process data without an intercept.** Mixture components
  beside process factors sum to 1, so they make up the intercept, as in a Scheffe model. The search
  of a standard model still added one: on the 7 blends of a simplex-centroid design at two settings
  of a process factor `T`, `analysis_type="model_selection"` with `model="main_effects"` selected
  `y ~ x1 + x2`, with an R-squared of 0.73, and the default `"interactions"` model selected
  `y ~ x1 + x2 + T + x1:x2 + x1:T + x2:T`, for a response with an `x1:x2` blending term and `x1:T`
  and `x3:T` terms. A standard model (`"main_effects"`, `"interactions"`, the default, or
  `"quadratic"`) is now the mixture-process model of that order (`mixture_process_model`), and a
  formula without an intercept over the components, such as
  `y ~ -1 + x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + x1:T + x2:T + x3:T`, keeps its linear blending
  terms. Every linear blending term stays in and the search chooses among the others: the
  interactions and quadratic models now select `y ~ -1 + x1 + x2 + x3 + x1:x2 + x1:T + x3:T`
  (R-squared 0.997). A categorical process factor is coded in `L - 1` columns, as
  `analyze_experiment` codes it beside components, so the search scores, and `n_terms` counts, the
  model `analyze_experiment` fits for the selected formula: `T:catalyst` was three columns where it
  fits two. Scheffe models and data without process factors are searched as before. (#1169)
