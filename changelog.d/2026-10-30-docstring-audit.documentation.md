- Clarified `loading_plot`'s `loadings_type` docstring: the function default is `'p'`
  for every model (not only PCA), and passing anything other than `'p'` to a PCA
  model raises `AttributeError` rather than being silently ignored. The list of
  recognised values is unchanged.
