- **`mva_fit` returns the cross-validated Q² as lists.** `cross_validation.q2_cumulative` and
  `q2_standard_error` were pandas Series, which are not JSON: `json.dumps` of the result raised
  `TypeError`, and the MCP server sent each one as the Series' printed text. With
  `cross_validate=True`, the default, or `n_components="auto"`, both are now lists of floats, one per
  number of components from 1, as `r2x_per_component` is. (#952)
