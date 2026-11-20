- **`fit_gaussian_process`, `variance_components` and `multistratum_anova` are tools** (#1267), so an agent on the
  MCP server and the `doe-designer` skill can fit a Gaussian process to a computer experiment (with predictions
  and each input's sensitivity), estimate how much of a result's spread each random source gives, and test each
  term of a split-plot or strip-plot design against its own stratum, or see a design's skeleton analysis of
  variance before it is run. The Gaussian process takes at most 200 runs, the other two 2,000 rows.
