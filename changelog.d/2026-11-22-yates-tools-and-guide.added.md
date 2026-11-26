- **`yates_table_3k`: Yates's algorithm for a full three-level factorial.** Each pass takes successive
  triples and writes their sums, linear and quadratic contrasts; the table gives each of the
  3^k - 1 single-degree-of-freedom effects (`"A"`, `"A^2"`, `"A:B"`, `"A^2:B"`, ...) with its divisor
  z'z, its orthogonal effect, its sum of squares, its standardized component and its normal score.
  It reproduces Box and Draper's Table 7G.1 and the columns of Table 7G.2. (#803)
- **The factorial-analysis tables are agent tools.** `yates_table` works Yates's algorithm on a full
  2^k or 3^k factorial, and with `keep` runs a 2^k in reverse for a model's fitted values;
  `interaction_components` splits the interactions of up to six three-level factors into their
  two-degree-of-freedom components. The MCP server and `skills/doe-designer/` can now call them. (#1097)
