- **`variance_components` refuses a random term that cannot be told apart from the residual.** With one
  reading at each level of a term, such as `part:operator` in a gauge study with one reading per part
  and operator, only the sum of the term's variance and the residual's is estimable. It used to report
  an arbitrary split of that sum, with standard errors near 2e7; now it raises a `ValueError`, before
  fitting, that names the term. (#1004)
