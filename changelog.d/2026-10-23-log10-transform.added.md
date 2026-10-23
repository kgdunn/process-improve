- **`analyze_experiment(transform="log10")` fits the common log of the response.** Its F tests and p-values are
  those of `transform="log"`; its sums of squares, coefficients and predictions are on the log10 scale, as
  Design-Expert prints them, so Table 4-4 of Anderson and Whitcomb's *DOE Simplified* reproduces as printed.
  The `analyze_experiment` tool accepts it too.
