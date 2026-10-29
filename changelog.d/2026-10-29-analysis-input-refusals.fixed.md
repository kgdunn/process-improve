- **`analyze_experiment` refuses three inputs by name that used to fail deep inside numpy or statsmodels:**
  data with no complete run (#1000); a zero in a binomial GLM's trials *column*, which was refused as a
  number or a list but not as a column (#1001); and a response so large in its units that its sums of
  squares overflow, with the advice to rescale it, which changes no test or R-squared (#1005).
- **A response named after a Python keyword, such as `yield`, is analysed when the data also has a column
  `yield_`.** Its stand-in name became `yield__`, which the formula check refused as dunder access (#1002).
