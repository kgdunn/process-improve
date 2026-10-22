- **`analyze_experiment` refuses effects for mixture-process data.** `"effects"` and `"lenth_method"` measure
  each factor between its -1 and +1 settings, so on mixture components beside process factors they reported
  numbers that described no change a formulator can make. They now raise `ValueError` and point to
  `"coefficients"` and `"anova"`.
