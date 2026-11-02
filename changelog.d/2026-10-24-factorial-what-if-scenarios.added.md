- **Factorial: settings kept from the "What if" panel under a name (#899).** "Keep these settings" adds the panel's
  settings to the study under the name typed, or "Settings 1" and on, and Learn lists them below the panel with each
  response's prediction and intervals there. Only the settings are kept: the predictions are worked out again from
  the models and the results each time, so they follow a result that changes. Kept settings that a study edited
  since no longer allows (a factor they do not set, a setting outside its range, a level the factor lost, a limit
  they break) say so instead of predicting. Each can become three confirmation runs at the end of the run sheet, or
  be deleted. They go with the study through this browser, the study file, a link (with its results or without) and
  the workbook, and into the report and the panel's Python code. A study with kept settings is saved in study
  format 5, so an earlier Factorial refuses it as saved by a later one rather than dropping them on its next save; a
  study without keeps the format it had.
