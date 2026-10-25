- **A refused name now comes with a name that works.** A factor, response or column name that a model formula
  cannot hold is refused with a suggestion close to it: `'Water %'` gives `Water_pct`, `'Temp (°C)'` gives
  `Temp_deg_C`, `'2nd stage'` gives `x2nd_stage` and `'lambda'` gives `lambda_`. `analyze_experiment`,
  `evaluate_design`, `augment_design` and `ASCA` refuse every unfit factor name in one error, with a map to paste
  into `DataFrame.rename(columns=...)`; a suggestion that another column already has gets `_2`, `_3`, and so on.
  On factori.al, the factor table offers the suggestion as you type the name.
