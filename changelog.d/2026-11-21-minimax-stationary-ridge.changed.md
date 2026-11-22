- **A line of saddle centers is a stationary ridge of minimax form** (#971): when an eigenvalue of B is flat (within
  `flat_tolerance`) beside curved ones of both signs and the response has no slope along it, `canonical_analysis` and
  `stationary_point` now give `classification="stationary_ridge"` with `ridge_of="minimax"`, Box and Draper's Table
  11.1 (E); they gave `"saddle_point"`. A saddle that keeps changing along its flat axis is still `"saddle_point"`.
  Table 11.1 (E), recorded as a disagreement, now agrees, and Exercise 11.12's "(E)" is checked with
  `flat_tolerance=0.05`.
