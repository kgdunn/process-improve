- **`budget` sizes a fractional factorial (#956).** `generate_design(factors,
  design_type="fractional_factorial", budget=32)` builds the minimum-aberration fraction with that many
  runs, so sizes between two resolutions, such as 2^(7-2) and 2^(8-3) in 32 runs, no longer need
  hand-written generators. The budget counts centre runs and replicates. A budget that leaves a number
  of runs that is not a power of two warns (`UserWarning`) and rounds down to the largest power of two
  that fits; a budget beyond the half fraction warns and gives the half fraction. With `resolution` as
  well, the fraction must reach it, or a `ValueError` says so; before, a budget with a resolution gave
  the fewest runs that reached the resolution, and it now gives the budget's size (`budget=32,
  resolution=4` for 7 factors: 32 runs, not 16). With `generators`, or when the design type is chosen
  automatically, the budget is a ceiling, as before.
