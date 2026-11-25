- **Minimum-aberration fractions of up to 50 factors in 32, 64 and 128 runs.** Factors after Z are named a, b, c, ...
  (without i), as FrF2 names them, so 50 factors have single-letter names, and the exhaustive search now gives 26 to
  31 factors in 32 runs. The fractions of 12 to 50 factors in 64 and 128 runs, too many for any exhaustive search,
  are read from `datasets/experiments/minimum_aberration.json`, which `scripts/minimum_aberration_table.py` writes
  with an iterated tabu search. Each has the word-length pattern of the minimum-aberration design of the published
  catalogs (Chen, Sun and Wu 1993; Mee 2009; Xu 2009; Block and Mee 2005), which the tests check.
  `minimum_aberration_generators`, `get_trade_off_table_entry`, `trade_off_table` and
  `generate_design(design_type="fractional_factorial")` give them where they raised `SearchLimitError` or the factor
  names ran out, and the `trade_off_table` tool no longer has cells marked "?" (A12 of #788).
