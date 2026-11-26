- **Minimum-aberration fractions of 12 to 25 factors in 32 runs.** `minimum_aberration_generators`,
  `get_trade_off_table_entry`, `trade_off_table` and `generate_design(design_type="fractional_factorial")` now give
  them, where they raised `SearchLimitError`. The search scores a generator set from the weights of the code its
  columns span (the MacWilliams identities) instead of from its defining relation, which finds the same designs
  many times faster: an exhaustive search of all 10.4 million sets for 18 factors takes seconds. The 32-run designs
  it finds are listed, since the search takes seconds, and each has the word-length pattern Chen, Sun and Wu (1993)
  give (A12 of #788; 64 and 128 runs stay open).
