- **A factor whose name Python normalises can appear in a limit.** Python's parser writes every name
  in its Unicode NFKC form, so a factor named with a decomposed accent (`e` and a combining acute,
  as macOS and some keyboards type it) or a full-width letter (from an East Asian input method)
  never matched its own name in a limit, and the refusal named two strings that print alike.
  `parse_constraint`, `DesignRegion`, `rename_factors` and Factorial's limits now match each name
  through its NFKC form, keep the name as it was given, and refuse two factors whose names share one.
  A refusal shows the code points of a name that is not ASCII.
