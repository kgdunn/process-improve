- **`get_trade_off_table_entry` no longer lists a defining relation of more than 1,023 words.** A design of more
  than ten generators, such as 15 factors in 16 runs or 25 factors in 32 runs, gets its resolution from its
  word-length pattern, and its `defining_relation` and `aliases` are empty, as `generate_design` already leaves
  such a relation out of its metadata. Listing the 2^20 - 1 words of 25 factors in 32 runs, with their alias chains,
  took hours; it now takes under a second. The printed report says how many words the relation has.
