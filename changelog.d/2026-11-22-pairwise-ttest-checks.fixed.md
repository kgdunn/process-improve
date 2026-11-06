- **`ttest_independent_from_df` and `ttest_paired_from_df` check their arguments with fewer than two
  groups.** The confidence level was checked for each pair of groups, and the `correction` name
  only once there were comparisons to correct, so data with one group returned an empty table for
  `conflevel=95` or `correction="bonferroni"`. Both are now checked first, with the messages the
  other functions give: `ValueError: conflevel must lie in (0, 1); got 95.` and `ValueError:
  correction must be one of 'holm', 'bh', or None; got 'bonferroni'.` (#1169)
