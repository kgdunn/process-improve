- **`ttest_paired`'s `"z value"` and `"Standard deviation"`**, which hold the t statistic and the
  standard error of the mean difference, are deprecated since 2.1.0 and will be removed in 3.0; use
  `"t value"` and `"Std error of mean difference"`. Reading either key by name gives a
  `DeprecationWarning`; reading the whole result, as `dict(result)` or `json.dumps(result)` do, does
  not. The same two columns of `ttest_paired_from_df`, and the `ttest_paired_samples` tool's
  `z_value`, are deprecated on the same schedule.
