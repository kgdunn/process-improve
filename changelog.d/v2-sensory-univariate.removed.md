- **`process_improve.sensory.discriminate_observational`.** Deprecated since 1.77.0. Use
  `find_predictive_descriptors`, which takes the same arguments and returns the same result. The old
  name now raises `ImportError` on import and `AttributeError` on attribute access.
- **The `discriminator=` keyword of `relate_observational` and `analyze_descriptive`.** Deprecated
  since 1.77.0. Use `find_predictive=`. Passing `discriminator=` now raises `TypeError`. The
  `sensory_analyze_descriptive` tool no longer declares a `discriminator` field, and because its
  schema forbids unknown keys, a call that sends it now fails validation with
  `ToolInputInvalidError`.
- **The `discriminator` keys in sensory results.** Deprecated since 1.77.0. Read
  `result.relate["predictive_descriptors"]` instead of `result.relate["discriminator"]`, and
  `result.config["find_predictive"]` instead of `result.config["discriminator"]`.
- **`q_value` and `discriminator_significant` on the `find_predictive_descriptors` descriptor
  records.** Deprecated since 1.77.0. Read `p_value_fwer` and `is_predictive`, which carry the same
  values. The `q_value` on the marginal `associations` rows of `relate_observational` is a
  Benjamini-Hochberg q-value and stays.
- **The `"z value"` and `"Pooled standard deviation"` keys of `ttest_independent`.** Deprecated
  since 1.70.0. Read `"t value"` and `"Std error of difference"`, which carry the same values. The
  statistic is a t-statistic, and the second key held the standard error of the difference, not the
  pooled standard deviation, which is `"Pooled std dev"`. The two columns are also gone from
  `ttest_independent_from_df`. Reading an old key now raises `KeyError`. The paired test,
  `ttest_paired`, is unchanged.
- **The `z_value` and `pooled_std` fields of the `ttest_two_samples` tool.** Deprecated since 1.70.0.
  Read `t_value` and `std_error_of_difference`. The `pooled_std_dev` field stays. The
  `ttest_paired_samples` tool is unchanged.
