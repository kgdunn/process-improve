- **`midcourse_correction` no longer reports `"ok"` while an SPE or T2 cap is broken (#681).** With
  both caps binding, the polishing step lowered one multiplier while checking only its own
  statistic, which could push the other statistic over its cap: 48 of 720 two-cap solves read
  `"ok"` with a cap more than 1% over. A step is now taken only when both caps hold, so `"ok"`
  means both are met. `active_constraints` no longer flags a cap whose statistic ended below it
  with a multiplier bisected to almost nothing: such a cap is active only if dropping its
  multiplier from the schedule returned would break it.
