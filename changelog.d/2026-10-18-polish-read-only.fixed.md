- **`augment_design(..., "add_runs_optimal")` no longer fails on a custom `target_model` whose runs
  are polished.** Under pandas 3 the model rows patsy builds are read-only, so the coordinate
  exchange that moves runs off the candidate grid raised `ValueError: assignment destination is
  read-only` (seen adding runs to an 8-factor definitive screening design). The exchange now works
  on a copy.
