- **A benchmark for whole studies, `process_improve.experiments.benchmark`.** `run_study` follows
  `recommend_next_step` round by round on a process whose true model is known, makes each round's
  runs on it, and reports where the study ended: the actions, the runs used, the final settings,
  and how far they fall short of the true best, in noise standard deviations. `run_benchmark`
  repeats every case over many noise seeds and returns one summary row per case.
  `scripts/doe_benchmark.py` prints the table. The cases (`CASES`) include the adhesion problem of
  the "AI agents in DoE" study, processes hypothesised from coating chemistry for its Problems 02
  and 03 (not the study's hidden models), and synthetic cases for an interior optimum, curvature
  in eight factors, an optimum outside the ranges, and noise only.
