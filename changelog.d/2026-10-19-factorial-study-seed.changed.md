- **Each Factorial study puts its runs in random order with a seed of its own.** Every design
  used seed 42, so every reader got the same run order. A new study now takes the second it
  began, as a Unix timestamp, and keeps it while designs are tried in its place; the designs the
  comparison offers are made with it too. Each set of runs added to a study is drawn with the
  seed plus the number of runs before it. The report and the Python code name the study's seed.
  Studies saved before keep seed 42, as does the worked example.
